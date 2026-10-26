# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``sampler_tail``: the on-device sampler (``ttnn/device_sampler.py``'s composite) as one program on one core.

One data-movement kernel reproduces the composite's arithmetic on the RISC bit for bit (``kernels/sample.cpp``), plus
the presence penalty on the ids each row has emitted; :func:`device_sampler_reference` is the host model. The token is
lane r of the fp32 TILE token row. The guard row's lane 0 holds the rows' guard mask (bit r: row r's draw could depend
on a token it did not read); the host samples such a row over the full vocabulary instead.

The history is a per-row bit array over the vocabulary (uint32 ROW_MAJOR ``[1,1,rows,HIST_WORDS]``) in an L1 shard on
the program's core: the kernel sets the bit of the token it draws; the host zeroes it at request start and rewrites it
when the emitted token is not the device's draw. Frequency and repetition penalties keep the host loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

import ttnn
from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
)

from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

# ttnn/device_sampler.py and ttnn/embedding.py import the model stack, which imports this package: their names are
# pinned here and the modules imported inside the functions that need them.
NAME = "sampler_tail"
KERNEL = fp.kernel_source(NAME, "sample.cpp")
CB_STAGE = 0
MAX_ROWS = 5  # the decode step's row, or the MTP verify rows (k + 1)
LANES = 128  # candidates per row: four shards of 32 (device_sampler.LANES)
MAX_TOP_K = 32  # device_sampler.MAX_TOP_K
TABLE_SIZE = 65536  # device_sampler.TABLE_SIZE
VOCAB_SIZE = 248320  # embedding.VOCAB_SIZE
ROW_LANES = 2 * LANES  # 256: values and ids of the four shards (SAMPLING_CANDIDATE_ROW_SHAPE[-1])
TOKEN_ROW_SHAPE = (1, 1, 1, 32)  # embedding.TOKEN_ROW_SHAPE
HIST_WORDS = -(-VOCAB_SIZE // 32)  # 7,760 uint32 words of bits per row
SCALAR_LANES = 16  # the policy row and the draws row: one 64-byte page each
POLICY_TOP_K, POLICY_TOP_P, POLICY_MIN_WEIGHT, POLICY_GREEDY, POLICY_PRESENCE = range(5)
GRAIN = 64
TILE_BYTES = 4096


def _mapper(mesh):
    """The replicating mesh mapper of the served 1x4 mesh; a one-chip test mesh takes plain uploads."""

    return replicate_tensor_2d_mesh_mapper(mesh) if mesh.get_num_devices() > 1 else None


def _device_sampler():
    from models.demos.blackhole.qwen38_flash_next.ttnn import device_sampler

    return device_sampler


def _metadata(tensor) -> str:
    return f"{tensor.dtype} {tensor.layout} {tuple(int(v) for v in tensor.shape)}"


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def stage_bytes(rows: int) -> int:
    """The kernel's L1 stage size; must match sample.cpp's STAGE_BYTES."""

    work_words = 3 * LANES + 5 * MAX_TOP_K
    return TILE_BYTES + rows * ROW_LANES * 4 + 3 * GRAIN + MAX_TOP_K * GRAIN + work_words * 4


def history_image(rows: int, emitted: list[list[int]]) -> torch.Tensor:
    """The bit array of ``emitted`` (row r's emitted token ids) as the host writes it: uint32 ``[1,1,rows,HIST_WORDS]``."""

    if len(emitted) != rows:
        raise ValueError(f"history image needs {rows} rows of tokens, got {len(emitted)}")
    words = torch.zeros(rows, HIST_WORDS, dtype=torch.int64)
    for r, tokens in enumerate(emitted):
        for token in tokens:
            if not 0 <= int(token) < VOCAB_SIZE:
                raise ValueError(f"token {token} outside the vocabulary")
            words[r, int(token) >> 5] |= 1 << (int(token) & 31)
    return words.to(torch.int32).reshape(1, 1, rows, HIST_WORDS)  # the bit pattern; uint32 on the device


def _one_core(mesh):
    core = ttnn.CoreCoord(0, 0)
    return core, ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _rows_of(row) -> int:
    shape = tuple(int(v) for v in row.shape)
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or shape[3] != ROW_LANES
        or row.dtype != ttnn.float32
        or row.layout != ttnn.ROW_MAJOR_LAYOUT
        or not 1 <= shape[2] <= MAX_ROWS
    ):
        raise ValueError(
            f"sampler_tail takes fp32 ROW_MAJOR candidate rows [1,1,1..{MAX_ROWS},{ROW_LANES}], got {_metadata(row)}"
        )
    return shape[2]


@dataclass
class Qwen38TTNNSamplerTailConstants:
    """The program's replicated device tensors, behind the composite's interface so callers can hold either."""

    rows: int
    policy_row: Any  # fp32 ROW_MAJOR [1,1,1,16]: top_k, top_p, min_weight, greedy_flag, presence_penalty
    uniforms: Any  # fp32 ROW_MAJOR [1,1,1,16]: lane r = row r's draw
    weight_table: Any  # fp32 ROW_MAJOR [1,1,1,65536]
    history: Any  # uint32 ROW_MAJOR [1,1,rows,HIST_WORDS], L1 height-sharded on core (0, 0)
    guard: Any  # fp32 ROW_MAJOR [1,1,1,16]: lane 0 the rows' guard mask (bit r = row r)
    resident_temperature: float | None = None
    resident_policy: Any = None  # a Qwen38DeviceSamplerPolicy
    mesh_device: Any = field(default=None, repr=False)
    presence_on_device: bool = True

    HOST_WRITTEN = ("policy_row", "uniforms", "weight_table", "history")

    @classmethod
    def history_memory_config(cls, rows: int) -> "ttnn.MemoryConfig":
        _core, grid = _one_core(None)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [rows, HIST_WORDS], ttnn.ShardOrientation.ROW_MAJOR),
        )

    @classmethod
    def build(
        cls, mesh_device, mesh_contract: Qwen38MeshContract, *, rows: int = 1
    ) -> "Qwen38TTNNSamplerTailConstants":
        if not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"sampler_tail serves 1..{MAX_ROWS} rows, got {rows}")
        replicate = _mapper(mesh_device)

        def upload(host: torch.Tensor, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=memory_config,
                mesh_mapper=replicate,
            )

        ds = _device_sampler()
        greedy = ds.Qwen38DeviceSamplerPolicy.greedy_policy()
        result = cls(
            rows=rows,
            policy_row=upload(_policy_row(greedy), ttnn.float32),
            uniforms=upload(torch.zeros(1, 1, 1, SCALAR_LANES, dtype=torch.float32), ttnn.float32),
            weight_table=upload(ds.weight_table(1.0), ttnn.float32),
            history=upload(
                history_image(rows, [[] for _ in range(rows)]), ttnn.uint32, cls.history_memory_config(rows)
            ),
            guard=upload(torch.zeros(1, 1, 1, SCALAR_LANES, dtype=torch.float32), ttnn.float32),
            resident_temperature=1.0,
            resident_policy=greedy,
            mesh_device=mesh_device,
        )
        result.validate(mesh_contract)
        return result

    def tensors(self) -> tuple[Any, ...]:
        return (self.policy_row, self.uniforms, self.weight_table, self.history, self.guard)

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        expected = (
            ("policy_row", (1, 1, 1, SCALAR_LANES), ttnn.float32),
            ("uniforms", (1, 1, 1, SCALAR_LANES), ttnn.float32),
            ("weight_table", (1, 1, 1, TABLE_SIZE), ttnn.float32),
            ("history", (1, 1, self.rows, HIST_WORDS), ttnn.uint32),
            ("guard", (1, 1, 1, SCALAR_LANES), ttnn.float32),
        )
        for name, shape, dtype in expected:
            tensor = getattr(self, name)
            if _shape(tensor) != shape or tensor.dtype != dtype or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise RuntimeError(f"sampler_tail {name} must be {dtype} ROW_MAJOR {shape}, got {_metadata(tensor)}")
            if self.mesh_device.get_num_devices() > 1:
                mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
        config = self.history.memory_config()
        if (
            not config.is_sharded()
            or config.buffer_type != ttnn.BufferType.L1
            or config.shard_spec.grid.num_cores() != 1
        ):
            raise RuntimeError(f"sampler_tail history must be one L1 shard on the program's core, got {config}")

    def _write(self, name: str, host: torch.Tensor, dtype) -> None:
        image = ttnn.from_torch(host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=_mapper(self.mesh_device))
        ttnn.copy_host_to_device_tensor(image, getattr(self, name))

    def write_policy(self, policy) -> dict[str, Any]:
        """Request start: the policy row and, when the temperature changes, the table; returns what was written.

        Eager writes, ordered before the request's first TAIL."""

        ds = _device_sampler()
        if not isinstance(policy, ds.Qwen38DeviceSamplerPolicy):
            raise TypeError("policy must be a Qwen38DeviceSamplerPolicy")
        written = {"table": False, "scalars": False}
        if not policy.greedy and policy.temperature != self.resident_temperature:
            self._write("weight_table", ds.weight_table(policy.temperature), ttnn.float32)
            self.resident_temperature = policy.temperature
            written["table"] = True
        if self.resident_policy is None or policy != self.resident_policy:
            self._write("policy_row", _policy_row(policy), ttnn.float32)
            self.resident_policy = policy
            written["scalars"] = True
        return written

    def write_uniform(self, uniform: float) -> None:
        """One step's draw for the decode row (row 0)."""

        self.write_uniforms([uniform])

    def write_uniforms(self, uniforms: list[float]) -> None:
        """The rows' draws (``n * 2**-24``, exact in fp32), lane r = row r."""

        if not 1 <= len(uniforms) <= self.rows:
            raise ValueError(f"{len(uniforms)} draws for {self.rows} rows")
        row = torch.zeros(1, 1, 1, SCALAR_LANES, dtype=torch.float32)
        for r, uniform in enumerate(uniforms):
            if not 0 <= uniform < 1:
                raise ValueError(f"uniform must lie in [0, 1), got {uniform!r}")
            row[0, 0, 0, r] = uniform
        self._write("uniforms", row, ttnn.float32)

    def reset_history(self) -> None:
        """Request start: no token emitted yet, on every row."""

        self.rewrite_history([[] for _ in range(self.rows)])

    def rewrite_history(self, emitted: list[list[int]]) -> None:
        """The rows' emitted tokens as the host knows them (the two steps whose token is not the device's draw)."""

        self._write("history", history_image(self.rows, emitted), ttnn.uint32)

    def mark_corruptible(self) -> None:
        for name in (*self.HOST_WRITTEN, "guard"):  # the guard row: rewritten by every replay
            acknowledge_corruptible(getattr(self, name))

    def release(self) -> None:
        for tensor in self.tensors():
            ttnn.deallocate(tensor)


def _policy_row(policy) -> torch.Tensor:
    row = torch.zeros(1, 1, 1, SCALAR_LANES, dtype=torch.float32)
    row[0, 0, 0, POLICY_TOP_K] = float(policy.top_k)
    row[0, 0, 0, POLICY_TOP_P] = policy.top_p
    row[0, 0, 0, POLICY_MIN_WEIGHT] = policy.min_weight
    row[0, 0, 0, POLICY_GREEDY] = 1.0 if policy.greedy else 0.0
    row[0, 0, 0, POLICY_PRESENCE] = policy.presence_penalty
    return row


def sampler_tail(row: Any, greedy_token_row: Any, constants: "Qwen38TTNNSamplerTailConstants", *, memory_config=None):
    """The composite's contract: the candidate row(s), the greedy token tile and the constants -> the fp32 TILE token
    row (lane r = row r's id; the greedy tile itself when the flag is set), the rows' guard mask into
    ``constants.guard``.  Neither input is consumed."""

    if not isinstance(constants, Qwen38TTNNSamplerTailConstants):
        raise TypeError("sampler_tail needs Qwen38TTNNSamplerTailConstants (the composite takes its own)")
    rows = _rows_of(row)
    if rows != constants.rows:
        raise ValueError(f"candidate rows {rows} vs the constants' {constants.rows} rows")
    if (
        _shape(greedy_token_row) != TOKEN_ROW_SHAPE
        or greedy_token_row.dtype != ttnn.float32
        or greedy_token_row.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(f"greedy token row must be fp32 TILE {TOKEN_ROW_SHAPE}, got {_metadata(greedy_token_row)}")
    mesh = row.device()
    token = fp.allocate(TOKEN_ROW_SHAPE, ttnn.float32, ttnn.TILE_LAYOUT, mesh, memory_config or ttnn.DRAM_MEMORY_CONFIG)
    core, one = _one_core(mesh)
    accessed = [
        row,
        greedy_token_row,
        constants.policy_row,
        constants.uniforms,
        constants.weight_table,
        token,
        constants.guard,
    ]
    kernel = fp.reader_kernel(
        KERNEL,
        one,
        [a for tensor in accessed for a in fp.accessor_args(tensor)],
        [
            (
                core,
                [
                    row.buffer_address(),
                    greedy_token_row.buffer_address(),
                    constants.policy_row.buffer_address(),
                    constants.uniforms.buffer_address(),
                    constants.weight_table.buffer_address(),
                    constants.history.buffer_address(),
                    token.buffer_address(),
                    constants.guard.buffer_address(),
                ],
            )
        ],
        named={"cb_stage": CB_STAGE, "rows": rows, "lanes": LANES, "table_size": TABLE_SIZE, "hist_words": HIST_WORDS},
    )
    pages = -(-stage_bytes(rows) // TILE_BYTES)
    fp.run_program(
        [row, greedy_token_row, *constants.tensors(), token],
        fp.program_descriptor([kernel], cbs=[fp.cb_descriptor(CB_STAGE, ttnn.float32, TILE_BYTES, pages, one)]),
    )
    return fp.stamp_topology(token, greedy_token_row)


def sample_on_device_composed(row: Any, greedy_token_row: Any, constants: Any):
    """The composite (``ttnn/device_sampler.py``) on its own constants."""

    return _device_sampler().sample_on_device(row, greedy_token_row, constants)


register(
    FusedKernel(
        name=NAME,
        replaces="sample_on_device (57 ttnn calls, 76 programs) of the sampling chain's TAIL epilogue",
        tolerance=BITWISE,
        fused=sampler_tail,
        composed=sample_on_device_composed,
        gate=None,  # the single-chip device test against device_sampler_reference is the component gate
    )
)

__all__ = [
    "HIST_WORDS",
    "MAX_ROWS",
    "NAME",
    "Qwen38TTNNSamplerTailConstants",
    "history_image",
    "sample_on_device_composed",
    "sampler_tail",
    "stage_bytes",
]
