# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact TP4 gated residual for Qwen3.8-Flash-Next decode.

The persistent residual is represented as ``[1, 4, 1, 640]`` on every
coordinate.  The last dimension is the coordinate's slice of hidden size and
dimension one is the four hyper-connection branches.  Keeping the branch axis
explicit is important: sharding a flattened branch-major 10,240 vector would
put a different complete branch, rather than every branch's hidden slice, on
each device.  The branch axis leads (each branch owns its own tile row-block)
so the gate multiply consumes the residual directly, with no per-read branch
permute, and the same tiles read as one ``[1, 1, 1, 2560]`` row indexed
(branch, local hidden) for the decode matmuls.

Two exact host folds keep the read short.  The 1/4 branch mean is folded into
the RMS gamma (a power-of-two scale commutes with BF16 rounding), so the
normalized residual, both projections, the gate product and the branch mean
all carry the mean with no device multiply.  The up weight's K is zero-padded
to the fused row width so the reduced row feeds it whole, with no slice.

The two elementwise producers that feed the decode matmuls write the matmul
activation layouts directly: the gamma multiply lands in the down+inject
matmul's five-core width-sharded L1 row and the SiLU in the up matmul's
two-core row, so no interleaved-to-sharded copy precedes either matmul.  The
five-core shard is defined on the flat row (five 512-column shards do not
divide the 640-column branch-major view), so the gamma multiply and the gate
multiply both run on the flat row and the branch reduce reads the product
through the branch-major view.

The down and injection projections are row-parallel over that flat row.  Their
outputs are local partial sums until an explicit all-reduce over mesh axis 1.
They must never be treated as replicated merely because every coordinate has
the same local shape.  Both partials come out of one matmul and cross the mesh
in one collective: the injection columns ride as the eleventh tile of the
rank-320 down row and are sliced back out after the reduce.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import INDEX_SHA256, Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidualWeights
from models.demos.blackhole.qwen38_flash_next.ttnn import prefill_glue
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROW_COUNTS,
    CHUNK_ROWS,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    is_slab_rows,
    replicate_tensor_2d_mesh_mapper,
    require_lane_count,
    same_buffer,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    dram_sharded_matmul_configs,
    dram_sharded_row_tiles,
    dram_sharded_weight_memory_config,
    prefill_matmul_program_config,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.prefill_dense import Qwen38TTNNPrefillDense, prefill_linear

TP_AXIS = 1
TP_SIZE = 4
HIDDEN_SIZE = 2560
LOCAL_HIDDEN_SIZE = HIDDEN_SIZE // TP_SIZE
RESIDUAL_BRANCHES = 4
RESIDUAL_RANK = 320
RESIDUAL_LOCAL_SHAPE = (1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE)
# The branch-major residual read as one row.  Tile pages of [1,4,32,640] and
# [1,1,32,2560] coincide ((branch b, tile j) -> page 20b+j), so the view moves
# no data; (branch, local hidden) is K for down/injection and N for up.
FLAT_LOCAL_WIDTH = RESIDUAL_BRANCHES * LOCAL_HIDDEN_SIZE
FLAT_LOCAL_SHAPE = (1, 1, 1, FLAT_LOCAL_WIDTH)
BLOCK_LOCAL_SHAPE = (1, 1, 1, LOCAL_HIDDEN_SIZE)
INJECTION_SHAPE = (1, 1, 1, RESIDUAL_BRANCHES)
# One FP32 tile row carries both TP4 partials: ten down tiles, the tile holding
# the four injection coefficients, then one zero tile (zero weight columns pad
# both).  Twelve tiles let the K-padded up matmul split the whole row over two
# storage cores in one K block (decode_matmul needs K % (32 x cores) == 0).
PARTIAL_WIDTH = RESIDUAL_RANK + 2 * 32
PARTIAL_REDUCTION_SHAPE = (1, 1, 1, PARTIAL_WIDTH)
PARTIAL_REDUCTION_PADDED_SHAPE = (1, 1, 32, PARTIAL_WIDTH)
# The prefill chunk's 32-row forms.  With 32 tokens the flat (branch, local hidden) row of token j
# spans the four branch tile row-blocks, so the two zero-copy views of the 1-row read become real
# permutes; every matmul still sees one 32-row tile and keeps its decode program.
RESIDUAL_ROWS_LOCAL_SHAPE = (1, RESIDUAL_BRANCHES, CHUNK_ROWS, LOCAL_HIDDEN_SIZE)
TOKEN_MAJOR_ROWS_SHAPE = (1, CHUNK_ROWS, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE)
FLAT_ROWS_LOCAL_SHAPE = (1, 1, CHUNK_ROWS, FLAT_LOCAL_WIDTH)
BLOCK_ROWS_LOCAL_SHAPE = (1, 1, CHUNK_ROWS, LOCAL_HIDDEN_SIZE)
INJECTION_ROWS_SHAPE = (1, 1, CHUNK_ROWS, RESIDUAL_BRANCHES)
PARTIAL_REDUCTION_ROWS_SHAPE = (1, 1, CHUNK_ROWS, PARTIAL_WIDTH)


def residual_rows_shape(rows: int) -> tuple[int, int, int, int]:
    """Branch-major residual rows of a chunk form (32 or 128 rows), of a prefill slab, or of 1..32 batched lanes."""

    lanes = not isinstance(rows, bool) and type(rows) is int and 1 <= rows <= CHUNK_ROWS
    if rows not in CHUNK_ROW_COUNTS and not is_slab_rows(rows) and not lanes:
        raise ValueError(
            f"GR rows path admits 1..{CHUNK_ROWS} rows, {CHUNK_ROW_COUNTS} rows or a slab row count, got {rows}"
        )
    return (1, RESIDUAL_BRANCHES, rows, LOCAL_HIDDEN_SIZE)


def block_rows_shape(rows: int) -> tuple[int, int, int, int]:
    return (1, 1, residual_rows_shape(rows)[2], LOCAL_HIDDEN_SIZE)


def injection_rows_shape(rows: int) -> tuple[int, int, int, int]:
    return (1, 1, residual_rows_shape(rows)[2], RESIDUAL_BRANCHES)


# Resident weights: local shape, dtype, mesh shard dim.  The two matmul weights
# are DRAM width-sharded across the banks for the decode program.
DEVICE_WEIGHTS = {
    "norm_scale": ((1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE), ttnn.float32, 3),
    "down_inject": ((1, 1, FLAT_LOCAL_WIDTH, PARTIAL_WIDTH), ttnn.bfloat16, 2),
    "up": ((1, 1, PARTIAL_WIDTH, FLAT_LOCAL_WIDTH), ttnn.bfloat16, 3),
}
MATMUL_WEIGHTS = ("down_inject", "up")


def gr_rows_shapes(rows_or_shape, *, label: str = "GR rows") -> dict[str, Any]:
    """The rows path's local shapes for ``rows`` tokens (32 = the chunk constants above, 1..32 = batched lanes).

    ``rows_or_shape`` is the row count or a rank-4 local shape whose dim 2 is the row count.  The flat and partial
    tensors are one 32-row tile tall whatever the row count (the DRAM-sharded matmuls keep the decode program).
    """

    rows = rows_or_shape
    if isinstance(rows_or_shape, tuple):
        if len(rows_or_shape) != 4:
            raise ValueError(f"{label} must be rank 4 [1,4,rows,640] / [1,1,rows,...], got {list(rows_or_shape)}")
        rows = rows_or_shape[2]
    rows = require_lane_count(rows, label=label)
    return {
        "rows": rows,
        "residual": (1, RESIDUAL_BRANCHES, rows, LOCAL_HIDDEN_SIZE),
        "token_major": (1, rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE),
        "flat": (1, 1, rows, FLAT_LOCAL_WIDTH),
        "flat_padded": (1, 1, CHUNK_ROWS, FLAT_LOCAL_WIDTH),
        "block": (1, 1, rows, LOCAL_HIDDEN_SIZE),
        "injection": (1, 1, rows, RESIDUAL_BRANCHES),
        "partial": (1, 1, rows, PARTIAL_WIDTH),
        "partial_padded": (1, 1, CHUNK_ROWS, PARTIAL_WIDTH),
    }


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.shape)


def _padded_shape(tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.padded_shape)


def _deallocate(*tensors) -> None:
    for tensor in tensors:
        if tensor is not None:
            ttnn.deallocate(tensor)


def _tensor_key(tensor) -> tuple[str, int]:
    return (str(tensor.device()), int(tensor.tensor_id))


def _cache_dir(
    root: str | Path,
    checkpoint: Qwen38Checkpoint,
    mesh_contract: Qwen38MeshContract,
    tt_metal_sha: str,
    namespace: str,
    layer_index: int,
    block: str,
) -> Path:
    if namespace not in {"backbone", "mtp"}:
        raise ValueError(f"unsupported GR namespace {namespace!r}")
    if block not in {"attn", "mlp"}:
        raise ValueError(f"GR block must be attn or mlp, got {block!r}")
    if layer_index < 0:
        raise ValueError("GR layer index must be nonnegative")
    if len(tt_metal_sha) != 40 or any(character not in "0123456789abcdef" for character in tt_metal_sha):
        raise ValueError(f"tt_metal_sha must be a lowercase 40-hex revision, got {tt_metal_sha!r}")
    physical = "-".join(str(value) for value in mesh_contract.physical_ids)
    path = (
        Path(root).resolve()
        / "gr"
        / f"index-{INDEX_SHA256}"
        / f"config-{checkpoint.config.config_sha256}"
        / f"tt-metal-{tt_metal_sha}"
        / f"mesh-1x4-physical-{physical}"
        / namespace
        / f"layer-{layer_index:02d}"
        / f"{block}-gr"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_exact_config(source: Qwen38GatedResidualWeights) -> None:
    config = source.config
    exact = {
        "hidden_size": HIDDEN_SIZE,
        "residual_branches": RESIDUAL_BRANCHES,
        "residual_rank": RESIDUAL_RANK,
    }
    for name, expected in exact.items():
        actual = int(getattr(config, name))
        if actual != expected:
            raise ValueError(f"Qwen3.8 GR requires {name}={expected}, got {actual}")
    if float(config.rms_norm_eps) != 1e-6:
        raise ValueError(f"Qwen3.8 GR requires rms_norm_eps=1e-6, got {config.rms_norm_eps}")
    # The mesh mappers split one axis contiguously in mesh-column order; the
    # per-device weight blocks below are stacked in that same order.
    mesh_ranges = tuple((index * LOCAL_HIDDEN_SIZE, (index + 1) * LOCAL_HIDDEN_SIZE) for index in range(TP_SIZE))
    if tuple(source.placement.hidden_ranges) != mesh_ranges:
        raise ValueError(f"Qwen3.8 GR requires hidden ranges {mesh_ranges}, got {source.placement.hidden_ranges}")


def _prepare_host_weights(source: Qwen38GatedResidualWeights) -> dict[str, torch.Tensor]:
    """Put every weight in its device-local decode layout before mesh sharding.

    The sharded axis of each matmul weight is ordered (device, branch, local
    hidden): device ``d`` receives rows/columns ``[2560 d, 2560 (d + 1))``, its
    hidden slice of all four branches in flat-row order.  ``down`` and
    ``inject`` stack their four input-branch blocks along K into one weight
    whose N is the fused partial row (ten rank tiles, the injection tile, a
    zero tile); ``up`` stacks its four output-branch blocks along N and
    zero-pads its K to that row.  The 1/4 branch mean lives in ``norm_scale``.
    """

    _validate_exact_config(source)
    norm = source.norm.unflatten(0, (RESIDUAL_BRANCHES, HIDDEN_SIZE))
    shards = [source.device_shard(device) for device in range(TP_SIZE)]
    injection_pad = torch.zeros(
        FLAT_LOCAL_WIDTH, PARTIAL_WIDTH - RESIDUAL_RANK - RESIDUAL_BRANCHES, dtype=source.inject.dtype
    )
    up_pad = torch.zeros(PARTIAL_WIDTH - RESIDUAL_RANK, TP_SIZE * FLAT_LOCAL_WIDTH, dtype=source.up.dtype)
    prepared = {
        # Qwen4Exp uses zero-centred RMSNorm: gamma = 1 + checkpoint weight.
        # Preserve the addition in FP32 and fold in the 1/4 branch mean (an
        # exact exponent shift); the normalized activation is cast back to
        # BF16 only after this multiplication.  Branch-major to match the
        # persistent residual layout.
        "norm_scale": ((1.0 + norm.float()) / RESIDUAL_BRANCHES)
        .reshape(1, RESIDUAL_BRANCHES, 1, HIDDEN_SIZE)
        .contiguous(),
        "down_inject": torch.cat(
            [
                torch.cat(
                    [
                        # [rank, input_branch, local hidden]
                        #     -> [(input_branch, local hidden), rank]
                        shard.down.permute(1, 2, 0).reshape(FLAT_LOCAL_WIDTH, RESIDUAL_RANK),
                        # [output_branch, input_branch, local hidden]
                        #     -> [(input_branch, local hidden), output_branch]
                        shard.inject.permute(1, 2, 0).reshape(FLAT_LOCAL_WIDTH, RESIDUAL_BRANCHES),
                        injection_pad,
                    ],
                    dim=1,
                )
                for shard in shards
            ],
            dim=0,
        ).reshape(1, 1, TP_SIZE * FLAT_LOCAL_WIDTH, PARTIAL_WIDTH),
        # [output_branch, local hidden, rank] -> [rank | zero pad, (output_branch, local hidden)]
        "up": torch.cat(
            [
                torch.cat(
                    [shard.up.permute(2, 0, 1).reshape(RESIDUAL_RANK, FLAT_LOCAL_WIDTH) for shard in shards],
                    dim=1,
                ),
                up_pad,
            ],
            dim=0,
        ).reshape(1, 1, PARTIAL_WIDTH, TP_SIZE * FLAT_LOCAL_WIDTH),
    }
    expected = {
        "norm_scale": (1, RESIDUAL_BRANCHES, 1, HIDDEN_SIZE),
        "down_inject": (1, 1, TP_SIZE * FLAT_LOCAL_WIDTH, PARTIAL_WIDTH),
        "up": (1, 1, PARTIAL_WIDTH, TP_SIZE * FLAT_LOCAL_WIDTH),
    }
    for name, value in prepared.items():
        if tuple(value.shape) != expected[name]:
            raise RuntimeError(f"prepared GR {name} has shape {tuple(value.shape)}, expected {expected[name]}")
    return prepared


@dataclass(frozen=True)
class Qwen38TTNNGatedResidualWeights:
    """Resident BF16/FP32 tensors for one attention or MLP GR read/write."""

    norm_scale: Any
    down_inject: Any
    up: Any
    replicated_anchor: Any
    epsilon: float
    layer_index: int
    block: Literal["attn", "mlp"]
    namespace: Literal["backbone", "mtp"]
    # norm_scale with its one row repeated over the 32 tile rows, [1, 4, 32, local] fp32: the fused read's gamma
    # operand (its reader streams tiles as they are; copying the row on the RISC cost 175 us per read).
    norm_scale_rows: Any = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache_root: str | Path,
        *,
        layer_index: int,
        block: Literal["attn", "mlp"],
        namespace: Literal["backbone", "mtp"] = "backbone",
        tt_metal_sha: str,
    ) -> "Qwen38TTNNGatedResidualWeights":
        mesh_contract.validate_mesh(mesh_device)
        if namespace == "backbone":
            source = Qwen38GatedResidualWeights.from_checkpoint(
                checkpoint,
                placement,
                layer_index=layer_index,
                block=block,
            )
        elif namespace == "mtp":
            source = Qwen38GatedResidualWeights.from_mtp_checkpoint(
                checkpoint,
                placement,
                mtp_layer_index=layer_index,
                block=block,
            )
        else:
            raise ValueError(f"unsupported GR namespace {namespace!r}")
        prepared = _prepare_host_weights(source)
        cache_dir = _cache_dir(
            cache_root,
            checkpoint,
            mesh_contract,
            tt_metal_sha,
            namespace,
            layer_index,
            block,
        )

        hidden_output_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))
        hidden_input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 2))

        def upload(value: torch.Tensor, name: str, dtype, mapper, memory_config):
            return ttnn.as_tensor(
                value.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=memory_config,
                mesh_mapper=mapper,
                cache_file_name=cache_dir / name,
            )

        # New cache names: the quarter-scaled gamma and the twelve-tile row
        # weights must never load a stale cache written by the unfolded read.
        norm_scale = upload(
            prepared["norm_scale"], "norm_scale_q_bm", ttnn.float32, hidden_output_mapper, ttnn.DRAM_MEMORY_CONFIG
        )
        norm_scale_rows = upload(
            prepared["norm_scale"].expand(-1, -1, ttnn.TILE_SIZE, -1),
            "norm_scale_rows_q_bm",
            ttnn.float32,
            hidden_output_mapper,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        # The matmul weights are DRAM width-sharded for the decode program.
        down_inject = upload(
            prepared["down_inject"],
            "down_inject_w384_dram_sharded",
            ttnn.bfloat16,
            hidden_input_mapper,
            dram_sharded_weight_memory_config(mesh_device, FLAT_LOCAL_WIDTH, PARTIAL_WIDTH),
        )
        up = upload(
            prepared["up"],
            "up_k384_dram_sharded",
            ttnn.bfloat16,
            hidden_output_mapper,
            dram_sharded_weight_memory_config(mesh_device, PARTIAL_WIDTH, FLAT_LOCAL_WIDTH),
        )
        replicated_anchor = ttnn.from_torch(
            torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
        )
        weights = cls(
            norm_scale=norm_scale,
            down_inject=down_inject,
            up=up,
            replicated_anchor=replicated_anchor,
            epsilon=float(source.config.rms_norm_eps),
            layer_index=layer_index,
            block=block,
            namespace=namespace,
            norm_scale_rows=norm_scale_rows,
        )
        weights.validate(mesh_contract)
        return weights

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        for name, (expected_shape, expected_dtype, shard_dim) in DEVICE_WEIGHTS.items():
            tensor = getattr(self, name)
            if _shape(tensor) != expected_shape or tensor.dtype != expected_dtype or tensor.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(
                    f"GR {name} must be TILE {expected_dtype} with local shape {expected_shape}; "
                    f"got {tensor.layout} {tensor.dtype} {_shape(tensor)}"
                )
            if (name in MATMUL_WEIGHTS) != (
                tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            ):
                raise RuntimeError(f"GR {name} has memory layout {tensor.memory_config().memory_layout}")
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=shard_dim)
        mesh_contract.validate_tensor(self.replicated_anchor, placement=TensorPlacement.REPLICATED)
        if _shape(self.replicated_anchor) != (1, 1, 1, 1):
            raise RuntimeError("GR replicated topology anchor must have local shape [1,1,1,1]")
        if float(self.epsilon) != 1e-6:
            raise RuntimeError(f"GR epsilon must be 1e-6, got {self.epsilon}")


@dataclass(frozen=True)
class Qwen38TTNNGatedResidualState:
    """State retained between one GR read and its matching write."""

    residual: Any
    injection: Any


class Qwen38TTNNGatedResidual:
    """One exact global-B1 TP4 gated-residual module."""

    # The prefill slab's dense-linear policy (ttnn/prefill_dense: the QWEN38_PREFILL_DENSE_* switches); the decode
    # linears never read it.
    prefill_dense: Qwen38TTNNPrefillDense | None = None

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNGatedResidualWeights,
        *,
        tt_ccl,
        collective_topology=None,
        prefill_dense: Qwen38TTNNPrefillDense | None = None,
    ) -> None:
        mesh_contract.validate_mesh(mesh_device)
        weights.validate(mesh_contract)
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.weights = weights
        self.tt_ccl = tt_ccl
        self.collective_topology = collective_topology or ttnn.Topology.Linear
        self.prefill_dense = Qwen38TTNNPrefillDense.resolve(prefill_dense, mesh_device)
        # The prefill glue policy (QWEN38_PREFILL_GLUE), resolved once; read by read_rows' slab branch only.
        self.glue = prefill_glue.policy()
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Decode matmul configs.  The flat normalized row splits its K=2560
        # over five cores (eight-tile K blocks); the twelve-tile low-rank row
        # splits over two cores (one six-tile K block) for the wide up weight.
        self.down_inject_act_memory_config, self.down_inject_program_config = dram_sharded_matmul_configs(
            mesh_device, FLAT_LOCAL_WIDTH, PARTIAL_WIDTH, num_cores=5
        )
        self.up_act_memory_config, self.up_program_config = dram_sharded_matmul_configs(
            mesh_device, PARTIAL_WIDTH, FLAT_LOCAL_WIDTH, num_cores=2
        )
        # The gamma multiply runs on the flat row (see _normalize); this view
        # shares the resident weight's buffer.
        self.norm_scale_flat = ttnn.experimental.view(weights.norm_scale, FLAT_LOCAL_SHAPE)
        # The read resolves through the fused registry at construction (ttnn/fused/gr_read when it is on: in
        # registry.DEFAULT_ON or QWEN38_FUSED=gr_read; QWEN38_FUSED_OFF keeps the chain) and stays so through trace capture; the class body of
        # read() is the composed chain the fused kernel is gated against.
        from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_kernels

        self._read_fused = None
        if fused_kernels.enabled("gr_read"):
            # gr_fold (on by default) is gr_read's read with its collectives folded into the programs; it needs the mesh's
            # global semaphores before any trace capture, so they are created here
            name = "gr_fold" if fused_kernels.enabled("gr_fold") else "gr_read"
            if fused_kernels.enabled("gr_recip_last"):
                # the COMPONENT-class front (ttnn/fused/gr_recip_last: the rsqrt applied after the down projection),
                # on by default, takes the fold's place; it runs on the same line transports (QWEN38_FUSED_OFF=gr_recip_last
                # restores the fold)
                name = "gr_recip_last"
            if name in ("gr_fold", "gr_recip_last"):
                fused_kernels.gr_fold.line_semaphores(mesh_device)
            self._read_fused = fused_kernels.kernel(name).fused
            self.read = functools.partial(self._read_fused, self)
        # QWEN38_FUSED=gr_write: write and write_rows (one tile row) resolve to ttnn/fused/gr_write the same way.
        self._write_fused = None
        if fused_kernels.enabled("gr_write"):
            from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_write as fused_gr_write

            self._write_fused = fused_gr_write.gr_write
            self.write = functools.partial(fused_gr_write.write_fused, self)
            self.write_rows = functools.partial(fused_gr_write.write_rows_fused, self)

    def _validate_residual(self, residual) -> None:
        if _shape(residual) != RESIDUAL_LOCAL_SHAPE:
            raise ValueError(
                f"GR residual must have local branch-major shape {RESIDUAL_LOCAL_SHAPE}, got {_shape(residual)}"
            )
        if residual.layout != ttnn.TILE_LAYOUT or residual.dtype != ttnn.bfloat16:
            raise ValueError("GR residual must be TILE BFLOAT16")
        self.mesh_contract.validate_tensor(
            residual,
            placement=TensorPlacement.HIDDEN_SHARDED,
            shard_dim=3,
        )

    def _validate_block(self, block_output) -> None:
        if _shape(block_output) != BLOCK_LOCAL_SHAPE:
            raise ValueError(f"GR block output must have local shape {BLOCK_LOCAL_SHAPE}, got {_shape(block_output)}")
        if block_output.layout != ttnn.TILE_LAYOUT or block_output.dtype != ttnn.bfloat16:
            raise ValueError("GR block output must be TILE BFLOAT16")
        self.mesh_contract.validate_tensor(
            block_output,
            placement=TensorPlacement.HIDDEN_SHARDED,
            shard_dim=3,
        )

    def _mark_partial(self, tensor, expected_shape: tuple[int, ...]) -> None:
        self.mesh_contract.mark_local_partial(
            tensor,
            replicated_reference=self.weights.replicated_anchor,
            expected_shape=expected_shape,
        )

    def _all_reduce_partial(self, tensor, expected_shape: tuple[int, ...]):
        if expected_shape != PARTIAL_REDUCTION_SHAPE:
            raise RuntimeError(f"GR partial reduction shape {expected_shape} is not the fused down+injection tile row")
        self._mark_partial(tensor, expected_shape)
        if (
            _padded_shape(tensor) != PARTIAL_REDUCTION_PADDED_SHAPE
            or tensor.dtype != ttnn.float32
            or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise RuntimeError(
                "GR partial reduction requires its exact FP32 TILE DRAM logical/padded contract; "
                f"got {_shape(tensor)}/{_padded_shape(tensor)} {tensor.dtype} {tensor.layout}"
            )
        if self.collective_topology != ttnn.Topology.Linear:
            raise RuntimeError("GR partial reduction is qualified only for TP4 Linear topology")
        if self.tt_ccl is None:
            raise RuntimeError("GR partial reduction requires the builder-owned TT-CCL manager")
        num_links = self.tt_ccl.get_num_links(TP_AXIS)
        if type(num_links) is not int or num_links < 1:
            raise RuntimeError(f"GR partial reduction received invalid TP-axis link count {num_links!r}")

        original_shape = tensor.shape
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=0,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(TP_AXIS),
            num_links=num_links,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(TP_AXIS),
            chunks_per_sync=1,
            num_workers_per_link=1,
            num_buffers_per_channel=2,
        )
        reduced = ttnn.experimental.fast_reduce_nc(
            gathered,
            dims=[0],
            output=None,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        reduced = ttnn.reshape(reduced, original_shape, reduced.padded_shape)
        _deallocate(gathered)
        self.mesh_contract.validate_tensor(reduced, placement=TensorPlacement.REPLICATED)
        if (
            _shape(reduced) != expected_shape
            or _padded_shape(reduced) != PARTIAL_REDUCTION_PADDED_SHAPE
            or reduced.dtype != ttnn.float32
            or reduced.layout != ttnn.TILE_LAYOUT
            or reduced.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise RuntimeError(
                "GR partial reduction returned a different FP32 TILE DRAM logical/padded contract; "
                f"got {_shape(reduced)}/{_padded_shape(reduced)} {reduced.dtype} {reduced.layout}"
            )
        return reduced

    def _normalize(self, residual):
        """Per-branch distributed zero-centred RMSNorm over global H=2560, scaled by gamma/4.

        Returns the normalized residual as the flat (branch, local hidden) row
        in the down+inject matmul's width-sharded L1 activation layout.
        """

        stats = ttnn.rms_norm_pre_all_gather(
            residual,
            # The post-all-gather kernel consumes the canonical BF16 stats tile.
            # FP32 here changes that tile encoding and produces incorrect norms.
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        stats = ttnn.reshape(stats, (1, RESIDUAL_BRANCHES, 1, 32))
        self.mesh_contract.validate_tensor(stats, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        gathered_stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(stats)
        self.mesh_contract.validate_tensor(gathered_stats, placement=TensorPlacement.REPLICATED)
        if _shape(gathered_stats) != (1, RESIDUAL_BRANCHES, 1, 32 * TP_SIZE):
            raise RuntimeError(f"GR gathered RMS statistics have shape {_shape(gathered_stats)}, expected [1,4,1,128]")
        unit = ttnn.rms_norm_post_all_gather(
            residual,
            gathered_stats,
            epsilon=self.weights.epsilon,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        _deallocate(gathered_stats)
        self.mesh_contract.validate_tensor(unit, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        # The fused post kernel accepts one hidden-width gamma vector and
        # broadcasts it over the row axis.  Qwen3.8 instead has a distinct
        # learned scale for each of its four hyper-connection branches, so
        # apply the full branch-major scale elementwise after computing the
        # unscaled RMS unit.  norm_scale already carries the 1/4 branch mean.
        # The multiply reads both operands through their flat views (the same
        # tile pages) and writes the down+inject activation shard directly.
        normalized_ws = ttnn.multiply(
            ttnn.experimental.view(unit, FLAT_LOCAL_SHAPE),
            self.norm_scale_flat,
            dtype=ttnn.bfloat16,
            memory_config=self.down_inject_act_memory_config,
        )
        _deallocate(unit)
        self.mesh_contract.validate_tensor(normalized_ws, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if (
            _shape(normalized_ws) != FLAT_LOCAL_SHAPE
            or normalized_ws.memory_config() != self.down_inject_act_memory_config
        ):
            raise RuntimeError(
                f"GR normalization produced {_shape(normalized_ws)} {normalized_ws.memory_config()}, "
                f"expected {FLAT_LOCAL_SHAPE} in the down+inject activation layout"
            )
        return normalized_ws

    def read(self, residual) -> tuple[Any, Qwen38TTNNGatedResidualState]:
        """Read the H-wide block input and retain the exact write state (the composed chain; see __init__)."""

        self._validate_residual(residual)
        # The flat (branch, local hidden) row, already scaled by 1/4, in the
        # fused projection's activation layout; the gate multiply reads the
        # same shard at the end of the read.
        normalized_ws = self._normalize(residual)

        # Every coordinate owns its hidden/K slice of all four branches: one
        # K=2560 matmul sums the branches, then the collective sums the
        # distinct hidden-slice contributions across TP4.
        partial_ws = ttnn.linear(
            normalized_ws,
            self.weights.down_inject,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.down_inject_program_config,
            # Do not round the K-partials to BF16 before the TP sum.
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_config,
        )
        partial = ttnn.to_memory_config(partial_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(partial_ws)
        if _shape(partial) != PARTIAL_REDUCTION_SHAPE:
            raise RuntimeError(f"GR fused down+injection projection has unexpected shape {_shape(partial)}")
        reduced = self._all_reduce_partial(partial, PARTIAL_REDUCTION_SHAPE)
        _deallocate(partial)
        # One BF16 rounding of the TP sum; the row already carries both 1/4
        # means through the folded gamma.  The injection tile is split off at
        # the tile boundary before the SiLU.
        reduced_bf16 = ttnn.typecast(reduced, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(reduced)
        inject_row = ttnn.slice(
            reduced_bf16,
            (0, 0, 0, RESIDUAL_RANK),
            (1, 1, 1, RESIDUAL_RANK + RESIDUAL_BRANCHES),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # SiLU over the whole row, written straight into the up matmul's
        # two-core activation shard: columns 320-383 meet the zero K rows of
        # the padded up weight, so no slice is needed on the down half.
        low_rank_ws = ttnn.silu(reduced_bf16, memory_config=self.up_act_memory_config)
        _deallocate(reduced_bf16)
        if _shape(low_rank_ws) != PARTIAL_REDUCTION_SHAPE or _shape(inject_row) != INJECTION_SHAPE:
            raise RuntimeError(
                f"GR partial split produced {_shape(low_rank_ws)} and {_shape(inject_row)}, "
                f"expected {PARTIAL_REDUCTION_SHAPE} and {INJECTION_SHAPE}"
            )
        if low_rank_ws.memory_config() != self.up_act_memory_config:
            raise RuntimeError(f"GR low-rank row has memory config {low_rank_ws.memory_config()}")
        self.mesh_contract.validate_tensor(low_rank_ws, placement=TensorPlacement.REPLICATED)

        # The up weight stacks its four output-branch matrices along N, so the
        # now-global row yields the flat branch-major gate in one matmul.  Up
        # shards only N/hidden, so no reduction is missing here.
        up_ws = ttnn.linear(
            low_rank_ws,
            self.weights.up,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.up_program_config,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(low_rank_ws)
        up_flat = ttnn.to_memory_config(up_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(up_ws)
        self.mesh_contract.validate_tensor(up_flat, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(up_flat) != FLAT_LOCAL_SHAPE:
            raise RuntimeError(f"GR up projection has unexpected shape {_shape(up_flat)}")
        # The gate multiplies the normalized shard in place: sharded lhs,
        # interleaved rhs, interleaved product.  The branch-major view owns
        # gated_flat's buffer from here on.
        gate_flat = ttnn.sigmoid(up_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(up_flat)
        gated_flat = ttnn.multiply(normalized_ws, gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(gate_flat, normalized_ws)
        gated = ttnn.experimental.view(gated_flat, RESIDUAL_LOCAL_SHAPE)
        # Branch mean without a permute: fast_reduce_nc sums the four branch
        # tile row-blocks (generic dim-1 reduce would transpose); the 1/4 is
        # already in normalized_ws.  fast_reduce_nc reports the 32-row tile
        # padding as its logical row count; restore the logical shape without
        # moving data.
        gated_sum = ttnn.experimental.fast_reduce_nc(
            gated,
            dims=[1],
            output=None,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        block_input = ttnn.reshape(gated_sum, BLOCK_LOCAL_SHAPE, gated_sum.padded_shape)
        _deallocate(gated)
        self._validate_block(block_input)

        # 2 * sigmoid in one kernel: the fused input activation packs
        # BF16(sigmoid) before the exact x2, the same two rounding points as
        # separate sigmoid and multiply ops (vector mode RC, accurate mode).
        injection = ttnn.multiply(
            inject_row,
            2.0,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(inject_row)
        self.mesh_contract.validate_tensor(injection, placement=TensorPlacement.REPLICATED)
        if _shape(injection) != INJECTION_SHAPE:
            raise RuntimeError(f"GR injection has shape {_shape(injection)}, expected {INJECTION_SHAPE}")
        return block_input, Qwen38TTNNGatedResidualState(residual=residual, injection=injection)

    def write(self, block_output, state: Qwen38TTNNGatedResidualState):
        """Write one block result into every persistent hidden-sharded branch."""

        self._validate_block(block_output)
        self._validate_residual(state.residual)
        self.mesh_contract.validate_tensor(state.injection, placement=TensorPlacement.REPLICATED)
        if _shape(state.injection) != INJECTION_SHAPE:
            raise ValueError(f"GR injection must have shape {INJECTION_SHAPE}, got {_shape(state.injection)}")
        coefficient = ttnn.reshape(state.injection, (1, RESIDUAL_BRANCHES, 1, 1))
        update = ttnn.multiply(block_output, coefficient, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(coefficient)
        self.mesh_contract.validate_tensor(update, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(update) != RESIDUAL_LOCAL_SHAPE:
            raise RuntimeError(f"GR branch update has shape {_shape(update)}, expected {RESIDUAL_LOCAL_SHAPE}")
        output = ttnn.add(state.residual, update, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(update)
        self._validate_residual(output)
        return output

    # ------------------------------------------------------------------ rows path (prefill chunk)
    # read_rows / write_rows are read / write over the rows of one chunk (32 or 128): the same ops row
    # for row (every op is per row or per element), with the two zero-copy views replaced by real
    # permutes at 32 rows and by branch slices + concats at 128.  The two DRAM-sharded matmuls read one
    # 32-row tile per call on this runtime: at 32 rows the tile is the chunk, at 128 rows the four row
    # tiles run the same program one after another (row j sees the same program on the same values either
    # way) and their FP32 partials are reduced over TP in one collective.  The 1-row bodies above are untouched.

    def _validate_rows(self, tensor, expected: tuple[int, ...], *, label: str) -> None:
        if _shape(tensor) != expected or tensor.layout != ttnn.TILE_LAYOUT or tensor.dtype != ttnn.bfloat16:
            raise ValueError(
                f"{label} must be TILE BF16 {expected}, got {tensor.layout} {tensor.dtype} {_shape(tensor)}"
            )
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _slab_program_config(self, rows: int, k: int, n: int):
        """The slab's 2D-multicast matmul config for one of the two linears, built once per (rows, k, n): today's
        config, or the prefill dense policy's wide grid (QWEN38_PREFILL_DENSE_GRID=wide)."""

        configs = self.__dict__.setdefault("_slab_program_configs", {})
        key = (rows, k, n)
        if key not in configs:
            if self.prefill_dense.policy.grid == "today":
                configs[key] = prefill_matmul_program_config(self.mesh_device, rows, k, n)
            else:
                configs[key] = self.prefill_dense.program_config(rows, k, n)
        return configs[key]

    def read_rows(self, residual_rows, *, flat_views: bool = False) -> tuple[Any, Qwen38TTNNGatedResidualState]:
        """:meth:`read` for the residual rows ``[1,4,rows,640]`` of a chunk -> block rows ``[1,1,rows,640]``.

        ``flat_views``: the branch-major <-> flat walks as the 1-row read's ``ttnn.experimental.view`` (every branch
        is one 32-row tile row, so the branch-major tile sequence is the flat rows' tile sequence: the same pages,
        no permute and no relayout; the forms of at most 32 rows: the chunk and the batched lanes); the default keeps
        the permutes (the prefill chunks' form, 32 or 128 rows).  Below 32 rows (B lanes) every flat tensor is one
        padded tile and the two fast_reduce_nc calls report the tile as their row count: the logical rows are restored.
        """

        rows = _shape(residual_rows)[2]  # residual_rows_shape admits 1..32, 128 and the slab counts
        self._validate_rows(residual_rows, residual_rows_shape(rows), label="GR residual rows")
        if flat_views and getattr(self, "_read_fused", None) is not None:
            return self._read_fused(self, residual_rows)
        flat_rows_shape = (1, 1, rows, FLAT_LOCAL_WIDTH)
        tile_rows = max(rows, CHUNK_ROWS)
        flat_rows_padded = (1, 1, tile_rows, FLAT_LOCAL_WIDTH)
        dram = ttnn.DRAM_MEMORY_CONFIG
        stats = ttnn.rms_norm_pre_all_gather(
            residual_rows, dtype=ttnn.bfloat16, memory_config=dram, compute_kernel_config=self.compute_config
        )
        stats = ttnn.reshape(stats, (1, RESIDUAL_BRANCHES, rows, 32))
        self.mesh_contract.validate_tensor(stats, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        gathered_stats = ttnn.all_gather(stats, dim=3, cluster_axis=TP_AXIS, memory_config=dram)
        _deallocate(stats)
        self.mesh_contract.validate_tensor(gathered_stats, placement=TensorPlacement.REPLICATED)
        if _shape(gathered_stats) != (1, RESIDUAL_BRANCHES, rows, 32 * TP_SIZE):
            raise RuntimeError(f"GR rows RMS statistics have shape {_shape(gathered_stats)}, expected [1,4,{rows},128]")
        unit = ttnn.rms_norm_post_all_gather(
            residual_rows,
            gathered_stats,
            epsilon=self.weights.epsilon,
            memory_config=dram,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        _deallocate(gathered_stats)
        self._validate_rows(unit, residual_rows_shape(rows), label="GR rows RMS unit")
        # Branch-major -> one flat (branch, local hidden) row per token.  At 128 rows the four branches are
        # whole row-tile blocks: sliced out and concatenated on the width, no token-major intermediate (whose
        # 4-row branch dim would pad to a tile: 8x the bytes through a permute and a reshape).
        if flat_views:
            if rows > CHUNK_ROWS:
                raise ValueError(f"GR rows flat views are the option of at most {CHUNK_ROWS} rows, got {rows} rows")
            unit_flat = ttnn.experimental.view(unit, flat_rows_shape)  # the view owns unit's buffer
        elif rows > CHUNK_ROWS:
            branches = [
                ttnn.slice(unit, (0, b, 0, 0), (1, b + 1, rows, LOCAL_HIDDEN_SIZE), memory_config=dram)
                for b in range(RESIDUAL_BRANCHES)
            ]
            _deallocate(unit)
            unit_flat = ttnn.concat(branches, dim=3, memory_config=dram)
            _deallocate(*branches)
        else:
            unit_tokens = ttnn.permute(unit, (0, 2, 1, 3), memory_config=dram)
            _deallocate(unit)
            unit_flat = ttnn.reshape(unit_tokens, flat_rows_shape)
            if not same_buffer(
                unit_flat, unit_tokens
            ):  # a copy today (the last dim changes); never free a view's source
                _deallocate(unit_tokens)
        if _shape(unit_flat) != flat_rows_shape or _padded_shape(unit_flat) != flat_rows_padded:
            raise RuntimeError(
                f"GR rows flat unit must be {flat_rows_shape} backed by {flat_rows_padded}, "
                f"got {_shape(unit_flat)}/{_padded_shape(unit_flat)}"
            )
        # One tile is normalized straight into the down+inject activation shard; the long chunk normalizes
        # interleaved (the same elementwise multiply) and moves each row tile into the shard for its call.
        normalized_ws = ttnn.multiply(
            unit_flat,
            self.norm_scale_flat,
            dtype=ttnn.bfloat16,
            memory_config=self.down_inject_act_memory_config if rows <= CHUNK_ROWS else dram,
        )
        _deallocate(unit_flat)
        self.mesh_contract.validate_tensor(normalized_ws, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(normalized_ws) != flat_rows_shape or (
            rows <= CHUNK_ROWS and normalized_ws.memory_config() != self.down_inject_act_memory_config
        ):
            raise RuntimeError(
                f"GR rows normalization produced {_shape(normalized_ws)} {normalized_ws.memory_config()}, "
                f"expected {flat_rows_shape} in the down+inject activation layout"
            )
        if self.collective_topology != ttnn.Topology.Linear or self.tt_ccl is None:
            raise RuntimeError("GR rows partial reduction requires the TP4 Linear topology and the TT-CCL manager")

        slab = is_slab_rows(rows)
        normalized_tiles = (
            [normalized_ws]
            if rows <= CHUNK_ROWS
            else []
            if slab
            else dram_sharded_row_tiles(normalized_ws, self.down_inject_act_memory_config)
        )
        partial_tiles = []
        for normalized_tile in normalized_tiles:
            partial_ws = ttnn.linear(
                normalized_tile,
                self.weights.down_inject,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.down_inject_program_config,
                dtype=ttnn.float32,
                compute_kernel_config=self.compute_config,
            )
            partial_tiles.append(ttnn.to_memory_config(partial_ws, dram))
            _deallocate(partial_ws)
        partial_rows_shape = (1, 1, rows, PARTIAL_WIDTH)
        partial_rows_padded = (1, 1, tile_rows, PARTIAL_WIDTH)
        if slab:
            # The slab: one 2D-multicast matmul over every row on an interleaved copy of the weight (fp32 out), or on
            # the resident prefill copy and with the prefill dense policy's fidelity when its switches say so.
            partial = prefill_linear(
                normalized_ws,
                self.weights.down_inject,
                self._slab_program_config(rows, FLAT_LOCAL_WIDTH, PARTIAL_WIDTH),
                compute_kernel_config=self.prefill_dense.compute_config(self.compute_config),
                dtype=ttnn.float32,
                resident_weight=self.prefill_dense.resident("down_inject"),
            )
        elif rows <= CHUNK_ROWS:
            partial = partial_tiles[0]
        else:
            # The four FP32 tile partials stacked on the rows: one gather and one reduce for the 128 rows (the
            # same four device terms per element, summed in the same device order).
            _deallocate(*normalized_tiles)
            partial = ttnn.concat(partial_tiles, dim=2, memory_config=dram)
            _deallocate(*partial_tiles)
        self._mark_partial(partial, partial_rows_shape)
        if (
            _shape(partial) != partial_rows_shape
            or _padded_shape(partial) != partial_rows_padded
            or partial.dtype != ttnn.float32
            or partial.layout != ttnn.TILE_LAYOUT
            or partial.memory_config() != dram
        ):
            raise RuntimeError(
                f"GR rows partial must be FP32 TILE DRAM {partial_rows_shape}; "
                f"got {_shape(partial)}/{_padded_shape(partial)} {partial.dtype} {partial.layout}"
            )
        # The 1-row partial reduction over the row tiles: the same gather + fast reduce, the logical shape
        # already the padded one.  Slab glue forms (prefill_glue): gr_gather_generic (a default; 0.23 ms per read
        # against the async op's 0.59 at one worker and one chunk per sync, ``today``) / gr_gather_tuned gather the
        # same [4, rows, 384] fp32 bytes into the same page order by the generic op or by the async op with two
        # workers per link and ten chunks per sync, and fast_reduce_nc reads the same tensor (bitwise: data
        # movement); gr_partial_rs_ag (tolerance) sums the partials by a reduce-scatter over the rows + all-gather,
        # in the collective's hop order instead of fast_reduce_nc's device order.
        if slab and self.glue.enabled("gr_partial_rs_ag"):
            scattered = ttnn.reduce_scatter(
                partial, dim=2, cluster_axis=TP_AXIS, memory_config=dram, topology=ttnn.Topology.Linear
            )
            _deallocate(partial)
            reduced = ttnn.all_gather(scattered, dim=2, cluster_axis=TP_AXIS, memory_config=dram)
            _deallocate(scattered)
        else:
            if slab and self.glue.enabled("gr_gather_generic"):
                gathered = ttnn.all_gather(partial, dim=0, cluster_axis=TP_AXIS, memory_config=dram)
            else:
                tuned = slab and self.glue.enabled("gr_gather_tuned")
                gathered = ttnn.experimental.all_gather_async(
                    partial,
                    persistent_output_buffer=None,
                    dim=0,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(TP_AXIS),
                    num_links=self.tt_ccl.get_num_links(TP_AXIS),
                    cluster_axis=TP_AXIS,
                    memory_config=dram,
                    topology=ttnn.Topology.Linear,
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(TP_AXIS),
                    chunks_per_sync=10 if tuned else 1,
                    num_workers_per_link=2 if tuned else 1,
                    num_buffers_per_channel=2,
                )
            _deallocate(partial)
            if slab and _shape(gathered) != (TP_SIZE, 1, rows, PARTIAL_WIDTH):
                raise RuntimeError(
                    f"GR rows partial gather returned {_shape(gathered)}, expected [4,1,{rows},{PARTIAL_WIDTH}]"
                )
            reduced = ttnn.experimental.fast_reduce_nc(
                gathered, dims=[0], output=None, compute_kernel_config=self.compute_config, memory_config=dram
            )
            _deallocate(gathered)
        if rows < CHUNK_ROWS and _shape(reduced) == partial_rows_padded:
            reduced = ttnn.reshape(reduced, partial_rows_shape, reduced.padded_shape)  # the tile reported as rows
        self.mesh_contract.validate_tensor(reduced, placement=TensorPlacement.REPLICATED)
        if _shape(reduced) != partial_rows_shape or reduced.dtype != ttnn.float32:
            raise RuntimeError(
                f"GR rows partial reduction returned {_shape(reduced)} {reduced.dtype}, "
                f"expected FP32 {partial_rows_shape}"
            )
        reduced_bf16 = ttnn.typecast(reduced, ttnn.bfloat16, memory_config=dram)
        _deallocate(reduced)
        inject_rows = ttnn.slice(
            reduced_bf16,
            (0, 0, 0, RESIDUAL_RANK),
            (1, 1, rows, RESIDUAL_RANK + RESIDUAL_BRANCHES),
            memory_config=dram,
        )
        if _shape(inject_rows) != injection_rows_shape(rows):
            raise RuntimeError(
                f"GR rows partial split produced {_shape(inject_rows)}, expected {injection_rows_shape(rows)}"
            )

        up_tiles = []
        low_rank_tile_shape = (1, 1, min(rows, CHUNK_ROWS), PARTIAL_WIDTH)
        for tile in range(0 if slab else tile_rows // CHUNK_ROWS):
            reduced_tile = (
                reduced_bf16
                if rows <= CHUNK_ROWS
                else ttnn.slice(
                    reduced_bf16,
                    (0, 0, tile * CHUNK_ROWS, 0),
                    (1, 1, (tile + 1) * CHUNK_ROWS, PARTIAL_WIDTH),
                    memory_config=dram,
                )
            )
            low_rank_ws = ttnn.silu(reduced_tile, memory_config=self.up_act_memory_config)
            _deallocate(reduced_tile)
            if _shape(low_rank_ws) != low_rank_tile_shape:
                raise RuntimeError(f"GR rows low-rank tile is {_shape(low_rank_ws)}, expected {low_rank_tile_shape}")
            if low_rank_ws.memory_config() != self.up_act_memory_config:
                raise RuntimeError(f"GR rows low-rank rows have memory config {low_rank_ws.memory_config()}")
            self.mesh_contract.validate_tensor(low_rank_ws, placement=TensorPlacement.REPLICATED)
            up_ws = ttnn.linear(
                low_rank_ws,
                self.weights.up,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.up_program_config,
                compute_kernel_config=self.compute_config,
            )
            _deallocate(low_rank_ws)
            up_tiles.append(ttnn.to_memory_config(up_ws, dram))
            _deallocate(up_ws)
        if slab:
            low_rank = ttnn.silu(reduced_bf16, memory_config=dram)
            up_flat = prefill_linear(
                low_rank,
                self.weights.up,
                self._slab_program_config(rows, PARTIAL_WIDTH, FLAT_LOCAL_WIDTH),
                compute_kernel_config=self.prefill_dense.compute_config(self.compute_config),
                resident_weight=self.prefill_dense.resident("up"),
            )
            _deallocate(low_rank, reduced_bf16)
        elif rows <= CHUNK_ROWS:
            up_flat = up_tiles[0]
        else:
            up_flat = ttnn.concat(up_tiles, dim=2, memory_config=dram)
            _deallocate(*up_tiles)
        self.mesh_contract.validate_tensor(up_flat, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        if _shape(up_flat) != flat_rows_shape:
            raise RuntimeError(f"GR rows up projection has unexpected shape {_shape(up_flat)}")
        gate_flat = ttnn.sigmoid(up_flat, memory_config=dram)
        _deallocate(up_flat)
        gated_flat = ttnn.multiply(normalized_ws, gate_flat, memory_config=dram)
        _deallocate(gate_flat, normalized_ws)
        # Flat rows -> branch-major, then the branch mean (the 1/4 is in norm_scale).  At 128 rows the four
        # branch columns are sliced out and stacked on dim 1 (the fold's inverse, no token-major intermediate).
        if flat_views:
            gated = ttnn.experimental.view(gated_flat, residual_rows_shape(rows))  # owns gated_flat's buffer
        elif rows > CHUNK_ROWS:
            branches = [
                ttnn.slice(
                    gated_flat,
                    (0, 0, 0, b * LOCAL_HIDDEN_SIZE),
                    (1, 1, rows, (b + 1) * LOCAL_HIDDEN_SIZE),
                    memory_config=dram,
                )
                for b in range(RESIDUAL_BRANCHES)
            ]
            _deallocate(gated_flat)
            gated = ttnn.concat(branches, dim=1, memory_config=dram)
            _deallocate(*branches)
        else:
            gated_tokens = ttnn.reshape(gated_flat, (1, rows, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE))
            if not same_buffer(gated_tokens, gated_flat):  # a copy today (the last dim changes)
                _deallocate(gated_flat)
            gated = ttnn.permute(gated_tokens, (0, 2, 1, 3), memory_config=dram)
            _deallocate(gated_tokens)
        self._validate_rows(gated, residual_rows_shape(rows), label="GR rows gated unit")
        block_rows = ttnn.experimental.fast_reduce_nc(
            gated, dims=[1], output=None, compute_kernel_config=self.compute_config, memory_config=dram
        )
        _deallocate(gated)
        if rows < CHUNK_ROWS and _shape(block_rows) == (1, 1, CHUNK_ROWS, LOCAL_HIDDEN_SIZE):
            block_rows = ttnn.reshape(block_rows, block_rows_shape(rows), block_rows.padded_shape)  # the tile as rows
        self._validate_rows(block_rows, block_rows_shape(rows), label="GR rows block input")

        injection = ttnn.multiply(
            inject_rows,
            2.0,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0)],
            memory_config=dram,
        )
        _deallocate(inject_rows)
        self.mesh_contract.validate_tensor(injection, placement=TensorPlacement.REPLICATED)
        if _shape(injection) != injection_rows_shape(rows):
            raise RuntimeError(
                f"GR rows injection has shape {_shape(injection)}, expected {injection_rows_shape(rows)}"
            )
        return block_rows, Qwen38TTNNGatedResidualState(residual=residual_rows, injection=injection)

    def write_rows(self, block_rows, state: Qwen38TTNNGatedResidualState):
        """:meth:`write` for the block rows of a chunk into its branch-major residual rows."""

        rows = _shape(state.residual)[2]
        self._validate_rows(block_rows, block_rows_shape(rows), label="GR block rows")
        self._validate_rows(state.residual, residual_rows_shape(rows), label="GR residual rows")
        self.mesh_contract.validate_tensor(state.injection, placement=TensorPlacement.REPLICATED)
        if _shape(state.injection) != injection_rows_shape(rows):
            raise ValueError(
                f"GR rows injection must have shape {injection_rows_shape(rows)}, got {_shape(state.injection)}"
            )
        # [1,1,rows,4] -> [1,4,rows,1]: branch b's coefficient column, broadcast over the local hidden.
        coefficient = ttnn.permute(state.injection, (0, 3, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if _shape(coefficient) != (1, RESIDUAL_BRANCHES, rows, 1):
            raise RuntimeError(f"GR rows coefficient has shape {_shape(coefficient)}, expected [1,4,{rows},1]")
        update = ttnn.multiply(block_rows, coefficient, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(coefficient)
        self._validate_rows(update, residual_rows_shape(rows), label="GR rows branch update")
        output = ttnn.add(state.residual, update, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(update)
        self._validate_rows(output, residual_rows_shape(rows), label="GR rows output")
        return output
