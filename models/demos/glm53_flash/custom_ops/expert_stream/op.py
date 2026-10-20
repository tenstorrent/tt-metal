# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash routed experts as DRAM-bank-streaming generic ops (one chip's slice of the hybrid placement).

After ``models/demos/gpt_oss/tt/fused_decode/stream.py`` (ExpertGateUpStream / ExpertDownStream), adapted to a
32-token block (M <= 32 rows, 32x32 tiles), the chip's slice of the hybrid placement (E_local = 72 experts of the
column, I_local = 256-wide intermediate slice of the row, K = H = 4096), bfp8 weights, no biases, the router's dense
routing matrix and the clamped SwiGLU (``silu(min(g, 10)) * clamp(u, -10, 10)``).

Weights are stored width-sharded over the DRAM banks (``stream_memory_config``), bank-major and column-major, so a
worker core next to a bank streams a whole weight column (K tiles) as one contiguous byte range in large NOC
packets, expert-major so an expert's columns for a core are contiguous:

* gate|up: ``[E, K, 2 I]`` -> bank ``b`` holds the column-tile pair ``(gate b, up b)`` of every expert
  (``2 I / 32 / 2 = 8`` pairs = the 8 banks); per expert ``2 * K/32`` tiles, ``expert_stride`` bytes.
* down: ``[E, I, H]`` -> bank ``b`` holds output tile columns ``[b C, (b+1) C)`` (``C = H / 32 / banks = 16``) of
  every expert, ``I / 32`` K tiles per column: expert e's columns ``[c0, c0 + cols)`` of a core are contiguous.

Ops (every kernel reads the routing matrix ``[1, 1, M, E]`` BF16 TILE itself and scans the active experts, so the
calls are trace-safe with data-dependent routing and need no host-side counts):

* ``ExpertGateUpStream(x, routing, w13, act)``: ``G`` groups of 8 cores (one core per bank per group); group ``g``
  streams the active experts ``g, g + G, ...``; per expert and bank: gate and up columns against the resident
  ``[32, K]`` activation (HiFi2, fp32 acc), ``clamped_silu_glu`` in DST, the routing weights as ``D_e @ h``
  (``D_e = diag(w[:, e])``, exact), written to ``act [1, E, 32, I]`` (tile ``(e, b)``).
* ``ExpertDownStream(act, routing, w2, out)``: ``R`` readers per bank, ``cols = C / R`` output columns each
  (``<= 8``: the fp32 DST capacity); for every active expert the core multiplies ``act[e]`` (``I/32`` tiles) against
  its columns' segments, accumulating the expert sum in DST, and writes its ``[32, 32]`` output tiles of
  ``out [1, 1, 32, H]`` once (zeros when no expert is active).

Exactness: the same products (BF16 x BFP8, HiFi2) and fp32 accumulation as the ``sparse_matmul`` block, the expert
sum in fp32 (the slab path sums BF16 tiles), the routing weight as one BF16 x BF16 product per element (bit-exact
with the multiply it replaces). ``golden`` is the torch reference on the bfp8-rounded weights.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
BF16_TILE_BYTES = 2048
BFP8_TILE_BYTES = 1088
MAX_PACKET_BYTES = 16384  # Blackhole NOC max burst
SWIGLU_LIMIT = 10.0  # the fused clamped_silu_glu SFPU op's compile-time limit (GLM-5.3-Flash swiglu_limit)


# ---------------------------------------------------------------------------------------------------- layouts
def _bank_columns(cols: torch.Tensor, banks: int) -> torch.Tensor:
    """[E, C * banks, k, 32, 32] column tiles -> [rows, banks * 32]: bank b's shard = its C columns, expert-major,
    as one column of tiles."""
    C = cols.shape[1] // banks
    return torch.cat([cols[:, b * C : (b + 1) * C].reshape(-1, TILE) for b in range(banks)], dim=-1)


def gate_up_stream_layout(w13: torch.Tensor, banks: int = 8) -> torch.Tensor:
    """[E, K, 2 I] (``[gate | up]`` columns) -> [rows, banks * 32]: bank b holds (gate tile column b, up tile column
    b) of every expert, expert -> (gate, up) -> K tile. Requires I / 32 == banks."""
    E, K, W = w13.shape
    half = W // TILE // 2
    assert half == banks, f"{half} column pairs must equal the {banks} banks (I = 32 * banks)"
    cols = w13.reshape(E, K // TILE, TILE, W // TILE, TILE).permute(0, 3, 1, 2, 4)  # [E, n, k, 32, 32]
    order = [c for b in range(banks) for c in (b, half + b)]
    return _bank_columns(cols[:, order], banks)


def down_stream_layout(w2: torch.Tensor, banks: int = 8) -> torch.Tensor:
    """[E, I, H] -> [rows, banks * 32]: bank b holds output tile columns [b C, (b+1) C), expert -> column -> K tile."""
    E, I, H = w2.shape
    assert (H // TILE) % banks == 0, (H, banks)
    cols = w2.reshape(E, I // TILE, TILE, H // TILE, TILE).permute(0, 3, 1, 2, 4)
    return _bank_columns(cols, banks)


def gate_up_rows(E: int, K: int) -> int:
    return E * 2 * (K // TILE) * TILE


def down_rows(E: int, I: int, H: int, banks: int = 8) -> int:
    return E * (H // TILE // banks) * (I // TILE) * TILE


def stream_memory_config(rows: int, banks: int = 8) -> ttnn.MemoryConfig:
    """DRAM width-sharded over the banks: bank b holds the [rows, 32] shard b."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (rows, TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def upload_stream_weights(device, layout: torch.Tensor, dtype=ttnn.bfloat8_b, mesh_mapper=None) -> ttnn.Tensor:
    """One chip: [rows, banks * 32] -> the width-sharded DRAM tensor [1, 1, rows, banks * 32]. On a mesh: the
    chips' layouts stacked [D, rows, banks * 32] with ``mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0)`` (each
    chip its own [1, 1, rows, banks * 32] shard)."""
    rows = layout.shape[-2]
    lead = layout.shape[0] if layout.dim() == 3 else 1
    return ttnn.from_torch(
        layout.reshape(lead, 1, rows, -1).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=stream_memory_config(rows),
        mesh_mapper=mesh_mapper,
    )


# ---------------------------------------------------------------------------------------------------- golden
def bfp8_round(w: torch.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


def expert_stream_golden(
    x: torch.Tensor, routing: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, limit=SWIGLU_LIMIT
):
    """x [M, K] bf16, routing [M, E] (bf16 weights, 0 = not routed), w13 [E, K, 2I], w2 [E, I, H] (bfp8-rounded
    floats) -> (act [E, M, I], y [M, H]) in fp32: act[e] = w[:, e, None] * silu(min(x W_g, limit)) * clamp(x W_u),
    y = sum_e act[e] @ W2[e] (inactive experts' act rows are zero)."""
    M, K = x.shape
    E, _, W = w13.shape
    I = W // 2
    xf = x.float()
    act = torch.zeros(E, M, I)
    y = torch.zeros(M, w2.shape[-1])
    rw = routing.float()
    for e in range(E):
        if rw[:, e].abs().sum() == 0:
            continue
        g = (xf @ w13[e, :, :I]).clamp(max=limit)
        u = (xf @ w13[e, :, I:]).clamp(-limit, limit)
        h = torch.nn.functional.silu(g) * u
        act[e] = rw[:, e : e + 1] * h.bfloat16().float()  # the device packs h to bf16 before the weight product
        y += act[e].bfloat16().float() @ w2[e]
    return act, y


# ---------------------------------------------------------------------------------------------------- placement
class _BankStreamOp:
    """Worker placement shared by the streaming ops (after gpt_oss fused_decode/stream.py): ``readers`` cores per
    DRAM bank, from the bank's optimal NOC0 worker core outwards (same row first, then the rows above / below), each
    with a NOC virtual channel distinct from earlier cores on its row; coalesced into core ranges."""

    def __init__(self, device, readers: int):
        self.device = device
        self.banks = device.dram_grid_size().x
        self.readers = readers
        bank_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        assert len(bank_workers) == self.banks, (len(bank_workers), self.banks)
        grid = device.compute_with_storage_grid_size()
        used = {(c.x, c.y) for c in bank_workers}
        placed = []
        for bank, core in enumerate(bank_workers):
            cores = [core]
            # candidates by distance: same row first, then neighbouring rows
            cands = sorted(
                ((abs(x - core.x) + 4 * abs(y - core.y), x, y) for y in range(grid.y) for x in range(grid.x)),
            )
            for _, x, y in cands:
                if len(cores) >= readers:
                    break
                if (x, y) not in used:
                    cores.append(ttnn.CoreCoord(x, y))
                    used.add((x, y))
            assert len(cores) == readers, f"no room for {readers} readers for bank {bank}"
            placed += [(c, bank, r) for r, c in enumerate(cores)]
        vcs = []
        for i, (core, bank, r) in enumerate(placed):
            vc = (bank + r) & 0x3
            for j in range(i):
                if placed[j][0].y == core.y and vcs[j] == vc:
                    vc = (vc + 1) & 0x3
            vcs.append(vc)
        self.assignments = [(c, b, r, v) for (c, b, r), v in zip(placed, vcs)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.assignments[0][0], self.assignments[0][0])])
        for c, *_ in self.assignments[1:]:
            self.cores = self.cores.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))

    def _cb(self, index: int, dtype, page: int, pages: int) -> ttnn.CBDescriptor:
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    def _program(
        self, name: str, cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct
    ) -> ttnn.ProgramDescriptor:
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"reader_{name}.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"writer_{name}.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"compute_{name}.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


class ExpertGateUpStream(_BankStreamOp):
    """(x, routing, w13, act) -> act. x [1, 1, 32, K] BF16 TILE (M <= 32 token rows, zero-padded); routing
    [1, 1, 32, E] BF16 TILE; w13: ``upload_stream_weights(gate_up_stream_layout(...))``; act [1, E, 32, I] BF16
    TILE DRAM (written at the active experts' tiles). ``groups`` = cores per bank = how many experts stream in
    parallel; the activation (K / 32 tiles, 256 KB at K = 4096) stays resident on every core."""

    def __init__(self, device, E: int, K: int, I: int, groups: int = 12):
        super().__init__(device, groups)
        self.E, self.K, self.I = E, K, I
        self.kt = K // TILE
        self.i_tiles = I // TILE
        assert self.i_tiles == self.banks, f"I / 32 = {self.i_tiles} tile columns must equal the {self.banks} banks"
        self.groups = groups
        self.col_bytes = self.kt * BFP8_TILE_BYTES
        self.expert_stride = 2 * self.col_bytes
        self.page_bytes = (
            max(d for d in range(1, self.kt + 1) if self.kt % d == 0 and d * BFP8_TILE_BYTES <= MAX_PACKET_BYTES)
            * BFP8_TILE_BYTES
        )

    def __call__(self, x, routing, w13, act):
        cb_x, cb_w, cb_r0, cb_r1, cb_d, cb_h, cb_out = 0, 1, 2, 3, 4, 5, 16
        e_tiles = -(-self.E // TILE)
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, BF16_TILE_BYTES, self.kt),
            self._cb(cb_w, ttnn.bfloat8_b, BFP8_TILE_BYTES, 3 * self.kt),
            self._cb(cb_r0, ttnn.bfloat16, BF16_TILE_BYTES, e_tiles),
            self._cb(cb_r1, ttnn.bfloat16, BF16_TILE_BYTES, e_tiles),
            self._cb(cb_d, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            self._cb(cb_h, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            self._cb(cb_out, ttnn.bfloat16, BF16_TILE_BYTES, 2),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [w13.buffer_address(), routing.buffer_address(), bank, vc, r]
            writer_rt[core.x][core.y] = [x.buffer_address(), routing.buffer_address(), act.buffer_address(), bank, r]
        reader_ct = [
            cb_w,
            cb_r0,
            self.kt,
            self.E,
            TILE,
            self.groups,
            self.expert_stride,
            BFP8_TILE_BYTES,
            self.page_bytes,
        ]
        reader_ct += ttnn.TensorAccessorArgs(routing).get_compile_time_args()
        writer_ct = [cb_x, cb_r1, cb_d, cb_out, self.kt, self.E, TILE, self.groups, self.i_tiles]
        for t in (x, routing, act):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_d, cb_h, cb_out, self.kt]
        program = self._program("gate_up", cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct)
        ttnn.generic_op([w13, x, routing, act], program)
        return act


class ExpertDownStream(_BankStreamOp):
    """(act, routing, w2, out) -> out. act [1, E, 32, I] BF16 TILE (the gate|up stream's output), routing as above,
    w2: ``upload_stream_weights(down_stream_layout(...))``, out [1, 1, 32, H] BF16 TILE DRAM (the expert sum).
    ``readers`` cores per bank split the bank's C = H / 32 / banks output columns (cols = C / readers <= 8)."""

    def __init__(self, device, E: int, I: int, H: int, readers: int = 2):
        super().__init__(device, readers)
        self.E, self.I, self.H = E, I, H
        self.seg_tiles = I // TILE
        self.C = H // TILE // self.banks
        assert self.C % readers == 0, f"{self.C} columns per bank do not split over {readers} readers"
        self.cols = self.C // readers
        assert self.cols <= 8, f"{self.cols} output columns per core exceed the 8 fp32 DST tiles"
        self.seg_bytes = self.seg_tiles * BFP8_TILE_BYTES
        assert self.seg_bytes <= MAX_PACKET_BYTES
        self.bank_expert_stride = self.C * self.seg_bytes

    def __call__(self, act, routing, w2, out):
        cb_a, cb_w, cb_r0, cb_r1, cb_out = 0, 1, 2, 3, 16
        e_tiles = -(-self.E // TILE)
        cbs = [
            self._cb(cb_a, ttnn.bfloat16, BF16_TILE_BYTES, 2 * self.seg_tiles),
            self._cb(cb_w, ttnn.bfloat8_b, BFP8_TILE_BYTES, 3 * self.cols * self.seg_tiles),
            self._cb(cb_r0, ttnn.bfloat16, BF16_TILE_BYTES, e_tiles),
            self._cb(cb_r1, ttnn.bfloat16, BF16_TILE_BYTES, e_tiles),
            self._cb(cb_out, ttnn.bfloat16, BF16_TILE_BYTES, self.cols),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [w2.buffer_address(), routing.buffer_address(), bank, vc, r * self.cols]
            writer_rt[core.x][core.y] = [
                routing.buffer_address(),
                act.buffer_address(),
                out.buffer_address(),
                bank * self.C + r * self.cols,
            ]
        reader_ct = [cb_w, cb_r0, self.seg_tiles, self.E, TILE, self.cols, self.bank_expert_stride, BFP8_TILE_BYTES]
        reader_ct += ttnn.TensorAccessorArgs(routing).get_compile_time_args()
        writer_ct = [cb_r1, cb_a, cb_out, self.seg_tiles, self.E, TILE, self.cols]
        for t in (routing, act, out):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_a, cb_w, cb_out, self.seg_tiles, self.cols]
        program = self._program("down", cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct)
        ttnn.generic_op([w2, act, routing, out], program)
        return out


class ExpertStream:
    """The chip's routed-expert block as the two streaming ops: ``forward(x, routing, act, out)``."""

    def __init__(self, device, E: int, K: int, I: int, H: int, groups: int = 12, down_readers: int = 2):
        self.gate_up = ExpertGateUpStream(device, E, K, I, groups)
        self.down = ExpertDownStream(device, E, I, H, down_readers)
        self.E, self.K, self.I, self.H = E, K, I, H

    def alloc(self, device, act_l1: bool = False):
        """The act and out buffers for one call (allocate inside a trace or keep persistent). act_l1 keeps the
        E x 32 x I activations in L1 interleaved (1.2 MB at the chip shapes, ~2 % off the down op)."""
        act = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.E, TILE, self.I]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG if act_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        return act, out

    def forward(self, x, routing, w13, w2, act, out):
        self.gate_up(x, routing, w13, act)
        self.down(act, routing, w2, out)
        return out

    __call__ = forward
    golden = staticmethod(expert_stream_golden)  # (x [M,K], routing [M,E], w13 [E,K,2I], w2 [E,I,H]) -> (act, y)
