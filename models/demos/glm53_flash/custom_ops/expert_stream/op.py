# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash routed experts as DRAM-bank-streaming generic ops (one chip's slice of the hybrid placement).

After ``models/demos/gpt_oss/tt/fused_decode/stream.py`` (ExpertGateUpStream / ExpertDownStream), adapted to the
chip's slice of the hybrid placement (E_local experts of the column, I_local-wide intermediate slice of the row:
256 on the 8x4, 1024 on a (2,4) block, 2048 on one chip; K = H = 4096), bfp8 weights, no biases, the router's dense
routing matrix and the clamped SwiGLU (``silu(min(g, 10)) * clamp(u, -10, 10)``).

Weights are stored width-sharded over the DRAM banks (``stream_memory_config``), bank-major and column-major, so a
worker core next to a bank streams whole weight columns (K tiles each) as contiguous byte ranges in large NOC
packets, expert-major so an expert's columns for a core are contiguous:

* gate|up: ``[E, K, 2 I]`` -> ``ppb = I / 32 / banks`` column pairs per bank: bank ``b`` holds the pairs
  ``(gate b ppb + q, up b ppb + q)``, ``q < ppb``, of every expert (``2 ppb`` columns of ``K/32`` tiles per expert,
  ``expert_stride`` bytes).
* down: ``[E, I, H]`` -> bank ``b`` holds output tile columns ``[b C, (b+1) C)`` (``C = H / 32 / banks = 16``) of
  every expert, ``I / 32`` K tiles per column: expert e's columns ``[c0, c0 + cols)`` of a core are contiguous.

Block ops (every kernel reads the routing matrix ``[1, 1, 32, E]`` BF16 TILE itself and scans the active experts, so
the calls are trace-safe with data-dependent routing and need no host-side counts); one 32-token block per call:

* ``ExpertGateUpStream(x, routing, w13, act)``: ``G`` groups of 8 cores (one core per bank per group); group ``g``
  streams the active experts ``g, g + G, ...``; per expert and column pair: gate and up columns against the resident
  ``[32, K]`` activation (HiFi2, fp32 acc), ``clamped_silu_glu`` in DST, the routing weights as ``D_e @ h``
  (``D_e = diag(w[:, e])``, exact), written to ``act [1, E, 32, I]`` (tile ``(e, b ppb + q)``).
* ``ExpertDownStream(act, routing, w2, out)``: ``R`` readers per bank, ``cols = C / R`` output columns each
  (``<= 8``: the fp32 DST capacity, ``cols * I/32 <= 128`` tiles per expert block); for every active expert the core
  multiplies ``act[e]`` (``I/32`` tiles) against its columns' segments, accumulating the expert sum in DST, and
  writes its ``[32, 32]`` output tiles of ``out [1, 1, 32, H]`` once (zeros when no expert is active).

Dispatch ops (phase 2, ``T`` gathered tokens > 32, every active expert's weights read once): the per-expert token
counts ``counts = sum_t (routing != 0)`` (``[1, 1, 1, E]`` BF16, two ttnn ops) give every kernel the same pass
enumeration (expert ascending, 32 routed rows per pass) and the same LPT expert -> core-group assignment:

* ``ExpertGateUpDispatch(x_rm, routing, counts, w13, act, table)``: per pass the writer gathers the pass's rows of the
  ROW_MAJOR ``x [1, 1, T, K]`` (one 8 KB read per row) into a row-major block that the compute tilizes, then the
  gate|up / SwiGLU / ``D @ h`` as above into ``act [1, P, 32, I]`` (``P = T/4 + E`` passes at most); bank 0 writes
  the pass table ``[1, 1, P, 64]`` uint16 (``[n_rows, e, t_0 .. t_31]``). With one pair per bank an expert's columns
  stay resident over its passes; with more pairs they are re-streamed per pass.
* ``ExpertDownDispatch(act, table, counts, w2, out)``: 8 readers per bank (2 output columns each, 64 cores); phase 1
  computes ``h_k = act[k] @ W2[e][:, cols]`` per pass into a resident BF16 buffer, phase 2 gathers every token's
  contributions (``<= 8`` per token) into slot tiles and sums them in fp32 DST (identity-tile matmuls, HiFi4) into
  ``out [1, 1, T, H]``.

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
MAX_PICKS = 8  # routed experts per token (top-8): bounds the passes and the gather slots of the dispatch ops
TABLE_WIDTH = 64  # uint16 entries per pass-table row: [n_rows, e, t_0 .. t_31, pad]


def _packet_bytes(tiles: int) -> int:
    """The largest NOC packet (a divisor of `tiles` bfp8 tiles) that fits the max burst."""
    return (
        max(d for d in range(1, tiles + 1) if tiles % d == 0 and d * BFP8_TILE_BYTES <= MAX_PACKET_BYTES)
        * BFP8_TILE_BYTES
    )


# ---------------------------------------------------------------------------------------------------- layouts
def _bank_columns(cols: torch.Tensor, banks: int) -> torch.Tensor:
    """[E, C * banks, k, 32, 32] column tiles -> [rows, banks * 32]: bank b's shard = its C columns, expert-major,
    as one column of tiles."""
    C = cols.shape[1] // banks
    return torch.cat([cols[:, b * C : (b + 1) * C].reshape(-1, TILE) for b in range(banks)], dim=-1)


def gate_up_stream_layout(w13: torch.Tensor, banks: int = 8) -> torch.Tensor:
    """[E, K, 2 I] (``[gate | up]`` columns) -> [rows, banks * 32]: bank b holds the column pairs
    (gate b ppb + q, up b ppb + q), q < ppb = I / 32 / banks, of every expert: expert -> pair -> (gate, up) -> K tile.
    """
    E, K, W = w13.shape
    half = W // TILE // 2
    assert half % banks == 0, f"{half} column pairs do not split over the {banks} banks (I must be a multiple of 256)"
    ppb = half // banks
    cols = w13.reshape(E, K // TILE, TILE, W // TILE, TILE).permute(0, 3, 1, 2, 4)  # [E, n, k, 32, 32]
    order = [c for b in range(banks) for q in range(ppb) for c in (b * ppb + q, half + b * ppb + q)]
    return _bank_columns(cols[:, order], banks)


def down_stream_layout(w2: torch.Tensor, banks: int = 8) -> torch.Tensor:
    """[E, I, H] -> [rows, banks * 32]: bank b holds output tile columns [b C, (b+1) C), expert -> column -> K tile."""
    E, I, H = w2.shape
    assert (H // TILE) % banks == 0, (H, banks)
    cols = w2.reshape(E, I // TILE, TILE, H // TILE, TILE).permute(0, 3, 1, 2, 4)
    return _bank_columns(cols, banks)


def gate_up_rows(E: int, K: int, I: int, banks: int = 8) -> int:
    return E * 2 * (I // TILE // banks) * (K // TILE) * TILE


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


def dispatch_passes(routing: torch.Tensor) -> list[tuple[int, list[int]]]:
    """The dispatch ops' pass enumeration of a [T, E] routing matrix: (expert, token rows) per pass, experts
    ascending, 32 rows per pass."""
    passes = []
    for e in range(routing.shape[1]):
        rows = torch.nonzero(routing[:, e] != 0).flatten().tolist()
        for p in range(0, len(rows), TILE):
            passes.append((e, rows[p : p + TILE]))
    return passes


def expert_dispatch_golden(
    x: torch.Tensor, routing: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, limit=SWIGLU_LIMIT
):
    """As ``expert_stream_golden`` for T tokens plus the per-pass activations the dispatch gate|up op writes:
    (act_pass [P, 32, I] (rows past a pass's count zero), y [T, H], passes)."""
    act, y = expert_stream_golden(x, routing, w13, w2, limit)
    passes = dispatch_passes(routing)
    act_pass = torch.zeros(len(passes), TILE, act.shape[-1])
    for k, (e, rows) in enumerate(passes):
        act_pass[k, : len(rows)] = act[e, rows]
    return act_pass, y, passes


def routing_counts(routing: ttnn.Tensor) -> ttnn.Tensor:
    """[1, 1, T, E] BF16 routing matrix -> [1, 1, 1, E] BF16 per-expert token counts (exact up to T = 256)."""
    return ttnn.sum(ttnn.ne(routing, 0.0), dim=2, keepdim=True)


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
        self,
        name: str,
        cbs,
        reader_ct,
        reader_rt,
        writer_ct,
        writer_rt,
        compute_ct,
        fidelity=ttnn.MathFidelity.HiFi2,
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
                    math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def _down_readers(C: int, seg_tiles: int) -> int:
    """Readers per bank for the down ops: the fewest that keep cols <= 8 (fp32 DST) and an expert's block of
    cols * seg_tiles tiles <= 128 (three 139 KB slots): 2 at I = 256, 4 at 1024, 8 at 2048."""
    for r in (1, 2, 4, 8, 16):
        if C % r == 0 and C // r <= 8 and (C // r) * seg_tiles <= 128:
            return r
    raise ValueError(f"no reader split for {C} columns x {seg_tiles} K tiles")


# ---------------------------------------------------------------------------------------------------- block ops
class ExpertGateUpStream(_BankStreamOp):
    """(x, routing, w13, act) -> act. x [1, 1, 32, K] BF16 TILE (M <= 32 token rows, zero-padded); routing
    [1, 1, 32, E] BF16 TILE; w13: ``upload_stream_weights(gate_up_stream_layout(...))``; act [1, E, 32, I] BF16
    TILE (written at the active experts' tiles). ``groups`` = cores per bank = how many experts stream in
    parallel; the activation (K / 32 tiles, 256 KB at K = 4096) stays resident on every core."""

    def __init__(self, device, E: int, K: int, I: int, groups: int = 12):
        super().__init__(device, groups)
        self.E, self.K, self.I = E, K, I
        self.kt = K // TILE
        self.i_tiles = I // TILE
        assert (
            self.i_tiles % self.banks == 0
        ), f"I / 32 = {self.i_tiles} tile columns do not split over {self.banks} banks"
        self.ppb = self.i_tiles // self.banks  # column pairs per bank
        self.groups = groups
        self.col_bytes = self.kt * BFP8_TILE_BYTES
        self.expert_stride = 2 * self.ppb * self.col_bytes
        self.page_bytes = _packet_bytes(self.kt)

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
            self.ppb,
        ]
        reader_ct += ttnn.TensorAccessorArgs(routing).get_compile_time_args()
        writer_ct = [cb_x, cb_r1, cb_d, cb_out, self.kt, self.E, TILE, self.groups, self.i_tiles, self.ppb]
        for t in (x, routing, act):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_d, cb_h, cb_out, self.kt, self.ppb]
        program = self._program("gate_up", cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct)
        ttnn.generic_op([w13, x, routing, act], program)
        return act


class ExpertDownStream(_BankStreamOp):
    """(act, routing, w2, out) -> out. act [1, E, 32, I] BF16 TILE (the gate|up stream's output), routing as above,
    w2: ``upload_stream_weights(down_stream_layout(...))``, out [1, 1, 32, H] BF16 TILE DRAM (the expert sum).
    ``readers`` cores per bank split the bank's C = H / 32 / banks output columns (cols = C / readers <= 8;
    None = the fewest readers whose expert block fits, ``_down_readers``)."""

    def __init__(self, device, E: int, I: int, H: int, readers: int | None = None):
        banks = device.dram_grid_size().x
        self.seg_tiles = I // TILE
        self.C = H // TILE // banks
        if readers is None:
            readers = _down_readers(self.C, self.seg_tiles)
        super().__init__(device, readers)
        self.E, self.I, self.H = E, I, H
        assert self.C % readers == 0, f"{self.C} columns per bank do not split over {readers} readers"
        self.cols = self.C // readers
        assert self.cols <= 8, f"{self.cols} output columns per core exceed the 8 fp32 DST tiles"
        self.seg_bytes = self.seg_tiles * BFP8_TILE_BYTES
        self.block_tiles = self.cols * self.seg_tiles
        self.page_bytes = _packet_bytes(self.seg_tiles)
        self.bank_expert_stride = self.C * self.seg_bytes

    def __call__(self, act, routing, w2, out):
        cb_a, cb_w, cb_r0, cb_r1, cb_out = 0, 1, 2, 3, 16
        e_tiles = -(-self.E // TILE)
        cbs = [
            self._cb(cb_a, ttnn.bfloat16, BF16_TILE_BYTES, 2 * self.seg_tiles),
            self._cb(cb_w, ttnn.bfloat8_b, BFP8_TILE_BYTES, 3 * self.block_tiles),
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
        reader_ct = [
            cb_w,
            cb_r0,
            self.seg_tiles,
            self.E,
            TILE,
            self.cols,
            self.bank_expert_stride,
            BFP8_TILE_BYTES,
            self.page_bytes,
        ]
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

    def __init__(self, device, E: int, K: int, I: int, H: int, groups: int = 12, down_readers: int | None = None):
        self.gate_up = ExpertGateUpStream(device, E, K, I, groups)
        self.down = ExpertDownStream(device, E, I, H, down_readers)
        self.E, self.K, self.I, self.H = E, K, I, H

    def alloc(self, device, act_l1: bool = False):
        """The act and out buffers for one call (allocate inside a trace or keep persistent). act_l1 keeps the
        E x 32 x I activations in L1 interleaved (1.2 MB at the 8x4 shapes, ~2 % off the down op)."""
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


# ---------------------------------------------------------------------------------------------------- dispatch ops
def max_passes(T: int, E: int) -> int:
    """Upper bound of the pass count: sum_e ceil(n_e / 32) with sum_e n_e <= MAX_PICKS * T."""
    return MAX_PICKS * T // TILE + E


class ExpertGateUpDispatch(_BankStreamOp):
    """(x_rm, routing, counts, w13, act, table) -> act, table. x_rm [1, 1, T, K] BF16 ROW_MAJOR; routing
    [1, 1, T, E] BF16 TILE; counts = ``routing_counts(routing)``; w13 as the block op; act [1, P, 32, I] BF16 TILE
    (the per-pass activations, P = ``max_passes``); table [1, 1, P, 64] UINT16 ROW_MAJOR (the pass rows)."""

    def __init__(self, device, E: int, K: int, I: int, T: int, groups: int = 12):
        super().__init__(device, groups)
        assert T % TILE == 0 and T <= 256, T  # counts are exact BF16 integers up to 256
        assert groups <= 32, groups
        self.E, self.K, self.I, self.T = E, K, I, T
        self.kt = K // TILE
        self.i_tiles = I // TILE
        assert (
            self.i_tiles % self.banks == 0
        ), f"I / 32 = {self.i_tiles} tile columns do not split over {self.banks} banks"
        self.ppb = self.i_tiles // self.banks
        self.groups = groups
        self.col_bytes = self.kt * BFP8_TILE_BYTES
        self.expert_stride = 2 * self.ppb * self.col_bytes
        self.page_bytes = _packet_bytes(self.kt)
        self.row_bytes = K * 2
        self.P = max_passes(T, E)

    def __call__(self, x_rm, routing, counts, w13, act, table):
        cb_xrm, cb_x, cb_w, cb_r, cb_c0, cb_c1, cb_d, cb_h, cb_t, cb_out = 0, 1, 2, 3, 4, 5, 6, 7, 8, 16
        e_tiles = -(-self.E // TILE)
        rt = self.T // TILE
        cbs = [
            self._cb(cb_xrm, ttnn.bfloat16, BF16_TILE_BYTES, self.kt),  # 32 row-major rows of K: kt tile-sized pages
            self._cb(cb_x, ttnn.bfloat16, BF16_TILE_BYTES, self.kt),
            self._cb(cb_w, ttnn.bfloat8_b, BFP8_TILE_BYTES, 4 * self.kt),  # two (gate, up) pair slots
            self._cb(cb_r, ttnn.bfloat16, BF16_TILE_BYTES, rt * e_tiles),
            self._cb(cb_c0, ttnn.bfloat16, BF16_TILE_BYTES, e_tiles),
            self._cb(cb_c1, ttnn.bfloat16, BF16_TILE_BYTES, e_tiles),
            self._cb(cb_d, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            self._cb(cb_h, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            self._cb(cb_t, ttnn.uint16, TABLE_WIDTH * 2, 2),
            self._cb(cb_out, ttnn.bfloat16, BF16_TILE_BYTES, 2),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [w13.buffer_address(), counts.buffer_address(), bank, vc, r]
            writer_rt[core.x][core.y] = [
                x_rm.buffer_address(),
                routing.buffer_address(),
                counts.buffer_address(),
                act.buffer_address(),
                table.buffer_address(),
                bank,
                r,
            ]
        reader_ct = [
            cb_w,
            cb_c0,
            self.kt,
            self.E,
            self.groups,
            self.expert_stride,
            BFP8_TILE_BYTES,
            self.page_bytes,
            self.ppb,
        ]
        reader_ct += ttnn.TensorAccessorArgs(counts).get_compile_time_args()
        writer_ct = [
            cb_xrm,
            cb_r,
            cb_c1,
            cb_d,
            cb_t,
            cb_out,
            self.kt,
            self.E,
            self.T,
            self.groups,
            self.i_tiles,
            self.row_bytes,
            TABLE_WIDTH * 2,
            self.ppb,
        ]
        for t in (x_rm, routing, counts, act, table):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_xrm, cb_x, cb_w, cb_d, cb_h, cb_out, self.kt, self.ppb]
        program = self._program("gate_up_dispatch", cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct)
        ttnn.generic_op([w13, x_rm, routing, counts, act, table], program)
        return act


class ExpertDownDispatch(_BankStreamOp):
    """(act, table, counts, w2, out) -> out. act / table from the gate|up dispatch, counts as above, w2 as the block
    op, out [1, 1, T, H] BF16 TILE. ``readers`` cores per bank (default 8: 2 output columns each; the resident
    per-pass h buffer is P x cols tiles)."""

    def __init__(self, device, E: int, I: int, H: int, T: int, readers: int = 8):
        super().__init__(device, readers)
        assert T % TILE == 0 and T <= 256, T
        self.E, self.I, self.H, self.T = E, I, H, T
        self.seg_tiles = I // TILE
        self.C = H // TILE // self.banks
        assert self.C % readers == 0, (self.C, readers)
        self.cols = self.C // readers
        assert self.cols <= 8, self.cols
        self.block_tiles = self.cols * self.seg_tiles
        assert self.block_tiles <= 128, f"{self.block_tiles} tiles per expert block: use more readers"
        self.seg_bytes = self.seg_tiles * BFP8_TILE_BYTES
        self.page_bytes = _packet_bytes(self.seg_tiles)
        self.bank_expert_stride = self.C * self.seg_bytes
        self.P = max_passes(T, E)
        self.J = MAX_PICKS

    def __call__(self, act, table, counts, w2, out):
        cb_a, cb_w, cb_c0, cb_c1, cb_t, cb_id, cb_h, cb_g, cb_m, cb_out = 0, 1, 2, 3, 4, 5, 6, 7, 8, 16
        e_tiles = -(-self.E // TILE)
        cbs = [
            self._cb(cb_a, ttnn.bfloat16, BF16_TILE_BYTES, 6 * self.seg_tiles),  # two batches of three pass slots
            self._cb(cb_w, ttnn.bfloat8_b, BFP8_TILE_BYTES, 3 * self.block_tiles),
            self._cb(cb_c0, ttnn.bfloat16, BF16_TILE_BYTES, e_tiles),
            self._cb(cb_c1, ttnn.bfloat16, BF16_TILE_BYTES, e_tiles),
            self._cb(cb_t, ttnn.uint16, TABLE_WIDTH * 2, self.P),
            self._cb(cb_id, ttnn.bfloat16, BF16_TILE_BYTES, 1),
            self._cb(cb_h, ttnn.bfloat16, BF16_TILE_BYTES, self.P * self.cols),
            self._cb(cb_g, ttnn.bfloat16, BF16_TILE_BYTES, 2 * self.J * self.cols),
            self._cb(cb_m, ttnn.uint32, 512, 1),  # [n_active, passes per active expert]
            self._cb(cb_out, ttnn.bfloat16, BF16_TILE_BYTES, 2 * self.cols),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [w2.buffer_address(), counts.buffer_address(), bank, vc, r * self.cols]
            writer_rt[core.x][core.y] = [
                counts.buffer_address(),
                table.buffer_address(),
                act.buffer_address(),
                out.buffer_address(),
                bank * self.C + r * self.cols,
            ]
        reader_ct = [
            cb_w,
            cb_c0,
            self.seg_tiles,
            self.E,
            self.cols,
            self.bank_expert_stride,
            BFP8_TILE_BYTES,
            self.page_bytes,
        ]
        reader_ct += ttnn.TensorAccessorArgs(counts).get_compile_time_args()
        writer_ct = [
            cb_c1,
            cb_t,
            cb_a,
            cb_id,
            cb_h,
            cb_g,
            cb_out,
            self.seg_tiles,
            self.E,
            self.T,
            self.cols,
            self.J,
            self.P,
            self.H // TILE,
            TABLE_WIDTH * 2,
            cb_m,
        ]
        for t in (counts, table, act, out):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_a, cb_w, cb_id, cb_h, cb_g, cb_out, self.seg_tiles, self.cols, self.J, self.T // TILE, cb_m]
        program = self._program(
            "down_dispatch",
            cbs,
            reader_ct,
            reader_rt,
            writer_ct,
            writer_rt,
            compute_ct,
            fidelity=ttnn.MathFidelity.HiFi4,
        )
        ttnn.generic_op([w2, act, table, counts, out], program)
        return out


class ExpertDispatch:
    """The chip's routed experts for T gathered tokens as the two dispatch ops (weights read once):
    ``forward(x_rm, routing, counts, w13, w2, act, table, out)``."""

    def __init__(self, device, E: int, K: int, I: int, H: int, T: int, groups: int = 12, down_readers: int = 8):
        self.gate_up = ExpertGateUpDispatch(device, E, K, I, T, groups)
        self.down = ExpertDownDispatch(device, E, I, H, T, down_readers)
        self.E, self.K, self.I, self.H, self.T = E, K, I, H, T
        self.P = self.gate_up.P

    def alloc(self, device, act_l1: bool = True):
        act = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.P, TILE, self.I]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG if act_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        table = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.P, TABLE_WIDTH]), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.T, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        return act, table, out

    @staticmethod
    def prepare(x, routing):
        """The dispatch inputs from the block inputs: x [1, 1, T, K] BF16 TILE -> ROW_MAJOR (L1), the counts."""
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        return x_rm, routing_counts(routing)

    def forward(self, x_rm, routing, counts, w13, w2, act, table, out):
        self.gate_up(x_rm, routing, counts, w13, act, table)
        self.down(act, table, counts, w2, out)
        return out

    __call__ = forward
    golden = staticmethod(expert_dispatch_golden)
