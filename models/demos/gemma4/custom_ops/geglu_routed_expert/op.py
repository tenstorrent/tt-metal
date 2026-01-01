# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``GegluRoutedExpert``: the Gemma 4 GeGLU routed-expert FFN over a dispatched token buffer as one program.

For every local expert ``e`` (``experts_per_chip`` of them, global id ``idx_table[e]``) the program reads the
expert's rows ``x[start[gid] : start[gid] + counts[gid]]`` of the shared dispatched buffer and writes
``down_e(gelu_tanh(x @ gate_e) * (x @ up_e))`` into the same rows of the output, all device-side: the counts,
region offsets and index table are read by the kernels, so the call is data-independent on the host and
trace-safe. It is the Gemma 4 model-local form of ``ttnn.experimental.deepseek_prefill.unified_routed_expert_moe``
(the DeepSeek dispatch stack's fused routed-expert op) with the GeGLU activation baked in, so nothing outside
``models/demos/gemma4`` carries a Gemma 4 activation variant. Kernels in ``kernels/`` (reader / writer copied from
the shared op without its bias path, compute adapted to ``gelu_tanh_tile``); the program built here mirrors the
shared op's program factory (11x8 worker grid, per-K-block x and weight multicasts, writer-side NoC-1 up read and
down split, runtime chunking from each expert's own count, packer-L1 accumulation, LoFi).

Contract (per device of the mesh; the same program runs on every chip with its own weights and buffer):

| tensor | shape / dtype / layout | role |
|---|---|---|
| ``x`` | ``[M_buf, K]`` (leading 1s allowed) ROW_MAJOR bf16 (the dispatch output, tilized in-op) or TILE bf8 | the dispatched buffer: expert ``e``'s rows at ``start[gid]`` |
| ``counts`` | ``[G]`` or ``[1, G]`` uint32 ROW_MAJOR DRAM | tokens per GLOBAL expert |
| ``region_offsets`` | same as counts | first row of each global expert's region |
| ``idx_table`` | ``[E]`` uint32 ROW_MAJOR DRAM | local slot -> global expert id |
| ``gate_projs`` / ``up_projs`` | ``E`` x ``[K, H]`` TILE bf4/bf8, DRAM interleaved or ND-sharded ``[32, per_core_N*32]`` | per-expert weights (the ``TtRoutedExpert`` layout) |
| ``down_projs`` | ``E`` x ``[H, K]`` | |
| output | ``[M_buf, K]`` TILE bf8 (fresh for ROW_MAJOR x, ``x`` itself in place for TILE x unless ``output`` is given) | expert outputs at the expert regions; other rows untouched |

``golden`` is the torch reference on the same (count, offset, index) contract.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn
from models.demos.gemma4.custom_ops.grid import BLACKHOLE_L1_PER_CORE, compute_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
U32_MAX = 0xFFFFFFFF

TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _nd_shard_n_tiles(w: ttnn.Tensor) -> int:
    """Tile columns per DRAM ND shard of a weight tensor, 0 when it is DRAM-interleaved (the kernels coalesce a
    shard row into one NoC transaction; see kernels/weight_runs.hpp)."""
    mc = w.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM:
        return 0
    spec = mc.nd_shard_spec
    if spec is None:
        return 0
    shape = list(spec.shard_shape)
    if len(shape) < 2 or shape[-1] % TILE != 0:
        return 0
    return shape[-1] // TILE


class GegluRoutedExpert:
    """``GegluRoutedExpert(mesh_device, gate_projs, up_projs, down_projs, idx_table, max_tokens)(x, counts,
    region_offsets)``: see the module docstring for the contract.

    Capacity contract: ``max_tokens`` is the per-expert row capacity (M_tiles_full tiles) every program of the op
    is sized for; the caller guarantees no expert's count exceeds it (the dispatch plan passes the chunk's row
    count, the maximum when each row names an expert at most once -- distinct top-k ids, sentinel pad rows). The
    kernels clamp a larger count to the capacity (adaptive_chunk::clamp_count_tiles) so Release builds stay in
    bounds, and ASSERT on it under the watcher: a clamp is a caller bug, never a silent mode of operation."""

    # The worker rectangle the program always runs on (the shared op's kCoreGridX x kCoreGridY): the K-axis split,
    # the activated L1 multicast and the padded per_core_N all assume it. The kernel family is Blackhole-only; the
    # rectangle is taken from the device grid (``custom_ops/grid.compute_grid``, capped by GEMMA4_TEST_GRID), and a
    # grid that cannot host it (Wormhole 8x8) makes ``unsupported_reason`` non-None: callers and the unit tests skip
    # with that reason instead of failing.
    GRID_X = 11
    GRID_Y = 8

    @classmethod
    def grid_reason(cls, gx: int, gy: int, blackhole: bool) -> str | None:
        """None when the op can run on a ``gx`` x ``gy`` compute grid of a Blackhole (or not) device, else why not."""
        if not blackhole:
            return "Blackhole-only: the GeGLU routed-expert kernels assume the Blackhole compute grid"
        if gx < cls.GRID_X or gy < cls.GRID_Y:
            return f"Blackhole-only: needs an {cls.GRID_X}x{cls.GRID_Y} compute rectangle, the grid is {gx}x{gy}"
        return None

    @classmethod
    def unsupported_reason(cls, mesh_device) -> str | None:
        gx, gy = compute_grid(mesh_device)
        return cls.grid_reason(gx, gy, mesh_device.arch() == ttnn.Arch.BLACKHOLE)

    # CB index allocation (shared across the three kernels through compile-time args).
    CB_IN0_X = 0
    CB_IN1_GATE = 1
    CB_IN1_UP = 2
    CB_IN1_DOWN = 3
    CB_GATE_INT = 4
    CB_ACTIVATED = 6
    CB_PARTIALS_GU = 7
    CB_PARTIALS_D = 8
    CB_OUT = 9
    CB_COUNTS_SCRATCH = 10
    CB_IDX_SCRATCH = 11
    CB_IN0_DOWN_FULL = 12
    CB_PARTIALS_UP = 13
    CB_START_SCRATCH = 14  # writer's copy of the region-offset page
    CB_START_SCRATCH_READER = 15  # reader's own copy (the two RISCs never share one L1 page)
    CB_X_RM = 16  # row-major bf16 x staging (row-major x only)
    CB_ZERO = 17  # row-major x only: one K-block window of zeros for the padding sticks of a partial tile-row

    # Per-core M (tile rows) the CBs are sized for: 4 (chunk 32 rows x GRID_Y). per_core_M and the gate/up K-block
    # width compete for L1; the fit below may narrow the width or, on shapes where each extra K-block buys enough
    # avoided weight traffic, double M (kWidenMMinTilesPerBlock, calibrated on the shared op). A narrower or wider
    # plan was measured null (the per-expert cost is bytes at ~450 GB/s plus a fixed ~27 us), so there are no knobs.
    PER_CORE_M_MAX = 4
    IN0_BLOCK_W_GU_REQUEST = 16
    MIN_CHUNK_M_TILES = 16
    WIDEN_M_MIN_TILES_PER_BLOCK = 4096
    DST_CAPACITY = 8  # bf16 dst (fp32_dest_acc_en=False)
    L1_SCRATCH_MARGIN = 48 * 1024
    # IN1_WRITER_MCAST: the writer multicasts the gate/up weights on NoC 1 (overlapping the reader's NoC-0 reads),
    # a worker multicast on NoC 1 beside fabric CCL traffic, as the shared op ships it.
    WRITER_MCASTS_IN1 = True

    def __init__(
        self,
        mesh_device,
        gate_projs: list,
        up_projs: list,
        down_projs: list,
        idx_table: ttnn.Tensor,
        max_tokens: int,
        min_active_tokens: int = 0,
        max_active_tokens: int = U32_MAX,
    ):
        reason = self.unsupported_reason(mesh_device)
        assert reason is None, reason
        self.per_core_m_max, self.in0_block_w_request = self.PER_CORE_M_MAX, self.IN0_BLOCK_W_GU_REQUEST
        assert len(gate_projs) == len(up_projs) == len(down_projs) > 0, "one gate/up/down tensor per local expert"
        self.device = mesh_device
        self.gate_projs, self.up_projs, self.down_projs = list(gate_projs), list(up_projs), list(down_projs)
        self.idx_table = idx_table
        self.E = len(self.gate_projs)
        self.m_tiles = _ceil_div(max_tokens, TILE)
        self.min_active_tokens, self.max_active_tokens = min_active_tokens, max_active_tokens
        self.writer_mcasts_in1 = self.WRITER_MCASTS_IN1
        g0, u0, d0 = self.gate_projs[0], self.up_projs[0], self.down_projs[0]
        for e in range(self.E):
            for name, w, ref in (
                ("gate", self.gate_projs[e], g0),
                ("up", self.up_projs[e], u0),
                ("down", self.down_projs[e], d0),
            ):
                assert w.layout == ttnn.TILE_LAYOUT, f"{name}[{e}] must be TILE"
                assert w.padded_shape == ref.padded_shape and w.dtype == ref.dtype, f"{name}[{e}] differs from expert 0"
                assert w.memory_config() == ref.memory_config(), f"{name}[{e}] placement differs from expert 0"
        assert g0.padded_shape[-1] == u0.padded_shape[-1] == d0.padded_shape[-2], "gate/up N must equal down K"
        assert d0.padded_shape[-1] == g0.padded_shape[-2], "down N must equal gate K"
        self._plans: dict = {}

    # ------------------------------------------------------------------------------------------------ golden
    @staticmethod
    def golden(x, counts, region_offsets, idx_table, gate_projs, up_projs, down_projs):
        """torch reference. ``x`` [M_buf, K] float; ``counts`` / ``region_offsets`` indexed by global expert id;
        ``idx_table`` the local slot -> global id list; weights per local slot in the device ``(K, H)`` / ``(H, K)``
        orientation. Rows outside every expert region are returned as zeros (the op leaves them untouched)."""
        out = torch.zeros_like(x, dtype=torch.float32)
        for e, gid in enumerate(int(g) for g in idx_table):
            n, s = int(counts[gid]), int(region_offsets[gid])
            if n == 0:
                continue
            xe = x[s : s + n].float()
            h = F.gelu(xe @ gate_projs[e].float(), approximate="tanh") * (xe @ up_projs[e].float())
            out[s : s + n] = h @ down_projs[e].float()
        return out

    # ------------------------------------------------------------------------------------------------ the op
    def __call__(self, x: ttnn.Tensor, counts: ttnn.Tensor, region_offsets: ttnn.Tensor, output=None) -> ttnn.Tensor:
        x_is_row_major = x.layout == ttnn.ROW_MAJOR_LAYOUT
        if x_is_row_major:
            assert x.dtype == ttnn.bfloat16, "a ROW_MAJOR x must be bf16 (tilized to bf8 in-op)"
        else:
            assert x.dtype == ttnn.bfloat8_b and x.layout == ttnn.TILE_LAYOUT, "a TILE x must be bf8"
        for t, name in ((counts, "counts"), (region_offsets, "region_offsets"), (self.idx_table, "idx_table")):
            assert t.dtype == ttnn.uint32 and t.layout == ttnn.ROW_MAJOR_LAYOUT, f"{name} must be uint32 ROW_MAJOR"
        assert counts.shape[-1] == region_offsets.shape[-1], "counts and region_offsets index the same global ids"
        if output is None:
            output = (
                ttnn.allocate_tensor_on_device(
                    x.shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
                )
                if x_is_row_major
                else x
            )
        assert output.dtype == ttnn.bfloat8_b and output.layout == ttnn.TILE_LAYOUT, "the output is TILE bf8"
        assert list(output.padded_shape)[-1] == list(x.padded_shape)[-1], "output N must match x K"
        plan = self._plan_for(x, counts, region_offsets, output)
        program = self._program(plan, x, counts, region_offsets, output)
        io = [x, counts, self.idx_table, region_offsets] + self.gate_projs + self.up_projs + self.down_projs
        if output is not x:
            io.append(output)
        else:
            io = io[1:] + [x]  # the in-place buffer is the output: generic_op takes the output as the last tensor
        ttnn.generic_op(io, program)
        return output

    # ------------------------------------------------------------------------------------------------ plan
    def _plan_for(self, x, counts, start, out):
        x_is_row_major = x.layout == ttnn.ROW_MAJOR_LAYOUT
        key = (
            x_is_row_major,
            x.dtype,
            tuple(x.padded_shape),
            out.dtype,
            tuple(out.padded_shape),
            counts.shape[-1],
            counts.buffer_aligned_page_size(),
            start.buffer_aligned_page_size(),
            self.idx_table.shape[-1],
            self.idx_table.buffer_aligned_page_size(),
            # the cached TensorAccessorArgs encode the page -> bank layout, not the address
            tuple(str(t.memory_config()) for t in (x, counts, start, out)),
        )
        if key not in self._plans:
            plan = self._build_plan(x, counts, start, out)
            self._attach_descriptors(plan, x, counts, start, out)
            self._plans[key] = plan
        return self._plans[key]

    def _l1_budget(self) -> int:
        """Per-core L1 available to the CBs: the Blackhole L1 (the arch is asserted in __init__; the Python
        MeshDevice has no l1_size_per_core) minus the allocator's reserved base and a margin for the small scratch
        CBs and per-CB alignment."""
        l1_size = BLACKHOLE_L1_PER_CORE
        base = ttnn._ttnn.device.GetAllocatorBaseAddress(self.device, ttnn.BufferType.L1)
        assert l1_size > base + self.L1_SCRATCH_MARGIN, (l1_size, base)
        return l1_size - base - self.L1_SCRATCH_MARGIN

    def _build_plan(self, x, counts, start, out) -> dict:
        """The shared op's create_descriptor, host side: block / subblock / chunk sizes, the L1 fit, CB sizes,
        compile-time args and the per-core runtime args (everything but the buffer addresses)."""
        GRID_X, GRID_Y = self.GRID_X, self.GRID_Y
        x_is_row_major = x.layout == ttnn.ROW_MAJOR_LAYOUT
        g0, u0, d0 = self.gate_projs[0], self.up_projs[0], self.down_projs[0]
        M_tiles_full = self.m_tiles
        K_gate_tiles = list(x.padded_shape)[-1] // TILE
        N_gate_tiles_full = list(g0.padded_shape)[-1] // TILE
        K_down_tiles = list(d0.padded_shape)[-2] // TILE
        N_down_tiles_full = list(d0.padded_shape)[-1] // TILE
        assert list(g0.padded_shape)[-2] // TILE == K_gate_tiles, "gate K must equal x K"
        assert M_tiles_full <= list(x.padded_shape)[-2] // TILE, "max_tokens exceeds the buffer"
        gx, gy = compute_grid(self.device)  # the rectangle lives at the grid origin; the cap may shrink the grid
        assert gx >= GRID_X and gy >= GRID_Y, self.grid_reason(gx, gy, True)

        in0_block_w_gu = min(self.in0_block_w_request, K_gate_tiles)
        while in0_block_w_gu > 1 and K_gate_tiles % in0_block_w_gu != 0:
            in0_block_w_gu -= 1
        per_core_N_gu = _ceil_div(N_gate_tiles_full, GRID_X)
        per_core_N_d = _ceil_div(N_down_tiles_full, GRID_X)
        N_gate_tiles_padded = per_core_N_gu * GRID_X
        K_down_tiles_padded = N_gate_tiles_padded  # down K = gate N
        in0_block_w_d = per_core_N_gu
        assert K_down_tiles_padded % in0_block_w_d == 0

        # DRAM ND-sharded weights: the shard width must be exactly a core's N slice (one shard = one NoC transaction
        # per K-row); gate and up share one width.
        gu_shard_w, up_shard_w, d_shard_w = _nd_shard_n_tiles(g0), _nd_shard_n_tiles(u0), _nd_shard_n_tiles(d0)
        assert gu_shard_w == up_shard_w, f"gate/up ND shard widths differ: {gu_shard_w} vs {up_shard_w}"
        for name, shard_w, per_core_n in (("gate/up", gu_shard_w, per_core_N_gu), ("down", d_shard_w, per_core_N_d)):
            assert shard_w in (
                0,
                per_core_n,
            ), f"{name}: ND shard width must be per_core_N ({per_core_n} tiles), got {shard_w}"

        # Subblocks: 1 x (largest divisor of per_core_N within the 8-tile dst). A down subblock narrower than 3 tiles
        # goes unpack-bound, so below that the down grid takes a ragged last subblock instead.
        gu_out_subblock_h = 1
        gu_out_subblock_w = max(c for c in range(1, self.DST_CAPACITY + 1) if per_core_N_gu % c == 0)
        d_out_subblock_h = 1
        d_sub_w = max(c for c in range(1, self.DST_CAPACITY + 1) if per_core_N_d % c == 0)
        if d_sub_w < 3:
            d_sub_w = min(self.DST_CAPACITY, per_core_N_d)
        d_out_subblock_w = d_sub_w
        d_in1_num_subblocks = _ceil_div(per_core_N_d, d_out_subblock_w)
        d_out_subblock_w_tail = per_core_N_d - (d_in1_num_subblocks - 1) * d_out_subblock_w
        d_subblocks_exact = d_out_subblock_w_tail == d_out_subblock_w
        cb_out_stage_count = 2
        cb_out_tiles = d_out_subblock_h * (d_out_subblock_w if d_subblocks_exact else per_core_N_d) * cb_out_stage_count

        # Formats / tile sizes. Partials (K-loop accumulators) are bf16; intermediates (post-activation operands of
        # the next matmul) bf8.
        in0_x_dtype = ttnn.bfloat8_b if x_is_row_major else x.dtype
        in0_x_ts, gate_ts, up_ts, down_ts = (TILE_BYTES[t] for t in (in0_x_dtype, g0.dtype, u0.dtype, d0.dtype))
        out_ts = TILE_BYTES[out.dtype]
        intermed_ts, partials_ts = TILE_BYTES[ttnn.bfloat8_b], TILE_BYTES[ttnn.bfloat16]

        def cb_footprint(M, w_gu):
            total = M * w_gu * (1 if x_is_row_major else 2) * in0_x_ts  # cb_in0_x (RM: single-buffered)
            if x_is_row_major:
                total += M * w_gu * 2 * partials_ts  # cb_x_rm (bf16 staging, multicast source: double-buffered)
                total += w_gu * TILE * 2  # cb_zero (one zero K-block stick)
            total += w_gu * per_core_N_gu * 2 * gate_ts  # cb_in1_gate
            total += w_gu * per_core_N_gu * 2 * up_ts  # cb_in1_up
            total += in0_block_w_d * per_core_N_d * 2 * down_ts  # cb_in1_down
            total += M * per_core_N_gu * intermed_ts  # cb_gate_intermed
            total += M * per_core_N_gu * intermed_ts  # cb_activated
            total += M * per_core_N_gu * partials_ts  # cb_mm_partials_gu
            total += M * per_core_N_gu * partials_ts  # cb_mm_partials_up
            total += M * per_core_N_d * partials_ts  # cb_mm_partials_d
            total += cb_out_tiles * out_ts  # cb_out
            total += M * in0_block_w_d * 2 * intermed_ts  # cb_in0_down_full
            return total

        l1_budget = self._l1_budget()

        def fit_config(requested_M):
            for M in range(requested_M, 0, -1):
                if requested_M % M != 0:
                    continue
                for w in range(min(in0_block_w_gu, K_gate_tiles), 0, -1):
                    if K_gate_tiles % w == 0 and cb_footprint(M, w) <= l1_budget:
                        return M, w
            return 0, 0

        per_core_M, fit_w = fit_config(self.per_core_m_max)
        assert per_core_M != 0, f"the per-core CBs do not fit L1 (need {cb_footprint(1, 1)} B of {l1_budget} B)"
        # Doubling per_core_M halves the chunk count (every chunk re-reads the full weights) but narrows the K-block
        # width that still fits; take it only when each extra K-block buys enough avoided weight traffic.
        wide_M, wide_w = fit_config(self.per_core_m_max * 2)
        if wide_M == self.per_core_m_max * 2 and wide_w < fit_w:
            extra_blocks = K_gate_tiles // wide_w - K_gate_tiles // fit_w
            weight_tiles = 2 * K_gate_tiles * N_gate_tiles_full + K_down_tiles * N_down_tiles_full
            if extra_blocks > 0 and weight_tiles // extra_blocks >= self.WIDEN_M_MIN_TILES_PER_BLOCK:
                per_core_M, fit_w = wide_M, wide_w
        in0_block_w_gu = fit_w
        chunk_M_tiles = per_core_M * GRID_Y
        assert K_gate_tiles % in0_block_w_gu == 0
        min_chunk = min(chunk_M_tiles, self.MIN_CHUNK_M_TILES)
        num_chunks = _ceil_div(M_tiles_full, min_chunk)

        gu_in0_num_subblocks = per_core_M // gu_out_subblock_h
        gu_in1_num_subblocks = per_core_N_gu // gu_out_subblock_w
        gu_in0_block_num_tiles = per_core_M * in0_block_w_gu
        gu_in0_subblock_num_tiles = gu_out_subblock_h * in0_block_w_gu
        gu_in1_block_num_tiles = in0_block_w_gu * per_core_N_gu
        gu_num_blocks = K_gate_tiles // in0_block_w_gu
        gu_out_block_num_tiles = per_core_M * per_core_N_gu
        d_in0_num_subblocks = per_core_M // d_out_subblock_h
        d_in0_block_num_tiles = per_core_M * in0_block_w_d
        d_in0_subblock_num_tiles = d_out_subblock_h * in0_block_w_d
        d_in1_block_num_tiles = in0_block_w_d * per_core_N_d
        d_num_blocks = K_down_tiles_padded // in0_block_w_d
        d_out_block_num_tiles = per_core_M * per_core_N_d
        dst_M_tiles = list(out.padded_shape)[-2] // TILE

        # Weight delivery: UP_SPLIT (the writer reads `up` on NoC 1, the reader multicasts it on NoC 0), DOWN_SPLIT
        # (the two RISCs read half the down K-rows each; needs >= 2 rows), IN1_WRITER_MCAST (see __init__).
        up_mode = 2
        reader_reads_up, reader_mcasts_up = 0, 1
        split_down = in0_block_w_d >= 2
        down_split_k = in0_block_w_d // 2 if split_down else in0_block_w_d

        # Semaphores, numbered in allocation order (the id is the index).
        sem_names = ["in1_ready", "in1_valid", "in0_ready", "in0_valid", "act_ready", "act_valid", "counts_valid"]
        if self.writer_mcasts_in1:
            sem_names += ["mcast_go", "mcast_done"]
        if split_down:
            sem_names += ["down_go", "down_done"]
        sem_names += ["up_go", "up_done"]  # up_mode == 2
        sem = {name: i for i, name in enumerate(sem_names)}
        sem_id = lambda name: sem.get(name, 0)  # noqa: E731

        # Circular buffers: (index, dtype, tiles, page bytes).
        cbs = [
            (self.CB_IN0_X, in0_x_dtype, gu_in0_block_num_tiles * (1 if x_is_row_major else 2), in0_x_ts),
        ]
        if x_is_row_major:
            cbs.append((self.CB_X_RM, ttnn.bfloat16, gu_in0_block_num_tiles * 2, partials_ts))
            # the zero stick the reader fills a partial tile-row's padding sticks from (one K-block window)
            cbs.append((self.CB_ZERO, ttnn.bfloat16, 1, in0_block_w_gu * TILE * 2))
        cbs += [
            (self.CB_IN1_GATE, g0.dtype, gu_in1_block_num_tiles * 2, gate_ts),
            (self.CB_IN1_UP, u0.dtype, gu_in1_block_num_tiles * 2, up_ts),
            (self.CB_IN1_DOWN, d0.dtype, d_in1_block_num_tiles * 2, down_ts),
            (self.CB_GATE_INT, ttnn.bfloat8_b, gu_out_block_num_tiles, intermed_ts),
            (self.CB_ACTIVATED, ttnn.bfloat8_b, gu_out_block_num_tiles, intermed_ts),
            (self.CB_PARTIALS_GU, ttnn.bfloat16, gu_out_block_num_tiles, partials_ts),
            (self.CB_PARTIALS_UP, ttnn.bfloat16, gu_out_block_num_tiles, partials_ts),
            (self.CB_PARTIALS_D, ttnn.bfloat16, d_out_block_num_tiles, partials_ts),
            (self.CB_OUT, out.dtype, cb_out_tiles, out_ts),
            (self.CB_IN0_DOWN_FULL, ttnn.bfloat8_b, d_in0_block_num_tiles * 2, intermed_ts),
        ]
        # A CB wraps only when a push lands exactly on fifo_limit: every ring must be a whole number of the finest
        # granule its kernels push (adaptive_chunk.hpp).
        d_row_granule = d_out_subblock_h * (d_out_subblock_w if d_subblocks_exact else per_core_N_d)
        for name, ring, granule in (
            ("cb_in0_x", gu_in0_block_num_tiles * (1 if x_is_row_major else 2), in0_block_w_gu),
            ("cb_x_rm", gu_in0_block_num_tiles * 2, in0_block_w_gu),
            ("cb_gate_intermed", gu_out_block_num_tiles, gu_out_subblock_h * gu_out_subblock_w),
            ("cb_partials_gu", gu_out_block_num_tiles, gu_out_subblock_h * gu_out_subblock_w),
            ("cb_partials_up", gu_out_block_num_tiles, gu_out_subblock_h * gu_out_subblock_w),
            ("cb_activated", gu_out_block_num_tiles, per_core_M * in0_block_w_d),
            ("cb_mm_partials_d", d_out_block_num_tiles, d_row_granule),
            ("cb_out", cb_out_tiles, d_row_granule),
            ("cb_in0_down_full", d_in0_block_num_tiles * 2, d_in0_block_num_tiles),
            ("cb_in1_gate", gu_in1_block_num_tiles * 2, gu_in1_block_num_tiles),
            ("cb_in1_up", gu_in1_block_num_tiles * 2, gu_in1_block_num_tiles),
            ("cb_in1_down", d_in1_block_num_tiles * 2, d_in1_block_num_tiles),
        ):
            assert granule > 0 and ring % granule == 0, f"{name}: ring {ring} tiles is not a whole number of {granule}"
        # Scratch pages for the device-side count / index / region-offset lookups (one page each).
        counts_n, idx_n = counts.shape[-1], self.idx_table.shape[-1]
        counts_scratch = max(counts.buffer_aligned_page_size(), counts_n * 4)
        idx_scratch = max(self.idx_table.buffer_aligned_page_size(), idx_n * 4)
        start_scratch = max(start.buffer_aligned_page_size(), counts_n * 4)
        scratch_cbs = [
            (self.CB_COUNTS_SCRATCH, ttnn.uint32, 1, counts_scratch),
            (self.CB_IDX_SCRATCH, ttnn.uint32, 1, idx_scratch),
            (self.CB_START_SCRATCH, ttnn.uint32, 1, start_scratch),
            (self.CB_START_SCRATCH_READER, ttnn.uint32, 1, start_scratch),
        ]

        reader_ct = [
            self.CB_IN0_X, self.CB_IN1_GATE, self.CB_IN1_UP, self.CB_IN1_DOWN, self.CB_IN0_DOWN_FULL,
            self.CB_COUNTS_SCRATCH, self.CB_IDX_SCRATCH,
            self.E, per_core_M, per_core_N_gu, per_core_N_d, K_gate_tiles, K_down_tiles, in0_block_w_gu, in0_block_w_d,
            N_gate_tiles_full, N_down_tiles_full, M_tiles_full, num_chunks, chunk_M_tiles,
            self.CB_ACTIVATED, GRID_X, K_down_tiles_padded, reader_reads_up, reader_mcasts_up,
            self.CB_START_SCRATCH_READER, int(x_is_row_major), self.CB_X_RM, TILE, 2,  # 28 TILE_HEIGHT, 29 bf16 bytes
            down_split_k, int(self.writer_mcasts_in1), self.min_active_tokens, self.max_active_tokens,
            gu_shard_w, d_shard_w, self.CB_ZERO,
        ]  # fmt: skip
        assert len(reader_ct) == 37
        writer_ct = [
            self.CB_OUT, per_core_M, per_core_N_gu, per_core_N_d, d_out_subblock_h, d_out_subblock_w,
            N_gate_tiles_full, N_down_tiles_full, num_chunks, chunk_M_tiles,
            self.CB_COUNTS_SCRATCH, self.CB_IDX_SCRATCH, self.E, M_tiles_full, dst_M_tiles, self.CB_START_SCRATCH,
            self.CB_IN1_UP, in0_block_w_gu, K_gate_tiles, int(up_mode == 2),
            self.CB_IN1_DOWN, in0_block_w_d, K_down_tiles, d_num_blocks, d_in1_block_num_tiles, down_split_k,
            self.CB_IN1_GATE, int(self.writer_mcasts_in1), self.min_active_tokens, self.max_active_tokens,
            gu_shard_w, d_shard_w, d_out_subblock_w_tail,
        ]  # fmt: skip
        assert len(writer_ct) == 33
        compute_ct = [
            in0_block_w_gu, gu_in0_num_subblocks, gu_in0_block_num_tiles, gu_in0_subblock_num_tiles,
            gu_in1_num_subblocks, gu_in1_block_num_tiles, per_core_N_gu, gu_num_blocks,  # gate
            in0_block_w_gu, gu_in0_num_subblocks, gu_in0_block_num_tiles, gu_in0_subblock_num_tiles,
            gu_in1_num_subblocks, gu_in1_block_num_tiles, per_core_N_gu, gu_num_blocks,  # up (same split)
            in0_block_w_d, d_in0_num_subblocks, d_in0_block_num_tiles, d_in0_subblock_num_tiles,
            d_in1_num_subblocks, d_in1_block_num_tiles, per_core_N_d, d_num_blocks,  # down
            gu_out_subblock_h, gu_out_subblock_w, gu_out_block_num_tiles,
            d_out_subblock_h, d_out_subblock_w, d_out_subblock_w_tail, d_out_block_num_tiles,
            num_chunks, self.E, chunk_M_tiles, int(x_is_row_major), K_down_tiles,
            self.min_active_tokens, self.max_active_tokens,
        ]  # fmt: skip
        assert len(compute_ct) == 38
        compute_named = [
            ("cb_x_rm", self.CB_X_RM), ("cb_in0_x", self.CB_IN0_X), ("cb_in1_gate", self.CB_IN1_GATE),
            ("cb_in1_up", self.CB_IN1_UP), ("cb_in1_down", self.CB_IN1_DOWN), ("cb_gate_intermed", self.CB_GATE_INT),
            ("cb_activated", self.CB_ACTIVATED), ("cb_in0_down_full", self.CB_IN0_DOWN_FULL),
            ("cb_mm_partials_gu", self.CB_PARTIALS_GU), ("cb_mm_partials_up", self.CB_PARTIALS_UP),
            ("cb_mm_partials_d", self.CB_PARTIALS_D), ("cb_out", self.CB_OUT),
            ("cb_counts_scratch", self.CB_COUNTS_SCRATCH), ("cb_idx_scratch", self.CB_IDX_SCRATCH),
            ("m_tiles_full", M_tiles_full),
        ]  # fmt: skip

        # Per-core runtime args: the schedule and multicast geometry (addresses are common args, set per call).
        # DRAM reads are issue-bound, so the weight sender of column gx sits on row gx % GRID_Y and the x sender of
        # row gy on column (gy + 1) % GRID_X: no core owns both streams on the 11x8 grid.
        noc = lambda gx, gy: self.device.worker_core_from_logical_core(ttnn.CoreCoord(gx, gy))  # noqa: E731
        grid_first, grid_last = noc(0, 0), noc(GRID_X - 1, GRID_Y - 1)
        reader_rt, writer_rt, compute_rt = {}, {}, {}
        for gy in range(GRID_Y):
            for gx in range(GRID_X):
                in1_sender_row = gx % GRID_Y
                in0_sender_col = (gy + 1) % GRID_X
                assert not (gy == in1_sender_row and gx == in0_sender_col), f"sender collision at ({gx}, {gy})"
                is_in1_sender = gy == in1_sender_row
                sender_noc, first_recv, last_recv = noc(gx, in1_sender_row), noc(gx, 0), noc(gx, GRID_Y - 1)
                is_in0_sender = gx == in0_sender_col
                in0_sender_noc, in0_first, in0_last = noc(in0_sender_col, gy), noc(0, gy), noc(GRID_X - 1, gy)
                args = [
                    gy, gx, gx,
                    int(is_in1_sender), sem_id("in1_ready"), sem_id("in1_valid"), GRID_Y - 1,
                    first_recv.x, first_recv.y, last_recv.x, last_recv.y, sender_noc.x, sender_noc.y,
                    int(is_in0_sender), sem_id("in0_ready"), sem_id("in0_valid"), GRID_X - 1,
                    in0_first.x, in0_first.y, in0_last.x, in0_last.y, in0_sender_noc.x, in0_sender_noc.y,
                    sem_id("act_ready"), sem_id("act_valid"), sem_id("up_go"), sem_id("up_done"),
                    # COUNTS_BCAST: logical (0,0) reads counts/idx once and multicasts them to the grid
                    int(gx == 0 and gy == 0), grid_first.x, grid_first.y, grid_last.x, grid_last.y,
                    sem_id("counts_valid"), GRID_X * GRID_Y - 1,
                    sem_id("down_go"), sem_id("down_done"), sem_id("mcast_go"), sem_id("mcast_done"),
                ]  # fmt: skip
                for gxi in range(GRID_X):  # M-row NoC table (phase-4 activated multicast)
                    c = noc(gxi, gy)
                    args += [c.x, c.y]
                reader_rt[(gx, gy)] = args
                writer_rt[(gx, gy)] = [
                    gy, gx, gx, int(is_in1_sender), sem_id("up_go"), sem_id("up_done"),
                    sem_id("down_go"), sem_id("down_done"),
                    first_recv.x, first_recv.y, last_recv.x, last_recv.y, GRID_Y - 1,
                    sem_id("in1_ready"), sem_id("in1_valid"), sem_id("mcast_go"), sem_id("mcast_done"),
                ]  # fmt: skip
                valid_n_gu = min(per_core_N_gu, max(N_gate_tiles_full - gx * per_core_N_gu, 0))
                valid_n_d = min(per_core_N_d, max(N_down_tiles_full - gx * per_core_N_d, 0))
                compute_rt[(gx, gy)] = [
                    _ceil_div(valid_n_gu, gu_out_subblock_w),
                    _ceil_div(valid_n_d, d_out_subblock_w),
                ]

        return dict(
            per_core_M=per_core_M,
            in0_block_w_gu=in0_block_w_gu,
            chunk_M_tiles=chunk_M_tiles,
            num_chunks=num_chunks,
            per_core_N_gu=per_core_N_gu,
            per_core_N_d=per_core_N_d,
            d_out_subblock_w=d_out_subblock_w,
            d_out_subblock_w_tail=d_out_subblock_w_tail,
            gu_shard_w=gu_shard_w,
            d_shard_w=d_shard_w,
            l1_budget=l1_budget,
            cb_bytes=cb_footprint(per_core_M, in0_block_w_gu),
            n_semaphores=len(sem_names),
            cbs=cbs + scratch_cbs,
            reader_ct=reader_ct,
            writer_ct=writer_ct,
            compute_ct=compute_ct,
            compute_named=compute_named,
            reader_rt=reader_rt,
            writer_rt=writer_rt,
            compute_rt=compute_rt,
        )

    # ------------------------------------------------------------------------------------------------ program
    def _attach_descriptors(self, plan, x, counts, start, out):
        """The descriptor objects of a plan, built once: CBs, semaphores, the three kernels with their compile-time
        args (TensorAccessorArgs included: they encode the buffers' page layout, not their addresses) and per-core
        runtime args. A call only sets the common runtime args (the buffer addresses) on the reader and writer."""
        core_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.GRID_X - 1, self.GRID_Y - 1))]
        )
        plan["cb_descriptors"] = [
            ttnn.CBDescriptor(
                total_size=tiles * page,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
            )
            for idx, dtype, tiles, page in plan["cbs"]
        ]
        plan["semaphore_descriptors"] = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(plan["n_semaphores"])
        ]

        def rt(table):
            args = ttnn.RuntimeArgs()
            for (gx, gy), values in table.items():
                args[gx][gy] = [int(v) for v in values]
            return args

        g0, u0, d0 = self.gate_projs[0], self.up_projs[0], self.down_projs[0]
        reader_ct = list(plan["reader_ct"])
        for t in (x, g0, u0, d0, counts, self.idx_table, start):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = list(plan["writer_ct"])
        for t in (out, start, u0, d0):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        plan["kernel_descriptors"] = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_geglu_routed_expert.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=rt(plan["reader_rt"]),
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_geglu_routed_expert.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=rt(plan["writer_rt"]),
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_geglu_routed_expert.cpp"),
                core_ranges=core_set,
                compile_time_args=list(plan["compute_ct"]),
                named_compile_time_args=list(plan["compute_named"]),
                # PACKER_L1_ACC: cross-K-block accumulation through the packer's L1 read-modify-write.
                defines=[("PACKER_L1_ACC", "1")],
                runtime_args=rt(plan["compute_rt"]),
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=False,
                    dst_full_sync_en=False,
                    math_approx_mode=False,
                ),
            ),
        ]

    def _program(self, plan, x, counts, start, out) -> ttnn.ProgramDescriptor:
        addr = lambda t: t.buffer_address()  # noqa: E731
        gate_a, up_a, down_a = ([addr(w) for w in ws] for ws in (self.gate_projs, self.up_projs, self.down_projs))
        reader, writer, compute = plan["kernel_descriptors"]
        # Reader: x, counts, index table, region offsets, gate/up/down weights. Writer: output, region offsets,
        # up/down weights (the order the kernels' get_common_arg_val indexes).
        reader.common_runtime_args = [addr(x), addr(counts), addr(self.idx_table), addr(start)] + gate_a + up_a + down_a
        writer.common_runtime_args = [addr(out), addr(start)] + up_a + down_a
        # ProgramDescriptor copies the kernel descriptors, so the cached objects can be re-pointed on the next call.
        return ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute], semaphores=plan["semaphore_descriptors"], cbs=plan["cb_descriptors"]
        )
