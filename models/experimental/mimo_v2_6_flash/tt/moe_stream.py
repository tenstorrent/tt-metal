# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM-streaming expert path for MiMo MoE decode (a ``ttnn.generic_op`` program).

Blackhole streams DRAM fastest when each bank is read by the worker core next to it, in large
contiguous blocks with several reads in flight. ``GateUpStream`` selects every user's top-k experts and
computes gate|up, SwiGLU and the routing weight of every (user, expert) pair of a decode step on the eight
bank cores; its device kernels live in ``tt/kernels/`` and are JIT-compiled from the checkout.
"""

import math

import torch

import ttnn

from .common import TILE, pad_to

KERNEL_DIR = "models/experimental/mimo_v2_6_flash/tt/kernels"


_BANK_CORES = {}  # device ids of a (sub)mesh -> bank cores registered by set_dram_bank_cores


def set_dram_bank_cores(device, cores):
    """Use ``cores`` as the bank-reader cores of every chip in ``device`` (a mesh or submesh).

    A mesh answers get_optimal_dram_bank_to_logical_worker_assignment for its first chip only, but chips are
    harvested differently. On the Galaxy, row 2's first chip yields logical x = 6 for banks 4-7 where x = 7 is
    right for the other chips of that row, and the streaming MoE kernels run 23% slower there (250 vs 204 us per
    layer). The 4x8 parent's answer (x = 7) is fast on all 32 chips, so the generator registers it for every row."""
    _BANK_CORES[tuple(device.get_device_ids())] = list(cores)


def dram_bank_cores(device):
    """One worker core per DRAM bank (NOC 0 placement): core ``i`` streams bank ``i``."""
    cores = _BANK_CORES.get(tuple(device.get_device_ids()))
    if cores is not None:
        return cores
    return device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)


def zero_output(plan, hidden, rows=TILE, memory_config=None):
    """Persistent zero ``[1, 1, rows, hidden]`` bf16 tile tensor on the row, for a kernel that writes only user rows."""
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, hidden),
        device=plan.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=plan.replicate(),
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def is_dram_interleaved(t):
    mc = t.memory_config()
    return mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def fp32_bits(value):
    return int(torch.tensor([float(value)], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF


def bank_vcs(cores):
    """Readers in the same core row get different VCs (as DeepSeek-B1's DRAM streaming matmul)."""
    vcs = []
    for idx, core in enumerate(cores):
        vc = idx & 0x3
        for j in range(idx):
            if cores[j].y == core.y and (j & 0x3) == (idx & 0x3):
                vc = (vc + 1) & 0x3
                break
        vcs.append(vc)
    return vcs


class GateUpStream:
    """Top-k routing, then silu(x @ gate) * (x @ up) * w for every decode pair, streamed bank-locally from the
    interleaved w_gate_up.

    Routing runs on the data-movement RISCs of every bank core (reader: user 0, so it starts streaming at once;
    writer: the rest, handed over before the stream reaches their pairs): from rows 0-3 of the fp32 sigmoid scores
    and biased scores, each user's top-k by exact fp32 order on the
    biased scores (ties to the lower expert id, sorted descending like ttnn.topk); the unbiased scores of the chosen
    experts are normalized (sum + 1e-20), scaled, rounded to bf16 and multiplied by the bf16 token mask, as the
    TTNN routing ops did. The bank-0 reader writes the uint16 expert-id stick the down sparse_matmul indexes with.

    The per-chip w_gate_up [1, E, K, 2 * I] has 2 * I / 32 = 2 * banks tile columns, so DRAM bank b holds gate
    column b and up column b of every expert, contiguous per expert as [kt][gate, up]. Core b streams bank b and
    writes row u of pair (u, k)'s [4, 32] h tile b into page p of the zeroed row-major ``out`` (DownStream's
    activation: page p = pair p's 8 tiny tiles; the other rows stay zero).
    """

    BLOCK_K = 16  # K tiles per block: 16 x (gate, up) x 576 B = 18 KB
    # 12 blocks (216 KB, ~1.5 pairs) keep the bank streaming through a pair's SwiGLU epilogue; trids 1..12.
    NUM_BUFFERS = 12
    IN_FLIGHT = 6
    MAX_PACKET = 16384
    MAX_USERS = 4  # routing stages rows 0-3 of each tile face
    FP32_TILE_BYTES = 4096

    def __init__(
        self, plan, hidden, inter_local, num_pairs, topk, num_experts, norm_topk_prob=True, routed_scaling_factor=1.0
    ):
        device = plan.mesh_device
        self.cores = list(dram_bank_cores(device))
        self.banks = device.dram_grid_size().x
        assert len(self.cores) == self.banks, (len(self.cores), self.banks)
        assert inter_local == TILE * self.banks, "gate|up column b must live in bank b"
        assert hidden % (TILE * self.BLOCK_K) == 0
        assert num_experts % TILE == 0
        self.users = num_pairs // topk
        assert self.users * topk == num_pairs and 1 <= self.users <= self.MAX_USERS and topk <= 16
        assert num_pairs * 4 <= 1024
        assert self.IN_FLIGHT < self.NUM_BUFFERS <= 15
        self.num_pairs, self.topk = num_pairs, topk
        # The reader routes user 0 and streams its pairs while the writer routes the others.
        self.reader_users = 1
        self.expert_tiles = num_experts // TILE
        self.norm = 1 if norm_topk_prob else 0
        self.has_scale = 0 if routed_scaling_factor == 1.0 else 1
        self.scale_bits = fp32_bits(routed_scaling_factor)
        # rows 0-3 of biased and score tiles, 256 B of mask rows, the uint16 id stick
        stage_bytes = 2 * self.expert_tiles * 512 + 256 + 2 * num_pairs
        self.stage_pages = math.ceil(stage_bytes / self.FP32_TILE_BYTES)

        self.kt = hidden // TILE
        self.blocks_per_pair = self.kt // self.BLOCK_K
        self.w_tile_bytes = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat4_b)
        self.block_tiles = 2 * self.BLOCK_K
        self.block_bytes = self.block_tiles * self.w_tile_bytes
        packets = 1
        while self.block_bytes // packets > self.MAX_PACKET or self.block_bytes % packets:
            packets += 1
        self.packets_per_block = packets
        self.packet_bytes = self.block_bytes // packets
        self.expert_bytes = 2 * self.kt * self.w_tile_bytes
        self.out_tiles_per_pair = inter_local // TILE
        self.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

        self.vcs = bank_vcs(self.cores)

        tiny = ttnn.Tile([4, TILE])
        full = ttnn.Tile([TILE, TILE])
        self.tiny_bytes = tiny.get_tile_size(ttnn.bfloat16)

        def cb(index, dtype, tile, pages):
            page = tile.get_tile_size(dtype)
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(tile)
            )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.core_ranges, format_descriptors=[fmt])

        self.cb_x, self.cb_w, self.cb_h, self.cb_route = 0, 1, 2, 3
        self.cb_rstage, self.cb_wstage, self.cb_sel = 4, 5, 6
        self.cbs = [
            cb(self.cb_x, ttnn.bfloat16, tiny, self.kt),
            cb(self.cb_w, ttnn.bfloat4_b, full, self.NUM_BUFFERS * self.block_tiles),
            # one h tile per pair: compute never waits on the writer while it is still routing
            cb(self.cb_h, ttnn.bfloat16, tiny, num_pairs),
            # page 0: the reader's pair weights, page 1: the writer's
            cb(self.cb_route, ttnn.float32, full, 2),
            cb(self.cb_rstage, ttnn.float32, full, self.stage_pages),
            cb(self.cb_wstage, ttnn.float32, full, self.stage_pages),
            cb(self.cb_sel, ttnn.float32, full, 1),
        ]
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )

    def __call__(self, x, scores, biased, token_mask, w_gate_up, out, ids_out):
        """x [1, 1, 32, K] bf16, scores / biased [1, 1, 32, E] fp32, token_mask [1, 1, 32, 1] bf16 or None (all DRAM
        interleaved tiles); writes the routed h rows into ``out`` and the uint16 expert ids into ``ids_out``."""
        for t in (x, scores, biased, w_gate_up):
            assert t.layout == ttnn.TILE_LAYOUT and is_dram_interleaved(t)
        assert x.dtype == ttnn.bfloat16 and w_gate_up.dtype == ttnn.bfloat4_b
        assert scores.dtype == ttnn.float32 and biased.dtype == ttnn.float32
        assert scores.shape[-1] == self.expert_tiles * TILE and biased.shape[-1] == self.expert_tiles * TILE
        has_mask = token_mask is not None
        if has_mask:
            assert token_mask.layout == ttnn.TILE_LAYOUT and token_mask.dtype == ttnn.bfloat16
            assert is_dram_interleaved(token_mask)
        assert ids_out.layout == ttnn.ROW_MAJOR_LAYOUT and ids_out.dtype == ttnn.uint16 and is_dram_interleaved(ids_out)
        assert ids_out.shape[-1] == self.num_pairs
        mask_addr = token_mask.buffer_address() if has_mask else 0

        route_ct = [self.expert_tiles, self.norm, self.has_scale, self.scale_bits, 1 if has_mask else 0]
        reader_ct = [
            self.cb_w,
            self.cb_route,
            self.cb_rstage,
            self.cb_sel,
            self.num_pairs,
            self.topk,
            self.block_bytes,
            self.block_tiles,
            self.blocks_per_pair,
            self.packet_bytes,
            self.packets_per_block,
            self.expert_bytes,
            self.NUM_BUFFERS,
            self.IN_FLIGHT,
            self.reader_users,
            self.stage_pages,
        ] + route_ct
        writer_ct = [
            self.cb_x,
            self.cb_h,
            self.kt,
            self.num_pairs,
            self.topk,
            self.out_tiles_per_pair,
            self.cb_wstage,
            self.cb_sel,
            self.reader_users,
            self.users,
            self.stage_pages,
        ] + route_ct
        compute_ct = [
            self.cb_x,
            self.cb_w,
            self.cb_h,
            self.cb_route,
            self.num_pairs,
            self.blocks_per_pair,
            self.BLOCK_K,
            self.reader_users * self.topk,
        ]

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for bank, core in enumerate(self.cores):
            reader_rt[core.x][core.y] = [
                bank,
                self.vcs[bank],
                w_gate_up.buffer_address(),
                biased.buffer_address(),
                scores.buffer_address(),
                mask_addr,
                ids_out.buffer_address(),
            ]
            writer_rt[core.x][core.y] = [
                x.buffer_address(),
                out.buffer_address(),
                bank,
                biased.buffer_address(),
                scores.buffer_address(),
                mask_addr,
            ]
            compute_rt[core.x][core.y] = []

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/moe_gate_up_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/moe_gate_up_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default
            ),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/moe_gate_up_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=self.compute_config,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=self.cbs, semaphores=[])
        io = [x, scores, biased] + ([token_mask] if has_mask else []) + [w_gate_up, ids_out, out]
        ttnn.generic_op(io, program)
        return out


class DownStream:
    """Down projection and the sum over pairs for every decode pair, streamed bank-locally from the interleaved
    w_down, into a compact [1, 1, 32, hidden / 8] output: rows 0-3 of output tile column t go to rows 4c..4c+3 of
    compact tile t % compact_tiles (c = t / compact_tiles), so the row all-gather moves only the user rows;
    ``ReduceExpand`` sums the chips' copies and restores the [1, 1, 32, hidden] layout.

    The per-chip w_down [1, E, I, H] has H / 32 tile columns, so DRAM bank b holds expert e as I / 32 rows of
    H / 32 / banks tiles (output columns b, b + banks, ...), contiguous per expert. Core b streams bank b. The
    activation (GateUpStream's output) holds pair (u, k)'s h only in row u of its [4, 32] tiny tiles, so the
    accumulation over all pairs in fp32 dest leaves user u's expert sum in row u.
    """

    BLOCK_COLS = 4  # output tiles per dest pass: 4 fp32 dest tiles (half sync)
    NUM_BUFFERS = 8
    IN_FLIGHT = 6
    TINY_ROWS = 4

    def __init__(self, plan, hidden, inter_local, num_pairs):
        device = plan.mesh_device
        self.cores = list(dram_bank_cores(device))
        self.banks = device.dram_grid_size().x
        assert len(self.cores) == self.banks, (len(self.cores), self.banks)
        self.kt = inter_local // TILE
        assert self.kt == self.banks, "h tile b comes from bank core b"
        nt = hidden // TILE
        assert nt % self.banks == 0
        self.cols = nt // self.banks
        assert self.cols % self.BLOCK_COLS == 0
        self.groups = self.cols // self.BLOCK_COLS
        assert self.IN_FLIGHT < self.NUM_BUFFERS <= 15
        self.hidden, self.inter_local, self.num_pairs = hidden, inter_local, num_pairs
        # Compact output: 8 row blocks of TINY_ROWS rows per tile, so hidden / 8 columns hold all 4 user rows.
        self.row_blocks = TILE // self.TINY_ROWS
        assert nt % self.row_blocks == 0
        self.compact_tiles = nt // self.row_blocks
        self.compact_width = self.compact_tiles * TILE

        self.w_tile_bytes = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat4_b)
        self.block_tiles = self.kt * self.BLOCK_COLS
        self.group_bytes = self.BLOCK_COLS * self.w_tile_bytes
        self.block_bytes = self.kt * self.group_bytes
        self.row_bytes = self.cols * self.w_tile_bytes
        self.expert_bytes = self.kt * self.row_bytes
        tiny = ttnn.Tile([self.TINY_ROWS, TILE])
        full = ttnn.Tile([TILE, TILE])
        self.tiny_bytes = tiny.get_tile_size(ttnn.bfloat16)
        self.act_page_bytes = self.kt * self.tiny_bytes
        self.ids_bytes = pad_to(2 * num_pairs, 64)
        assert self.ids_bytes <= 4096
        self.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.vcs = bank_vcs(self.cores)

        def cb(index, dtype, tile, pages):
            page = tile.get_tile_size(dtype)
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(tile)
            )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.core_ranges, format_descriptors=[fmt])

        self.cb_h, self.cb_w, self.cb_out, self.cb_ids = 0, 1, 2, 3
        self.cbs = [
            cb(self.cb_h, ttnn.bfloat16, tiny, num_pairs * self.kt),
            cb(self.cb_w, ttnn.bfloat4_b, full, self.NUM_BUFFERS * self.block_tiles),
            cb(self.cb_out, ttnn.bfloat16, tiny, self.cols),
            cb(self.cb_ids, ttnn.float32, full, 1),
        ]
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )

    def activation_buffer(self, plan):
        """Zeroed row-major [1, 1, pairs, kt * 4 * 32] bf16 in L1: page p = pair p's kt [4, 32] tiny tiles."""
        return ttnn.from_torch(
            torch.zeros(1, 1, self.num_pairs, self.act_page_bytes // 2),
            device=plan.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=plan.replicate(),
        )

    def compact_output(self, plan):
        """Persistent [1, 1, 32, hidden / 8] bf16 DRAM tiles; every call rewrites all of it."""
        return zero_output(plan, self.compact_width)

    def __call__(self, act, ids, w_down, out):
        """act from ``activation_buffer``, ids the uint16 [1, 1, 1, pairs] stick, w_down [1, E, I, H] BFP4 (DRAM
        interleaved); fills ``out`` from ``compact_output`` (bf16 DRAM tiles) and returns it."""
        assert act.layout == ttnn.ROW_MAJOR_LAYOUT and act.dtype == ttnn.bfloat16
        assert act.shape[-2] == self.num_pairs and act.shape[-1] * 2 == self.act_page_bytes
        assert ids.layout == ttnn.ROW_MAJOR_LAYOUT and ids.dtype == ttnn.uint16 and is_dram_interleaved(ids)
        assert ids.shape[-1] == self.num_pairs
        assert w_down.layout == ttnn.TILE_LAYOUT and w_down.dtype == ttnn.bfloat4_b and is_dram_interleaved(w_down)
        assert w_down.shape[-2] == self.inter_local and w_down.shape[-1] == self.hidden
        assert out.layout == ttnn.TILE_LAYOUT and out.dtype == ttnn.bfloat16 and is_dram_interleaved(out)
        assert out.shape[-2] == TILE and out.shape[-1] == self.compact_width

        reader_ct = [
            self.cb_w,
            self.cb_ids,
            self.num_pairs,
            self.groups,
            self.kt,
            self.block_tiles,
            self.block_bytes,
            self.group_bytes,
            self.row_bytes,
            self.expert_bytes,
            self.NUM_BUFFERS,
            self.IN_FLIGHT,
            self.ids_bytes,
        ]
        writer_ct = [
            self.cb_h,
            self.cb_out,
            self.num_pairs,
            self.kt,
            self.act_page_bytes,
            self.groups,
            self.BLOCK_COLS,
            self.banks,
            self.tiny_bytes,
            self.compact_tiles,
        ]
        compute_ct = [
            self.cb_h,
            self.cb_w,
            self.cb_out,
            self.num_pairs,
            self.kt,
            self.BLOCK_COLS,
            self.groups,
        ]

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for bank, core in enumerate(self.cores):
            reader_rt[core.x][core.y] = [bank, self.vcs[bank], w_down.buffer_address(), ids.buffer_address()]
            writer_rt[core.x][core.y] = [act.buffer_address(), out.buffer_address(), bank]
            compute_rt[core.x][core.y] = []

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/moe_down_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/moe_down_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default
            ),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/moe_down_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=self.compute_config,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=self.cbs, semaphores=[])
        ttnn.generic_op([act, ids, w_down, out], program)
        return out


class ReduceExpand:
    """Row all-gather of DownStream's compact output ([1, 1, 32, chips * hidden / 8]: chip c's compact tile tc is tile
    column c * compact_tiles + tc) -> the sum over the chips (chip order 0, 1, ..., fp32 accumulation, one bf16
    rounding; identical on every chip) -> [1, 1, 32, hidden] bf16 tiles with the user rows in rows 0-3 and zeros in
    rows 4-31. Bank core i handles compact tiles [i * n, (i + 1) * n): its NOC 0 RISC reads the chips' copies, the
    TRISC sums them, its NOC 1 RISC writes the expanded rows."""

    def __init__(self, plan, hidden, chips, compute_config, rows=4):
        device = plan.mesh_device
        assert rows == 4, "the kernel moves 4-row face blocks"
        self.cores = list(dram_bank_cores(device))
        self.hidden = hidden
        self.chips = chips
        nt = hidden // TILE
        self.row_blocks = TILE // rows
        assert nt % self.row_blocks == 0
        self.compact_tiles = nt // self.row_blocks
        self.compact_width = self.compact_tiles * TILE
        assert self.compact_tiles % len(self.cores) == 0
        self.tiles_per_core = self.compact_tiles // len(self.cores)
        self.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.compute_config = compute_config
        full = ttnn.Tile([TILE, TILE])
        page = full.get_tile_size(ttnn.bfloat16)

        def cb(index, pages):
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=ttnn.bfloat16, page_size=page, tile=ttnn.TileDescriptor(full)
            )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.core_ranges, format_descriptors=[fmt])

        self.cb_in, self.cb_sum, self.cb_zero = 0, 1, 2
        self.cbs = [
            cb(self.cb_in, chips * self.tiles_per_core),
            cb(self.cb_sum, self.tiles_per_core),
            cb(self.cb_zero, 1),
        ]
        # With a residual: the bf16-rounded sum, the residual rows packed like the compact tiles, the residual tiles.
        self.cb_mid, self.cb_res, self.cb_x = 3, 4, 5
        self.residual_cbs = [
            cb(self.cb_in, chips * self.tiles_per_core),
            cb(self.cb_sum, self.tiles_per_core),
            cb(self.cb_mid, 2),
            cb(self.cb_res, self.tiles_per_core),
            cb(self.cb_x, self.tiles_per_core * self.row_blocks),
        ]

    def __call__(self, gathered, out, residual=None):
        """gathered [1, 1, 32, chips * hidden / 8], out [1, 1, 32, hidden] (bf16 DRAM interleaved tiles); returns
        ``out``. ``residual`` [1, 1, 32, hidden] (bf16 DRAM interleaved tiles): out = residual + the bf16-rounded sum
        in rows 0-3 and the residual's rows 4-31 (the all-reduce and the residual add in one program)."""
        tensors = (gathered, out) + ((residual,) if residual is not None else ())
        for t in tensors:
            assert t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16 and is_dram_interleaved(t)
            assert t.shape[-2] == TILE
        assert gathered.shape[-1] == self.chips * self.compact_width and out.shape[-1] == self.hidden
        if residual is not None:
            assert residual.shape[-1] == self.hidden

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        res_args = [residual.buffer_address()] if residual is not None else []
        for i, core in enumerate(self.cores):
            reader_rt[core.x][core.y] = [gathered.buffer_address(), i * self.tiles_per_core] + res_args
            writer_rt[core.x][core.y] = [out.buffer_address(), i * self.tiles_per_core] + res_args
            compute_rt[core.x][core.y] = []

        if residual is None:
            sources = ("moe_reduce_reader.cpp", "moe_expand_rows.cpp", "moe_reduce_compute.cpp")
            reader_ct = [self.cb_in, self.tiles_per_core, self.compact_tiles, self.chips]
            writer_ct = [self.cb_sum, self.cb_zero, self.tiles_per_core, self.compact_tiles, self.row_blocks]
            compute_ct = [self.cb_in, self.cb_sum, self.tiles_per_core, self.chips]
            cbs = self.cbs
        else:
            sources = ("moe_reduce_residual_reader.cpp", "moe_expand_residual.cpp", "moe_reduce_residual_compute.cpp")
            reader_ct = [self.cb_in, self.cb_res, self.tiles_per_core, self.compact_tiles, self.chips, self.row_blocks]
            writer_ct = [self.cb_sum, self.cb_x, self.tiles_per_core, self.compact_tiles, self.row_blocks]
            compute_ct = [self.cb_in, self.cb_mid, self.cb_res, self.cb_sum, self.tiles_per_core, self.chips]
            cbs = self.residual_cbs

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/{sources[0]}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/{sources[1]}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default
            ),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/{sources[2]}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=self.compute_config,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs, semaphores=[])
        io = [gathered] + ([residual] if residual is not None else []) + [out]
        ttnn.generic_op(io, program)
        return out


class CompactRows:
    """Rows 0-3 of a [1, 1, 32, hidden] bf16 tile tensor (L1 or DRAM interleaved) -> DownStream's compact
    [1, 1, 32, hidden / 8] layout (rows 4c..4c+3 of compact tile t % compact_tiles hold rows 0-3 of tile column t,
    c = t / compact_tiles), so a row all-gather and ``ReduceExpand`` all-reduce only the user rows. One data-movement
    kernel on 16 cores, each moving the rows of hidden / 32 / 16 tile columns."""

    ROWS = 4
    GRID = (8, 2)

    def __init__(self, plan, hidden):
        nt = hidden // TILE
        self.row_blocks = TILE // self.ROWS
        assert nt % self.row_blocks == 0
        self.hidden = hidden
        self.compact_tiles = nt // self.row_blocks
        self.compact_width = self.compact_tiles * TILE
        gx, gy = self.GRID
        self.cores = [ttnn.CoreCoord(x, y) for y in range(gy) for x in range(gx)]
        assert nt % len(self.cores) == 0
        self.tiles_per_core = nt // len(self.cores)
        self.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        tiny = ttnn.Tile([self.ROWS, TILE])
        page = tiny.get_tile_size(ttnn.bfloat16)
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=0, data_format=ttnn.bfloat16, page_size=page, tile=ttnn.TileDescriptor(tiny)
        )
        self.cbs = [
            ttnn.CBDescriptor(
                total_size=self.tiles_per_core * page, core_ranges=self.core_ranges, format_descriptors=[fmt]
            )
        ]

    def output(self, plan):
        """Persistent [1, 1, 32, hidden / 8] bf16 DRAM tiles; every call rewrites all of it."""
        return zero_output(plan, self.compact_width)

    def __call__(self, x, out):
        """x [1, 1, 32, hidden] bf16 interleaved tiles -> ``out`` from ``output``; returns ``out``."""
        mc = x.memory_config()
        assert x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16
        assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        assert x.shape[-2] == TILE and x.shape[-1] == self.hidden
        assert out.layout == ttnn.TILE_LAYOUT and out.dtype == ttnn.bfloat16 and is_dram_interleaved(out)
        assert out.shape[-2] == TILE and out.shape[-1] == self.compact_width
        in_dram = 1 if mc.buffer_type == ttnn.BufferType.DRAM else 0

        rt = ttnn.RuntimeArgs()
        for i, core in enumerate(self.cores):
            rt[core.x][core.y] = [x.buffer_address(), out.buffer_address(), i * self.tiles_per_core]
        kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/attn/attn_compact_rows.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=[0, self.tiles_per_core, self.compact_tiles, in_dram],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        program = ttnn.ProgramDescriptor(kernels=[kernel], cbs=self.cbs, semaphores=[])
        ttnn.generic_op([x, out], program)
        return out
