# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM-streaming expert path for MiMo MoE decode (a ``ttnn.generic_op`` program).

Blackhole streams DRAM fastest when each bank is read by the worker core next to it, in large
contiguous blocks with several reads in flight. ``GateUpStream`` selects every user's top-k experts and
computes gate|up, SwiGLU and the routing weight of every (user, expert) pair of a decode step on the eight
bank cores; its device kernels live in ``tt/kernels/`` and are JIT-compiled from the checkout.
"""

import math
import os

import torch

import ttnn

from .common import TILE, pad_to

KERNEL_DIR = "models/experimental/mimo_v2_6_flash/tt/kernels"
# MIMO_MOE_DEDUP=0: stream every (user, top-k) pair's expert, as before the deduplicated stream (GateUpStream).
DEDUP = os.environ.get("MIMO_MOE_DEDUP", "1") == "1"
FUSED = os.environ.get("MIMO_MOE_FUSED", "1") == "1"
# Timing experiments on FusedStream only, results wrong: 1 = no weight reads, 2 = no matmuls.
STREAM_DEBUG = int(os.environ.get("MIMO_MOE_STREAM_DEBUG", "0"))


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


def semaphore_descriptor(ranges):
    """Semaphore 0 (initial value 0) of a program on ``ranges``, or None if the bindings take other arguments."""
    attempts = [dict(id=0, core_ranges=ranges, initial_value=0), dict(core_ranges=ranges, initial_value=0)]
    core_type = getattr(getattr(ttnn, "CoreType", None), "Worker", None)
    if core_type is not None:
        attempts.append(dict(core_type=core_type, core_ranges=ranges, initial_value=0))
    for kwargs in attempts:
        try:
            return ttnn.SemaphoreDescriptor(**kwargs)
        except (TypeError, AttributeError):
            continue
    return None


def noc_coords(device, cores):
    """Virtual NoC coordinates of logical worker ``cores`` (the same on every chip), or None without the lookup."""
    try:
        return [device.worker_core_from_logical_core(c) for c in cores]
    except (AttributeError, TypeError, RuntimeError):
        return None


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


def user_rows(users):
    """The rows the decode streams carry for ``users`` users: tiny [4, 32] tiles up to 4 users, [8, 32] up to 8."""
    assert 1 <= users <= 8, users
    return 4 if users <= 4 else 8


def stick_bytes(num_pairs):
    """Bytes of the deduplicated stream's expert stick (uint16 [count, ids...] for up to ``num_pairs`` experts), whole
    64-byte DRAM write units."""
    return pad_to(2 * (num_pairs + 1), 64)


def stick_buffer(plan, num_pairs, dedup=None):
    """The row-major uint16 [1, 1, 1, n] DRAM stick GateUpStream writes the experts into and DownStream reads: the
    pairs' ids, or deduplicated [count, distinct ids...]."""
    dedup = DEDUP if dedup is None else dedup
    n = stick_bytes(num_pairs) // 2 if dedup else num_pairs
    ids32 = ttnn.from_torch(
        torch.zeros(1, 1, 1, n, dtype=torch.int32),
        device=plan.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=plan.replicate(),
    )
    out = ttnn.typecast(ids32, ttnn.uint16)
    ids32.deallocate(True)
    return out


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

    Deduplicated (``dedup``, the default; MIMO_MOE_DEDUP=0 turns it off): every distinct expert the active users (the
    token mask's nonzero rows) selected is streamed once. The users of one step are often one request at consecutive
    positions (a speculative verify), whose top-8 sets overlap (4 consecutive tokens select ~23 distinct experts, 8
    ~37), and an idle slot selects nothing. The reader routes users [0, users / 2) and streams their experts while the
    writer routes the rest; both list the distinct experts in user order. Expert j's h tile carries every user's row,
    scaled by that user's routing weight of j (0 when it did not select j: the writer's weight tile), so page j of
    ``out`` is expert j's whole [4, 32] tiny tiles, and ``ids_out`` is the stick [count, ids...] DownStream reads.
    """

    BLOCK_K = 16  # K tiles per block: 16 x (gate, up) x 576 B = 18 KB
    # 12 blocks (216 KB, ~1.5 pairs) keep the bank streaming through a pair's SwiGLU epilogue; trids 1..12.
    NUM_BUFFERS = 12
    IN_FLIGHT = 6
    MAX_PACKET = 16384
    MAX_USERS = 4  # the pair stream: routing stages rows 0-3 of each tile face
    MAX_USERS_DEDUP = 8  # the deduplicated stream stages rows 0-7 when it serves more than 4 users
    FP32_TILE_BYTES = 4096

    def __init__(
        self,
        plan,
        hidden,
        inter_local,
        num_pairs,
        topk,
        num_experts,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        dedup=None,
    ):
        device = plan.mesh_device
        self.dedup = DEDUP if dedup is None else dedup
        self.cores = list(dram_bank_cores(device))
        self.banks = device.dram_grid_size().x
        assert len(self.cores) == self.banks, (len(self.cores), self.banks)
        assert inter_local == TILE * self.banks, "gate|up column b must live in bank b"
        assert hidden % (TILE * self.BLOCK_K) == 0
        assert num_experts % TILE == 0
        self.users = num_pairs // topk
        max_users = self.MAX_USERS_DEDUP if self.dedup else self.MAX_USERS
        assert self.users * topk == num_pairs and 1 <= self.users <= max_users and topk <= 16
        assert num_pairs * 4 <= 1024
        assert self.IN_FLIGHT < self.NUM_BUFFERS <= 15
        self.num_pairs, self.topk = num_pairs, topk
        # The reader routes user 0 and streams its pairs while the writer routes the others; deduplicated, every
        # weight waits for every user's routing, so the two split it.
        self.reader_users = (self.users + 1) // 2 if self.dedup else 1
        self.rows = user_rows(self.users)
        self.stick_bytes = stick_bytes(num_pairs) if self.dedup else 2 * num_pairs
        self.expert_tiles = num_experts // TILE
        self.norm = 1 if norm_topk_prob else 0
        self.has_scale = 0 if routed_scaling_factor == 1.0 else 1
        self.scale_bits = fp32_bits(routed_scaling_factor)
        # the user rows of the biased and score tiles (faces 0 and 1), 512 B of mask rows, the uint16 id stick
        stage_bytes = 2 * self.expert_tiles * 2 * self.rows * 64 + 512 + self.stick_bytes
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

        tiny = ttnn.Tile([self.rows, TILE])
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
        # Deduplicated: the reader's users' selections for the writer, the distinct-expert count for compute, and one
        # routing-weight tile per distinct expert.
        self.cb_local, self.cb_cnt, self.cb_wt = 7, 8, 9
        if self.dedup:
            self.cbs += [
                cb(self.cb_local, ttnn.float32, full, 1),
                cb(self.cb_cnt, ttnn.float32, full, 1),
                cb(self.cb_wt, ttnn.bfloat16, full, num_pairs),
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
        assert ids_out.shape[-1] * 2 == self.stick_bytes
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
        sources = ("moe_gate_up_reader.cpp", "moe_gate_up_writer.cpp", "moe_gate_up_compute.cpp")
        if self.dedup:
            sources = ("moe_gate_up_uniq_reader.cpp", "moe_gate_up_uniq_writer.cpp", "moe_gate_up_uniq_compute.cpp")
            reader_ct[1] = self.cb_local
            reader_ct += [self.stick_bytes, self.rows]
            writer_ct += [self.cb_local, self.cb_cnt, self.cb_wt, self.rows]
            compute_ct = [
                self.cb_x,
                self.cb_w,
                self.cb_h,
                self.cb_cnt,
                self.cb_wt,
                self.blocks_per_pair,
                self.BLOCK_K,
                self.rows,
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

    def __init__(self, plan, hidden, inter_local, num_pairs, dedup=None, rows=4):
        """``rows``: the user rows of the tiny tiles and the compact output (4, or 8 with the deduplicated stream)."""
        device = plan.mesh_device
        self.dedup = DEDUP if dedup is None else dedup
        assert rows == 4 or (rows == 8 and self.dedup), rows
        self.rows = rows
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
        # Compact output: TILE / rows row blocks of ``rows`` rows per tile, so hidden * rows / 32 columns hold every
        # user row.
        self.row_blocks = TILE // rows
        assert nt % self.row_blocks == 0
        self.compact_tiles = nt // self.row_blocks
        self.compact_width = self.compact_tiles * TILE

        self.w_tile_bytes = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat4_b)
        self.block_tiles = self.kt * self.BLOCK_COLS
        self.group_bytes = self.BLOCK_COLS * self.w_tile_bytes
        self.block_bytes = self.kt * self.group_bytes
        self.row_bytes = self.cols * self.w_tile_bytes
        self.expert_bytes = self.kt * self.row_bytes
        tiny = ttnn.Tile([rows, TILE])
        full = ttnn.Tile([TILE, TILE])
        self.tiny_bytes = tiny.get_tile_size(ttnn.bfloat16)
        self.act_page_bytes = self.kt * self.tiny_bytes
        self.ids_bytes = stick_bytes(num_pairs) if self.dedup else pad_to(2 * num_pairs, 64)
        assert self.ids_bytes <= 2048  # the reader's and the writer's copies share cb_ids
        self.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.vcs = bank_vcs(self.cores)

        def cb(index, dtype, tile, pages):
            page = tile.get_tile_size(dtype)
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(tile)
            )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.core_ranges, format_descriptors=[fmt])

        self.cb_h, self.cb_w, self.cb_out, self.cb_ids, self.cb_cnt = 0, 1, 2, 3, 4
        self.cbs = [
            cb(self.cb_h, ttnn.bfloat16, tiny, num_pairs * self.kt),
            cb(self.cb_w, ttnn.bfloat4_b, full, self.NUM_BUFFERS * self.block_tiles),
            cb(self.cb_out, ttnn.bfloat16, tiny, self.cols),
            cb(self.cb_ids, ttnn.float32, full, 1),
        ]
        if self.dedup:  # the distinct experts' count, reader -> compute
            self.cbs.append(cb(self.cb_cnt, ttnn.float32, full, 1))
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )

    _ACTIVATIONS = {}  # (device ids, pairs, page bytes) -> the activation buffer every layer of a (sub)mesh shares

    def activation_buffer(self, plan):
        """Zeroed row-major [1, 1, pairs, kt * rows * 32] bf16 in L1: page p = pair (or distinct expert) p's kt
        [rows, 32] tiny tiles. One buffer for every MoE layer of the (sub)mesh: a layer's down stream reads it before
        the next layer's gate|up stream rewrites it, and the pair stream writes the same rows of it in every layer (its
        other rows stay zero). Persistent L1, so one copy instead of one per layer (47 x 4 KB per bank with 8 rows)."""
        key = (tuple(plan.mesh_device.get_device_ids()), self.num_pairs, self.act_page_bytes)
        cached = self._ACTIVATIONS.get(key)
        if cached is None or not cached.is_allocated():  # a mesh opened again in the same process
            self._ACTIVATIONS[key] = ttnn.from_torch(
                torch.zeros(1, 1, self.num_pairs, self.act_page_bytes // 2),
                device=plan.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=plan.replicate(),
            )
        return self._ACTIVATIONS[key]

    def compact_output(self, plan):
        """Persistent [1, 1, 32, hidden / 8] bf16 DRAM tiles; every call rewrites all of it."""
        return zero_output(plan, self.compact_width)

    def __call__(self, act, ids, w_down, out, signal=None, launch=True):
        """act from ``activation_buffer``, ids the uint16 [1, 1, 1, pairs] stick, w_down [1, E, I, H] BFP4 (DRAM
        interleaved); fills ``out`` from ``compact_output`` (bf16 DRAM tiles) and returns it.

        ``signal``: NoC coordinates of cores whose semaphore 0 each bank core raises once its output is written.
        ``launch=False``: return the launch as a section (kernels, cbs, tensors, semaphores) for a merged program."""
        assert act.layout == ttnn.ROW_MAJOR_LAYOUT and act.dtype == ttnn.bfloat16
        assert act.shape[-2] == self.num_pairs and act.shape[-1] * 2 == self.act_page_bytes
        assert ids.layout == ttnn.ROW_MAJOR_LAYOUT and ids.dtype == ttnn.uint16 and is_dram_interleaved(ids)
        assert ids.shape[-1] == (self.ids_bytes // 2 if self.dedup else self.num_pairs)
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
            (4 if self.dedup else 3) if signal else 0,  # signal args follow the writer's other runtime args
            0,
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
        sources = ("moe_down_reader.cpp", "moe_down_writer.cpp", "moe_down_compute.cpp")
        if self.dedup:
            sources = ("moe_down_uniq_reader.cpp", "moe_down_uniq_writer.cpp", "moe_down_uniq_compute.cpp")
            reader_ct.append(self.cb_cnt)
            writer_ct += [self.ids_bytes, self.cb_ids]
            compute_ct.append(self.cb_cnt)

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for bank, core in enumerate(self.cores):
            reader_rt[core.x][core.y] = [bank, self.vcs[bank], w_down.buffer_address(), ids.buffer_address()]
            args = [act.buffer_address(), out.buffer_address(), bank]
            if self.dedup:
                args.append(ids.buffer_address())
            if signal:
                args += [len(signal)] + [v for c in signal for v in (c.x, c.y)]
            writer_rt[core.x][core.y] = args
            compute_rt[core.x][core.y] = []

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
        section = dict(kernels=[reader, writer, compute], cbs=list(self.cbs), semaphores=[], tensors=[act, ids, w_down])
        if not launch:
            return section
        program = ttnn.ProgramDescriptor(kernels=section["kernels"], cbs=section["cbs"], semaphores=[])
        ttnn.generic_op([act, ids, w_down, out], program)
        return out


class FusedStream:
    """A deduplicated GateUpStream and DownStream in one launch on the bank cores (MIMO_MOE_FUSED=0 turns it off).

    Routing is spread over the bank cores: core b selects user b's top-k and weights (one user, ~6 us, instead of the
    streams' four per RISC) and writes them into every bank core's routing table, so the stream starts ~10 us into the
    launch instead of ~30-40. Each bank core then streams, back to back from its DRAM bank, the gate|up slice and then the down slice of every
    distinct expert. Its h tile (h column ``bank``) of each expert goes straight into every bank core's L1 (cb_hall),
    so the down matmul starts once the last expert's h lands everywhere, without a second launch or the activation's
    DRAM round trip, and with its weights already prefetched. Down accumulates all of a core's output tiles in fp32
    dest at once (the Compact layout), so its blocks are each expert's contiguous slice, two K rows of ``cols``
    tiles: the gate|up block's tile count, so one weight ring serves both. Routing, the expert list, the h values and
    the compact output are the two streams' (same order, same arithmetic)."""

    DOWN_ROWS = 2  # K rows of w_down per block
    CB_HALL, CB_OUT, CB_TABLE = 10, 11, 12
    READY_SEM, DONE_SEM, ROUTE_SEM = 0, 1, 2
    ENTRY_BYTES = 128  # a routing-table entry: [active, ids[topk], weight bits[topk]], 16-byte aligned

    def __init__(self, plan, gate_up, down):
        assert gate_up.dedup and down.dedup and gate_up.rows == down.rows
        assert gate_up.cores == down.cores and gate_up.num_pairs == down.num_pairs
        self.gu, self.dn = gate_up, down
        self.cols = down.cols
        assert self.DOWN_ROWS * self.cols == gate_up.block_tiles, (self.cols, gate_up.block_tiles)
        assert down.kt % self.DOWN_ROWS == 0
        self.down_blocks = down.kt // self.DOWN_ROWS
        assert self.down_blocks * gate_up.block_bytes == down.expert_bytes
        self.peers = noc_coords(plan.mesh_device, gate_up.cores)
        assert self.peers is not None, "the h exchange needs the bank cores' NoC coordinates"
        tiny = ttnn.Tile([gate_up.rows, TILE])
        page = tiny.get_tile_size(ttnn.bfloat16)

        def cb(index, pages):
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=ttnn.bfloat16, page_size=page, tile=ttnn.TileDescriptor(tiny)
            )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=gate_up.core_ranges, format_descriptors=[fmt])

        assert gate_up.users <= len(gate_up.cores) and (2 * gate_up.topk + 1) * 4 <= self.ENTRY_BYTES
        table_bytes = len(gate_up.cores) * self.ENTRY_BYTES
        table = ttnn.CBDescriptor(
            total_size=table_bytes,
            core_ranges=gate_up.core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(self.CB_TABLE, ttnn.uint32, table_bytes)],
        )
        self.cbs = list(gate_up.cbs) + [
            cb(self.CB_HALL, gate_up.num_pairs * down.kt),
            cb(self.CB_OUT, self.cols),
            table,
        ]
        self.semaphores = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=gate_up.core_ranges, initial_value=0)
            for i in (self.READY_SEM, self.DONE_SEM, self.ROUTE_SEM)
        ]

    def __call__(self, x, scores, biased, token_mask, w_gate_up, w_down, out, signal=None, launch=True):
        """GateUpStream's inputs plus w_down [1, E, I, H] BFP4 and ``out`` from DownStream.compact_output; fills
        ``out`` and returns it. ``signal`` / ``launch=False``: as DownStream (a section for a merged program)."""
        gu, dn = self.gu, self.dn
        for t in (x, scores, biased, w_gate_up, w_down, out):
            assert t.layout == ttnn.TILE_LAYOUT and is_dram_interleaved(t)
        assert x.dtype == ttnn.bfloat16 and w_gate_up.dtype == ttnn.bfloat4_b and w_down.dtype == ttnn.bfloat4_b
        assert scores.dtype == ttnn.float32 and biased.dtype == ttnn.float32
        assert w_down.shape[-2] == dn.inter_local and w_down.shape[-1] == dn.hidden
        assert out.dtype == ttnn.bfloat16 and out.shape[-2] == TILE and out.shape[-1] == dn.compact_width
        has_mask = token_mask is not None
        if has_mask:
            assert token_mask.layout == ttnn.TILE_LAYOUT and token_mask.dtype == ttnn.bfloat16
            assert is_dram_interleaved(token_mask)
        mask_addr = token_mask.buffer_address() if has_mask else 0

        route_ct = [gu.expert_tiles, gu.norm, gu.has_scale, gu.scale_bits, 1 if has_mask else 0]
        banks = len(gu.cores)
        reader_ct = [
            gu.cb_w,
            gu.cb_rstage,
            self.CB_TABLE,
            gu.num_pairs,
            gu.topk,
            gu.block_bytes,
            gu.block_tiles,
            gu.blocks_per_pair,
            gu.packet_bytes,
            gu.packets_per_block,
            gu.expert_bytes,
            gu.NUM_BUFFERS,
            gu.IN_FLIGHT,
            gu.stage_pages,
            *route_ct,
            gu.rows,
            self.down_blocks,
            dn.expert_bytes,
            banks,
            self.ROUTE_SEM,
            self.ENTRY_BYTES,
            STREAM_DEBUG,
        ]
        writer_ct = [
            gu.cb_x,
            gu.cb_h,
            gu.kt,
            gu.num_pairs,
            gu.topk,
            self.CB_TABLE,
            gu.users,
            self.ROUTE_SEM,
            self.ENTRY_BYTES,
            gu.cb_cnt,
            gu.cb_wt,
            gu.rows,
            self.CB_HALL,
            dn.kt,
            banks,
            self.READY_SEM,
            self.DONE_SEM,
            self.CB_OUT,
            self.cols,
            dn.compact_tiles,
            3 + 2 * banks if signal else 0,  # signal args follow the writer's other runtime args
            0,
        ]
        compute_ct = [
            gu.cb_x,
            gu.cb_w,
            gu.cb_h,
            gu.cb_cnt,
            gu.cb_wt,
            gu.blocks_per_pair,
            gu.BLOCK_K,
            gu.rows,
            self.CB_HALL,
            self.CB_OUT,
            dn.kt,
            self.down_blocks,
            self.DOWN_ROWS,
            self.cols,
            STREAM_DEBUG,
        ]
        peers = [v for c in self.peers for v in (c.x, c.y)]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for bank, core in enumerate(gu.cores):
            reader_rt[core.x][core.y] = [
                bank,
                gu.vcs[bank],
                w_gate_up.buffer_address(),
                biased.buffer_address(),
                scores.buffer_address(),
                mask_addr,
                w_down.buffer_address(),
            ] + peers
            args = [x.buffer_address(), out.buffer_address(), bank] + peers
            if signal:
                args += [len(signal)] + [v for c in signal for v in (c.x, c.y)]
            writer_rt[core.x][core.y] = args
            compute_rt[core.x][core.y] = []

        def kernel(name, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_DIR}/{name}",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=gu.core_ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )

        kernels = [
            kernel(
                "moe_fused_uniq_reader.cpp",
                reader_ct,
                reader_rt,
                ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
                ),
            ),
            kernel(
                "moe_fused_uniq_writer.cpp",
                writer_ct,
                writer_rt,
                ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default
                ),
            ),
            kernel("moe_fused_uniq_compute.cpp", compute_ct, compute_rt, gu.compute_config),
        ]
        tensors = [x, scores, biased] + ([token_mask] if has_mask else []) + [w_gate_up, w_down]
        section = dict(kernels=kernels, cbs=list(self.cbs), semaphores=list(self.semaphores), tensors=tensors)
        if not launch:
            return section
        program = ttnn.ProgramDescriptor(kernels=kernels, cbs=section["cbs"], semaphores=section["semaphores"])
        ttnn.generic_op(tensors + [out], program)
        return out


class ReduceExpand:
    """Row all-gather of DownStream's compact output ([1, 1, 32, chips * hidden / 8]: chip c's compact tile tc is tile
    column c * compact_tiles + tc) -> the sum over the chips (chip order 0, 1, ..., fp32 accumulation, one bf16
    rounding; identical on every chip) -> [1, 1, 32, hidden] bf16 tiles with the user rows in rows 0-3 and zeros in
    rows 4-31. Bank core i handles compact tiles [i * n, (i + 1) * n): its NOC 0 RISC reads the chips' copies, the
    TRISC sums them, its NOC 1 RISC writes the expanded rows."""

    # A row of cores clear of the bank cores and the bank rows the producers (DownStream, the o projection's
    # BankStreamMatmul) run on, so a producer and this reduce can share one launch (FabricRowGather).
    CORE_ROW = 5

    def __init__(self, plan, hidden, chips, compute_config, rows=4):
        """``rows``: the user rows of the compact tiles, 4 or 8."""
        device = plan.mesh_device
        assert rows in (4, 8), rows
        self.rows = rows
        self.cores = [ttnn.CoreCoord(x, self.CORE_ROW) for x in range(device.dram_grid_size().x)]
        self.hidden = hidden
        self.chips = chips
        nt = hidden // TILE
        self.row_blocks = TILE // rows
        assert nt % self.row_blocks == 0
        self.compact_tiles = nt // self.row_blocks
        self.compact_width = self.compact_tiles * TILE
        assert self.compact_tiles % len(self.cores) == 0
        self.tiles_per_core = self.compact_tiles // len(self.cores)
        # The fused norm's gather tiles: every core's user rows of its partial, 32 / rows cores per tile.
        self.gather_tiles = -(-len(self.cores) * rows // TILE)
        self.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(self.cores[0], self.cores[-1])])
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
        # With a norm as well: y (the residual sum, for the norm), y^2, the J / M constants, the row sums, this
        # core's partial, every core's user rows of their partials, gamma and the normalized tiles.
        self.cb_y, self.cb_sq, self.cb_const, self.cb_rs, self.cb_part = 6, 7, 8, 9, 10
        self.cb_gather, self.cb_gamma, self.cb_h = 11, 12, 13
        fp32_page = full.get_tile_size(ttnn.float32)

        def cb32(index, pages):
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=ttnn.float32, page_size=fp32_page, tile=ttnn.TileDescriptor(full)
            )
            return ttnn.CBDescriptor(
                total_size=pages * fp32_page, core_ranges=self.core_ranges, format_descriptors=[fmt]
            )

        self.norm_cbs = self.residual_cbs + [
            cb(self.cb_y, self.tiles_per_core),
            cb32(self.cb_sq, 1),
            cb32(self.cb_const, 2),
            cb32(self.cb_rs, 1),
            cb32(self.cb_part, 1),
            cb32(self.cb_gather, self.gather_tiles),
            cb(self.cb_gamma, self.tiles_per_core),
            cb(self.cb_h, self.tiles_per_core),
        ]
        self.norm_compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )
        self.noc_cores = noc_coords(device, self.cores)
        self.norm_consts = None
        if self.noc_cores is not None and semaphore_descriptor(self.core_ranges) is not None:
            # J (all ones) and M (M[r][c] = 1 when r % rows == c % rows) as two fp32 tiles.
            r = torch.arange(TILE)
            m = (r.reshape(-1, 1) % rows == r.reshape(1, -1) % rows).float()
            self.norm_consts = ttnn.from_torch(
                torch.cat([torch.ones(TILE, TILE), m], dim=1).reshape(1, 1, TILE, 2 * TILE),
                device=device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=plan.replicate(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def compact_gamma(self, plan, gamma):
        """RMSNorm weight ``gamma`` (torch [hidden]) packed like the compact tiles: row 4c + u of compact tile tc holds
        gamma over tile column c * compact_tiles + tc, for every user u. [1, 1, 32, hidden / 8] bf16 DRAM tiles."""
        g = gamma.float().reshape(self.row_blocks, self.compact_tiles, TILE)  # [c, tc, j]
        rows = TILE // self.row_blocks
        packed = g.reshape(self.row_blocks, 1, self.compact_width).expand(-1, rows, -1)
        packed = packed.reshape(1, 1, TILE, self.compact_width)
        return ttnn.from_torch(
            packed.contiguous(),
            device=plan.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=plan.replicate(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @property
    def can_norm(self):
        return self.norm_consts is not None

    def __call__(self, gathered, out, residual=None, norm=None):
        """gathered [1, 1, 32, chips * hidden / 8], out [1, 1, 32, hidden] (bf16 DRAM interleaved tiles); returns
        ``out``. ``residual`` [1, 1, 32, hidden] (bf16 DRAM interleaved tiles): out = residual + the bf16-rounded sum
        in rows 0-3 and the residual's rows 4-31 (the all-reduce and the residual add in one program).

        ``norm`` (with ``residual``): (gamma from ``compact_gamma``, norm_out, eps). Also writes RMSNorm(out) * gamma of
        the user rows into rows 0-3 of ``norm_out`` ([1, 1, 32, hidden] bf16 DRAM tiles whose rows 4-31 the caller
        keeps zero), so the next block's norm is part of this program."""
        if norm is not None:
            assert residual is not None and self.can_norm
            return self._call_norm(gathered, out, residual, *norm)
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
            writer_ct = [self.cb_sum, self.cb_zero, self.tiles_per_core, self.compact_tiles, self.row_blocks, self.rows]
            compute_ct = [self.cb_in, self.cb_sum, self.tiles_per_core, self.chips]
            cbs = self.cbs
        else:
            sources = ("moe_reduce_residual_reader.cpp", "moe_expand_residual.cpp", "moe_reduce_residual_compute.cpp")
            reader_ct = [self.cb_in, self.cb_res, self.tiles_per_core, self.compact_tiles, self.chips, self.row_blocks]
            reader_ct.append(self.rows)
            writer_ct = [self.cb_sum, self.cb_x, self.tiles_per_core, self.compact_tiles, self.row_blocks, self.rows]
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

    def call_fabric(self, compact, out, residual, norm, fabric):
        if isinstance(compact, PendingCompact):  # the producer runs in the same launch
            return self._call_fabric(compact.rows, out, residual, norm, fabric, producer=compact)
        return self._call_fabric(compact, out, residual, norm, fabric)

    def _call_fabric(self, compact, out, residual, norm, fabric, producer=None):
        """The row all-gather and the fused reduce + residual + RMSNorm in one launch per chip: ``compact`` is this
        chip's [1, 1, 32, hidden / 8] compact rows (DRAM tiles), sent over the fabric to every chip of the row
        (``fabric``: the row's FabricRowGather); returns (out, norm_out) as ``__call__`` with ``norm``. ``producer``
        (a PendingCompact): the launch that writes ``compact`` runs in this one, on other cores."""
        gamma, norm_out, eps = norm
        for t in (compact, out, residual, gamma, norm_out):
            assert t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16 and is_dram_interleaved(t)
        assert compact.shape[-1] == self.compact_width
        phase = fabric.next_phase()
        kernels, cbs, semaphores = self._norm_kernels(
            out, residual, gamma, norm_out, eps, gather_sem=fabric.sem_addr(phase)
        )
        # cb_in is the gather buffer of this phase (every chip's copies of this core's tiles land there)
        cbs[self.cb_in] = ttnn.cb_descriptor_from_sharded_tensor(
            self.cb_in, fabric.gather, address_offset=phase * fabric.phase_bytes, total_size=fabric.phase_bytes
        )
        extra_tensors = []
        if producer is not None:
            kernels = kernels + producer.section["kernels"]
            cbs = cbs + producer.section["cbs"]
            semaphores = semaphores + producer.section["semaphores"]
            extra_tensors = producer.section["tensors"]
        waits = producer.signals if producer is not None else 0
        mesh_program = fabric.programs(kernels, cbs, semaphores, compact, phase, wait_count=waits)
        io = [compact, residual, gamma, self.norm_consts, fabric.gather] + list(extra_tensors) + [out, norm_out]
        ttnn.generic_op(io, mesh_program)
        fabric.advance()
        return out, norm_out

    def _call_norm(self, gathered, out, residual, gamma, norm_out, eps):
        for t in (gathered, out, residual, gamma, norm_out):
            assert t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16 and is_dram_interleaved(t)
            assert t.shape[-2] == TILE
        assert gathered.shape[-1] == self.chips * self.compact_width
        kernels, cbs, semaphores = self._norm_kernels(out, residual, gamma, norm_out, eps, gathered=gathered)
        program = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
        ttnn.generic_op([gathered, residual, gamma, self.norm_consts, out, norm_out], program)
        return out, norm_out

    def _norm_kernels(self, out, residual, gamma, norm_out, eps, gathered=None, gather_sem=None):
        """Kernels, CBs and semaphores of the fused reduce + residual + RMSNorm on the bank cores. The chips' copies
        come from ``gathered`` (a row all-gather in DRAM) or, with ``gather_sem``, from the fabric gather buffer."""
        assert out.shape[-1] == self.hidden and residual.shape[-1] == self.hidden and norm_out.shape[-1] == self.hidden
        assert gamma.shape[-1] == self.compact_width
        cores = len(self.cores)
        assert cores * self.rows == self.gather_tiles * TILE, "the cores' user rows fill the gather tiles"
        noc = []
        for c in self.noc_cores:
            noc += [c.x, c.y]

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(self.cores):
            first = i * self.tiles_per_core
            reader_rt[core.x][core.y] = [
                gathered.buffer_address() if gathered is not None else gather_sem,
                first,
                residual.buffer_address(),
                gamma.buffer_address(),
                self.norm_consts.buffer_address(),
            ]
            writer_rt[core.x][core.y] = [
                out.buffer_address(),
                first,
                residual.buffer_address(),
                norm_out.buffer_address(),
                i,
            ] + noc
            compute_rt[core.x][core.y] = []
        sem_id = 0
        reader_ct = [
            self.cb_in,
            self.cb_res,
            self.tiles_per_core,
            self.compact_tiles,
            self.chips,
            self.row_blocks,
            self.cb_gamma,
            self.cb_const,
            self.cb_gather,
            cores,
            sem_id,
            self.rows,
            self.gather_tiles,
        ]
        writer_ct = [
            self.cb_sum,
            self.cb_x,
            self.tiles_per_core,
            self.compact_tiles,
            self.row_blocks,
            self.cb_part,
            self.cb_gather,
            self.cb_h,
            cores,
            sem_id,
            self.rows,
        ]
        compute_ct = [
            self.cb_in,
            self.cb_mid,
            self.cb_res,
            self.cb_sum,
            self.tiles_per_core,
            self.chips,
            self.cb_y,
            self.cb_sq,
            self.cb_const,
            self.cb_rs,
            self.cb_part,
            self.cb_gather,
            self.cb_gamma,
            self.cb_h,
            fp32_bits(1.0 / self.hidden),
            fp32_bits(eps),
            self.gather_tiles,
        ]
        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/moe_reduce_norm_reader{'' if gathered is not None else '_fabric'}.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/moe_expand_norm.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default
            ),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/moe_reduce_norm_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=self.norm_compute_config,
        )
        return [reader, writer, compute], list(self.norm_cbs), [semaphore_descriptor(self.core_ranges)]


class FabricRowGather:
    """The row all-gather of the compact rows feeding ReduceExpand.call_fabric, over the fabric instead of a separate
    all-gather op: one transport core per chip multicasts this chip's compact tiles (BRISC: the next 4 chips of the
    ring, NCRISC: the previous 3), each reduce core's two tiles as one fused scatter write + semaphore increment into
    that core's persistent gather buffer on every chip. Shared by every layer of a row.

    Chips run at most one gather phase apart (a chip finishes phase k only after every chip has sent its phase k
    rows), so two gather buffers and global semaphores alternated by phase never take data a chip still reads."""

    PHASES = 2
    CB_SCRATCH = (14, 15)  # forward, backward sender scratch on a transport core
    TRANSPORT_CORES = ((8, 5), (9, 5))  # one per fabric link; next to ReduceExpand's core row
    FORWARD_RANGE = 4

    def __init__(self, plan, reduce):
        device = plan.mesh_device
        self.device = device
        self.chips = plan.tp
        assert tuple(device.shape) == (1, self.chips), tuple(device.shape)
        self.cores = list(reduce.cores)
        self.tiles_per_core = reduce.tiles_per_core
        self.compact_tiles = reduce.compact_tiles
        assert self.tiles_per_core in (2, 4) and len(self.cores) * self.tiles_per_core == self.compact_tiles
        self.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        tile_bytes = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat16)
        self.phase_tiles = self.tiles_per_core * self.chips
        self.phase_bytes = self.phase_tiles * tile_bytes
        rows = self.PHASES * self.phase_tiles * TILE
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.core_ranges, [rows, TILE], ttnn.ShardOrientation.ROW_MAJOR),
        )
        # Allocated in place (no host copy, no staging buffer that would leave a hole above it in L1): every chip
        # writes a slot before the reduce reads it.
        self.gather = ttnn.allocate_tensor_on_device(
            ttnn.Shape([len(self.cores) * rows, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc
        )
        self.sems = [ttnn.create_global_semaphore(device, self.core_ranges, 0) for _ in range(self.PHASES)]
        self.transports = [ttnn.CoreCoord(*c) for c in self.TRANSPORT_CORES]
        assert len(self.cores) % len(self.transports) == 0
        self.nodes = [device.get_fabric_node_id(ttnn.MeshCoordinate(0, c)) for c in range(self.chips)]
        self.noc_cores = noc_coords(device, self.cores)
        self.phase = 0

    def next_phase(self):
        """The phase the next launch uses. Call ``advance`` once that launch is enqueued."""
        return self.phase

    def advance(self):
        """A launch of ``next_phase`` was enqueued. A captured trace replays the phases it was captured with, so it
        must make an even number of launches (see ``TtMiMoModel.capture_decode_trace``): otherwise the last launch
        of one replay and the first of the next would share a gather buffer and semaphore."""
        self.phase = (self.phase + 1) % self.PHASES

    def sem_addr(self, phase):
        return ttnn.get_global_semaphore_address(self.sems[phase])

    def programs(self, kernels, cbs, semaphores, compact, phase, wait_count=0):
        """MeshProgramDescriptor: per chip, ``kernels`` / ``cbs`` / ``semaphores`` (the reduce, the same on every
        chip) plus, per fabric link, a transport core whose two senders multicast its share of ``compact``."""
        transport = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.transports])
        per_link = len(self.cores) // len(self.transports)
        page = ttnn.Tile([TILE, TILE]).get_tile_size(ttnn.bfloat16)
        scratch = []
        for index in self.CB_SCRATCH:
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index,
                data_format=ttnn.bfloat16,
                page_size=page,
                tile=ttnn.TileDescriptor(ttnn.Tile([TILE, TILE])),
            )
            scratch.append(
                ttnn.CBDescriptor(
                    total_size=per_link * self.tiles_per_core * page, core_ranges=transport, format_descriptors=[fmt]
                )
            )
        gather_addr = self.gather.buffer_address() + phase * self.phase_bytes
        directions = (
            (1, self.CB_SCRATCH[0], self.FORWARD_RANGE, 1, ttnn.WriterConfigDescriptor()),
            (-1, self.CB_SCRATCH[1], self.chips - 1 - self.FORWARD_RANGE, 0, ttnn.ReaderConfigDescriptor()),
        )
        mesh_program = ttnn.MeshProgramDescriptor()
        for chip in range(self.chips):
            senders = []
            for _, cb, rng, local, config in directions:
                rt = ttnn.RuntimeArgs()
                for link, t in enumerate(self.transports):
                    first = link * per_link
                    noc = []
                    for c in self.noc_cores[first : first + per_link]:
                        noc += [c.x, c.y]
                    rt[t.x][t.y] = [compact.buffer_address(), gather_addr, self.sem_addr(phase), chip, first] + noc
                senders.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KERNEL_DIR}/fabric_gather_send.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=transport,
                        compile_time_args=[cb, rng, per_link, self.tiles_per_core, self.chips, local, 1, 0, wait_count],
                        runtime_args=rt,
                        config=config,
                    )
                )
            # Semaphore 0 of the transport cores: raised by each producer core in the same launch (before the
            # fabric connections take their semaphore ids).
            sems = list(semaphores) + ([semaphore_descriptor(transport)] if wait_count else [])
            program = ttnn.ProgramDescriptor(kernels=list(kernels) + senders, cbs=list(cbs) + scratch, semaphores=sems)
            n = len(kernels)
            for d, (step, *_rest) in enumerate(directions):
                dst = self.nodes[(chip + step) % self.chips]
                for link, t in enumerate(self.transports):
                    args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=self.nodes[chip],
                        dst_fabric_node_id=dst,
                        link_idx=link,
                        program_descriptor=program,
                        worker_core=t,
                    )
                    program.kernels[n + d].runtime_args[t.x][t.y].extend(args)
            coord = ttnn.MeshCoordinate(0, chip)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        return mesh_program


def fabric_transport_noc(device):
    """NoC coordinates of FabricRowGather's transport cores (what a producer signals)."""
    return [device.worker_core_from_logical_core(ttnn.CoreCoord(*c)) for c in FabricRowGather.TRANSPORT_CORES]


class PendingCompact:
    """A producer of this chip's compact rows that has not launched yet: ``section`` (kernels, cbs, semaphores,
    tensors) writes ``rows``, and each of its ``signals`` cores then raises semaphore 0 of every transport core.
    ReduceExpand.call_fabric runs it in the same launch as the all-gather and the reduce."""

    def __init__(self, section, rows, signals):
        self.section, self.rows, self.signals = section, rows, signals


def fabric_row_gather(plan, reduce):
    """The row's shared FabricRowGather, created on first use and kept on ``plan``: its L1 buffer and global
    semaphores belong to ``plan.mesh_device`` and must not outlive it (a later device with the same ids gets its own).
    """
    if getattr(plan, "fabric_gather", None) is None:
        plan.fabric_gather = FabricRowGather(plan, reduce)
    return plan.fabric_gather
