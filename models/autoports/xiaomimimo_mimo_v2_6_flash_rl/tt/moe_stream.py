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

from .common import TILE

KERNEL_DIR = "models/autoports/xiaomimimo_mimo_v2_6_flash_rl/tt/kernels"


def dram_bank_cores(device):
    """One worker core per DRAM bank (NOC 0 optimal placement): core ``i`` streams bank ``i``."""
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


class GateUpStream:
    """Top-k routing, then silu(x @ gate) * (x @ up) * w for every decode pair, streamed bank-locally from the
    interleaved w_gate_up.

    Routing runs on the data-movement RISCs of every bank core (reader: the first half of the users, writer: the
    rest): from rows 0-3 of the fp32 sigmoid scores and biased scores, each user's top-k by exact fp32 order on the
    biased scores (ties to the lower expert id, sorted descending like ttnn.topk); the unbiased scores of the chosen
    experts are normalized (sum + 1e-20), scaled, rounded to bf16 and multiplied by the bf16 token mask, as the
    TTNN routing ops did. The bank-0 reader writes the uint16 expert-id stick the down sparse_matmul indexes with.

    The per-chip w_gate_up [1, E, K, 2 * I] has 2 * I / 32 = 2 * banks tile columns, so DRAM bank b holds gate
    column b and up column b of every expert, contiguous per expert as [kt][gate, up]. Core b streams bank b and
    writes row u of pair (u, k)'s h tile column b into ``out`` [1, pairs, 32, I] (other rows stay zero).
    """

    BLOCK_K = 16  # K tiles per block: 16 x (gate, up) x 576 B = 18 KB
    NUM_BUFFERS = 6
    IN_FLIGHT = 4
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
        self.num_pairs, self.topk = num_pairs, topk
        self.reader_users = (self.users + 1) // 2
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

        # Readers in the same core row get different VCs (as DeepSeek-B1's DRAM streaming matmul).
        self.vcs = []
        for idx, core in enumerate(self.cores):
            vc = idx & 0x3
            for j in range(idx):
                if self.cores[j].y == core.y and (j & 0x3) == (idx & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            self.vcs.append(vc)

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
            cb(self.cb_h, ttnn.bfloat16, tiny, 2),
            cb(self.cb_route, ttnn.float32, full, 1),
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
