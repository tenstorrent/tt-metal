# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The DFlash verify pass's attention for the sliding-window layers, at every context length (fused_decode/kernels/
row_attn_*.cpp): M query rows of one sequence, row r at position pos[r] over the paged K / V cache, with per-head
attention sinks and the sliding window; one core per (row, kv head), every chunk of the row's window held on the core
(no cross-core reduction: the attended range is at most max_chunks x 32 positions, which the window bounds). Fused
into the verify QKV program (dflash/verify.py): the reader reads the older chunks while QKV runs, Q and the block's
chunks once the QKV writers signal. Output: [1, M, heads, head_dim] BF16 tiles in L1 (the SDPA output layout; the
o_proj stream reads rows straight from it)."""

import struct

import torch

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

TILE = ttnn.TILE_SIZE


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class RowAttention:
    def __init__(self, mesh_device, rows, n_heads, n_kv, head_dim, block_size, max_chunks, cores_start=(0, 0)):
        """cores: the 8-wide block of rows x n_kv logical cores at cores_start."""
        assert head_dim == 64 and n_heads <= TILE and n_heads // n_kv == 8, "one 32-row tile of heads, 8 per kv head"
        self.mesh, self.M, self.nq, self.nkv, self.hd = mesh_device, rows, n_heads, n_kv, head_dim
        self.block, self.max_chunks = block_size, max_chunks
        x0, y0 = cores_start
        self.core_list = [ttnn.CoreCoord(x0 + (i % 8), y0 + i // 8) for i in range(rows * n_kv)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.core_list[0], self.core_list[-1])])
        # the output in L1 (the o_proj stream reads rows straight from it; a DRAM source is staged whole tiles first)
        self.out = ttnn.from_torch(
            torch.zeros(1, rows, n_heads, head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def parts(self, q, k_cache, v_cache, pos, page_table, sinks, scale, window, wait):
        """(kernels, cbs, io) of the op. q: [1, M, heads, 64] BF16 tiles (row r's heads in tile row r); pos: INT32 [M];
        page_table: INT32 [*, blocks] (row 0 used); sinks: the heads' [heads, 32] BF16 sink tile (column 0 = sink /
        scale); wait = (semaphore id, count): the QKV writers' signal that q and the block's K / V rows are written
        (the reader reads the older chunks first, Q and the block's chunks after the count)."""
        NK = self.max_chunks
        cb_q, cb_k, cb_v, cb_aux, cb_scr, cb_qs, cb_s, cb_m, cb_p, cb_l, cb_o, cb_out, cb_scr2 = range(13)
        kv_dtype = k_cache.dtype
        kv_tile = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[kv_dtype]
        pt_page = page_table.shape[-1] * 4
        pos_page = pos.shape[-1] * 4
        f32, bf16 = ttnn.float32, ttnn.bfloat16

        def cb(i, dtype, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=self.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page)],
            )

        cbs = [
            cb(cb_q, bf16, 2048, 2),
            cb(cb_k, kv_dtype, kv_tile, 2 * NK),
            cb(cb_v, kv_dtype, kv_tile, 2 * NK),
            cb(cb_aux, bf16, 2048, 7),
            cb(cb_scr, bf16, 256 + max(pt_page, 64), 1),
            cb(cb_qs, bf16, 2048, 2),
            cb(cb_s, f32, 4096, NK + 1),
            cb(cb_m, f32, 4096, 1),
            cb(cb_p, bf16, 2048, NK + 1),
            cb(cb_l, f32, 4096, 1),
            cb(cb_o, f32, 4096, 2),
            cb(cb_out, bf16, 2048, 2),
            cb(cb_scr2, bf16, 64, 1),
        ]
        reader_ct = [cb_q, cb_k, cb_v, cb_aux, cb_scr, self.nkv, self.block, window, kv_tile, pt_page, pos_page, NK]
        for t in (q, k_cache, v_cache, pos, page_table, sinks):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cb_out, cb_scr2, window, pos_page, NK]
        for t in (self.out, pos):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_q, cb_k, cb_v, cb_aux, cb_qs, cb_s, cb_m, cb_p, cb_l, cb_o, cb_out, _f32_bits(scale)]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.core_list):
            r, g = i // self.nkv, i % self.nkv
            reader_rt[c.x][c.y] = [
                q.buffer_address(),
                k_cache.buffer_address(),
                v_cache.buffer_address(),
                pos.buffer_address(),
                page_table.buffer_address(),
                sinks.buffer_address(),
                r,
                g,
            ]
            writer_rt[c.x][c.y] = [self.out.buffer_address(), pos.buffer_address(), r, g]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "row_attn_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=[("RA_WAIT_SEM", str(wait[0])), ("RA_WAIT_N", str(wait[1]))],
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "row_attn_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "row_attn_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        return kernels, cbs, [q, k_cache, v_cache, pos, page_table, sinks, self.out]

    def noc_cores(self):
        return [self.mesh.worker_core_from_logical_core(c) for c in self.core_list]
