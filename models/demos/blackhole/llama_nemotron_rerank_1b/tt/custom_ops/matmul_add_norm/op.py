# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul + residual add + RMSNorm in one program: ``h = residual + x @ W``, ``y = rmsnorm(h) * gamma``.

Replaces WO (or FF2) followed by fused_add_rmsnorm. Separately, the matmul writes x @ W to DRAM
and the add-norm reads it back with the residual and writes h and y: the matmul output makes a
DRAM round trip and the add-norm is a second launch whose time is all data movement. Here the
matmul's output block never leaves L1; the epilogue reads the residual block, adds it, and
writes h and y directly.

The catch is RMSNorm's row statistic: a core holds only its N slice of each row (N is split
over the grid's y axis). Each core computes the sum of squares over its slice, sends those
32 values per tile-row to the other cores holding the same rows (one NOC write per peer per
block, then a semaphore increment), and normalizes once all of them have arrived.

Grid and blocking follow minimal_matmul's transposed layout (``_mm_grid``); one N block per
core, so each core keeps its whole slice of a block of rows in L1 for the second pass.
"""

import os
import struct

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops import _mm_grid as mg
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops._common import TILE_BYTES

_HERE = os.path.dirname(os.path.abspath(__file__))
WRITER_KERNEL = os.path.join(_HERE, "kernels", "writer_add_norm.cpp")
COMPUTE_KERNEL = os.path.join(_HERE, "kernels", "compute_add_norm.cpp")
TILE = 32


def make_matmul_add_norm_constants(gamma: torch.Tensor, device):
    """gamma ``[W]`` -> (gamma rows replicated over 32 rows ``[1, 1, 32, W]``, 1/W tile, 1.0 tile), bf16."""
    width = int(gamma.numel())
    rep = gamma.to(torch.float32).reshape(1, 1, 1, width).expand(1, 1, TILE, width).contiguous()
    g = ttnn.from_torch(rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def const(val):
        t = torch.full((1, 1, TILE, TILE), float(val), dtype=torch.float32)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return g, const(1.0 / width), const(1.0)


def matmul_add_norm(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    residual: ttnn.Tensor,
    consts,
    *,
    eps: float,
    block=(8, 8, 8, 1, 4),
    grid=None,
    fidelity=ttnn.MathFidelity.HiFi2,
    h_dtype=ttnn.bfloat16,
    y_dtype=ttnn.bfloat8_b,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    defines=(),
    in1_depth=2,
    in0_depth=2,
    y_memory_config=None,
):
    """x ``[1, 1, M, K]``, w ``[K, N]``, residual ``[1, 1, M, N]`` (TILE, interleaved) -> (h, y), each ``[1, 1, M, N]``.

    ``block`` = (M_block, K_block, N_block, subblock_h, subblock_w) in tiles; N_block must hold a
    core's whole N slice (ceil(N / grid_y) tiles) and be a multiple of 4."""
    gamma_tiles, inv_w_tile, one_tile = consts
    device = x.device()
    shape = list(x.padded_shape)
    M = 1
    for d in shape[:-1]:
        M *= int(d)
    M //= TILE
    K = shape[-1] // TILE
    N = int(w.padded_shape[-1]) // TILE
    gxy = grid or device.compute_with_storage_grid_size()
    gx, gy = int(gxy.x), int(gxy.y)
    mb, kb, nb, sh, sw = block
    g = mg.Grid(gx, gy, M, K, N, mb, kb, nb)
    if g.N_blocks_per_core != 1 or nb % 4:
        raise ValueError(f"N block {nb} must hold the core's {g.N_per_core} N tiles in one even block")
    if gy > 16:
        raise ValueError("the exchange gives each core of a column one row of a 16-wide face: grid y <= 16")
    if g.N_per_core * (gy - 1) >= N:
        raise ValueError(f"grid y {gy} leaves the last core row without real columns (N {N} tiles)")
    if mb % sh or nb % sw or sh * sw > 4:
        raise ValueError(f"bad subblock {sh}x{sw} for block {mb}x{nb}")

    out_shape = ttnn.Shape([1, 1, M * TILE, N * TILE])
    split = any(d == "SPLIT_RESIDUAL" for d, _ in defines)
    h_shape = ttnn.Shape([1, 1, gy * M * TILE, nb * TILE]) if split else out_shape
    h = ttnn.allocate_tensor_on_device(h_shape, h_dtype, ttnn.TILE_LAYOUT, device, memory_config)
    y = ttnn.allocate_tensor_on_device(out_shape, y_dtype, ttnn.TILE_LAYOUT, device, y_memory_config or memory_config)

    all_cores = g.all_cores()
    out_tiles = mb * nb
    cbs = [
        mg.cb(0, mb * kb * in0_depth, x.dtype, all_cores),
        mg.cb(1, kb * nb * in1_depth, w.dtype, all_cores),
        mg.cb(3, out_tiles, ttnn.float32, all_cores),
        mg.cb(4, out_tiles, residual.dtype, all_cores),
        mg.cb(5, nb, ttnn.bfloat16, all_cores),
        mg.cb(6, 1, ttnn.bfloat16, all_cores),
        mg.cb(8, nb, ttnn.bfloat16, all_cores),
        mg.cb(9, mb, ttnn.float32, all_cores),
        mg.cb(10, 2 * mb, ttnn.float32, all_cores),
        mg.cb(11, 1, ttnn.float32, all_cores),
        mg.cb(12, 1, ttnn.bfloat16, all_cores),
        mg.cb(16, out_tiles, h_dtype, all_cores),
        mg.cb(17, 2 * out_tiles, y_dtype, all_cores),
        mg.cb(18, out_tiles, ttnn.bfloat16, all_cores),
    ]
    semaphores = mg.matmul_semaphores(all_cores, extra=1)
    xchg_sem = 6

    kernels = mg.in0_kernels(device, g, x, h)

    w_bytes = TILE_BYTES[w.dtype]
    writer_prefix = [g.M, g.padded_M, g.K, g.K, g.N, gy * g.N_per_core, mb, kb, nb, g.M_blocks_per_core]
    writer_prefix += [w_bytes, TILE_BYTES[h_dtype], TILE_BYTES[y_dtype], TILE_BYTES[residual.dtype], 3, 4, 5]
    tail = [gy, xchg_sem] + mg.accessor_args(w, residual, gamma_tiles, inv_w_tile, one_tile, h, y)
    send_ct = writer_prefix + [1] + tail
    recv_ct = writer_prefix + [0] + tail
    eps_bits = struct.unpack("<I", struct.pack("<f", eps))[0]
    compute_ct = [g.K_blocks, mb, kb, nb, g.M_blocks_per_core, sh, sw, eps_bits]

    send_rt, recv_rt, compute_rt = [], [], []
    addrs = [
        w.buffer_address(),
        residual.buffer_address(),
        gamma_tiles.buffer_address(),
        inv_w_tile.buffer_address(),
        one_tile.buffer_address(),
        h.buffer_address(),
        y.buffer_address(),
    ]
    for xx in range(gx):
        peers = []
        for yy in range(gy):
            p = mg.physical(device, xx, yy)
            peers += [p.x, p.y]
        for yy in range(gy):
            m0, m1 = g.m_range(xx)
            n0, n1 = g.n_range(yy)
            rt = addrs + mg.in1_chain_args(device, g, xx, yy) + [m0, m1, n0, n1, g.defer_write_k_block(yy), yy]
            rt += mg.in1_mcast_args(device, g, yy)
            (send_rt if xx == 0 else recv_rt).append(((xx, yy), rt + peers))
            compute_rt.append(((xx, yy), [m0, m1, n0, n1, min(n1, N) - n0]))

    def dm(ct, rt, core_set):
        return ttnn.KernelDescriptor(
            kernel_source=WRITER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=ct,
            runtime_args=rt,
            defines=list(defines),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=mg.IN1_NOC),
            compiler_include_paths=[mg.KERNEL_ROOT],
        )

    kernels.append(dm(send_ct, send_rt, mg.cores(0, 0, 0, gy - 1)))
    if gx > 1:
        kernels.append(dm(recv_ct, recv_rt, mg.cores(1, 0, gx - 1, gy - 1)))
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=list(defines),
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
                bfp8_pack_precise=True,
            ),
        )
    )
    program = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    ttnn.generic_op([x, w, residual, gamma_tiles, inv_w_tile, one_tile, h, y], program)
    return h, y
