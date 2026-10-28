# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul + residual add + RMSNorm in one program: ``h = residual + x @ W`` (in place), ``y = rmsnorm(h) * gamma``.

Replaces WO (or FF2) followed by fused_add_rmsnorm. Separately, the matmul writes x @ W to DRAM,
and the add-norm reads it back with the residual and writes the new residual and y: per element
1 + 2 + 2 + 2 + 1 bytes of DRAM traffic and a second launch whose time is all data movement.

Here the residual stream lives in L1 for the whole forward, block-sharded exactly like this op's
core grid (``residual_memory_config``): core (x, y) holds rows block x and columns block y. The
matmul's output block never leaves L1, the epilogue adds the core's residual slice in place, and
only y goes to DRAM. Keeping the residual traffic off the NOC matters beyond its bytes: with the
residual read from and written to DRAM in the epilogue, the matmul itself ran ~40 % slower
(WO 16384 x 2048 x 2048: 1110 us vs 805 us without the residual traffic).

RMSNorm's row statistic needs all columns: each core computes the sum of squares over its slice,
sends those 32 values per tile-row to the other cores holding the same rows (one NOC write per
peer per block, then a semaphore increment), and normalizes once all of them have arrived.

Grid and blocking follow minimal_matmul's transposed layout (``_mm_grid``); one N block per core.
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


def residual_layout(rows, width, grid):
    """(padded rows, padded width, shard rows, shard width) in elements for a [rows, width] residual
    stream on ``grid``: rows split over x, columns over y, each padded to whole tiles per core."""
    gx, gy = int(grid.x), int(grid.y)
    mt, nt = rows // TILE, width // TILE
    mpc, npc = -(-mt // gx), -(-nt // gy)
    return gx * mpc * TILE, gy * npc * TILE, mpc * TILE, npc * TILE


def residual_memory_config(rows, width, grid):
    """L1 block sharding of the padded residual: shard (r, c) on core (x = r, y = c), i.e. column-major."""
    _, _, sr, sc = residual_layout(rows, width, grid)
    gx, gy = int(grid.x), int(grid.y)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            [sr, sc],
            ttnn.ShardOrientation.COL_MAJOR,
        ),
    )


def to_resident(x: ttnn.Tensor, grid) -> ttnn.Tensor:
    """[1, 1, rows, width] (TILE, interleaved) -> the padded, L1-resident residual stream."""
    rows, width = int(x.padded_shape[-2]), int(x.padded_shape[-1])
    pr, pw, _, _ = residual_layout(rows, width, grid)
    if (pr, pw) != (rows, width):
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, pr - rows), (0, pw - width)], 0.0)
    return ttnn.to_memory_config(x, residual_memory_config(rows, width, grid))


def from_resident(res: ttnn.Tensor, rows, width) -> ttnn.Tensor:
    """The L1-resident residual stream -> [1, 1, rows, width] interleaved in DRAM (tests, debugging)."""
    t = ttnn.to_memory_config(res, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.slice(t, [0, 0, 0, 0], [1, 1, rows, width])


def matmul_add_norm(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    residual: ttnn.Tensor,
    consts,
    *,
    eps: float,
    block=(4, 8, 8, 1, 4),
    grid=None,
    fidelity=ttnn.MathFidelity.HiFi2,
    y_dtype=ttnn.bfloat8_b,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    defines=(),
    fp32_acc=True,
    y_fidelity=None,
    skip_gamma=False,
):
    """x ``[1, 1, M, K]``, w ``[K, N]``, residual: the resident stream from ``to_resident`` for [M, N]
    (updated in place to residual + x @ w). Returns y = rmsnorm(residual) * gamma ``[1, 1, M, N]``.

    ``block`` = (M_block, K_block, N_block, subblock_h, subblock_w) in tiles; N_block must hold a
    core's whole N slice (ceil(N / grid_y) tiles; exactly that avoids computing padding), and M_block must divide
    a core's rows (ceil(M / grid_x) tiles). ``fp32_acc=False`` accumulates in bf16 DST (subblocks up
    to 8 tiles instead of 4). ``y_fidelity`` (e.g. "HiFi2") sets the fidelity of the y multiplies
    (default HiFi4). ``skip_gamma``: y = rmsnorm(h) only (gamma folded into the next matmul)."""
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
    if g.N_blocks_per_core != 1:
        raise ValueError(f"N block {nb} must hold the core's {g.N_per_core} N tiles")
    if g.M_per_core % mb:
        raise ValueError(f"M block {mb} must divide the core's {g.M_per_core} rows")
    if gy > 16:
        raise ValueError("the exchange gives each core of a column one row of a 16-wide face: grid y <= 16")
    if g.N_per_core * (gy - 1) >= N:
        raise ValueError(f"grid y {gy} leaves the last core row without real columns (N {N} tiles)")
    if mb % sh or nb % sw or sh * sw > (4 if fp32_acc else 8):
        raise ValueError(f"bad subblock {sh}x{sw} for block {mb}x{nb}")
    spec = residual.memory_config().shard_spec
    if spec is None or list(spec.shape) != [g.M_per_core * TILE, g.N_per_core * TILE]:
        raise ValueError("residual must be the resident stream from to_resident() on this grid")

    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, M * TILE, N * TILE]), y_dtype, ttnn.TILE_LAYOUT, device, memory_config
    )

    all_cores = g.all_cores()
    out_tiles = mb * nb
    res_shard_tiles = g.M_per_core * g.N_per_core
    cbs = [
        mg.cb(0, mb * kb * 2, x.dtype, all_cores),
        mg.cb(1, kb * nb * 2, w.dtype, all_cores),
        mg.cb(3, out_tiles, ttnn.float32 if fp32_acc else ttnn.bfloat16, all_cores),
        ttnn.cb_descriptor_from_sharded_tensor(4, residual),
        mg.cb(6, 1, ttnn.bfloat16, all_cores),
        mg.cb(8, nb, ttnn.bfloat16, all_cores),
        mg.cb(9, mb, ttnn.float32, all_cores),
        mg.cb(10, 2 * mb, ttnn.float32, all_cores),
        mg.cb(11, 1, ttnn.float32, all_cores),
        mg.cb(12, 1, ttnn.bfloat16, all_cores),
        mg.cb(17, 2 * out_tiles, y_dtype, all_cores),
    ] + ([] if skip_gamma else [mg.cb(5, nb, ttnn.bfloat16, all_cores)])
    semaphores = mg.matmul_semaphores(all_cores, extra=1)
    xchg_sem = 6

    defines = list(defines) + ([("Y_FIDELITY", y_fidelity)] if y_fidelity else [])
    defines += [("NO_GAMMA", "1")] if skip_gamma else []
    zoned = any(d == "ZONES" for d, _ in defines)
    in0_kernel = os.path.join(_HERE, "kernels", "dm_in0_zoned.cpp") if zoned else mg.IN0_KERNEL
    kernels = mg.in0_kernels(device, g, x, y, kernel=in0_kernel)

    w_bytes = TILE_BYTES[w.dtype]
    writer_prefix = [g.M, g.padded_M, g.K, g.K, g.N, gy * g.N_per_core, mb, kb, nb, g.M_blocks_per_core]
    writer_prefix += [w_bytes, res_shard_tiles, TILE_BYTES[y_dtype], 0, 3, 4, 5]
    tail = [gy, xchg_sem] + mg.accessor_args(w, gamma_tiles, inv_w_tile, one_tile, y)
    send_ct = writer_prefix + [1] + tail
    recv_ct = writer_prefix + [0] + tail
    eps_bits = struct.unpack("<I", struct.pack("<f", eps))[0]
    compute_ct = [g.K_blocks, mb, kb, nb, g.M_blocks_per_core, sh, sw, eps_bits, g.N_per_core, res_shard_tiles]

    send_rt, recv_rt, compute_rt = [], [], []
    addrs = [
        w.buffer_address(),
        gamma_tiles.buffer_address(),
        inv_w_tile.buffer_address(),
        one_tile.buffer_address(),
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
                fp32_dest_acc_en=fp32_acc,
                dst_full_sync_en=False,
                bfp8_pack_precise=True,
            ),
        )
    )
    program = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    ttnn.generic_op([x, w, residual, gamma_tiles, inv_w_tile, one_tile, y], program)
    return y
