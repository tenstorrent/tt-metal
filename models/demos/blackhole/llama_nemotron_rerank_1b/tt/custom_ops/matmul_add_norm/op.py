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

Grid and blocking follow minimal_matmul's layouts (``_mm_grid``, either orientation); one N block per core.
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
GATHER_KERNEL = os.path.join(_HERE, "kernels", "embed_gather.cpp")
TILE = 32
# embed_rmsnorm: whole embedding rows in flight per core (read transaction ids 1-8), and the
# semaphore of its start handshake (after matmul_semaphores' six and the exchange's).
EMBED_ROW_SLOTS = 8
EMBED_SEM = 7


def make_matmul_add_norm_constants(gamma: torch.Tensor, device):
    """gamma ``[W]`` -> (gamma rows replicated over 32 rows ``[1, 1, 32, W]``, 1/W tile, 1.0 tile), bf16."""
    width = int(gamma.numel())
    rep = gamma.to(torch.float32).reshape(1, 1, 1, width).expand(1, 1, TILE, width).contiguous()
    g = ttnn.from_torch(rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def const(val):
        t = torch.full((1, 1, TILE, TILE), float(val), dtype=torch.float32)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return g, const(1.0 / width), const(1.0)


def residual_layout(rows, width, grid, transposed=True):
    """(rows, width covered by the core grid, shard rows, shard width) in elements for a [rows, width]
    residual stream on ``grid``: rows split over x and columns over y (``transposed``, as
    minimal_matmul's M > N grid) or rows over y and columns over x; whole tiles per core."""
    gx, gy = int(grid.x), int(grid.y)
    n_m, n_n = (gx, gy) if transposed else (gy, gx)
    mt, nt = rows // TILE, width // TILE
    mpc, npc = -(-mt // n_m), -(-nt // n_n)
    return n_m * mpc * TILE, n_n * npc * TILE, mpc * TILE, npc * TILE


def residual_memory_config(rows, width, grid, transposed=True):
    """L1 block sharding of the residual: shard (r, c) on the core holding row block r and column
    block c, i.e. column-major orientation for the transposed grid, row-major otherwise."""
    _, _, sr, sc = residual_layout(rows, width, grid, transposed)
    gx, gy = int(grid.x), int(grid.y)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            [sr, sc],
            ttnn.ShardOrientation.COL_MAJOR if transposed else ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def to_resident(x: ttnn.Tensor, grid, transposed=True) -> ttnn.Tensor:
    """[1, 1, rows, width] (TILE, interleaved) -> the L1-resident residual stream. The shards are uneven
    (the last row / column of cores holds fewer real tiles); no padded copy is made."""
    rows, width = int(x.padded_shape[-2]), int(x.padded_shape[-1])
    return ttnn.to_memory_config(x, residual_memory_config(rows, width, grid, transposed))


def from_resident(res: ttnn.Tensor, rows, width) -> ttnn.Tensor:
    """The L1-resident residual stream -> [1, 1, rows, width] interleaved in DRAM (tests, debugging)."""
    return ttnn.to_memory_config(res, ttnn.DRAM_MEMORY_CONFIG)


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
    in0_depth=2,
    in1_depth=2,
    full_sync=False,
    transposed=True,
    norm_only_shape=None,
    acc_cb_fp32=None,
    embed=None,
    in0_mcast=False,
):
    """x ``[1, 1, M, K]``, w ``[K, N]``, residual: the resident stream from ``to_resident`` for [M, N]
    (updated in place to residual + x @ w). Returns y = rmsnorm(residual) * gamma ``[1, 1, M, N]``.

    ``block`` = (M_block, K_block, N_block, subblock_h, subblock_w) in tiles; N_block must hold a
    core's whole N slice (ceil(N / grid_y) tiles; exactly that avoids computing padding), and M_block must divide
    a core's rows (ceil(M / grid_x) tiles). ``fp32_acc=False`` accumulates in bf16 DST (subblocks up
    to 8 tiles instead of 4). ``y_fidelity`` (e.g. "HiFi2") sets the fidelity of the y multiplies
    (default HiFi4). ``skip_gamma``: y = rmsnorm(h) only (gamma folded into the next matmul).
    ``norm_only_shape`` = (rows, width) with x = w = None: no matmul, y = rmsnorm(residual) (use
    ``resident_rmsnorm``). ``embed`` = (token ids [B, S] uint32 row-major, embedding table [vocab, width]
    bf16 row-major), norm-only: the residual is first filled with the tokens' embeddings (``embed_rmsnorm``)."""
    gamma_tiles, inv_w_tile, one_tile = consts
    norm_only = norm_only_shape is not None
    device = residual.device()
    if norm_only:
        M, K, N = norm_only_shape[0] // TILE, 1, norm_only_shape[1] // TILE
    else:
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
    g = mg.Grid(gx, gy, M, K, N, mb, kb, nb, transposed=transposed)
    if g.N_blocks_per_core != 1:
        raise ValueError(f"N block {nb} must hold the core's {g.N_per_core} N tiles")
    if g.M_per_core % mb:
        raise ValueError(f"M block {mb} must divide the core's {g.M_per_core} rows")
    if g.n_n > 16:
        raise ValueError("the exchange gives each core of a column one row of a 16-wide face: grid y <= 16")
    if g.N_per_core * (g.n_n - 1) >= N:
        raise ValueError(f"{g.n_n} cores along N leave the last one without real columns (N {N} tiles)")
    if mb % sh or nb % sw or sh * sw > (4 if fp32_acc else 8) * (2 if full_sync else 1):
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
    matmul_cbs = (
        []
        if norm_only
        else [
            mg.cb(0, mb * kb * in0_depth, x.dtype, all_cores),
            mg.cb(1, kb * nb * in1_depth, w.dtype, all_cores),
            mg.cb(
                3,
                out_tiles,
                ttnn.float32 if (fp32_acc if acc_cb_fp32 is None else acc_cb_fp32) else ttnn.bfloat16,
                all_cores,
            ),
        ]
    )
    cbs = (
        matmul_cbs
        + [
            ttnn.cb_descriptor_from_sharded_tensor(4, residual),
            mg.cb(6, 1, ttnn.bfloat16, all_cores),
            mg.cb(8, 2, ttnn.bfloat16, all_cores),
            mg.cb(18, mb, ttnn.float32, all_cores),
            mg.cb(9, mb, ttnn.float32, all_cores),
            mg.cb(10, 2 * mb, ttnn.float32, all_cores),
            mg.cb(11, 4, ttnn.float32, all_cores),
            mg.cb(12, 1, ttnn.bfloat16, all_cores),
            mg.cb(17, 2 * out_tiles, y_dtype, all_cores),
        ]
        + ([] if skip_gamma else [mg.cb(5, nb, ttnn.bfloat16, all_cores)])
    )
    if embed is not None:
        # 13: this core's token ids (32 per tile-row); 14: the whole shard row-major (the gathered
        # embedding rows); 15: whole embedding rows in flight (EMBED_ROW_SLOTS); 16: per-tile-row arrival counters.
        cbs += [
            mg.cb(13, g.M_per_core, ttnn.uint32, all_cores, page_size=TILE * 4),
            mg.cb(14, res_shard_tiles, ttnn.bfloat16, all_cores),
            mg.cb(15, EMBED_ROW_SLOTS, ttnn.bfloat16, all_cores, page_size=N * TILE * 2),
            mg.cb(16, g.M_per_core, ttnn.uint32, all_cores, page_size=16),
        ]
    semaphores = mg.matmul_semaphores(all_cores, extra=2 if embed is not None else 1)
    xchg_sem = 6

    defines = list(defines) + ([("Y_FIDELITY", y_fidelity)] if y_fidelity else [])
    defines += [("NO_GAMMA", "1")] if skip_gamma else []
    defines += [("NORM_ONLY", "1")] if norm_only else []
    if embed is not None:
        defines += [("EMBED", "1")]
    w_slot = gamma_tiles if norm_only else w  # the writer's in1 accessor slot (unused without a matmul)
    zoned = any(d == "ZONES" for d, _ in defines)
    in0_kernel = os.path.join(_HERE, "kernels", "dm_in0_zoned.cpp") if zoned else mg.IN0_KERNEL
    if norm_only:
        kernels = []
    elif in0_mcast:
        kernels = mg.in0_mcast_kernels(device, g, x)
    else:
        kernels = mg.in0_kernels(device, g, x, y, kernel=in0_kernel)

    w_bytes = TILE_BYTES[w_slot.dtype]
    writer_prefix = [g.M, g.padded_M, g.K, g.K, g.N, g.n_n * g.N_per_core, mb, kb, nb, g.M_blocks_per_core]
    writer_prefix += [w_bytes, res_shard_tiles, TILE_BYTES[y_dtype], 0, 3, 4, 5]
    tail = [g.n_n, xchg_sem] + mg.accessor_args(w_slot, gamma_tiles, inv_w_tile, one_tile, y)
    send_ct = writer_prefix + [1] + tail
    recv_ct = writer_prefix + [0] + tail
    eps_bits = struct.unpack("<I", struct.pack("<f", eps))[0]
    compute_ct = [g.K_blocks, mb, kb, nb, g.M_blocks_per_core, sh, sw, eps_bits, g.N_per_core, res_shard_tiles]

    send_rt, recv_rt, compute_rt, gather_rt = [], [], [], []
    seq = int(embed[0].padded_shape[-1]) if embed is not None else 0
    addrs = [
        w_slot.buffer_address(),
        gamma_tiles.buffer_address(),
        inv_w_tile.buffer_address(),
        one_tile.buffer_address(),
        y.buffer_address(),
    ]

    def core_at(m_idx, n_idx):
        return (m_idx, n_idx) if transposed else (n_idx, m_idx)

    for mi in range(g.n_m):
        # Peers: the cores holding the other columns of these rows, in N-index order.
        peers = []
        for ni in range(g.n_n):
            p = mg.physical(device, *core_at(mi, ni))
            peers += [p.x, p.y]
        for ni in range(g.n_n):
            xx, yy = core_at(mi, ni)
            m0, m1 = g.m_range(xx, yy)
            n0, n1 = g.n_range(xx, yy)
            rt = addrs + mg.in1_chain_args(device, g, xx, yy)
            rt += [m0, m1, n0, n1, g.defer_write_k_block(xx, yy), ni]
            rt += mg.in1_mcast_args(device, g, yy)
            (send_rt if mi == 0 else recv_rt).append(((xx, yy), rt + peers))
            compute_rt.append(((xx, yy), [m0, m1, n0, n1, min(n1, N) - n0]))
            if embed is not None:
                gather_rt.append(
                    ((xx, yy), [embed[0].buffer_address(), embed[1].buffer_address(), seq, m0, m1, ni] + peers)
                )

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

    kernels.append(dm(send_ct, send_rt, g.in1_sender_cores()))
    if g.has_in1_receivers():
        kernels.append(dm(recv_ct, recv_rt, g.in1_receiver_cores()))
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
                dst_full_sync_en=full_sync,
                bfp8_pack_precise=True,
            ),
        )
    )
    if embed is not None:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=GATHER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=all_cores,
                compile_time_args=[M, N, nb, g.n_n, EMBED_ROW_SLOTS, EMBED_SEM] + mg.accessor_args(*embed),
                runtime_args=gather_rt,
                defines=[],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=mg.IN0_NOC),
                compiler_include_paths=[mg.KERNEL_ROOT],
            )
        )
    program = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    ins = (
        [residual, gamma_tiles, inv_w_tile, one_tile] + list(embed or ())
        if norm_only
        else [x, w, residual, gamma_tiles, inv_w_tile, one_tile]
    )
    ttnn.generic_op(ins + [y], program)
    return y


def resident_rmsnorm(residual, consts, *, rows, width, eps, block_m=4, grid=None, **kwargs):
    """y = rmsnorm(residual) * gamma for the L1-resident residual stream (no matmul), with the same
    exchange of partial sums as matmul_add_norm. Used for the first layer's attention norm."""
    gxy = grid or residual.device().compute_with_storage_grid_size()
    _, _, sr, sc = residual_layout(rows, width, gxy, kwargs.get("transposed", True))
    npc, mpc = sc // TILE, sr // TILE
    block_m = max(d for d in range(1, block_m + 1) if mpc % d == 0)
    return matmul_add_norm(
        None,
        None,
        residual,
        consts,
        eps=eps,
        block=(block_m, 1, npc, 1, 1),
        grid=gxy,
        norm_only_shape=(rows, width),
        **kwargs,
    )


def embed_rmsnorm(tokens, table, consts, *, rows, width, eps, block_m=4, grid=None, **kwargs):
    """tokens ``[B, S]`` (uint32, row-major), table ``[vocab, width]`` (bf16, row-major) -> (residual, y):
    the L1-resident residual stream holding the tokens' embeddings, and y = rmsnorm(residual) * gamma.
    Replaces embedding + to_resident + resident_rmsnorm: each core gathers its own rows and columns
    of the embedding straight into its residual shard."""
    device = tokens.device()
    gxy = grid or device.compute_with_storage_grid_size()
    residual = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, width]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        residual_memory_config(rows, width, gxy, kwargs.get("transposed", True)),
    )
    y = resident_rmsnorm(
        residual, consts, rows=rows, width=width, eps=eps, block_m=block_m, grid=gxy, embed=(tokens, table), **kwargs
    )
    return residual, y
