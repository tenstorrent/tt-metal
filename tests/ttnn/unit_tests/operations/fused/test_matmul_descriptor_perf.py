# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end dispatch throughput test for the matmul descriptor factories.

For each factory: 3 representative shape variants, measured by running N_ITERS
matmul calls in a tight loop followed by a single device sync. Reports µs/iter
and TFLOPs. Run with -s to see the logger output.

Factory coverage:
  MatmulMultiCoreProgramFactory               – auto-selected for small/irregular shapes
  MatmulMultiCoreReuseOptimizedProgramFactory – MatmulMultiCoreReuseProgramConfig
  MatmulMultiCoreReuseMcast2DProgramFactory   – MatmulMultiCoreReuseMultiCastProgramConfig
  MatmulMultiCoreReuseMcast1DProgramFactory   – MatmulMultiCoreReuseMultiCast1DProgramConfig
  MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory    – MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
  MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory    – MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig
"""

import time
import pytest
import torch
import ttnn
from loguru import logger

N_WARMUP = 5
N_ITERS = 100

# Subblock choices ordered largest to smallest (h*w), matching the matmul op's preference.
_SUBBLOCK_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),
    (7, 1),
    (1, 7),
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),
    (5, 1),
    (1, 5),
    (2, 2),
    (4, 1),
    (1, 4),
    (3, 1),
    (1, 3),
    (2, 1),
    (1, 2),
    (1, 1),
]


def _subblock(block_h, block_w):
    for sh, sw in _SUBBLOCK_CHOICES:
        if block_h % sh == 0 and block_w % sw == 0:
            return sh, sw
    return 1, 1


def _subblock_sharded_out(per_core_M, per_core_N):
    """Subblock when output is sharded (or MatmulMultiCoreReuseProgramConfig):
    requires sw==per_core_N OR sh==1."""
    for sh, sw in _SUBBLOCK_CHOICES:
        if per_core_M % sh == 0 and per_core_N % sw == 0:
            if sw == per_core_N or sh == 1:
                return sh, sw
    return 1, 1


def _ckconfig(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _measure(device, in0_t, in1_t, m, k, n, label, **kwargs):
    for _ in range(N_WARMUP):
        ttnn.matmul(in0_t, in1_t, **kwargs)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        ttnn.matmul(in0_t, in1_t, **kwargs)
    ttnn.synchronize_device(device)

    elapsed = time.perf_counter() - t0
    us = elapsed / N_ITERS * 1e6
    tflops = 2 * m * k * n / 1e12 / (elapsed / N_ITERS)
    logger.info(f"[{label}]  {m}x{k}x{n}: {us:.1f} µs/iter  {tflops:.3f} TFLOPs")


# --------------------------------------------------------------------------
# MatmulMultiCoreProgramFactory
#   Auto-selected (no explicit Python config) for small/fallback shapes.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant,m,k,n",
    [
        ("small_square", 128, 128, 128),
        ("tall_m", 512, 64, 64),
        ("wide_n", 64, 64, 512),
    ],
)
def test_factory_multicore(device, variant, m, k, n):
    """MatmulMultiCoreProgramFactory: auto-selected fallback, no explicit config."""
    in0 = torch.randn(1, 1, m, k).bfloat16()
    in1 = torch.randn(1, 1, k, n).bfloat16()
    in0_t = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    in1_t = ttnn.from_torch(
        in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _measure(
        device,
        in0_t,
        in1_t,
        m,
        k,
        n,
        variant,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_ckconfig(device),
    )


# --------------------------------------------------------------------------
# MatmulMultiCoreReuseOptimizedProgramFactory
#   Triggered by: MatmulMultiCoreReuseProgramConfig
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant,b,m,k,n,grid",
    [
        ("batched_b2_m512", 2, 512, 512, 1024, (4, 4)),
        ("large_m_m4096", 1, 4096, 512, 1024, (8, 4)),
        ("large_k_k2048", 1, 1024, 2048, 1024, (4, 4)),
    ],
)
def test_factory_optimized(device, variant, b, m, k, n, grid):
    """MatmulMultiCoreReuseOptimizedProgramFactory: batched BMM variants."""
    gx, gy = grid
    num_cores = gx * gy
    # N is NOT split across cores (N == per_core_N assertion in factory).
    # Only M (including batch) is distributed across cores.
    per_core_M = b * m // 32 // num_cores
    per_core_N = n // 32
    in0_block_w = 2  # Small K-block to keep L1 usage low
    sh, sw = _subblock_sharded_out(per_core_M, per_core_N)

    in0 = torch.randn(b, 1, m, k).bfloat16()
    in1 = torch.randn(b, 1, k, n).bfloat16()
    in0_t = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    in1_t = ttnn.from_torch(
        in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    prog = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )
    _measure(
        device,
        in0_t,
        in1_t,
        b * m,
        k,
        n,
        variant,
        program_config=prog,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_ckconfig(device),
    )


# --------------------------------------------------------------------------
# MatmulMultiCoreReuseMcast2DProgramFactory
#   Triggered by: MatmulMultiCoreReuseMultiCastProgramConfig
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant,m,k,n,transpose_mcast,in0_sharded",
    [
        ("dram_in0", 1024, 1024, 1024, False, False),
        ("block_sharded_in0", 1024, 1024, 1024, False, True),
        ("transposed_mcast", 1024, 1024, 1024, True, True),
    ],
)
def test_factory_mcast_2d(device, variant, m, k, n, transpose_mcast, in0_sharded):
    """MatmulMultiCoreReuseMcast2DProgramFactory: 2D multicast (prefill-style)."""
    grid = (8, 8)
    gx, gy = grid
    in0_block_w = k // 32 // gx
    per_core_M = m // 32 // gy
    per_core_N = n // 32 // gx
    # When output is sharded, constraint: out_subblock_w==per_core_N OR out_subblock_h==1
    sh, sw = _subblock_sharded_out(per_core_M, per_core_N) if in0_sharded else _subblock(per_core_M, per_core_N)

    in0 = torch.randn(1, 1, m, k).bfloat16()
    in1 = torch.randn(1, 1, k, n).bfloat16()

    if in0_sharded:
        in0_mem = ttnn.create_sharded_memory_config(
            (1, 1, m, k),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR if transpose_mcast else ttnn.ShardOrientation.ROW_MAJOR,
        )
        out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    else:
        in0_mem = ttnn.DRAM_MEMORY_CONFIG
        out_mem = ttnn.DRAM_MEMORY_CONFIG

    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem)
    in1_t = ttnn.from_torch(
        in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    prog = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=transpose_mcast,
        fused_activation=None,
        fuse_batch=True,
    )
    _measure(
        device,
        in0_t,
        in1_t,
        m,
        k,
        n,
        variant,
        program_config=prog,
        memory_config=out_mem,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_ckconfig(device),
    )


# --------------------------------------------------------------------------
# MatmulMultiCoreReuseMcast1DProgramFactory
#   Triggered by: MatmulMultiCoreReuseMultiCast1DProgramConfig
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant,m,k,n,mcast_in0",
    [
        ("mcast_in0_m64_k4096_n4096", 64, 4096, 4096, True),
        ("mcast_in0_m32_k8192_n2048", 32, 8192, 2048, True),
        ("mcast_in1_m2048_k4096_n128", 2048, 4096, 128, False),
    ],
)
def test_factory_mcast_1d(device, variant, m, k, n, mcast_in0):
    """MatmulMultiCoreReuseMcast1DProgramFactory: 1D multicast (decode/prefill)."""
    grid = (8, 8)
    gx, gy = grid
    num_cores = gx * gy

    if mcast_in0:
        # in0 WIDTH-sharded across all cores; each core handles a K-slice and a N-slice.
        in0_block_w = k // num_cores // 32
        per_core_M = m // 32
        per_core_N = n // num_cores // 32
        in0_mem = ttnn.create_sharded_memory_config(
            (1, 1, m, k),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    else:
        # in0 HEIGHT-sharded; each core handles a M-slice; full N on each core.
        in0_block_w = k // 32 // 2
        per_core_M = m // 32 // num_cores
        per_core_N = n // 32
        in0_mem = ttnn.create_sharded_memory_config(
            (1, 1, m, k),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    # Output is always sharded for 1D mcast; constraint: sw==per_core_N OR sh==1
    sh, sw = _subblock_sharded_out(per_core_M, per_core_N)

    in0 = torch.randn(1, 1, m, k).bfloat16()
    in1 = torch.randn(1, 1, k, n).bfloat16()
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem)
    in1_t = ttnn.from_torch(
        in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=mcast_in0,
    )
    _measure(
        device,
        in0_t,
        in1_t,
        m,
        k,
        n,
        variant,
        program_config=prog,
        memory_config=out_mem,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_ckconfig(device),
    )


# --------------------------------------------------------------------------
# MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory
#   Triggered by: MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
#   in0: L1 WIDTH-sharded; in1: DRAM WIDTH-sharded across 12 banks.
# --------------------------------------------------------------------------


def _dram_sharded_tensors(device, m, k, n_padded, num_in0_cores, num_dram_banks):
    gx = num_in0_cores
    in0_shard_shape = [m, k // gx]
    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=1, x=gx),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, [k, n_padded // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)

    in0 = torch.randn(1, 1, m, k).bfloat16()
    in1 = torch.randn(1, 1, k, n_padded).bfloat16()
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_mem)
    return in0_t, in1_t


@pytest.mark.parametrize(
    "variant,m,k,n_padded,num_in0_cores",
    [
        # n_padded must be divisible by 12 * 32 = 384; num_in0_cores must divide k/32.
        # M must == 1 (m must == 32); DRAM-sharded factory does not support larger M.
        ("mlp_k4096_n1536", 32, 4096, 1536, 8),
        ("mlp_k8192_n1536", 32, 8192, 1536, 8),
        ("mlp_k2048_n3072", 32, 2048, 3072, 8),
    ],
)
def test_factory_dram_sharded(device, variant, m, k, n_padded, num_in0_cores):
    """MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory: DRAM WIDTH-sharded weights."""
    num_dram_banks = 12
    if device.dram_grid_size().x < num_dram_banks:
        pytest.skip(f"Need {num_dram_banks} DRAM banks, device has {device.dram_grid_size().x}")

    num_out_cores = num_in0_cores
    in0_block_w = k // num_in0_cores // 32
    per_core_M = m // 32
    per_core_N = n_padded // num_out_cores // 32

    in0_t, in1_t = _dram_sharded_tensors(device, m, k, n_padded, num_in0_cores, num_dram_banks)
    prog = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fused_activation=None,
    )
    _measure(
        device,
        in0_t,
        in1_t,
        m,
        k,
        n_padded,
        variant,
        program_config=prog,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_ckconfig(device),
    )


# --------------------------------------------------------------------------
# MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory
#   Triggered by: MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig
#   in0: L1 HEIGHT-sharded by batch; in1: DRAM HEIGHT-sharded by batch.
# --------------------------------------------------------------------------


def _batched_dram_sharded_tensors(device, batch, m, k, n, num_dram_banks, optimal_worker_cores):
    # Pad batch to be divisible by num_dram_banks.
    batch_padded = ((batch + num_dram_banks - 1) // num_dram_banks) * num_dram_banks
    batches_per_bank = batch_padded // num_dram_banks

    in0 = torch.zeros(1, batch_padded, m, k).bfloat16()
    in0[:, :batch] = torch.randn(1, batch, m, k).bfloat16()
    in1 = torch.zeros(1, batch_padded, k, n).bfloat16()
    in1[:, :batch] = torch.randn(1, batch, k, n).bfloat16()

    in0_shard_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in optimal_worker_cores]
    )
    dram_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))})
    out_shard_grid = in0_shard_grid

    in0_shard_spec = ttnn.ShardSpec(in0_shard_grid, [batches_per_bank * m, k], ttnn.ShardOrientation.ROW_MAJOR)
    in0_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec)

    in1_shard_spec = ttnn.ShardSpec(dram_shard_grid, [batches_per_bank * k, n], ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)

    out_shard_spec = ttnn.ShardSpec(out_shard_grid, [batches_per_bank * m, n], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((32, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mem,
    )
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, 32)),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_mem,
    )
    return in0_t, in1_t, out_mem, batch_padded


@pytest.mark.parametrize(
    "variant,batch,m,k,n",
    [
        ("b48_m32_k256_n256", 48, 32, 256, 256),
        ("b48_m32_k512_n128", 48, 32, 512, 128),
        ("b48_m32_k128_n512", 48, 32, 128, 512),
    ],
)
def test_factory_batched_dram_sharded(device, variant, batch, m, k, n):
    """MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory: batched DRAM HEIGHT-sharded."""
    optimal_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_dram_banks = len(optimal_worker_cores)

    in0_t, in1_t, out_mem, batch_padded = _batched_dram_sharded_tensors(
        device, batch, m, k, n, num_dram_banks, optimal_worker_cores
    )
    prog = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=k // 32,
        per_core_M=m // 32,
        per_core_N=n // 32,
        fused_activation=None,
    )
    _measure(
        device,
        in0_t,
        in1_t,
        batch_padded * m,
        k,
        n,
        variant,
        program_config=prog,
        memory_config=out_mem,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_ckconfig(device),
    )
