# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DRAM-sharded decode matmul with an in0 block spanning several activation shards.

in0_block_w may be a multiple of the activation shard width; the sender gathers the shards over the
NoC before the multicast. These cases cover shards_per_block 1, 2, 4 and 8, the three weight dtypes
(bfloat4_b makes the factory fall back to one shard per block on a narrow N), a K that is not a tile
multiple (the last block's sender pads the last K tile), and the fp32 / bf16 accumulation formats.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc


def _dram_sharded_matmul(device, k, n, in1_dtype, num_storage_cores, in0_block_w, fp32_acc, k_logical=None):
    k_logical = k_logical or k
    dram_x = device.dram_grid_size().x
    grid = (
        ttnn.CoreGrid(x=8, y=num_storage_cores // 8)
        if num_storage_cores % 8 == 0
        else ttnn.CoreGrid(x=num_storage_cores, y=1)
    )
    in0_mc = ttnn.create_sharded_memory_config(
        shape=(32, k // num_storage_cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    torch.manual_seed(0)
    in0 = ttnn.from_torch(
        torch.randn(1, 1, 32, k_logical),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mc,
    )
    padded_n = math.ceil(n / (32 * dram_x)) * 32 * dram_x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_x - 1, 0))})
    in1_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (k, padded_n // dram_x), ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1 = ttnn.from_torch(
        torch.randn(k_logical, n), dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_mc
    )
    out_mc = ttnn.create_sharded_memory_config(
        shape=(32, n // num_storage_cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=math.ceil(n / (32 * num_storage_cores)),
        fused_activation=None,
    )
    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi if in1_dtype == ttnn.bfloat4_b else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )
    out = ttnn.linear(
        in0,
        in1,
        program_config=program_config,
        memory_config=out_mc,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute,
    )
    # reference from the on-device (quantized) operands, so the only difference is the accumulation
    ref = ttnn.to_torch(in0).float().reshape(32, -1)[:, :k_logical] @ ttnn.to_torch(in1).float()[:k_logical]
    return ttnn.to_torch(out).float().reshape(32, n), ref


@pytest.mark.parametrize(
    "k, n, num_storage_cores, in0_block_w, shards_per_block",
    [
        (4096, 6144, 32, 4, 1),  # stock: one block per shard (Llama-3.1-8B QKV)
        (4096, 6144, 32, 8, 2),
        (4096, 6144, 32, 16, 4),
        (2048, 2048, 32, 16, 8),  # Llama-3.2-1B WO at the widest block
        (2304, 7168, 72, 8, 8),  # DeepSeek-V3 decode shape, 72 storage cores, 1 tile per shard
        (7168, 256, 56, 16, 4),  # narrow N: the in1 block is small
    ],
)
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("fp32_acc", [True, False])
def test_dram_sharded_multishard_block(
    device, k, n, num_storage_cores, in0_block_w, shards_per_block, in1_dtype, fp32_acc
):
    if num_storage_cores > device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y:
        pytest.skip("storage grid larger than the device")
    assert in0_block_w == shards_per_block * (k // 32 // num_storage_cores)
    tile_bytes = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[in1_dtype]
    in1_cb = 3 * math.ceil(n / 32 / device.dram_grid_size().x) * in0_block_w * tile_bytes
    if in1_cb > 900 * 1024:
        pytest.skip("in1 CB alone exceeds the L1 budget; the model-side width rules never pick this")
    out, ref = _dram_sharded_matmul(device, k, n, in1_dtype, num_storage_cores, in0_block_w, fp32_acc)
    passed, pcc = comp_pcc(ref, out, 0.999)
    assert passed, pcc
    # fp32 accumulation keeps every K tile: an error the size of a whole term means a shard was dropped
    max_err = (out - ref).abs().max().item()
    assert max_err < 0.05 * ref.abs().max().item(), max_err


@pytest.mark.parametrize("k_logical", [4088, 4072])
@pytest.mark.parametrize("in0_block_w, shards_per_block", [(4, 1), (8, 2), (16, 4)])
def test_dram_sharded_multishard_block_padded_k(device, k_logical, in0_block_w, shards_per_block):
    """K is not a tile multiple, so whoever sends the last block pads the last K tile in place."""
    out, ref = _dram_sharded_matmul(device, 4096, 4096, ttnn.bfloat8_b, 32, in0_block_w, True, k_logical=k_logical)
    passed, pcc = comp_pcc(ref, out, 0.999)
    assert passed, pcc


def _dram_width_sharded_2d_matmul(device, m, k, n, grid_x, in0_block_w):
    """2D-mcast matmul (MatmulMultiCoreReuseMultiCastProgramConfig) with in1 DRAM width-sharded and
    per_core_N narrower than one DRAM shard -- the configuration from issue #57732."""
    dram_x = device.dram_grid_size().x
    grid_size = (grid_x, 1)
    per_core_M = m // 32
    per_core_N = math.ceil(n / 32 / grid_x)
    out_subblock_w = math.gcd(per_core_N, 4)
    out_subblock_h = 1

    torch.manual_seed(0)
    in0 = ttnn.from_torch(torch.randn(1, 1, m, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    padded_n = math.ceil(n / (32 * dram_x)) * 32 * dram_x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_x - 1, 0))})
    in1_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (k, padded_n // dram_x), ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1 = ttnn.from_torch(
        torch.randn(k, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_mc
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )
    out = ttnn.matmul(in0, in1, program_config=program_config)
    ref = ttnn.to_torch(in0).float().reshape(m, k) @ ttnn.to_torch(in1).float()
    return ttnn.to_torch(out).float().reshape(m, n), ref


@pytest.mark.parametrize(
    "n, grid_x",
    [
        (6144, 9),  # per_core_N (22 tiles) < one DRAM shard (24 tiles): issue #57732 regression
        (6144, 13),  # per_core_N (15 tiles) < one DRAM shard (24 tiles), narrower still
        (6144, 8),  # per_core_N == shard width: control, must already pass
    ],
)
def test_dram_width_sharded_2d_mcast_narrow_worker_block(device, n, grid_x):
    """A worker's per_core_N block narrower than a DRAM bank's shard width must not read into the
    next worker's tiles (issue #57732: the unclamped first read corrupted every later column)."""
    if grid_x > device.compute_with_storage_grid_size().x:
        pytest.skip("grid_x larger than the device compute grid")
    out, ref = _dram_width_sharded_2d_matmul(device, 128, 4096, n, grid_x, in0_block_w=8)
    passed, pcc = comp_pcc(ref, out, 0.999)
    assert passed, pcc
    assert torch.isfinite(out).all(), "non-finite output indicates the narrow-block clamp regressed"


@pytest.mark.parametrize("in0_block_w", [8, 16])
def test_dram_sharded_multishard_block_matches_single_shard(device, in0_block_w):
    """A wider block changes only where the partial sums are packed; with fp32 accumulation the result
    must stay within bf16 output rounding of the one-shard-per-block path."""
    wide, ref = _dram_sharded_matmul(device, 4096, 4096, ttnn.bfloat8_b, 32, in0_block_w, True)
    stock, _ = _dram_sharded_matmul(device, 4096, 4096, ttnn.bfloat8_b, 32, 4, True)
    # The fp32 partial sums differ only in summation order, so the two results may round to
    # neighbouring bf16 values: one ulp of the larger magnitude, with an absolute floor for the
    # elements near zero.
    mag = torch.maximum(wide.abs(), stock.abs()).clamp(min=1e-30)
    ulp = 2.0 ** (torch.floor(torch.log2(mag)) - 7)
    floor = 2.0**-8 * ref.abs().mean()
    assert ((wide - stock).abs() <= torch.maximum(ulp, floor)).all(), (wide - stock).abs().max()
