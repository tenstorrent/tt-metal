# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""1D mcast_in0 matmul (MatmulMultiCoreReuseMultiCast1DProgramConfig, mcast_in0=True) with the weight sharded in
DRAM, the layout the decode matmuls use. The weight cannot back the in1 circular buffer, so the program reads it
through its tensor accessor like an interleaved weight. Each case checks the result against torch on the weight
as stored, and against the same matmul with an interleaved copy of the weight; a second call with a re-uploaded
weight checks the program-cache hit path.
"""

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _dram_sharded_memory_config(device, k, n):
    num_banks = device.dram_grid_size().x
    n_padded = math.ceil(n / (ttnn.TILE_SIZE * num_banks)) * ttnn.TILE_SIZE * num_banks
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, (k, n_padded // num_banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _program_config(device, m, k, n, in0_block_w, fp32_dest_acc):
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = min(grid.x, 8), min(grid.y, 8)
    num_cores = grid_x * grid_y
    per_core_n = math.ceil(math.ceil(n / ttnn.TILE_SIZE) / num_cores)
    limit = 4 if fp32_dest_acc else 8
    out_subblock_w = max(d for d in range(1, per_core_n + 1) if per_core_n % d == 0 and d <= limit)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=math.ceil(m / ttnn.TILE_SIZE),
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


@pytest.mark.parametrize("m", [32, 64, 128])
@pytest.mark.parametrize(
    "k, n, in0_block_w",
    [
        (1024, 1024, 2),
        (2048, 3072, 4),
        (4096, 6144, 2),
        (4096, 14336, 2),
        (3584, 4096, 4),
        (1024, 1280, 8),
    ],
)
@pytest.mark.parametrize("fp32_dest_acc", [False, True], ids=["bf16_dest", "fp32_dest"])
def test_mcast_in0_dram_sharded_in1(device, m, k, n, in0_block_w, fp32_dest_acc):
    torch.manual_seed(0)
    in0 = torch.randn(1, 1, m, k).bfloat16().float()
    in1 = (torch.randn(1, 1, k, n) / math.sqrt(k)).bfloat16().float()
    expected = in0 @ in1

    program_config = _program_config(device, m, k, n, in0_block_w, fp32_dest_acc)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=True,
    )

    tt_in0 = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_in1_sharded = ttnn.from_torch(
        in1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_dram_sharded_memory_config(device, k, n),
    )
    tt_in1_interleaved = ttnn.from_torch(
        in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    def run(weight):
        return ttnn.linear(
            tt_in0,
            weight,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    out_sharded = ttnn.to_torch(run(tt_in1_sharded))
    out_interleaved = ttnn.to_torch(run(tt_in1_interleaved))
    assert_with_pcc(expected, out_sharded, 0.9995)
    # Same program, same weight values: only the way in1 is addressed differs.
    assert torch.equal(out_sharded, out_interleaved)

    # Program-cache hit with a new weight tensor at a new address.
    in1_second = (torch.randn(1, 1, k, n) / math.sqrt(k)).bfloat16().float()
    tt_in1_second = ttnn.from_torch(
        in1_second,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_dram_sharded_memory_config(device, k, n),
    )
    assert_with_pcc(in0 @ in1_second, ttnn.to_torch(run(tt_in1_second)), 0.9995)


@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfp8", "bfp4"])
def test_mcast_in0_dram_sharded_in1_block_float(device, weight_dtype):
    m, k, n = 128, 4096, 4096
    torch.manual_seed(1)
    in0 = torch.randn(1, 1, m, k).bfloat16().float()
    in1 = (torch.randn(1, 1, k, n) / math.sqrt(k)).bfloat16().float()
    tt_in0 = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_in1 = ttnn.from_torch(
        in1,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_dram_sharded_memory_config(device, k, n),
    )
    # Golden on the weight as stored on device (quantized).
    stored = ttnn.to_torch(tt_in1).float()
    out = ttnn.linear(
        tt_in0,
        tt_in1,
        program_config=_program_config(device, m, k, n, 2, False),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    assert_with_pcc(in0 @ stored, ttnn.to_torch(out), 0.999)
