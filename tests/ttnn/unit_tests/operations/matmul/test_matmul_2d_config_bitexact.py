# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def dram_width_sharded(device, k, n):
    grid = device.dram_grid_size()
    padded_n = math.ceil(n / (32 * grid.x)) * (32 * grid.x)
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
        (k, padded_n // grid.x),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def make_input(dist, m, k):
    x = torch.randn(1, 1, m, k)
    if dist == "outliers":
        x[torch.rand_like(x) < 0.01] *= 1e4
    elif dist == "colscale":
        x = x * torch.exp(torch.randn(1, 1, 1, k) * 2.0)
    return x


@pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole only")
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
@pytest.mark.parametrize("dist", ["randn", "outliers", "colscale"])
# One small shape is enough: the property under test (out_subblock_w doesn't change the result) holds per output tile.
# per_core_N = 12 exercises out_subblock_w 2, 3 and 4, and K = 512 gives 16 in0 blocks to accumulate over.
# The model QKV shapes (e.g. 128x4096x6144) took ~90 s per case under ttsim and pushed the sim CI job past its timeout.
@pytest.mark.parametrize("m, k, n", [(128, 512, 3072)])
def test_matmul_2d_config_bitexact(device, m, k, n, dist, fp32_dest_acc_en):
    torch.manual_seed(0)
    per_core_n = math.ceil(n / 32 / 8)
    torch_x = make_input(dist, m, k)
    torch_w = torch.randn(1, 1, k, n) * 0.02
    x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(
        torch_w,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_width_sharded(device, k, n),
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )

    def run(out_subblock_w):
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
        out = ttnn.linear(
            x, w, program_config=program_config, dtype=ttnn.bfloat16, compute_kernel_config=compute_kernel_config
        )
        return ttnn.to_torch(out)

    reference = run(1)
    assert_with_pcc(torch_x @ torch_w, reference, 0.99)
    for out_subblock_w in [d for d in (2, 3, 4) if per_core_n % d == 0]:
        assert torch.equal(run(out_subblock_w), reference), f"out_subblock_w={out_subblock_w}"
