# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.operations.activations import get_golden_function_for_activation

from tests.ttnn.utils_for_testing import assert_with_pcc


def _sharded_memory_config(shape, strategy):
    m = shape[-2]
    if strategy == ttnn.ShardStrategy.BLOCK:
        grid = ttnn.CoreGrid(y=8, x=8)
    elif strategy == ttnn.ShardStrategy.HEIGHT:
        grid = ttnn.CoreGrid(y=m // 32 // 8, x=8)
    else:
        grid = ttnn.CoreGrid(y=1, x=8)
    return ttnn.create_sharded_memory_config(shape, core_grid=grid, strategy=strategy)


@pytest.mark.parametrize(
    "m_size, k_size, n_size, strategy",
    [
        (1792, 768, 3072, ttnn.ShardStrategy.BLOCK),
        (1792, 3072, 768, ttnn.ShardStrategy.BLOCK),
        (1024, 768, 3072, ttnn.ShardStrategy.HEIGHT),
        (512, 1024, 1024, ttnn.ShardStrategy.HEIGHT),
        (32, 4096, 4096, ttnn.ShardStrategy.WIDTH),
        (32, 2048, 8192, ttnn.ShardStrategy.WIDTH),
    ],
)
@pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
@pytest.mark.parametrize("use_bias", [True, False])
def test_linear_sharded_input_with_activation(device, m_size, k_size, n_size, strategy, activation, use_bias):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 8:
        pytest.skip("needs an 8x8 compute grid")

    torch.manual_seed(0)
    shape = (1, 1, m_size, k_size)
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn((k_size, n_size), dtype=torch.bfloat16) / k_size**0.5
    torch_bias = torch.randn((1, n_size), dtype=torch.bfloat16) * 0.1 if use_bias else None

    torch_out = torch_a.float() @ torch_b.float()
    if use_bias:
        torch_out = torch_out + torch_bias.float()
    torch_out = get_golden_function_for_activation(activation)(torch_out)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.to_memory_config(a, _sharded_memory_config(shape, strategy))
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device) if use_bias else None

    out = ttnn.linear(a, b, bias=bias, activation=activation)

    assert_with_pcc(torch_out, ttnn.to_torch(out).float(), 0.999)


def test_linear_sharded_input_with_activation_and_program_config_is_rejected(device, expect_error):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 8:
        pytest.skip("needs an 8x8 compute grid")

    shape = (1, 1, 1792, 768)
    a = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.to_memory_config(a, _sharded_memory_config(shape, ttnn.ShardStrategy.BLOCK))
    b = ttnn.from_torch(torch.randn((768, 3072), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=3,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=7,
        per_core_N=12,
        transpose_mcast=False,
        fused_activation=None,
    )

    with expect_error(RuntimeError, "fused_activation"):
        ttnn.linear(a, b, program_config=program_config, activation="gelu")
