# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
import tt_lib as ttl

from models.demos.wormhole.mamba.tt.full_model import MambaSsmBlockTransformer
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


@pytest.mark.parametrize(
    "batch, pcc",
    (
        (
            32,
            0.99,
        ),
    ),
)
def test_mamba_ssm_block_repeat_interleave(
    device: ttnn.Device,
    use_program_cache,
    batch: int,
    pcc: float,
):
    n = 16
    hidden_size = 2560
    input = torch.rand(1, 1, batch, hidden_size * 2)
    dtype = ttnn.bfloat16
    fidelity = ttl.tensor.MathFidelity.LoFi

    expected = torch.repeat_interleave(input, n, dim=3)

    transformer = MambaSsmBlockTransformer(device, hidden_size * 2, n, dtype=dtype)
    input = ttnn.to_device(
        ttnn.from_torch(input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    core_grid = ttnn.CoreGrid(y=7, x=8)
    actual = transformer.repeat_interleave(input, ttnn.L1_MEMORY_CONFIG, compute_kernel_config, core_grid)

    print(comp_allclose(expected, ttnn.to_torch(actual)))


@pytest.mark.parametrize(
    "batch, pcc",
    (
        (
            32,
            0.99,
        ),
    ),
)
def test_mamba_ssm_block_repeat(
    device: ttnn.Device,
    use_program_cache,
    batch: int,
    pcc: float,
):
    n = 16
    hidden_size = 2560
    input = torch.rand(1, 1, batch, n)
    dtype = ttnn.bfloat16
    fidelity = ttl.tensor.MathFidelity.LoFi

    # (1, 1, B, n) -> (1, 1, B, hidden * 2 * n)
    expected = input.repeat((1, 1, 1, hidden_size * 2))

    transformer = MambaSsmBlockTransformer(device, hidden_size * 2, n, dtype=dtype)
    input = ttnn.to_device(
        ttnn.from_torch(input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    core_grid = ttnn.CoreGrid(y=7, x=8)
    actual = transformer.repeat(input, ttnn.L1_MEMORY_CONFIG, compute_kernel_config, core_grid)

    print(comp_allclose(expected, ttnn.to_torch(actual)))


@pytest.mark.parametrize(
    "batch, pcc",
    (
        (
            32,
            0.99,
        ),
    ),
)
def test_mamba_ssm_block_reduce(
    device: ttnn.Device,
    use_program_cache,
    batch: int,
    pcc: float,
):
    n = 16
    hidden_size = 2560
    input = torch.rand(1, 1, batch, hidden_size * 2 * n)
    dtype = ttnn.bfloat16
    fidelity = ttl.tensor.MathFidelity.LoFi

    # (1, 1, b, hidden_size * 2 * n) -> (1, b, hidden_size * 2)
    expected = torch.sum(torch.reshape(input, (1, batch, hidden_size * 2, n)), dim=-1).unsqueeze(0)

    transformer = MambaSsmBlockTransformer(device, hidden_size * 2, n, dtype=dtype)
    input = ttnn.to_device(
        ttnn.from_torch(input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    core_grid = ttnn.CoreGrid(y=7, x=8)
    actual = transformer.reduce(input, ttnn.L1_MEMORY_CONFIG, compute_kernel_config, core_grid)

    print(comp_allclose(expected, ttnn.to_torch(actual)))
