# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn

import pytest
from loguru import logger

from models.utility_functions import tt2torch_tensor, comp_pcc


def run_ssm_1d_sum_reduce(H: int, W: int, latent_size: int, dtype, in_mem_config, out_mem_config, device):
    torch.manual_seed(0)

    input_shape = [1, 1, H, W]
    x = torch.randn(input_shape)
    expected = torch.sum(x.reshape((1, 1, H, W // latent_size, latent_size)), dim=-1)

    x = ttnn.Tensor(x, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)
    actual = ttnn.experimental.hc_sum_reduce(x, memory_config=out_mem_config, dtype=dtype)

    assert list(actual.shape.with_tile_padding()) == [1, 1, H, W // latent_size]
    assert actual.dtype == dtype

    actual = tt2torch_tensor(actual)
    passing_pcc, output_pcc = comp_pcc(actual, expected, 0.9997)
    logger.debug(f"Out passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
)
@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
)
@pytest.mark.parametrize(
    "H, W, latent_size",
    (
        (32, 1024, 32),
        (32, 163840, 32),
        (128, 1024, 32),
        (64, 163840, 32),
    ),
)
def test_ssm_reduce(H, W, latent_size, dtype, out_mem_config, in_mem_config, device):
    run_ssm_1d_sum_reduce(H, W, latent_size, dtype, out_mem_config, in_mem_config, device)


def test_ssm_1d_sum_reduce_with_program_cache(device, use_program_cache):
    H, W, latent = 32, 163840, 32
    mem_config = ttnn.L1_MEMORY_CONFIG
    dtype = ttnn.bfloat16

    for _ in range(2):
        H, W, latent = 32, 163840, 32
        run_ssm_1d_sum_reduce(H, W, latent, dtype, mem_config, mem_config, device)
        H, W, latent = 64, 163840, 32
        run_ssm_1d_sum_reduce(H, W, latent, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2
