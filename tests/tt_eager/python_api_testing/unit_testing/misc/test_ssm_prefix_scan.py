# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
import pytest
from loguru import logger

from models.utility_functions import tt2torch_tensor, comp_pcc, skip_for_grayskull


def sequential_prefix_scan(a, bx):
    (_, _, L, EN) = bx.shape
    hidden_states = torch.zeros((1, 1, L, EN), device=a.device)
    for i in range(L):
        hidden_states[:, :, i] = a[:, :, i] * hidden_states[:, :, i - 1] + bx[:, :, i]
    return hidden_states


def run_ssm_prefix_scan(L: int, E: int, N: int, num_cores: int, dtype, device):
    torch.manual_seed(0)

    a = torch.randn((1, 1, L, E * N))
    bx = torch.randn((1, 1, L, E * N))

    expected = sequential_prefix_scan(a, bx)

    compute_grid_size = device.compute_with_storage_grid_size()
    shard_grid = ttl.tensor.CoreRangeSet(ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True))
    shard_spec = ttl.tensor.ShardSpec(
        shard_grid,
        [L, E * N // num_cores],
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )
    memory_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, ttl.tensor.BufferType.L1, shard_spec
    )
    a = ttl.tensor.Tensor(a, dtype).to(ttl.tensor.Layout.TILE).to(device, memory_config)
    bx = ttl.tensor.Tensor(bx, dtype).to(ttl.tensor.Layout.TILE).to(device, memory_config)

    actual = ttl.operations.primary.transformers.ssm_prefix_scan(
        a, bx, output_mem_config=memory_config, output_dtype=dtype
    )
    assert list(actual.get_legacy_shape()) == list(expected.shape)
    assert actual.dtype == dtype

    actual = tt2torch_tensor(actual)

    passing_pcc, output_pcc = comp_pcc(actual, expected, 0.999)
    logger.debug(f"Out passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


@skip_for_grayskull("Grayskull not supported")
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B),
)
@pytest.mark.parametrize(
    "L, E, N, num_cores",
    (
        (32, 32, 32, 1),
        (32, 64, 32, 1),
        (64, 32, 32, 1),
        (64, 64, 32, 1),
        (32, 2560, 32, 32),
        (32, 5120, 32, 40),
        # (32, 5120, 32, 64), #-> 8x8 grid not supported on CI
        # (64, 5120, 32, 64) #-> 8x8 grid not supported on CI
    ),
)
def test_ssm_prefix_scan(L: int, E: int, N: int, num_cores: int, dtype, device):
    run_ssm_prefix_scan(L, E, N, num_cores, dtype, device)


@skip_for_grayskull("Grayskull not supported")
def test_ssm_prefix_scan_with_program_cache(device, use_program_cache):
    L, E, N = 32, 64, 32
    num_cores = 1
    dtype = ttl.tensor.DataType.BFLOAT8_B
    run_ssm_prefix_scan(L, E, N, num_cores, dtype, device)

    dummy_memory_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    dummy_shape = [1, 1, 128, 128]

    for _ in range(2):
        run_ssm_prefix_scan(L, E, N, num_cores, dtype, device)
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = (
            ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, dummy_memory_config)
        )

    assert device.num_program_cache_entries() == 1
