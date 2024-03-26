# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

import tt_lib as ttl
from loguru import logger
from models.utility_functions import is_grayskull
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from models.utility_functions import skip_for_grayskull


def transpose(
    input_shape,
    device,
    dim0,
    dim1,
    input_mem_config=ttl.tensor.MemoryConfig(),
    output_mem_config=ttl.tensor.MemoryConfig(),
    input_dtype=ttl.tensor.DataType.BFLOAT16,
    expected_program_cache_size=None,
):
    output_shape = list(input_shape)
    output_shape[dim0], output_shape[dim1] = input_shape[dim1], input_shape[dim0]

    if input_dtype == ttl.tensor.DataType.UINT16:
        x = torch.randint(0, 100, input_shape).to(torch.int16)
    else:
        x = torch.randn(input_shape).bfloat16().float()

    xt = ttl.tensor.Tensor(
        x,
        input_dtype,
    ).to(ttl.tensor.Layout.TILE)

    xt = xt.to(device, input_mem_config)
    xtt = ttl.tensor.transpose(xt, dim0, dim1, output_mem_config)
    assert list(xtt.get_legacy_shape()) == output_shape
    transposed_ref = x.transpose(dim0, dim1)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    if input_dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(transposed_ref, tt_got_back)
    else:
        target_pcc = 0.95 if input_dtype == ttl.tensor.DataType.BFLOAT4_B else 0.99
        passing, output = comp_pcc(transposed_ref, tt_got_back, target_pcc)
    logger.info(output)
    assert passing

    if expected_program_cache_size != None:
        assert device.num_program_cache_entries() == expected_program_cache_size


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.FLOAT32),
    ids=["bfloat16", "float"],
)
def test_transpose_hc(dtype, device):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    logger.info("transpose on C H dim")

    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2, input_dtype=dtype)


@skip_for_grayskull("Integer formats not supported on Grayskull")
def test_transpose_wh_uint16(device):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=-2, dim1=-1, input_dtype=ttl.tensor.DataType.UINT16)


@skip_for_grayskull("Bfp4 format not supported on Grayskull")
def test_transpose_wh_bfp4(device):
    N = 1
    C = 32
    H = 32 * 2
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=-2, dim1=-1, input_dtype=ttl.tensor.DataType.BFLOAT4_B)


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.FLOAT32),
    ids=["bfloat16", "float"],
)
def test_transpose_hc_program_cache(dtype, device, use_program_cache):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=1, input_dtype=dtype)

    # changing shape
    N = 1
    C = C * 2
    H = H * 3
    W = W
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=1, input_dtype=dtype)

    # changing shape, single core
    N = 1
    C = 1
    H = 32
    W = 32
    input_shape = (N, C, H, W)
    # CACHE MISS since its single core
    # Cache size 2 more because of pad op in single core impl + transpose
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=3, input_dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.FLOAT32),
    ids=["bfloat16", "float"],
)
def test_transpose_cn_program_cache(dtype, device, use_program_cache):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=0, dim1=1, expected_program_cache_size=1, input_dtype=dtype)

    N = 1
    C = 32
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=0, dim1=1, expected_program_cache_size=1, input_dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.FLOAT32),
    ids=["bfloat16", "float"],
)
def test_transpose_wh_program_cache(dtype, device, use_program_cache):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=1, input_dtype=dtype)

    # changing shape
    N = 1
    C = C * 2
    H = H * 3
    W = W
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=1, input_dtype=dtype)

    # changing shape, single core
    N = 1
    C = 1
    H = 32
    W = 32
    input_shape = (N, C, H, W)
    # CACHE MISS since its single core
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=2, input_dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.FLOAT32),
    ids=["bfloat8_b", "float"],
)
def test_transpose_wh_sharded_program_cache(dtype, device, use_program_cache):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    compute_grid_size = device.compute_with_storage_grid_size()
    input_dtype = dtype

    N = 32
    C = 2
    H = 32 * 4
    W = 64
    input_shape = torch.Size([N, C, H, W])

    num_cores = min(N, compute_grid_size.x * compute_grid_size.y)
    shard_grid = ttl.tensor.CoreRangeSet(ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True))
    input_shard_spec = ttl.tensor.ShardSpec(
        shard_grid,
        [
            input_shape.numel() // input_shape[-1] // num_cores,
            input_shape[-1],
        ],
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )
    mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
    )

    transpose(
        input_shape,
        device,
        dim0=-2,
        dim1=-1,
        input_mem_config=mem_config,
        output_mem_config=mem_config,
        input_dtype=input_dtype,
        expected_program_cache_size=1,
    )

    # changing shape
    N = 32
    C = 8
    H = 32 * 4
    W = 64

    input_shape = torch.Size([N, C, H, W])

    num_cores = min(N, compute_grid_size.x * compute_grid_size.y)
    shard_grid = ttl.tensor.CoreRangeSet(ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True))
    input_shard_spec = ttl.tensor.ShardSpec(
        shard_grid,
        [
            input_shape.numel() // input_shape[-1] // num_cores,
            input_shape[-1],
        ],
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )

    mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
    )
    # shape change also changes shard_spec as shard_shape is dependent on input_shape (resulting in CACHE MISS)
    transpose(
        input_shape,
        device,
        dim0=-2,
        dim1=-1,
        input_mem_config=mem_config,
        output_mem_config=mem_config,
        input_dtype=input_dtype,
        expected_program_cache_size=2,
    )

    # changing shape
    N = 32
    C = 2
    H = 2048
    W = 64

    input_shape = torch.Size([N, C, H, W])

    num_cores = min(N, compute_grid_size.x * compute_grid_size.y)
    shard_grid = ttl.tensor.CoreRangeSet(ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True))
    input_shard_spec = ttl.tensor.ShardSpec(
        shard_grid,
        [
            input_shape.numel() // input_shape[-1] // num_cores,
            input_shape[-1],
        ],
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )

    mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
    )

    # shape change also changes shard_spec as shard_shape is dependent on input_shape (resulting in CACHE MISS)
    # tensor cannot fit in L1 for fp32
    if input_dtype != ttl.tensor.DataType.FLOAT32:
        transpose(
            input_shape,
            device,
            dim0=-2,
            dim1=-1,
            input_mem_config=mem_config,
            output_mem_config=mem_config,
            input_dtype=input_dtype,
            expected_program_cache_size=3,
        )
