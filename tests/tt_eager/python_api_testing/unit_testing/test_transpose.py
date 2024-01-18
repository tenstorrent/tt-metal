# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal


def transpose(
    input_shape,
    device,
    dim0,
    dim1,
    input_mem_config=ttl.tensor.MemoryConfig(),
    output_mem_config=ttl.tensor.MemoryConfig(),
    input_shard_spec=None,
    input_dtype=ttl.tensor.DataType.BFLOAT16,
    expected_program_cache_size=None,
):
    output_shape = list(input_shape)
    output_shape[dim0], output_shape[dim1] = input_shape[dim1], input_shape[dim0]

    x = torch.randn(input_shape).bfloat16().float()

    xt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        input_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE)
    if input_shard_spec is not None:
        xt = xt.to(device, input_mem_config, input_shard_spec)
    else:
        xt = xt.to(device, input_mem_config)
    xtt = ttl.tensor.transpose(xt, dim0, dim1, output_mem_config)
    assert list(xtt.shape()) == output_shape
    transposed_ref = x.transpose(dim0, dim1)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    if input_dtype == ttl.tensor.DataType.BFLOAT16:
        passing, output = comp_equal(transposed_ref, tt_got_back)
    else:
        passing, output = comp_pcc(transposed_ref, tt_got_back)
    logger.info(output)
    assert passing

    if expected_program_cache_size != None:
        assert ttl.program_cache.num_entries() == expected_program_cache_size


def test_transpose_hc(device):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2)


def test_transpose_hc_program_cache(device, use_program_cache):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=1)

    # changing shape
    N = 1
    C = C * 2
    H = H * 3
    W = W
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=1)

    # changing shape, single core
    N = 1
    C = 1
    H = 32
    W = 32
    input_shape = (N, C, H, W)
    # CACHE MISS since its single core
    # Cache size 2 more because of pad op in single core impl + transpose
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=3)


def test_transpose_cn_program_cache(device, use_program_cache):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=0, dim1=1, expected_program_cache_size=1)

    N = 1
    C = 32
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=0, dim1=1, expected_program_cache_size=1)


def test_transpose_wh_program_cache(device, use_program_cache):
    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=1)

    # changing shape
    N = 1
    C = C * 2
    H = H * 3
    W = W
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=1)

    # changing shape, single core
    N = 1
    C = 1
    H = 32
    W = 32
    input_shape = (N, C, H, W)
    # CACHE MISS since its single core
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=2)


def test_transpose_wh_sharded_program_cache(device, use_program_cache):
    compute_grid_size = device.compute_with_storage_grid_size()
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    input_dtype = ttl.tensor.DataType.BFLOAT8_B

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

    transpose(
        input_shape,
        device,
        dim0=-2,
        dim1=-1,
        input_mem_config=mem_config,
        output_mem_config=mem_config,
        input_shard_spec=input_shard_spec,
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

    transpose(
        input_shape,
        device,
        dim0=-2,
        dim1=-1,
        input_mem_config=mem_config,
        output_mem_config=mem_config,
        input_shard_spec=input_shard_spec,
        input_dtype=input_dtype,
        expected_program_cache_size=1,
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

    transpose(
        input_shape,
        device,
        dim0=-2,
        dim1=-1,
        input_mem_config=mem_config,
        output_mem_config=mem_config,
        input_shard_spec=input_shard_spec,
        input_dtype=input_dtype,
        expected_program_cache_size=1,
    )
