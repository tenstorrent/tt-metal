# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

import ttnn

from loguru import logger
from models.utility_functions import is_grayskull, is_blackhole, torch_random
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from models.utility_functions import skip_for_grayskull, skip_for_blackhole, run_for_blackhole, skip_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def transpose(
    input_shape,
    device,
    dim0,
    dim1,
    input_mem_config=ttnn.MemoryConfig(),
    output_mem_config=ttnn.MemoryConfig(),
    input_dtype=ttnn.bfloat16,
    expected_program_cache_size=None,
):
    torch.manual_seed(2005)
    output_shape = list(input_shape)
    output_shape[dim0], output_shape[dim1] = input_shape[dim1], input_shape[dim0]

    if input_dtype == ttnn.uint16:
        x = torch.randint(0, 100, input_shape).to(torch.int16)
    else:
        x = torch.randn(input_shape).bfloat16().float()

    ttnn_input = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, device=device, memory_config=input_mem_config
    )
    xtt = ttnn.transpose(ttnn_input, dim0, dim1, memory_config=output_mem_config)

    assert list(xtt.shape) == output_shape
    transposed_ref = x.transpose(dim0, dim1)

    tt_got_back = ttnn.to_torch(xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

    if input_dtype == ttnn.bfloat16:
        passing, output = comp_equal(transposed_ref, tt_got_back)
    else:
        target_pcc = 0.95 if input_dtype == ttnn.bfloat4_b else 0.99
        passing, output = comp_pcc(transposed_ref, tt_got_back, target_pcc)
    logger.info(output)
    assert passing

    if expected_program_cache_size != None:
        assert device.num_program_cache_entries() == expected_program_cache_size


def test_fold_transpose(device, use_program_cache):
    N = 16
    C = 4
    H = 256
    W = 224
    input_shape = (N, C, H, W)
    ## 64

    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = min(N, compute_grid_size.x * compute_grid_size.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)

    sharded_config = ttnn.create_sharded_memory_config_(
        input_shape,
        shard_grid,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    transpose(input_shape, device, dim0=2, dim1=3, input_mem_config=sharded_config, output_mem_config=sharded_config)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
def test_transpose_hc_unit(dtype, device):
    if is_grayskull() and dtype == ttnn.float32:
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
    transpose(input_shape, device, dim0=-2, dim1=-1, input_dtype=ttnn.uint16)


@skip_for_grayskull("Bfp4 format not supported on Grayskull")
def test_transpose_wh_bfp4(device):
    N = 1
    C = 32
    H = 32 * 2
    W = 32 * 3
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=-2, dim1=-1, input_dtype=ttnn.bfloat4_b)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
def test_transpose_hc_program_cache(dtype, device, use_program_cache):
    if is_grayskull() and dtype == ttnn.float32:
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
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=2, input_dtype=dtype)

    # changing shape, single core
    N = 1
    C = 1
    H = 32
    W = 32
    input_shape = (N, C, H, W)
    transpose(input_shape, device, dim0=1, dim1=-2, expected_program_cache_size=3, input_dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
def test_transpose_cn_program_cache(dtype, device, use_program_cache):
    if is_grayskull() and dtype == ttnn.float32:
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
    transpose(input_shape, device, dim0=0, dim1=1, expected_program_cache_size=2, input_dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b),
    ids=["bfloat16", "float", "bfloat8_b"],
)
def test_transpose_wh_program_cache(dtype, device, use_program_cache):
    if is_grayskull() and dtype == ttnn.float32:
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
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=2, input_dtype=dtype)

    # changing shape, single core
    N = 1
    C = 1
    H = 32
    W = 32
    input_shape = (N, C, H, W)
    # CACHE MISS since its single core
    transpose(input_shape, device, dim0=-2, dim1=-1, expected_program_cache_size=3, input_dtype=dtype)


@skip_for_blackhole("GH #15234")
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.float32),
    ids=["bfloat8_b", "float"],
)
def test_transpose_wh_sharded_program_cache(dtype, device, use_program_cache):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    compute_grid_size = device.compute_with_storage_grid_size()
    input_dtype = dtype

    N = 32
    C = 2
    H = 32 * 4
    W = 64
    input_shape = torch.Size([N, C, H, W])

    num_cores = min(N, compute_grid_size.x * compute_grid_size.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        [
            input_shape.numel() // input_shape[-1] // num_cores,
            input_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

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
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        [
            input_shape.numel() // input_shape[-1] // num_cores,
            input_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
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
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        [
            input_shape.numel() // input_shape[-1] // num_cores,
            input_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # shape change also changes shard_spec as shard_shape is dependent on input_shape (resulting in CACHE MISS)
    # tensor cannot fit in L1 for fp32
    if input_dtype != ttnn.float32:
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


@skip_for_grayskull("Grayskull has pcc issue when transpose used untilize")
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("h", [230])
@pytest.mark.parametrize("w", [256])
def test_transpose_hw_rm_with_padding(device, n, c, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)
    activation_pyt_padded = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    activation_pyt_padded = ttnn.pad(
        activation_pyt_padded,
        padding=(
            (0, 0),
            (0, 26),
            (0, 0),
        ),
        value=0,
    )
    activation_pyt_padded = ttnn.transpose(activation_pyt_padded, 2, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.to_memory_config(activation_pyt_padded, ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.from_device(activation_pyt_padded_out)
    activation_pyt_padded_out = ttnn.to_torch(activation_pyt_padded_out)
    activation_pyt_padded_out = activation_pyt_padded_out[:n, :c, :w, :h]
    assert_with_pcc(torch_output_tensor, activation_pyt_padded_out, 0.9999)


@skip_for_grayskull("Grayskull has pcc issue when transpose used untilize")
@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [8])
@pytest.mark.parametrize("w", [256])
def test_transpose_hw_rm_no_padding(device, n, c, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)
    activation_pyt_padded = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    activation_pyt_padded = ttnn.transpose(activation_pyt_padded, 2, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.to_memory_config(activation_pyt_padded, ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.from_device(activation_pyt_padded_out)
    activation_pyt_padded_out = ttnn.to_torch(activation_pyt_padded_out)
    assert_with_pcc(torch_output_tensor, activation_pyt_padded_out, 0.9999)


def run_transpose_hw_rm_program_cache(device, n, c, h, w, use_program_cache):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)
    activation_pyt_padded = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    activation_pyt_padded = ttnn.transpose(activation_pyt_padded, 2, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.to_torch(activation_pyt_padded)
    assert_with_pcc(torch_output_tensor, activation_pyt_padded_out, 0.9999)


@skip_for_grayskull("Grayskull has pcc issue when transpose used untilize")
@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [8])
@pytest.mark.parametrize("w", [256])
def test_transpose_hw_rm_with_program_cache(device, n, c, h, w, use_program_cache):
    for _ in range(2):
        run_transpose_hw_rm_program_cache(device, n, c, h, w, use_program_cache)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 1


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [224])
@pytest.mark.parametrize("h", [16])
@pytest.mark.parametrize("w", [112])
def test_transpose_hw_sharded_rm(device, n, c, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # shard config
    num_cores_x = 8
    num_cores_y = 8
    if num_cores_y > device.core_grid.y:
        num_cores_y = device.core_grid.y
    grid_size = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(
        shard_grid, (n * h * c // (grid_size.x * grid_size.y), w), ttnn.ShardOrientation.COL_MAJOR
    )
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_mem_config)

    tt_output_tensor = ttnn.transpose(tt_input_tensor, 2, 3, memory_config=sharded_mem_config)
    tt_output_tensor = ttnn.to_memory_config(tt_output_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9999)


def run_transpose_hw_sharded_rm_with_program_cache(device, n, c, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # shard config
    grid_size = ttnn.CoreGrid(y=4, x=8)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(
        shard_grid, (n * h * c // (grid_size.x * grid_size.y), w), ttnn.ShardOrientation.COL_MAJOR
    )
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_mem_config)

    tt_output_tensor = ttnn.transpose(tt_input_tensor, 2, 3, memory_config=sharded_mem_config)
    tt_output_tensor = ttnn.to_memory_config(tt_output_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9999)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [16])
def test_transpose_hw_sharded_rm_with_program_cache(device, n, c, h, w, use_program_cache):
    for _ in range(2):
        run_transpose_hw_sharded_rm_with_program_cache(device, n, c, h, w)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 3


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [16])
def test_transpose_hc_rm(device, n, c, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(1, 2)
    activation_pyt_padded = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    activation_pyt_padded = ttnn.transpose(activation_pyt_padded, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.to_memory_config(activation_pyt_padded, ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.from_device(activation_pyt_padded_out)
    activation_pyt_padded_out = ttnn.to_torch(activation_pyt_padded_out)

    assert_with_pcc(torch_output_tensor, activation_pyt_padded_out, 0.9999)


def run_transpose_hc_rm_with_program_cache(device, n, c, h, w, use_program_cache):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(1, 2)
    activation_pyt_padded = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    activation_pyt_padded = ttnn.transpose(activation_pyt_padded, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.to_memory_config(activation_pyt_padded, ttnn.L1_MEMORY_CONFIG)
    activation_pyt_padded_out = ttnn.from_device(activation_pyt_padded_out)
    activation_pyt_padded_out = ttnn.to_torch(activation_pyt_padded_out)
    assert_with_pcc(torch_output_tensor, activation_pyt_padded_out, 0.9999)


@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [256])
@pytest.mark.parametrize("w", [16])
def test_transpose_hc_rm_with_program_cache(device, n, c, h, w, use_program_cache):
    for _ in range(2):
        run_transpose_hc_rm_with_program_cache(device, n, c, h, w, use_program_cache)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 1


def run_transpose_hc_sharded(device, n, c, h, w, grid_size):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(1, 2)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(
        shard_grid, (n * h * c // (grid_size.x * grid_size.y), w), ttnn.ShardOrientation.ROW_MAJOR
    )
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_mem_config)

    tt_output_tensor = ttnn.transpose(tt_input_tensor, 1, 2, memory_config=sharded_mem_config)
    tt_output_tensor = ttnn.to_memory_config(tt_output_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9999)


@pytest.mark.parametrize(
    "n, c, h, w, grid_size",
    [
        (1, 8, 4, 16, ttnn.CoreGrid(y=4, x=1)),
        (4, 3, 3, 16, ttnn.CoreGrid(y=2, x=1)),
        (2, 8, 4, 32, ttnn.CoreGrid(y=8, x=4)),
        (2, 2, 8, 64, ttnn.CoreGrid(y=8, x=1)),
        (16, 4, 224, 224, ttnn.CoreGrid(y=8, x=8)),
        (20, 4, 224, 224, ttnn.CoreGrid(y=8, x=7)),
        (24, 3, 224, 224, ttnn.CoreGrid(y=8, x=7)),
        (16, 128, 256, 16, ttnn.CoreGrid(y=8, x=8)),
        (16, 128, 128, 16, ttnn.CoreGrid(y=8, x=8)),
    ],
)
def test_transpose_hc_sharded_with_program_cache(device, n, c, h, w, grid_size, use_program_cache):
    if grid_size.y > device.core_grid.y:
        pytest.skip("grid size not for N300")
    for _ in range(2):
        run_transpose_hc_sharded(device, n, c, h, w, grid_size)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 3


@pytest.mark.parametrize(
    "shape, swap_dims",
    [
        ((32, 32, 32, 32), (0, 2)),
        ((32, 32, 32, 32), (1, 2)),
        ((32, 32, 32, 32), (0, 3)),
        ((32, 32, 32, 32), (1, 3)),
        ((32, 32, 32, 32), (2, 3)),
        ((32, 32, 32, 32), (0, 1)),
    ],
)
def test_transpose_bfloat8_b(device, shape, swap_dims):
    torch.manual_seed(2005)
    input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = input.transpose(*swap_dims)

    tt_input = ttnn.from_torch(input, dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.transpose(tt_input, *swap_dims)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "shape",
    [(1, 32, 12, 100), (1, 12, 32, 100), (1, 35, 7, 7), (1, 1, 1, 1), (1, 12, 32, 100)],
)
def test_transpose_hc(dtype, shape, device):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    logger.info("transpose on C H dim")

    transpose(shape, device, dim0=1, dim1=-2, input_dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "shape",
    [(9216, 128), (1, 32), (1, 12), (1, 35), (16, 32), (34, 8), [21843, 768]],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
)
def test_transpose_2D(dtype, shape, layout, device):
    pytest.skip("Unstable see #16779")
    torch.manual_seed(2005)
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch_input.transpose(0, 1)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.DataType.BFLOAT16, layout=layout, device=device)
    tt_output = ttnn.transpose(tt_input, 0, 1)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["bfloat16", "float"],
)
@pytest.mark.parametrize(
    "shape",
    [[32, 1, 32], [32, 1, 12], [1, 1, 35], [1, 16, 32], [2, 34, 8], (32, 12, 100), (6, 33, 34)],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize(
    "dims",
    [
        [0, 1],
        [0, 2],
        [2, 1],
        [-3, -2],
        [-3, -1],
        [-2, -1],
    ],
)
def test_transpose_3D(dtype, shape, layout, dims, device):
    torch.manual_seed(2005)
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    if layout == ttnn.ROW_MAJOR_LAYOUT and dtype == ttnn.bfloat16 and (shape[-1] % 2 or shape[dims[-1]] % 2):
        pytest.skip("Skipping RM odd inner dim test cases")

    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch_input.transpose(dims[0], dims[1])

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.DataType.BFLOAT16, layout=layout, device=device)
    tt_output = ttnn.transpose(tt_input, dims[0], dims[1])
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "shape",
    [[4, 3, 1280, 40], [1, 4096, 4096]],
)
def test_transpose_4d_wh_rm(shape, device):
    torch.manual_seed(2005)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch_input.transpose(-1, -2)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.transpose(tt_input, -1, -2)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "shape",
    [[4, 3, 1280, 40], [1, 1, 1200, 1280], [1, 1, 4096, 4096]],
)
def test_transpose_4d_wh_tile(shape, device):
    torch.manual_seed(2005)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch_input.transpose(-1, -2)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.transpose(tt_input, -1, -2)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.skip("Issue: #16141 Skipping due to hang on to_layout to tile where input shape has 1 in it.")
@pytest.mark.parametrize(
    "config",
    [
        [[1, 1370, 1, 3, 1280], [0, -2], ttnn.TILE_LAYOUT],  # hang
        [[1, 50, 1, 3, 768], [0, -2], ttnn.TILE_LAYOUT],  # hang
        [[1, 50, 1, 3, 1024], [0, -2], ttnn.TILE_LAYOUT],  # hang
        [[1, 197, 1, 3, 768], [0, -2], ttnn.TILE_LAYOUT],  # hang
        [[1, 197, 1, 3, 1024], [0, -2], ttnn.TILE_LAYOUT],  # hang
        [[2, 7, 2, 7, 384], [-4, -3], ttnn.TILE_LAYOUT],  # hang
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_transpose_failures(config, memory_config, device):
    torch.manual_seed(2005)
    torch_input = torch.randn(config[0], dtype=torch.bfloat16)
    torch_output = torch_input.transpose(config[1][0], config[1][1])

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.DataType.BFLOAT16, layout=config[2], device=device, memory_config=memory_config
    )
    tt_output = ttnn.transpose(tt_input, config[1][0], config[1][1])
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "config",
    [
        [
            [1, 16, 6, 64],
            [-1, -2],
            ttnn.ROW_MAJOR_LAYOUT,
        ],
        [
            [1, 16, 64, 6],
            [-1, -2],
            ttnn.ROW_MAJOR_LAYOUT,
        ],
        [
            [1, 16, 64, 6],
            [1, 2],
            ttnn.ROW_MAJOR_LAYOUT,
        ],
        [[1, 9, 8, 18], [1, 2], ttnn.ROW_MAJOR_LAYOUT],
        [[1, 9, 8, 14], [1, 2], ttnn.ROW_MAJOR_LAYOUT],
        [[1, 9, 8, 2], [1, 2], ttnn.ROW_MAJOR_LAYOUT],
        [[1, 2, 8, 2], [1, 2], ttnn.ROW_MAJOR_LAYOUT],
        [[64, 4, 49, 32], [-2, -1], ttnn.ROW_MAJOR_LAYOUT],
        [[12, 3], [0, 1], ttnn.ROW_MAJOR_LAYOUT],
        [
            [1, 8, 4096, 40],
            [1, 2],
            ttnn.ROW_MAJOR_LAYOUT,
        ],
        [[1, 9, 8, 40], [1, 2], ttnn.ROW_MAJOR_LAYOUT],
        [[1, 8, 8, 8], [1, 2], ttnn.ROW_MAJOR_LAYOUT],
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_transpose_former_failures(config, memory_config, device):
    torch.manual_seed(2005)
    # this will convert to tiled for now
    torch_input = torch.randn(config[0], dtype=torch.bfloat16)
    torch_output = torch_input.transpose(config[1][0], config[1][1])
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.DataType.BFLOAT16, layout=config[2], device=device, memory_config=memory_config
    )
    tt_output = ttnn.transpose(tt_input, config[1][0], config[1][1])
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "shape",
    [(1, 2, 32, 100), (1, 35, 7, 7), (1, 1, 1, 1)],
)
def test_transpose_hc_padded_c(shape, device):
    # this will convert to tiled for now
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch_input.transpose(1, 2)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.transpose(tt_input, 1, 2)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "shape",
    [[1, 197, 1, 3, 1024], [1, 197, 1, 3, 768], [1, 50, 1, 3, 1024], [1, 50, 1, 3, 768], [1, 1370, 1, 3, 1280]],
)
@pytest.mark.parametrize(
    "dims",
    [
        (0, -2),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.ROW_MAJOR_LAYOUT],
)
def test_transpose_5d(shape, dims, layout, device):
    torch.manual_seed(2005)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch_input.transpose(dims[0], dims[1])

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.DataType.BFLOAT16, layout=layout, device=device)
    tt_output = ttnn.transpose(tt_input, dims[0], dims[1])
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 5, 10, 15],
        [1, 1, 1, 2],
        [1, 3, 2, 1],
        [1, 17, 1, 1],
        [1, 1, 16, 1],
        [1, 1, 17, 1],
        [1, 1, 1, 17],
        [2, 1, 1, 1],
        [2, 33, 33, 33],
    ],
)
@pytest.mark.parametrize(
    "dims",
    [
        (1, 2),
        (0, 2),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.float32, ttnn.bfloat16],
)
def test_transpose_issue_11650_10350(shape, dims, layout, dtype, device):
    torch.manual_seed(2005)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch_input.transpose(dims[0], dims[1])

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)
    tt_output = ttnn.transpose(tt_input, dims[0], dims[1])
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 17, 1, 1],
        [1, 1, 16, 1],
        [1, 1, 17, 1],
        [1, 1, 1, 17],
        [2, 1, 1, 1],
        [2, 33, 33, 33],
    ],
)
@pytest.mark.parametrize(
    "dims",
    [
        (1, 2),
        (0, 2),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.float32, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "pad_value",
    [None, float("-inf")],
)
def test_transpose_unpadded(shape, dims, layout, dtype, pad_value, device):
    torch.manual_seed(2005)
    if pad_value is not None and is_blackhole():
        pytest.skip("Blackhole reduce is needed for the full test to work")
    elif dtype == ttnn.float32 and is_grayskull():
        pytest.skip("Grayskull does not support float32")
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch_input.transpose(dims[0], dims[1])

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)
    tt_output = ttnn.transpose(tt_input, dims[0], dims[1], pad_value=pad_value)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("h", [18])
@pytest.mark.parametrize("w", [65])
@pytest.mark.parametrize("dim0", [1])
@pytest.mark.parametrize("dim1", [2])
def test_transpose_forge_llama(device, b, h, w, dim0, dim1):
    torch.manual_seed(2005)

    torch_input_tensor = torch_random((b, h, w), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(dim0, dim1)

    input_tensor = ttnn.to_device(ttnn.from_torch(torch_input_tensor), device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.transpose(input_tensor, dim0, dim1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("w", [3])
@pytest.mark.parametrize("dim0", [-1])
@pytest.mark.parametrize("dim1", [-2])
def test_transpose_forge_basic(device, b, h, w, dim0, dim1):
    torch.manual_seed(2005)
    torch_input_tensor = torch_random((1, b, h, w), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(dim0, dim1)
    input_tensor = ttnn.to_device(ttnn.from_torch(torch_input_tensor), device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.transpose(input_tensor, dim0, dim1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("b", [6])
@pytest.mark.parametrize("h", [33])
@pytest.mark.parametrize("w", [34])
@pytest.mark.parametrize("dim0", [1])
@pytest.mark.parametrize("dim1", [0])
def test_transpose_forge_hc(device, b, h, w, dim0, dim1):
    torch.manual_seed(2005)
    torch_input_tensor = torch_random((1, b, h, w), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(dim0, dim1)
    input_tensor = ttnn.to_device(ttnn.from_torch(torch_input_tensor), device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.transpose(input_tensor, dim0, dim1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("h", [256])
@pytest.mark.parametrize("w", [32])
def test_transpose_hw_sharded_tiled_8_cores(device, n, c, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    sharded_mem_config = ttnn.create_sharded_memory_config(
        (32, 32),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0)),
            }
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    output_sharded_mem_config = ttnn.create_sharded_memory_config(
        (32, 32),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0)),
            }
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_mem_config)

    tt_output_tensor = ttnn.transpose(tt_input_tensor, 2, 3, memory_config=output_sharded_mem_config)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9999)


@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [32])
def test_transpose_hw_sharded_tiled_n_cores(device, n, c, h, w):
    torch.manual_seed(2005)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor.transpose(2, 3)
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    sharded_mem_config = ttnn.create_sharded_memory_config(
        (32, 32),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, h // 32 - 1)),
            }
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    output_sharded_mem_config = ttnn.create_sharded_memory_config(
        (32, 32),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, h // 32 - 1)),
            }
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_mem_config)

    tt_output_tensor = ttnn.transpose(tt_input_tensor, 2, 3, memory_config=output_sharded_mem_config)
    tt_output_tensor = ttnn.to_memory_config(tt_output_tensor, ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9999)


@pytest.mark.parametrize("shape", [[16, 4, 256, 256], [16, 128, 8, 256], [1, 1, 32, 32]])
def test_transpose_hw_rm(shape, device):
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_output = torch_input.transpose(2, 3)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = ttnn.transpose(tt_input, 2, 3)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@skip_for_grayskull("Grayskull does not support float32")
def test_transpose_16411(device):
    torch.manual_seed(2005)
    input_shape = (5, 3, 1, 1, 12, 8)
    a = torch.rand(input_shape, dtype=torch.bfloat16)
    p_b2 = torch.transpose(a, 1, 3)
    p_b3 = torch.transpose(a, 1, 5)
    p_c = torch.transpose(a, 0, 4)
    p_c2 = torch.transpose(a, 1, 4)
    p_c3 = torch.transpose(a, 2, 4)
    p_c4 = torch.transpose(a, 3, 4)

    b = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    b2 = ttnn.transpose(b, 1, 3)
    b3 = ttnn.transpose(b, 1, 5)
    c = ttnn.transpose(b, 0, 4)
    c2 = ttnn.transpose(b, 1, 4)
    c3 = ttnn.transpose(b, 2, 4)
    c4 = ttnn.transpose(b, 3, 4)

    assert_with_pcc(p_b2, ttnn.to_torch(b2), 0.9999)
    assert_with_pcc(p_b3, ttnn.to_torch(b3), 0.9999)
    assert_with_pcc(p_c, ttnn.to_torch(c), 0.9999)
    assert_with_pcc(p_c2, ttnn.to_torch(c2), 0.9999)
    assert_with_pcc(p_c3, ttnn.to_torch(c3), 0.9999)
    assert_with_pcc(p_c4, ttnn.to_torch(c4), 0.9999)


@pytest.mark.parametrize("rank", [5])
@pytest.mark.parametrize("indices", [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_transpose_high_rank(*, device: ttnn.Device, rank: int, indices, layout):
    torch.manual_seed(2005)
    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    shape = [2] * rank

    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(a, device=device, layout=layout)
    tt_b = ttnn.from_torch(b, device=device, layout=layout)

    a = a.transpose(*indices)
    b = b.transpose(*indices)

    tt_a = ttnn.transpose(tt_a, *indices)
    tt_b = ttnn.transpose(tt_b, *indices)

    output_a = ttnn.to_torch(tt_a)
    output_b = ttnn.to_torch(tt_b)

    assert torch.allclose(a, output_a)
    assert torch.allclose(b, output_b)


@pytest.mark.parametrize(
    "n, c, h, w, dim0, dim1",
    [(16, 128, 128, 16, 1, 2)],
)
def test_resnet50_fold(device, n, c, h, w, dim0, dim1):
    core_grid = ttnn.CoreGrid(y=8, x=8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if core_grid.x > compute_grid_size.x or core_grid.y > compute_grid_size.y:
        pytest.skip(f"Need {core_grid} grid size to run this test but core grid is {compute_grid_size}")

    torch.manual_seed(0)
    input_shape = (n, c, h, w)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    ## WH -> HW
    torch_output = torch_input.transpose(dim0, dim1)

    mem_config = ttnn.create_sharded_memory_config(
        input_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    tt_output = ttnn.transpose(tt_input, dim0, dim1)
    tt_output = ttnn.to_torch(tt_output.cpu())

    assert_with_pcc(torch_output, tt_output, 0.9999)


def test_transpose_21803(device):
    torch.manual_seed(2005)
    # Test parameters
    dim1, dim2, dim3, dim4, dim5 = 1, 1, 8, 64, 256
    dtype = ttnn.bfloat8_b

    for i in range(100):
        torch_input = torch.randn(dim1, dim2, dim3, dim4, dim5, dtype=torch.bfloat16)
        ttnn_input = ttnn.from_torch(torch_input, dtype, layout=ttnn.Layout.TILE, device=device)
        ttnn_output = ttnn.permute(ttnn_input, [0, 1, 2, 4, 3])

        torch_output = ttnn.to_torch(ttnn_output, dtype=torch.bfloat16)

        if torch.any(torch.isnan(torch_input)) or torch.any(torch.isinf(torch_input)):
            continue

        if torch.any(torch.isinf(torch_output)) or (torch.any(torch.isnan(torch_output))):
            assert False, f"Found infinity values at iteration {i} in ttnn but not in pytorch"
