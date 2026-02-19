# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import copy
import torch
import torch.nn as nn
import ttnn
from models.common.utility_functions import comp_allclose
from loguru import logger

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],  # single tile
        [5, 96, 64],  # multiple tiles
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [3, -1],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_full_like_int(device, input_shape, fill_value, layout):
    torch_input = torch.randint(0, 100, (input_shape), dtype=torch.int32)
    torch_output = torch.full_like(torch_input, fill_value, dtype=torch.int32)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],  # single tile
        [5, 96, 64],  # multiple tiles
        [3, 91, 67, 77],  # not multiple of 32
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3.14,
        2.00781250,  # mantissa: 0000 0001, bf16 round down test
        2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
        2.02343750,  # mantissa: 0000 0011, bf16 round up test
        -3.9921875,  # test mantissa overflow. answer should be 4
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_full_like_float(device, input_shape, fill_value, dtype, layout):
    torch_input = torch.rand((input_shape), dtype=dtype)
    torch_output = torch.full_like(torch_input, fill_value, dtype=dtype)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],  # single tile
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [3],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_full_like_callback(device, input_shape, fill_value, layout):
    for i in range(2):
        torch_input = torch.randint(0, 100, (input_shape), dtype=torch.int32)
        torch_output = torch.full_like(torch_input, fill_value)

        tt_input = ttnn.from_torch(torch_input, layout=layout, device=device)
        tt_output = ttnn.moreh_full_like(tt_input, fill_value)
        assert ttnn.is_tensor_storage_on_device(tt_output)
        tt_output_cpu = ttnn.to_torch(tt_output)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, nd_shard_shape",
    [
        ([32, 32], [32, 32]),
        ([5, 96, 64], [1, 32, 32]),
        ([3, 160, 160], [2, 64, 64]),
        ([2, 4, 128, 160], [2, 3, 96, 96]),
    ],
)
@pytest.mark.parametrize("fill_value", [3, -1])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 2))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
    ],
)
def test_full_like_int_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    torch_output = torch.full_like(torch_input, fill_value, dtype=torch.int32)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, nd_shard_shape",
    [
        ([32, 32], [32, 32]),
        ([5, 96, 64], [1, 32, 32]),
        ([3, 160, 160], [2, 64, 64]),
        ([2, 4, 128, 160], [2, 3, 96, 96]),
        ([7, 67, 77], [3, 32, 64]),  # not multiple of 32
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3.14,
        2.00781250,  # mantissa: 0000 0001, bf16 round down test
        2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
        2.02343750,  # mantissa: 0000 0011, bf16 round up test
        -3.9921875,  # test mantissa overflow. answer should be 4
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 2))}),
    ],
)
def test_full_like_float_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, dtype, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_input = torch.rand(input_shape, dtype=dtype)
    torch_output = torch.full_like(torch_input, fill_value, dtype=dtype)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, nd_shard_shape",
    [
        ([4, 128, 128], [3, 96, 96]),
    ],
)
@pytest.mark.parametrize("fill_value", [3])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
    ],
)
def test_full_like_callback_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    for i in range(2):
        torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
        torch_output = torch.full_like(torch_input, fill_value)

        tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
        tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
        assert ttnn.is_tensor_storage_on_device(tt_output)
        tt_output_cpu = ttnn.to_torch(tt_output)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, nd_shard_shape",
    [
        ([3, 160, 160], [3, 160, 64]),
        ([3, 160, 160], [2, 64, 64]),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3.14,
        2.00781250,  # mantissa: 0000 0001, bf16 round down test
        2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
        2.02343750,  # mantissa: 0000 0011, bf16 round up test
        -3.9921875,  # test mantissa overflow. answer should be 4
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
    ],
)
def test_full_like_float_DRAM_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, dtype, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    torch_input = torch.rand(input_shape, dtype=dtype)
    torch_output = torch.full_like(torch_input, fill_value, dtype=dtype)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, shard_shape, memory_layout, shard_grid",
    [
        # Height sharded
        (
            [4, 128, 64],
            [128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        (
            [2, 64, 64],
            [64, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # Width sharded
        (
            [2, 64, 128],
            [128, 32],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Block sharded (shard_shape must be [height/grid.y, width/grid.x])
        (
            [4, 128, 128],
            [256, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [2, 128, 128],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # Block sharded, uneven (shard coverage > tensor size, last shards have padding)
        (
            [2, 96, 96],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize("fill_value", [3, -1])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_full_like_int_legacy_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    torch_output = torch.full_like(torch_input, fill_value, dtype=torch.int32)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, shard_shape, memory_layout, shard_grid",
    [
        # Height sharded
        (
            [4, 128, 64],
            [128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        (
            [2, 64, 64],
            [64, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # Width sharded
        (
            [2, 64, 128],
            [128, 32],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Block sharded
        (
            [4, 128, 128],
            [256, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [2, 128, 128],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        # Block sharded, uneven
        (
            [2, 96, 96],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3.14,
        2.00781250,  # mantissa: 0000 0001, bf16 round down test
        2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
        2.02343750,  # mantissa: 0000 0011, bf16 round up test
        -3.9921875,  # test mantissa overflow. answer should be 4
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_full_like_float_legacy_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, dtype, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    torch_input = torch.rand(input_shape, dtype=dtype)
    torch_output = torch.full_like(torch_input, fill_value, dtype=dtype)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, shard_shape, memory_layout, shard_grid",
    [
        # Height sharded, uneven, DRAM (1D grid required for DRAM)
        # height=192, shard_height=128, 2 cores: 128*2=256 > 192, last core has 64 valid rows
        (
            [2, 96, 96],
            [128, 96],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("fill_value", [3, -1])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_full_like_int_legacy_DRAM_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.DRAM, shard_spec)

    torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
    torch_output = torch.full_like(torch_input, fill_value, dtype=torch.int32)

    tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
    tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, shard_shape, memory_layout, shard_grid",
    [
        # Height sharded
        (
            [4, 128, 64],
            [128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Width sharded
        (
            [2, 64, 128],
            [128, 32],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # Block sharded
        (
            [2, 128, 128],
            [128, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize("fill_value", [3])
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR],
)
def test_full_like_callback_legacy_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    for i in range(2):
        torch_input = torch.randint(0, 100, input_shape, dtype=torch.int32)
        torch_output = torch.full_like(torch_input, fill_value)

        tt_input = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=sharded_mem_config)
        tt_output = ttnn.moreh_full_like(tt_input, fill_value, memory_config=sharded_mem_config)
        assert ttnn.is_tensor_storage_on_device(tt_output)
        tt_output_cpu = ttnn.to_torch(tt_output)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
    assert torch.equal(torch_output, tt_output_cpu)
