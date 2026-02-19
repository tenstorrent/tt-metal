# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn
from models.common.utility_functions import comp_allclose, skip_for_blackhole
from loguru import logger
from tests.ttnn.utils_for_testing import assert_equal, tt_dtype_to_torch_dtype


@pytest.mark.parametrize(
    "input_shape",
    [
        [],  # single tile with rank 0
        [3],  # single tile with rank 1
        [1, 3],  # single tile
        [32, 32],  # single tile
        [5, 17, 31],  # multiple tiles
        [3, 300, 1, 300],  # multiple tiles not aligned by 32
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
def test_full_int(device, input_shape, fill_value, layout):
    torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

    tt_output = ttnn.full(input_shape, fill_value, layout=layout, device=device)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape",
    [
        [],  # single tile with rank 0
        [3],  # single tile with rank 1
        [1, 3],  # single tile
        [32, 32],  # single tile
        [5, 96, 64],  # multiple tiles
        [3, 300, 1, 300],  # multiple tiles not aligned by 32
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
        1e-8,  # Casted to different values in fp32->bf16 truncate vs round
    ],
)
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_full_float(device, input_shape, fill_value, tt_dtype, layout):
    torch_dtype = tt_dtype_to_torch_dtype[tt_dtype]
    torch_output = torch.full(input_shape, fill_value, dtype=torch_dtype)

    tt_output = ttnn.full(input_shape, fill_value, dtype=tt_dtype, layout=layout, device=device)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


####################################################################################################
# moreh_full device operation tests: interleaved, ND sharded, legacy sharded
####################################################################################################


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
        [3, 160, 160],
        [2, 4, 128, 160],
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
def test_moreh_full_int(device, input_shape, fill_value, layout):
    torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

    tt_output = ttnn.moreh_full(input_shape, fill_value, device, dtype=ttnn.int32, layout=layout)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
        [3, 160, 160],
        [2, 4, 128, 160],
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
    "tt_dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_moreh_full_float(device, input_shape, fill_value, tt_dtype, layout):
    torch_dtype = tt_dtype_to_torch_dtype[tt_dtype]
    torch_output = torch.full(input_shape, fill_value, dtype=torch_dtype)

    tt_output = ttnn.moreh_full(input_shape, fill_value, device, dtype=tt_dtype, layout=layout)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


####################################################################################################
# ND sharded tests
####################################################################################################


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
def test_moreh_full_int_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

    tt_output = ttnn.moreh_full(
        input_shape, fill_value, device, dtype=ttnn.int32, layout=layout, memory_config=sharded_mem_config
    )
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
def test_moreh_full_float_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, dtype, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_dtype = tt_dtype_to_torch_dtype[ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32]
    tt_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    torch_output = torch.full(input_shape, fill_value, dtype=torch_dtype)

    tt_output = ttnn.moreh_full(
        input_shape, fill_value, device, dtype=tt_dtype, layout=layout, memory_config=sharded_mem_config
    )
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
def test_moreh_full_callback_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    for i in range(2):
        torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

        tt_output = ttnn.moreh_full(
            input_shape, fill_value, device, dtype=ttnn.int32, layout=layout, memory_config=sharded_mem_config
        )
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
def test_moreh_full_float_DRAM_nd_sharded(
    device, input_shape, nd_shard_shape, fill_value, dtype, layout, shard_orientation, shard_core_grid
):
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=shard_core_grid, orientation=shard_orientation
    )
    sharded_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    torch_dtype = tt_dtype_to_torch_dtype[ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32]
    tt_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    torch_output = torch.full(input_shape, fill_value, dtype=torch_dtype)

    tt_output = ttnn.moreh_full(
        input_shape, fill_value, device, dtype=tt_dtype, layout=layout, memory_config=sharded_mem_config
    )
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


####################################################################################################
# Legacy 2D sharded tests
####################################################################################################


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
def test_moreh_full_int_legacy_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

    tt_output = ttnn.moreh_full(
        input_shape, fill_value, device, dtype=ttnn.int32, layout=layout, memory_config=sharded_mem_config
    )
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
def test_moreh_full_float_legacy_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, dtype, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    torch_dtype = tt_dtype_to_torch_dtype[ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32]
    tt_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    torch_output = torch.full(input_shape, fill_value, dtype=torch_dtype)

    tt_output = ttnn.moreh_full(
        input_shape, fill_value, device, dtype=tt_dtype, layout=layout, memory_config=sharded_mem_config
    )
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape, shard_shape, memory_layout, shard_grid",
    [
        # Height sharded, uneven, DRAM (1D grid required for DRAM)
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
def test_moreh_full_int_legacy_DRAM_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.DRAM, shard_spec)

    torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

    tt_output = ttnn.moreh_full(
        input_shape, fill_value, device, dtype=ttnn.int32, layout=layout, memory_config=sharded_mem_config
    )
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
def test_moreh_full_callback_legacy_sharded(
    device, input_shape, shard_shape, memory_layout, shard_grid, fill_value, layout, shard_orientation
):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    for i in range(2):
        torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

        tt_output = ttnn.moreh_full(
            input_shape, fill_value, device, dtype=ttnn.int32, layout=layout, memory_config=sharded_mem_config
        )
        assert ttnn.is_tensor_storage_on_device(tt_output)
        tt_output_cpu = ttnn.to_torch(tt_output)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
    assert torch.equal(torch_output, tt_output_cpu)
