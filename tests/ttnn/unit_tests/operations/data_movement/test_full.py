# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest

import torch
import torch.nn as nn
import ttnn
from models.common.utility_functions import comp_allclose, skip_for_blackhole
from loguru import logger
from tests.ttnn.utils_for_testing import assert_equal, tt_dtype_to_torch_dtype

FILL_FLOAT_VALUES = [
    3.14,
    2.00781250,  # mantissa: 0000 0001, bf16 round down test
    2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
    2.02343750,  # mantissa: 0000 0011, bf16 round up test
    -3.9921875,  # test mantissa overflow. answer should be 4
    1e-8,  # Casted to different values in fp32->bf16 truncate vs round
]

FILL_SHAPE_VALUES = [
    [],  # single tile with rank 0
    [3],  # single tile with rank 1
    [1, 3],  # single tile
    [32, 32],  # single tile
    [5, 17, 31],  # multiple tiles
    [3, 300, 1, 300],  # multiple tiles not aligned by 32
]


@pytest.mark.parametrize("input_shape", FILL_SHAPE_VALUES)
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
@pytest.mark.parametrize("fill_value", FILL_FLOAT_VALUES)
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


@pytest.mark.parametrize("input_shape", [[32, 32], [3, 300, 1, 300]])
@pytest.mark.parametrize("fill_value", [0.1, 1 / 3, 65504.0, 1e30, -1e-30, float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_full_tile_device_matches_host(device, input_shape, fill_value, tt_dtype, memory_config):
    # Whichever path fills it, a TILE ttnn.full on a device must hold the same values as the tensor built on the host
    host_output = ttnn.to_torch(ttnn.full(input_shape, fill_value, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT))
    tt_output = ttnn.full(
        input_shape, fill_value, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    assert ttnn.is_tensor_storage_on_device(tt_output)
    device_output = ttnn.to_torch(tt_output)

    assert device_output.shape == host_output.shape
    assert torch.equal(torch.isnan(device_output), torch.isnan(host_output))
    not_nan = ~torch.isnan(host_output)
    assert torch.equal(device_output[not_nan], host_output[not_nan])


# Mirrors k_min_device_fill_bytes in ttnn/cpp/ttnn/operations/creation/creation.cpp
MIN_DEVICE_FILL_BYTES = 128 * 1024
TILE_BYTES = {ttnn.bfloat16: 32 * 32 * 2, ttnn.float32: 32 * 32 * 4}


# One tile below and exactly at the threshold for float32 (32 tiles) and bfloat16 (64 tiles), plus shapes whose
# logical size is under the threshold while their tile-padded size is not
@pytest.mark.parametrize(
    "input_shape", [[32, 31 * 32], [32, 32 * 32], [31, 32 * 32], [32, 63 * 32], [32, 64 * 32], [32, 63 * 32 + 1]]
)
@pytest.mark.parametrize("fill_value", [1.0, float("nan")])
@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.float32])
def test_full_tile_device_fill_threshold(device, input_shape, fill_value, tt_dtype):
    # From MIN_DEVICE_FILL_BYTES of tile-padded data a TILE ttnn.full is filled on the device by one program; a smaller
    # one, or a bfloat16 NaN fill, keeps the host fill and upload. Either way it holds the values the host path builds.
    device.clear_program_cache()
    programs_before = device.num_program_cache_entries()
    tt_output = ttnn.full(input_shape, fill_value, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    programs_added = device.num_program_cache_entries() - programs_before

    tiles = math.prod(input_shape[:-2]) * math.ceil(input_shape[-2] / 32) * math.ceil(input_shape[-1] / 32)
    device_fill = tiles * TILE_BYTES[tt_dtype] >= MIN_DEVICE_FILL_BYTES and not (
        tt_dtype == ttnn.bfloat16 and math.isnan(fill_value)
    )
    assert programs_added == (1 if device_fill else 0)

    host_output = ttnn.to_torch(ttnn.full(input_shape, fill_value, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT))
    device_output = ttnn.to_torch(tt_output)
    assert torch.equal(torch.isnan(device_output), torch.isnan(host_output))
    not_nan = ~torch.isnan(host_output)
    assert torch.equal(device_output[not_nan], host_output[not_nan])


# TODO (issue #16579): Add program cache test when ttnn.full is run on device


@pytest.mark.parametrize("fill_value", FILL_FLOAT_VALUES)
@pytest.mark.parametrize("tt_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape",
    [[2, 2, 256, 512], [32, 32]],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "buffer_type,shard_core_grid",
    [
        (ttnn.BufferType.L1, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})),
        (
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
        ),
        (ttnn.BufferType.L1, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})),
        (ttnn.BufferType.L1, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})),
        (
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 0)),
                }
            ),
        ),
    ],
)
def test_full_nd_sharded(
    device,
    tensor_shape,
    fill_value,
    tt_dtype,
    shard_orientation,
    shard_core_grid,
    buffer_type,
    tensor_layout,
):
    torch_dtype = tt_dtype_to_torch_dtype[tt_dtype]
    torch_output = torch.full(tensor_shape, fill_value, dtype=torch_dtype)
    shard_dims = list(range(len(tensor_shape) - 2, len(tensor_shape)))

    tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=tt_dtype,
        layout=tensor_layout,
        buffer_type=buffer_type,
    ).sharded_across_dims(
        shard_dims,
        shard_core_grid,
        shard_orientation,
    )

    assert tensor_spec.memory_config.nd_shard_spec is not None

    tt_output = ttnn.full(
        tensor_shape,
        fill_value,
        dtype=tt_dtype,
        layout=tensor_spec.layout,
        device=device,
        memory_config=tensor_spec.memory_config,
    )

    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize("fill_value", FILL_FLOAT_VALUES)
@pytest.mark.parametrize("tt_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "tensor_layout,tensor_shape,shard_shape,buffer_type,shard_core_grid",
    [
        (
            ttnn.TILE_LAYOUT,
            [2, 2, 256, 512],
            [32, 32],
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            ttnn.ROW_MAJOR_LAYOUT,
            [2, 2, 256, 512],
            [32, 32],
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            ttnn.TILE_LAYOUT,
            [2, 2, 256, 512],
            [32, 32],
            ttnn.BufferType.L1,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            ttnn.TILE_LAYOUT,
            [64, 64, 64],
            [16, 32, 32],
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
        ),
        (
            ttnn.ROW_MAJOR_LAYOUT,
            [64, 64, 64],
            [16, 16, 16],
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
        ),
        (
            ttnn.TILE_LAYOUT,
            [8, 8, 8, 8, 8, 8, 8],
            [4, 32, 32],
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
        ),
        (
            ttnn.ROW_MAJOR_LAYOUT,
            [8, 8, 8, 8, 8, 8, 8],
            [4, 4, 4],
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
        ),
        (
            ttnn.TILE_LAYOUT,
            [17, 19, 41, 3, 44],
            [10, 32, 32],
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
        ),
        (
            # This test will create more pages than available in the card
            ttnn.TILE_LAYOUT,
            [3, 300, 1, 300],
            [32, 32],
            ttnn.BufferType.DRAM,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
        ),
    ],
)
def test_full_nd_sharded_manual_sharding(
    device,
    tensor_shape,
    shard_shape,
    fill_value,
    tt_dtype,
    shard_orientation,
    shard_core_grid,
    buffer_type,
    tensor_layout,
):
    torch_dtype = tt_dtype_to_torch_dtype[tt_dtype]
    torch_output = torch.full(tensor_shape, fill_value, dtype=torch_dtype)

    memory_config = ttnn.MemoryConfig(buffer_type, ttnn.NdShardSpec(shard_shape, shard_core_grid, shard_orientation))

    tt_output = ttnn.full(
        tensor_shape,
        fill_value,
        dtype=tt_dtype,
        layout=tensor_layout,
        device=device,
        memory_config=memory_config,
    )

    assert tt_output.memory_config().nd_shard_spec is not None
    assert tt_output.memory_config().is_sharded()
    assert ttnn.is_tensor_storage_on_device(tt_output)

    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)
