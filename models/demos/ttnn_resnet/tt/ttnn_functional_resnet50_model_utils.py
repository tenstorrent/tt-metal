# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.utility_functions import nearest_y


def get_core_grid_from_num_cores(num_cores: int, grid_rows: int, grid_cols: int):
    columns = num_cores // grid_rows
    assert columns <= grid_cols, "Not enough cores for specified core grid"
    ranges = []
    if columns != 0:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_rows - 1, columns - 1),
            )
        )
    remainder = num_cores % grid_rows
    if remainder != 0:
        assert columns + 1 <= grid_cols, "Not enough cores for specified core grid"
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, columns),
                ttnn.CoreCoord(remainder - 1, columns),
            )
        )
    return ttnn.CoreRangeSet({*ranges})


def find_closest_largest_divisor(num: int, start_divisor: int) -> int:
    divisor = start_divisor
    while num % divisor != 0:
        divisor -= 1
    return divisor


# Determins input memory config for a height sharded conv operation.
# If override_num_cores is set to True, the number of cores will be overriden to the closest largest divisor of the number of tiles
# This will avoid default conv codepath which can pad-up the nhw num tiles and produce padded output
# This can lead to issues with data-movment ops not handling padding correctly
def get_conv_input_memory_config(
    batch_size: int,
    input_channels: int,
    input_height: int,
    input_width: int,
    output_channels: int,
    output_height: int,
    output_width: int,
    compute_grid: ttnn.CoreGrid,
    input_channels_alignment: int,
    override_num_cores: bool,
) -> ttnn.MemoryConfig:
    parallel_config = ttnn._ttnn.operations.conv.determine_parallel_config(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        batch_size=batch_size,
        input_channels=input_channels,
        output_height=output_height,
        output_width=output_width,
        output_channels=output_channels,
        compute_grid_size=compute_grid,
        block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        enable_channels_padding=True,
    )

    if override_num_cores:
        nhw_ntiles = math.ceil(batch_size * output_height * output_width / 32)
        num_cores_nwh = find_closest_largest_divisor(nhw_ntiles, compute_grid.x * compute_grid.y)
        parallel_config.grid = get_core_grid_from_num_cores(num_cores_nwh, compute_grid.x, compute_grid.y)

    memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
        tensor_shape=ttnn.Shape(
            [
                1,
                1,
                input_width * input_height * batch_size,
                nearest_y(
                    input_channels,
                    input_channels_alignment,
                ),
            ]
        ),
        parallel_config=parallel_config,
        tile_size=32,
    )
    return memory_config
