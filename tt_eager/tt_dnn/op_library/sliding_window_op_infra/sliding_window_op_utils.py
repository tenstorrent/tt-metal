# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections import namedtuple

import tt_lib as ttl
from tt_lib.utils import _nearest_y

SlidingWindowOpParams = namedtuple(
    "SlidingWindowOpParams", "stride_h stride_w pad_h pad_w window_h window_w batch_size input_h input_w"
)
SlidingWindowOpParamsWithParallelConfig = namedtuple(
    "SlidingWindowOpParamsWithParallelConfig",
    "stride_h stride_w pad_h pad_w window_h window_w batch_size input_h input_w num_cores_w num_cores_h num_cores_nhw act_reshard_num_cores_nhw",
    defaults=[0],
)


def get_hash_from_sliding_window_op_params(sliding_window_op_params: SlidingWindowOpParamsWithParallelConfig):
    return f"{sliding_window_op_params.stride_h}_{sliding_window_op_params.stride_w}_{sliding_window_op_params.pad_h}_{sliding_window_op_params.pad_w}_{sliding_window_op_params.window_h}_{sliding_window_op_params.window_w}_{sliding_window_op_params.batch_size}_{sliding_window_op_params.input_h}_{sliding_window_op_params.input_w}_{sliding_window_op_params.num_cores_w}_{sliding_window_op_params.num_cores_h}_{sliding_window_op_params.num_cores_nhw}_{sliding_window_op_params.act_reshard_num_cores_nhw}"


def get_sliding_window_op_output_nhw_shape(
    input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
):
    output_h = ((int)((input_h + (2 * pad_h) - window_h) / stride_h)) + 1
    output_w = ((int)((input_w + (2 * pad_w) - window_w) / stride_w)) + 1
    return [input_n, output_h, output_w]


def get_sliding_window_op_output_shard_nhw_size(
    num_cores_nhw, input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w, is_out_tiled=True
):
    output_nhw_shape = get_sliding_window_op_output_nhw_shape(
        input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
    )
    if is_out_tiled:
        output_nhw_size_to_shard_evenly = _nearest_y(np.prod(output_nhw_shape), num_cores_nhw * 32)
    else:
        output_nhw_size_to_shard_evenly = _nearest_y(np.prod(output_nhw_shape), num_cores_nhw)
    output_shard_nhw_size = (int)(output_nhw_size_to_shard_evenly / num_cores_nhw)
    return output_shard_nhw_size


def calculate_shard_grid(grid_size, num_cores_nhw):
    num_cores_w, num_cores_h = grid_size
    shard_layout = (
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED
        if (num_cores_nhw == num_cores_w and num_cores_h > 1)
        else ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    )

    if shard_layout == ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED:
        core_range = ttl.tensor.CoreRange(
            ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(num_cores_w - 1, num_cores_h - 1)
        )
        shard_grid = ttl.tensor.CoreRangeSet({core_range})
    else:
        if num_cores_nhw >= num_cores_w:
            num_cores_height_excluding_remainder_last_row = num_cores_nhw // num_cores_w
            assert num_cores_h >= num_cores_height_excluding_remainder_last_row
            core_range_1 = ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(num_cores_w - 1, num_cores_height_excluding_remainder_last_row - 1),
            )
            num_cores_last = num_cores_nhw % num_cores_w
            if num_cores_last > 0:
                assert num_cores_h == num_cores_height_excluding_remainder_last_row + 1
                core_range_2 = ttl.tensor.CoreRange(
                    ttl.tensor.CoreCoord(0, num_cores_height_excluding_remainder_last_row),
                    ttl.tensor.CoreCoord(num_cores_last - 1, num_cores_height_excluding_remainder_last_row),
                )
                shard_grid = ttl.tensor.CoreRangeSet({core_range_1, core_range_2})
            else:
                assert num_cores_h == num_cores_height_excluding_remainder_last_row
                shard_grid = ttl.tensor.CoreRangeSet({core_range_1})
        else:
            core_range_1 = ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(num_cores_nhw - 1, 0),
            )
            shard_grid = ttl.tensor.CoreRangeSet({core_range_1})
    return shard_grid, shard_layout


class SWOParallelConfig:
    config_keys = ["num_cores", "grid_size", "shard_layout"]

    def __init__(
        self, num_cores=1, grid_size=(1, 1), shard_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    ) -> None:
        self.num_cores = num_cores
        self.grid_size = grid_size
        self.shard_layout = shard_layout
