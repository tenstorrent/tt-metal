# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math
import pytest

from tests.sweep_framework.sweep_utils.max_pool2d_with_indices_common import run_max_pool2d_with_indices


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("in_c", [32])
def test_mpwi_20_core_C_dims(device, in_c):
    in_n = 1
    in_h = 3
    in_w = 3
    kernel_size = [3, 3]
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    ceil_mode = False
    ttnn_dtype = ttnn.bfloat16
    # ttnn_dtype = ttnn.bfloat8_b

    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    # shield team memory config
    shard_width = math.ceil(in_c / 32) * 32
    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [in_n * in_h * in_w, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    run_max_pool2d_with_indices(
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ttnn_dtype,
        device,
        None,  # None means auto sharding
        ceil_mode,
        memory_config,
        False,  # not in place
    )


# @pytest.mark.parametrize(
#     "input_spec",
#     [
#         # Contains following parameters
#         # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
#         # DILATION / MULTI-BATCH CASES
#         [2, 40, 100, 100, 3, 3, 2, 2, 0, 1, 2, 2, True],
#         [3, 56, 85, 85, 3, 3, 3, 3, 1, 0, 2, 2, False],
#         [4, 24, 56, 64, 3, 3, 2, 1, 1, 1, 3, 2, True],
#     ],
# )
# @pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# def test_mpwi_general(device, ttnn_dtype, input_spec):
#     (
#         in_n,
#         in_c,
#         in_h,
#         in_w,
#         kernel_h,
#         kernel_w,
#         stride_h,
#         stride_w,
#         pad_h,
#         pad_w,
#         dilation_h,
#         dilation_w,
#         ceil_mode,
#     ) = input_spec

#     run_max_pool2d_with_indices(
#         in_n,
#         in_c,
#         in_h,
#         in_w,
#         kernel_h,
#         kernel_w,
#         stride_h,
#         stride_w,
#         pad_h,
#         pad_w,
#         dilation_h,
#         dilation_w,
#         ttnn_dtype,
#         device,
#         ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#         ceil_mode,
#         None,  # no memory_config
#         False,  # not in place
#     )
