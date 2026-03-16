# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math
import pytest

from tests.sweep_framework.sweep_utils.max_pool2d_with_indices_common import run_max_pool2d_with_indices


@pytest.mark.parametrize("in_c", [1, 16, 24, 32, 40, 48, 56, 64])
def test_mpwi_20_core_C_dims(device, in_c):
    in_n = 1
    in_h = 159
    in_w = 159
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
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
            [1280, shard_width],
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
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize(
    "input_spec",
    [
        # Contains following parameters
        # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
        # DILATION / MULTI-BATCH CASES
        [2, 16, 130, 130, 2, 2, 1, 1, 1, 1, 1, 1, False],
        [3, 16, 80, 80, 2, 2, 1, 1, 1, 1, 1, 1, False],
        [4, 16, 50, 60, 2, 2, 1, 1, 1, 1, 1, 1, False],
        [2, 48, 120, 120, 4, 4, 1, 1, 2, 2, 1, 1, False],
        [3, 48, 70, 70, 4, 4, 1, 1, 2, 2, 1, 1, False],
        [4, 48, 40, 50, 4, 4, 1, 1, 2, 2, 1, 1, False],
        [2, 64, 110, 110, 4, 8, 1, 1, 2, 4, 1, 1, False],
        [3, 64, 60, 60, 4, 8, 1, 1, 2, 4, 1, 1, False],
        [4, 64, 30, 40, 4, 8, 1, 1, 2, 4, 1, 1, False],
    ],
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_mpwi_small_kernel_sizes(device, ttnn_dtype, input_spec):
    (
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
        ceil_mode,
    ) = input_spec

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
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        memory_config=None,
        run_twice=True,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize(
    "input_spec",
    [
        # Contains following parameters
        # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
        [2, 64, 159, 159, 13, 13, 2, 2, 6, 6, 2, 2, True],
        [2, 40, 100, 100, 9, 9, 2, 2, 0, 1, 2, 2, True],
        [3, 56, 85, 85, 8, 8, 3, 3, 1, 0, 2, 2, False],
        [4, 24, 56, 64, 6, 6, 2, 1, 1, 1, 3, 2, True],
        [2, 72, 100, 225, 2, 64, 2, 2, 0, 1, 2, 2, True],
        [3, 64, 85, 180, 2, 48, 3, 3, 1, 0, 2, 2, False],
        [4, 16, 56, 140, 3, 32, 2, 1, 1, 1, 3, 2, True],
        [4, 32, 60, 140, 4, 24, 1, 2, 2, 4, 3, 2, True],
    ],
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_mpwi_large_kernel_sizes(device, ttnn_dtype, input_spec):
    (
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
        ceil_mode,
    ) = input_spec

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
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        memory_config=None,
        run_twice=True,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize(
    "input_spec",
    [
        # Contains following parameters
        # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
        [3, 16, 80, 80, 9, 9, 3, 3, 3, 1, 2, 2, True],
        [2, 48, 60, 60, 6, 6, 2, 2, 2, 0, 2, 2, False],
        [4, 56, 65, 55, 5, 5, 1, 2, 1, 1, 1, 2, False],
        [4, 24, 56, 64, 3, 3, 2, 1, 0, 1, 3, 2, True],
    ],
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "sharding_scheme",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_mpwi_general(device, ttnn_dtype, sharding_scheme, input_spec):
    (
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
        ceil_mode,
    ) = input_spec

    if sharding_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED and ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip("this case runs OOM")

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
        sharding=sharding_scheme,
        ceil_mode=ceil_mode,
        memory_config=None,
        run_twice=True,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize(
    "input_spec",
    [
        [1, 32, 384, 384, 3, 3, 1, 1, 1, 1, 1, 1, False],
        [1, 48, 350, 350, 5, 5, 1, 1, 2, 2, 1, 1, False],
        [1, 64, 350, 350, 6, 6, 1, 1, 3, 3, 1, 1, False],
        [3, 32, 300, 300, 7, 7, 1, 1, 3, 3, 1, 1, False],
        [2, 48, 300, 300, 9, 9, 2, 2, 4, 4, 1, 1, False],
    ],
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16])
def test_mpwi_32_bit_index(device, ttnn_dtype, input_spec):
    (
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
        ceil_mode,
    ) = input_spec

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
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        memory_config=None,
        run_twice=True,
        config_tensor_in_dram=True,
    )
