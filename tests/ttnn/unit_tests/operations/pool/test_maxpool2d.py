# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool

parameters = {
    "height_shard_tests": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [in_n, in_c, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [1, 128, 150, 150, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 16, 25, 23, 2, 2, 2, 2, 0, 0, 1, 1, False],  # C=16
            [1, 480, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, True],
            [1, 7, 24, 24, 3, 3, 1, 1, 0, 0, 2, 2, False],  # dilation, C = 7
            [1, 1, 59, 59, 3, 5, 4, 2, 1, 1, 5, 4, True],  # dilation with ceil mode, C = 1
            [1, 64, 400, 544, 3, 3, 2, 2, 1, 1, 1, 1, False],  # massive NHW
            [1, 832, 14, 14, 4, 4, 2, 2, 0, 0, 1, 1, True],  # > 800 channels, 16 kernel
            [1, 160, 30, 30, 15, 15, 1, 1, 7, 5, 1, 1, False],  # 15x15 kernel, uneven padding
            [1, 224, 20, 20, 8, 8, 6, 6, 2, 4, 1, 1, False],  # 8x8 kernel, uneven padding
            [1, 320, 48, 48, 36, 36, 1, 1, 0, 0, 1, 1, False],  # massive kernel, wide
            [1, 290, 47, 47, 36, 36, 1, 1, 0, 0, 1, 1, False],  # non-tile multiple NHW
            [1, 320, 48, 48, 36, 36, 1, 1, 0, 0, 1, 1, True],  # massive kernel, wide, ceil mode
            [1, 290, 47, 47, 36, 36, 1, 1, 0, 0, 1, 1, True],  # non-tile multiple NHW, ceil mode
            [1, 32, 6, 6, 3, 3, 1, 1, 1, 1, 1, 1, False],  # partial grid on WH to use noop cores
            [1, 32, 13, 8, 4, 3, 6, 5, 2, 1, 1, 1, True],  # ceil mode output shape adjustment edge case
            # requires reversed local reads on some cores, and forward reads on others
            [8, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, True],
            # requires reversed local reads on some cores, and forward reads on others, large kernel
            [32, 32, 264, 40, 5, 5, 2, 2, 2, 2, 1, 1, True],
        ],
    },
    "width_shard_tests": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            [1, 32768, 6, 6, 2, 2, 1, 1, 0, 0, 1, 1, False],  # wide in place untilize
            [1, 16384, 8, 8, 2, 2, 1, 1, 0, 0, 1, 1, False],  # normal in place untilize
            [1, 6144, 20, 20, 11, 11, 1, 1, 5, 5, 1, 1, False],  # 11x11 kernel
        ],
    },
    "block_shard_tests": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            [1, 4096, 16, 16, 2, 2, 1, 1, 0, 0, 1, 1, False],  # wide in place untilize
            [1, 2048, 16, 16, 2, 2, 1, 1, 0, 0, 1, 1, False],  # normal in place untilize
            # requires reversed local reads on some cores, and forward reads on others, wide in place untilize, large kernel
            [1, 4096, 16, 16, 5, 5, 2, 2, 2, 2, 1, 1, True],
            # requires reversed local reads on some cores, and forward reads on others, normal in place untilize, large kernel
            [1, 2048, 16, 16, 5, 5, 2, 2, 2, 2, 1, 1, True],
            [1, 512, 25, 25, 12, 12, 1, 1, 6, 6, 1, 1, False],  # 12x12 kernel
        ],
    },
    "mem_config_tests": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            [1, 32, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
}


@pytest.mark.parametrize("input_spec", parameters["height_shard_tests"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["height_shard_tests"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["height_shard_tests"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_height_shard(device, dtype, in_place, input_spec):
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
    if kernel_h == 36 and in_place:
        pytest.skip("36x36 kernel in place runs out of memory")

    torch_tensor_map = {}
    run_max_pool(
        [in_n, in_c, in_h, in_w],
        [kernel_h, kernel_w],
        [pad_h, pad_w],
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        device,
        torch_tensor_map,
        dtype,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
        nightly_skips=False,
    )


@pytest.mark.parametrize("input_spec", parameters["width_shard_tests"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["width_shard_tests"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["width_shard_tests"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_width_shard(device, dtype, in_place, input_spec):
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

    torch_tensor_map = {}
    run_max_pool(
        [in_n, in_c, in_h, in_w],
        [kernel_h, kernel_w],
        [pad_h, pad_w],
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        device,
        torch_tensor_map,
        dtype,
        shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
        nightly_skips=False,
    )


@pytest.mark.parametrize("input_spec", parameters["block_shard_tests"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["block_shard_tests"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["block_shard_tests"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_block_shard(device, dtype, in_place, input_spec):
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

    torch_tensor_map = {}
    run_max_pool(
        [in_n, in_c, in_h, in_w],
        [kernel_h, kernel_w],
        [pad_h, pad_w],
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        device,
        torch_tensor_map,
        dtype,
        shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
        nightly_skips=False,
    )


@pytest.mark.parametrize("input_spec", parameters["mem_config_tests"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["mem_config_tests"]["dtype"])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_mem_config(device, dtype, input_spec, memory_config):
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

    torch_tensor_map = {}
    run_max_pool(
        [in_n, in_c, in_h, in_w],
        [kernel_h, kernel_w],
        [pad_h, pad_w],
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        device,
        torch_tensor_map,
        dtype,
        memory_config=memory_config,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        in_place=False,
        nightly_skips=False,
    )
