# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import itertools
import random
import torch
import pytest
import math
from typing import Optional, Tuple, List

from models.utility_functions import is_wormhole_b0, is_grayskull, is_x2_harvested, torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d, mesh_device_fixture
from models.utility_functions import is_blackhole

import ttnn

parameters = {
    "max_pool2d_short_sweep_suite": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [1, 128, 112, 112, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 128, 150, 150, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 128, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 128, 64, 64, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 16, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 192, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 192, 56, 56, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 256, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 75, 75, 2, 2, 2, 2, 0, 0, 1, 1, True],
            [1, 32, 256, 256, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 320, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 4, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],  # requires padding along C
            [1, 480, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 480, 28, 28, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 512, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 512, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 512, 19, 19, 3, 3, 1, 1, 1, 1, 1, 1, False],
            [1, 512, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 512, 38, 38, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 528, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],  # required padding along C
            [1, 64, 112, 112, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 64, 128, 128, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 64, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 64, 24, 24, 2, 2, 1, 1, 0, 0, 1, 1, False],
            [1, 64, 300, 300, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 64, 360, 640, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 64, 400, 544, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 640, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 832, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, True],
            [1, 832, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 96, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 256, 20, 20, 8, 8, 6, 6, 0, 0, 1, 1, False],  # max rows per reduction multiple large kernel
            [1, 512, 20, 20, 8, 8, 6, 6, 0, 0, 1, 1, False],  # max rows per reduction multiple large kernel wide
            [1, 320, 48, 48, 36, 36, 1, 1, 0, 0, 1, 1, False],  # 3 reduction stages, multiple indexes per core, wide
            [1, 320, 47, 47, 36, 36, 1, 1, 0, 0, 1, 1, False],  # non-tile multiple NHW
        ],
    },
    "test_run_max_pool": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],  # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],  # vgg
            # [1, 512, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            # [1, 512, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            [1, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],  # resnet
            [2, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [4, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [8, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
    "test_run_max_pool_width_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False], # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            # [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False], # vgg
            [1, 32768, 8, 8, 2, 2, 1, 1, 0, 0, 1, 1, False],  # wide in place untilize
            [1, 16384, 8, 8, 2, 2, 1, 1, 0, 0, 1, 1, False],  # normal in place untilize
            [1, 32768, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            [1, 32768, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 32768, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            [1, 6144, 6, 6, 5, 5, 1, 1, 2, 2, 1, 1, False],
            [1, 6144, 6, 6, 9, 9, 1, 1, 2, 2, 1, 1, False],
            [1, 6144, 6, 6, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 6144, 6, 6, 13, 13, 1, 1, 2, 2, 1, 1, False],
            # [1, 6144, 6, 6, 13, 13, 1, 1, 4, 4, 1, 1, False],
            # [1, 6144, 6, 6, 13, 13, 1, 1, 6, 6, 1, 1, False],
            # [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            # [1, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False], #resnet
            # [2, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            # [4, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            # [8, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
    "test_run_max_pool_height_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32768, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            # [1, 32768, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 32768, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            # [1, 6144, 6, 6, 5, 5, 1, 1, 2, 2, 1, 1, False],
            # [1, 6144, 6, 6, 9, 9, 1, 1, 2, 2, 1, 1, False],
            # [1, 6144, 6, 6, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # requires reversed local reads on some cores, and forward reads on others
            [8, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, True],
            # requires reversed local reads on some cores, and forward reads on others, large kernel
            [32, 32, 264, 40, 5, 5, 2, 2, 2, 2, 1, 1, True],
            [1, 512, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            [1, 512, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            [1, 32, 6, 6, 3, 3, 1, 1, 1, 1, 1, 1, False],  # partial grid on WH to use noop cores
        ],
    },
    "test_run_max_pool_block_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],  # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            # [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],  # vgg
            [1, 4096, 16, 16, 2, 2, 1, 1, 0, 0, 1, 1, False],  # wide in place untilize
            [1, 2048, 16, 16, 2, 2, 1, 1, 0, 0, 1, 1, False],  # normal in place untilize
            # requires reversed local reads on some cores, and forward reads on others, wide in place untilize, large kernel
            [1, 4096, 16, 16, 5, 5, 2, 2, 2, 2, 1, 1, True],
            # requires reversed local reads on some cores, and forward reads on others, normal in place untilize, large kernel
            [1, 2048, 16, 16, 5, 5, 2, 2, 2, 2, 1, 1, True],
            [1, 512, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            [1, 512, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            # [1, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],  # resnet
            # [2, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            # [4, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            # [8, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
    "test_run_max_pool_mem_config": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],  # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],  # vgg
            # [1, 512, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            # [1, 512, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            [1, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],  # resnet
            [2, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [4, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [8, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    input_specs,
    dtype,
    *,
    device,
):
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
    ) = input_specs
    sharding = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    return run_max_pool2d(
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
        dtype,
        device,
        sharding,
        ceil_mode,
    )


import pytest


@pytest.mark.parametrize("input_spec", parameters["max_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["max_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_localrun(device, dtype, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_height_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_height_shard"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["test_run_max_pool_height_shard"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_localrun(device, dtype, in_place, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    if (kernel_height > 5 or kernel_width > 5) and in_place and dtype == ttnn.bfloat8_b:
        pytest.skip("this case runs out of memory due to combination of large remote temp CB and large untilize out CB")
    if input_spec[:4] == [1, 512, 10, 10] and in_place and dtype == ttnn.bfloat8_b and is_blackhole():
        pytest.skip(
            "this case runs out of memory on blackhole due to large remote temp CB, this is only an issue on blackhole since the larger number of cores results in a smaller nhe per core which results in more remote references and hence a larger remote temp CB"
        )
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool(device, dtype, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_width_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_width_shard"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["test_run_max_pool_width_shard"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_width_shard(device, dtype, in_place, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_block_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_block_shard"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["test_run_max_pool_block_shard"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_block_shard(device, dtype, in_place, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_mem_config"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_mem_config"]["dtype"])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_mem_config(device, dtype, input_spec, memory_config):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        ceil_mode=ceil_mode,
        memory_config=memory_config,
    )
