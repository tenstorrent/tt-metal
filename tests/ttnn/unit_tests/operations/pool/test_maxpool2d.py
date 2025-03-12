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
        ],
    },
    "test_run_max_pool_width_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False], # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            # [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False], # vgg
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
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32768, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            # [1, 32768, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 32768, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            [1, 6144, 6, 6, 5, 5, 1, 1, 2, 2, 1, 1, False],
            # [1, 6144, 6, 6, 9, 9, 1, 1, 2, 2, 1, 1, False],
            # [1, 6144, 6, 6, 9, 9, 1, 1, 4, 4, 1, 1, False],
            [1, 512, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            [1, 512, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
        ],
    },
    "test_run_max_pool_block_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],  # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            # [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],  # vgg
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_width_shard(device, dtype, input_spec):
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
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_block_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_block_shard"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_block_shard(device, dtype, input_spec):
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


@pytest.mark.parametrize(
    "act_shape, kernel_size, padding, stride, ceil_mode",
    [
        ((1, 64, 256, 256), (2, 2), (0, 0), (2, 2), False),
        ((1, 128, 128, 128), (2, 2), (0, 0), (2, 2), False),
        ((1, 256, 64, 64), (2, 2), (0, 0), (2, 2), False),
        ((1, 512, 32, 32), (2, 2), (0, 0), (2, 2), False),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vgg_unet_maxpool(act_shape, kernel_size, padding, stride, dilation, ceil_mode, device, dtype, memory_config):
    run_max_pool2d(
        act_shape[0],
        act_shape[1],
        act_shape[2],
        act_shape[3],
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        dtype,
        device,
        memory_config=memory_config,
    )
