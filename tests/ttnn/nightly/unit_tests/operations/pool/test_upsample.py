# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import torch
import torch.nn as nn
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.pool.test_upsample import upsample_multicore_common


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 128, 64, 64],
        [2, 64, 32, 32],
        [5, 32, 96, 96],
        [1, 96, 32, 32],
        [2, 32, 80, 32],
    ],
)
@pytest.mark.parametrize("scale_h", [2, 3])
@pytest.mark.parametrize("scale_w", [2, 3])
@pytest.mark.parametrize("memory_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "dtype_torch, dtype_ttnn",
    [[torch.bfloat16, ttnn.bfloat8_b], [torch.float32, ttnn.float32], [torch.bfloat16, ttnn.bfloat16]],
)
def test_upsample_nearest_interleaved(device, input_shapes, scale_h, scale_w, memory_layout, dtype_torch, dtype_ttnn):
    # Skip block datatypes if memory layout is not tiled
    if dtype_ttnn == ttnn.bfloat8_b and memory_layout != ttnn.TILE_LAYOUT:
        pytest.skip("Block datatypes require TILE_LAYOUT")

    batch_size, num_channels, height, width = input_shapes
    torch.manual_seed(0)

    # Generate appropriate test data based on dtype
    input = torch.rand(input_shapes, dtype=dtype_torch)
    tt_input = input.permute(0, 2, 3, 1)
    input_tensor = ttnn.from_torch(
        tt_input, device=device, layout=memory_layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype_ttnn
    )

    if input_tensor.padded_shape != input_tensor.shape and memory_layout == ttnn.TILE_LAYOUT:
        pytest.skip("Disabled until different logical and padded shapes are supported for TILE_LAYOUT")

    scale_factor = (scale_h, scale_w)
    torch_upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = torch_upsample(input)

    scale_factor = (scale_h, scale_w)

    output_tensor = ttnn.upsample(input_tensor, scale_factor)

    output_tensor = ttnn.to_torch(output_tensor)

    torch_result = torch_result.permute(0, 2, 3, 1)
    pcc_passed, pcc_message = assert_with_pcc(torch_result, output_tensor, 0.9999)
    logger.info(pcc_message)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_channels, height, width, scale_h, scale_w",
    (
        (1, 1280, 8, 8, 2, 2),
        (1, 256, 16, 16, 8, 8),
        (1, 256, 32, 32, 4, 4),
        (1, 256, 64, 64, 2, 2),
        (1, 256, 64, 64, 3, 3),
        (1, 1024, 8, 8, 2, 2),
        (1, 256, 28, 28, 2, 2),
        (1, 512, 14, 14, 2, 2),
    ),
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("math_approx_mode", [True, False])
def test_bilinear_interleaved_memory(
    device,
    batch_size,
    num_channels,
    height,
    width,
    scale_h,
    scale_w,
    math_fidelity,
    math_approx_mode,
):
    # Performs bilinear upsampling on interleaved inputs
    # Automatically height shards the input tensor

    torch.manual_seed(0)

    input_shape = [batch_size, num_channels, height, width]

    mode = "bilinear"

    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_input = torch_input.permute(0, 2, 3, 1)
    input_tensor = ttnn.from_torch(tt_input, device=device)
    scale_factor = (scale_h, scale_w)
    torch_upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
    torch_result = torch_upsample(torch_input)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=False,
    )

    output_tensor = ttnn.upsample(input_tensor, scale_factor, mode=mode, compute_kernel_config=compute_kernel_config)
    output_tensor = ttnn.to_torch(output_tensor)

    torch_result = torch_result.permute(0, 2, 3, 1)
    pcc_passed, pcc_message = assert_with_pcc(torch_result, output_tensor, pcc=0.999)
    logger.info(pcc_message)
    allclose = torch.allclose(output_tensor, torch_result, atol=1e-1, rtol=1e-1)
    assert allclose


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 640, 32, 32],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("scale_h", [2])
@pytest.mark.parametrize("scale_w", [2])
@pytest.mark.parametrize(
    "core_range",
    [
        [((0, 0), (4, 3))],
    ],
)
def test_rectangle_core_grid_bs(device, input_shape, scale_h, scale_w, core_range):
    (torch_result, output_tensor) = upsample_multicore_common(
        device=device,
        input_shape=input_shape,
        scale_h=scale_h,
        scale_w=scale_w,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        core_range=core_range,
    )
    ## compare the results
    torch_result = torch_result.permute(0, 2, 3, 1)

    isequal = torch.equal(output_tensor, torch_result)

    assert isequal


@pytest.mark.parametrize(
    "input_shape, core_range, scale_h, scale_w, shard_strategy, shard_orientation",
    [
        [
            [1, 1280, 32, 32],
            [((0, 0), (4, 7))],
            2,
            2,
            ttnn.ShardStrategy.BLOCK,
            ttnn.ShardOrientation.ROW_MAJOR,
        ],  # SDXL
        [[1, 640, 64, 64], [((0, 0), (4, 7))], 2, 2, ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR],  # SDXL
        [[1, 32, 8, 8], [((0, 0), (7, 7))], 1, 1, ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 120}], indirect=True)
def test_upsample_various(device, input_shape, core_range, scale_h, scale_w, shard_strategy, shard_orientation):
    if device.core_grid.y < 8:
        pytest.skip("n300 does not have 8 cores on y axis")
    (torch_result, output_tensor) = upsample_multicore_common(
        device=device,
        input_shape=input_shape,
        scale_h=scale_h,
        scale_w=scale_w,
        shard_strategy=shard_strategy,
        shard_orientation=shard_orientation,
        core_range=core_range,
    )
    ## compare the results
    torch_result = torch_result.permute(0, 2, 3, 1)

    isequal = torch.equal(output_tensor, torch_result)

    assert isequal
