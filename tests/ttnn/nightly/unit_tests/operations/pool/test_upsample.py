# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
@pytest.mark.parametrize("run_twice", [True])
def test_upsample_nearest_interleaved(
    device, input_shapes, scale_h, scale_w, memory_layout, dtype_torch, dtype_ttnn, run_twice
):
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

    if run_twice:
        ttnn.deallocate(output_tensor, True)
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
@pytest.mark.parametrize("run_twice", [True])
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
    run_twice,
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

    if run_twice:
        ttnn.deallocate(output_tensor, True)
        output_tensor = ttnn.upsample(
            input_tensor, scale_factor, mode=mode, compute_kernel_config=compute_kernel_config
        )

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
@pytest.mark.parametrize("run_twice", [True])
def test_rectangle_core_grid_bs(device, input_shape, scale_h, scale_w, core_range, run_twice):
    (torch_result, output_tensor) = upsample_multicore_common(
        device=device,
        input_shape=input_shape,
        scale_h=scale_h,
        scale_w=scale_w,
        shard_strategy=ttnn.ShardStrategy.BLOCK,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        core_range=core_range,
        run_twice=run_twice,
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
@pytest.mark.parametrize("run_twice", [True])
def test_upsample_various(
    device, input_shape, core_range, scale_h, scale_w, shard_strategy, shard_orientation, run_twice
):
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
        run_twice=run_twice,
    )
    ## compare the results
    torch_result = torch_result.permute(0, 2, 3, 1)

    isequal = torch.equal(output_tensor, torch_result)

    assert isequal


@pytest.mark.parametrize("device_params", [{"l1_small_size": 37888}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_channels, height, width, scale_h, scale_w, num_slices",
    (
        (1, 256, 64, 128, 2, 2, 2),
        (1, 128, 64, 128, 2, 2, 2),
    ),
)
@pytest.mark.parametrize("run_twice", [True])
def test_panoptic_upsample_sliced(
    device, batch_size, num_channels, height, width, scale_h, scale_w, num_slices, run_twice
):
    input_shape_nchw = [batch_size, num_channels, height, width]
    scale_factor = (scale_h, scale_w)
    mode_pytorch = "nearest"  # we only did nearest in panoptic due to pcc dropping very little and bilinear not being able to fit in memory as of now
    mode_ttnn = "nearest"
    dtype_torch = torch.bfloat16
    dtype_ttnn = ttnn.bfloat16

    batch_size, channels, input_h, input_w = input_shape_nchw
    assert channels % num_slices == 0, "Channels must be divisible by num_slices"
    slice_channels = channels // num_slices

    logger.info(f"Running Panoptic Upsample with Channel Slicing (slices={num_slices})")

    torch.manual_seed(0)
    torch_input_nchw = torch.rand(input_shape_nchw, dtype=dtype_torch)

    torch_upsample = nn.Upsample(scale_factor=scale_factor, mode=mode_pytorch)
    torch_output_nchw = torch_upsample(torch_input_nchw)

    ttnn_input_nhwc = ttnn.from_torch(
        torch_input_nchw.permute(0, 2, 3, 1), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype_ttnn
    )

    sliced_results = []
    for slice_idx in range(num_slices):
        start_ch = slice_idx * slice_channels
        end_ch_exclusive = (slice_idx + 1) * slice_channels

        x_slice_nhwc = ttnn.slice(
            ttnn_input_nhwc, [0, 0, 0, start_ch], [batch_size, input_h, input_w, end_ch_exclusive]
        )

        x_slice_upsampled = ttnn.upsample(
            x_slice_nhwc,
            scale_factor=scale_factor,
            mode=mode_ttnn,
        )

        if run_twice:
            ttnn.deallocate(x_slice_upsampled, True)
            x_slice_upsampled = ttnn.upsample(
                x_slice_nhwc,
                scale_factor=scale_factor,
                mode=mode_ttnn,
            )

        x_slice_upsampled = ttnn.to_memory_config(x_slice_upsampled, ttnn.DRAM_MEMORY_CONFIG)
        sliced_results.append(x_slice_upsampled)

        ttnn.deallocate(x_slice_nhwc)

    ttnn.deallocate(ttnn_input_nhwc)

    ttnn_output_nhwc = ttnn.concat(sliced_results, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    for slice_result in sliced_results:
        ttnn.deallocate(slice_result)

    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1)

    ttnn_output_torch_nhwc = ttnn.to_torch(ttnn_output_nhwc)

    passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch_nhwc, pcc=0.99)
    logger.info(pcc_message)
    assert passed, f"PCC check failed. {pcc_message}"


# ============================================================================
# Float scale factor tests with sharded memory
# ============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 64, 8, 8],
        [1, 128, 16, 16],
        [2, 64, 8, 8],
        [1, 32, 4, 4],
        [1, 64, 32, 32],
        [1, 128, 56, 56],
        [1, 32, 8, 8],
        [1, 256, 8, 8],
        [1, 512, 8, 8],
    ],
)
@pytest.mark.parametrize("scale_h, scale_w", [(2.0, 2.0), (1.5, 1.5), (2.5, 2.5), (0.5, 0.5), (0.75, 0.75), (2.0, 1.5)])
@pytest.mark.parametrize("shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.BLOCK])
def test_upsample_nearest_float_sharded(device, input_shape, scale_h, scale_w, shard_strategy):
    """Test upsample with float scale factors using sharded memory."""
    torch_result, output_tensor = upsample_multicore_common(
        device, input_shape, scale_h, scale_w, shard_strategy, ttnn.ShardOrientation.ROW_MAJOR
    )
    torch_result = torch_result.permute(0, 2, 3, 1)

    passing, pcc_msg = assert_with_pcc(torch_result, output_tensor, pcc=0.9999)
    logger.info(pcc_msg)
    assert passing
