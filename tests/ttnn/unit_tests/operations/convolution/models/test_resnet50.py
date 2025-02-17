from ..conv_test_utils import *

from loguru import logger

import ttnn
import torch
import pytest
from models.utility_functions import (
    skip_for_grayskull,
    is_grayskull,
    is_wormhole_b0,
    is_blackhole,
)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, None), HANGS!!
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 256}),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32}),  Out of Memory!!
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, {"act_block_h": 32}),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # rn50 layer3
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # rn50 layer4
        (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        ## small test
        (1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, BS, {"num_cores_nhw": 2, "grid_size": (2, 2)}),
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, BS, {"num_cores_nhw": 4, "grid_size": (2, 4)}),
        # (1, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, BS, None), sliding_window_op_infra/sliding_window.cpp:341: indices_length_last_core <= indices_length_per_core
        (8, 256, 256, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        # r50 1x1s2 shapes
        # Fails with packer_l1_acc = True (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, BS, None),  # r50 first bottleneck downsample shape
        (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, HS, None),  # r50 first bottleneck downsample shape
        # Fails with packer_l1_acc = True (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, BS, None),  # r50 second bottleneck downsample shape
        # (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, HS, None), - doesnt fit
        (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, BS, None),  # r50 third bottleneck downsample shape
        # (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, HS, None), - doesnt fit
        (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, BS, None),  # r50 fourth bottleneck downsample shape
        # (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, HS, None), - doesnt fit
        # (20, 128, 256, 56, 56, 1, 1, 2, 2, 0, 0, HS, None),  ## L2M1 DS: doesn't fit
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("packer_l1_acc", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_resnet50_conv_wh(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    packer_l1_acc,
    has_bias,
    auto_shard,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    use_shallow_conv_variant = (input_channels == 16) and device.arch() == ttnn.device.Arch.GRAYSKULL
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override=config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        packer_l1_acc=packer_l1_acc,
        fp32_accum=False,
        has_bias=has_bias,
        auto_shard=auto_shard,
        shard_layout=shard_layout,
    )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, None),
        # (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32}),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32}),
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        # (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        # (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, {"act_block_h": 32}),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # # rn50 layer3
        # (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        # (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        # (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        # (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # # rn50 layer4
        # (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        # (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        # (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        # (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        # (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        # (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.float32, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.float32, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("packer_l1_acc", [True])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_resnet50_conv_wh_fp32(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    fp32_accum,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    packer_l1_acc,
    auto_shard,
):
    if batch_size > 8 and (activations_dtype != ttnn.bfloat8_b or weights_dtype != ttnn.bfloat8_b):
        pytest.skip("Batch > 8 must be run fully bfp8")

    if (
        activations_dtype == ttnn.bfloat16
        and batch_size == 20
        and (
            output_channels == 64
            or (
                stride_h == 2
                and (output_channels == 256 or (output_channels == 128 and weights_dtype == ttnn.bfloat16))
            )
        )
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    use_shallow_conv_variant = (input_channels == 16) and device.arch() != ttnn.device.Arch.WORMHOLE_B0
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        shard_layout=shard_layout,
        config_override=config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        output_layout=ttnn.TILE_LAYOUT,
        auto_shard=auto_shard,
    )
