# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.common.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
)
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout

from tests.ttnn.nightly.unit_tests.operations.conv.test_conv_transpose2d import (
    run_conv_transpose2d,
    SliceHeight,
    SliceWidth,
    L1Full,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, config, shard_layout",
    (
        # Stride = 1
        (1, 8, 8, 256, 256, 3, 3, 1, 1, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (1, 16, 16, 256, 256, 3, 3, 1, 1, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (1, 256, 256, 32, 32, 3, 3, 1, 1, 1, 1, 0, 0, {"act_block_h": 64}, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 256, 256, 32, 32, 1, 1, 1, 1, 0, 0, 0, 0, {"act_block_h": 64}, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        # Stride = 2
        (1, 8, 8, 32, 64, 3, 3, 2, 2, 1, 1, 1, 1, None, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (1, 128, 128, 32, 64, 3, 3, 2, 2, 1, 1, 1, 1, {"act_block_h": 64}, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (1, 16, 16, 32, 32, 3, 3, 2, 2, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        # Vanilla Unet
        # Filter Size = 2 not supported in Block sharded
        (1, 30, 40, 512, 256, 3, 3, 2, 2, 1, 1, 1, 1, {"act_block_h": 64}, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("preprocess_weights", [True, False])
@pytest.mark.parametrize("mirror_kernel", [True, False])
def test_simple_conv_t2d(
    device,
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
    out_pad_h,
    out_pad_w,
    config,
    shard_layout,
    mirror_kernel,
    preprocess_weights,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 Grid for Wormhole_b0")
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        out_pad_h=out_pad_h,
        out_pad_w=out_pad_w,
        config_override=config,
        shard_layout=shard_layout,
        auto_shard=True,
        mirror_kernel=mirror_kernel,
        preprocess_weights_bias=preprocess_weights,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, config, shard_layout, num_slices, slice_type",
    (
        # fmt: off
        (1,  512, 512,  64, 64,  3, 3, 1, 1, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 4, SliceWidth ),
        (1,   32,  32,  64, 64,  8, 8, 4, 4, 2, 2, 2, 2, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 2 , SliceWidth ),
        (16,  16,  16, 256, 128, 2, 2, 2, 2, 0, 0, 0, 0, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED,  2, SliceWidth ),
        # fmt: on
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype, layout",
    [
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize("preprocess_weights", [False])
@pytest.mark.parametrize("mirror_kernel", [False])
def test_convt2d_dram(
    device,
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
    out_pad_h,
    out_pad_w,
    config,
    shard_layout,
    layout,
    mirror_kernel,
    preprocess_weights,
    num_slices,
    slice_type,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 Grid for Wormhole_b0")
    dram_slice_config = ttnn.Conv2dSliceConfig(
        slice_type=slice_type,
    )
    if is_blackhole() and config is not None:
        # Blackhole requires different act_block_h to be divisble
        config["act_block_h"] = 32
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        out_pad_h=out_pad_h,
        out_pad_w=out_pad_w,
        config_override=config,
        shard_layout=shard_layout,
        layout=layout,
        auto_shard=True,
        mirror_kernel=mirror_kernel,
        preprocess_weights_bias=preprocess_weights,
        dram_slice_config=dram_slice_config,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, config, shard_layout",
    (
        # Stride = 1
        (1, 8, 8, 256, 256, 3, 3, 1, 1, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (1, 16, 16, 256, 256, 3, 3, 1, 1, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (1, 256, 256, 32, 32, 3, 3, 1, 1, 1, 1, 0, 0, {"act_block_h": 64}, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        # Stride = 2
        (1, 8, 8, 32, 64, 3, 3, 2, 2, 1, 1, 1, 1, None, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (1, 128, 128, 32, 64, 3, 3, 2, 2, 1, 1, 1, 1, {"act_block_h": 64}, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("mirror_kernel", [True, False])
def test_simple_conv_t2d_weights_bias_match(
    device,
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
    out_pad_h,
    out_pad_w,
    config,
    shard_layout,
    mirror_kernel,
):
    """
    Test that verifies weights and bias returned by conv_transpose2d (with preprocess_weights=False)
    match exactly with weights and bias from prepare_conv_transpose2d_weights/bias functions.
    """
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 Grid for Wormhole_b0")

    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [input_channels, output_channels, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]

    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()

    # Create ttnn tensors
    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    if not mirror_kernel:
        torch_flipped_weights = torch.flip(torch_weight_tensor, [2, 3])
        tt_weight_tensor = ttnn.from_torch(
            torch_flipped_weights, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, device=device)

    # Setup configs
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        shard_layout=None,  # auto_shard
        deallocate_activation=False,
        enable_act_double_buffer=False,
        output_layout=ttnn.TILE_LAYOUT,
        config_tensors_in_dram=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    if config and "act_block_h" in config:
        conv_config.act_block_h_override = config["act_block_h"]
    if config and "act_block_w_div" in config:
        conv_config.act_block_w_div = config["act_block_w_div"]

    # Run conv_transpose2d with preprocess_weights_bias=False (weights on host)
    dilation = 1
    [tt_output_tensor_on_device, [out_height, out_width], [weights_from_conv, bias_from_conv]] = ttnn.conv_transpose2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        output_padding=(out_pad_h, out_pad_w),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=1,
        mirror_kernel=mirror_kernel,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=activations_dtype,
    )

    # Now call prepare functions separately
    tt_weight_tensor_host = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    if not mirror_kernel:
        torch_flipped_weights = torch.flip(torch_weight_tensor, [2, 3])
        tt_weight_tensor_host = ttnn.from_torch(
            torch_flipped_weights, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_bias_tensor_host = ttnn.from_torch(
        torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )

    weights_from_prepare = ttnn.prepare_conv_transpose2d_weights(
        weight_tensor=tt_weight_tensor_host,
        input_memory_config=ttnn.L1_MEMORY_CONFIG,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weights_format="IOHW",
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        has_bias=True,
        groups=1,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        mirror_kernel=mirror_kernel,
        input_dtype=activations_dtype,
    )

    bias_from_prepare = ttnn.prepare_conv_transpose2d_bias(
        bias_tensor=tt_bias_tensor_host,
        input_memory_config=ttnn.L1_MEMORY_CONFIG,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        device=device,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        groups=1,
        conv_config=conv_config,
        compute_config=compute_config,
        input_dtype=activations_dtype,
    )

    # Convert to torch tensors for comparison
    weights_conv_torch = ttnn.to_torch(weights_from_conv.cpu())
    weights_prepare_torch = ttnn.to_torch(weights_from_prepare.cpu())
    bias_conv_torch = ttnn.to_torch(bias_from_conv.cpu())
    bias_prepare_torch = ttnn.to_torch(bias_from_prepare.cpu())

    # Verify that weights and bias match exactly
    logger.info(f"Weights shape from conv: {weights_conv_torch.shape}, from prepare: {weights_prepare_torch.shape}")
    logger.info(f"Bias shape from conv: {bias_conv_torch.shape}, from prepare: {bias_prepare_torch.shape}")

    assert torch.equal(
        weights_conv_torch, weights_prepare_torch
    ), "Weights from conv_transpose2d don't match weights from prepare_conv_transpose2d_weights"
    assert torch.equal(
        bias_conv_torch, bias_prepare_torch
    ), "Bias from conv_transpose2d don't match bias from prepare_conv_transpose2d_bias"

    logger.info("Weights and bias from conv_transpose2d match prepare functions exactly!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
# fmt: off
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, config, enable_act_double_buffer",
    (
        (1, 64, 8, 64, 64, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True),
        (1, 128, 16, 128, 64, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True),
        (1, 256, 32, 128, 64, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True),
        (1, 512, 64, 128, 2, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True),
    ),
)
# fmt: on
@pytest.mark.parametrize(
    "weights_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
def test_conv_transpose2d_model_fruit(
    device,
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
    out_pad_h,
    out_pad_w,
    config,
    enable_act_double_buffer,
    mirror_kernel=False,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 Grid for Wormhole_b0")
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        out_pad_h=out_pad_h,
        out_pad_w=out_pad_w,
        config_override=config,
        auto_shard=True,
        enable_act_double_buffer=enable_act_double_buffer,
        mirror_kernel=mirror_kernel,
    )


@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w",
    (
        # Test config_tensors_in_dram parameter with various configurations
        # No l1_small_size fixture - the point is to test DRAM storage to avoid L1_SMALL usage
        (1, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 0, 0),
        (1, 8, 8, 256, 256, 3, 3, 1, 1, 1, 1, 0, 0),
        (1, 256, 256, 32, 32, 3, 3, 1, 1, 1, 1, 0, 0),
    ),
)
def test_conv_transpose2d_config_tensors_in_dram(
    device,
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
    out_pad_h,
    out_pad_w,
):
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        out_pad_h=out_pad_h,
        out_pad_w=out_pad_w,
        config_tensors_in_dram=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 1024}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, groups",
    (
        # Grouped transposed convolution tests (Issue #34497)
        # Test 1: Original issue with C_in=1088, C_out=1088, groups=17
        (1, 7, 7, 1088, 1088, 3, 3, 2, 2, 1, 1, 1, 1, 17),
        # Test 2: Secondary issue with C_in=119, C_out=119, groups=17 (divisible by 7)
        (1, 7, 7, 119, 119, 3, 3, 2, 2, 1, 1, 1, 1, 17),
        # Test 3: From comment - C_in=160, C_out=80, groups=2 (numerical accuracy issue)
        (1, 14, 14, 160, 80, 3, 3, 2, 2, 0, 0, 0, 0, 2),
    ),
)
def test_conv_transpose2d_with_groups(
    device,
    batch_size,
    input_height,
    input_width,
    input_channels,
    output_channels,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    out_pad_h,
    out_pad_w,
    groups,
):
    """
    Test grouped transposed convolution.
    Reproduces and validates fix for issue #34497.
    """
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 Grid for Wormhole_b0")
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        out_pad_h=out_pad_h,
        out_pad_w=out_pad_w,
        groups=groups,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        auto_shard=True,
    )


# l1_small_size is part of what the slicer sees: the L1 it can allocate from is the core's L1 minus the circular-buffer
# base and this value. With 32768 on Blackhole the single-slice plan for lengths 384 and 396 needs about 13 KB per core
# more than that pool, so the slicer has to choose two slices there. With 16384 the pool is 16 KB larger. The
# single-slice plan then fits by 3 KB if the op frees its own 8 KB sharded copy of the input after the halo op, and
# does not fit if it keeps that copy; the auto_slice_config case runs conv_transpose2d_L1 with the caller's config and
# so checks the release.
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768}, {"l1_small_size": 16384}],
    ids=["l1_small_32k", "l1_small_16k"],
    indirect=True,
)
@pytest.mark.parametrize("auto_slice", [False, True], ids=["default_slicing", "auto_slice_config"])
@pytest.mark.parametrize("input_width", [352, 384, 396, 416])
def test_conv_transpose2d_1d_upsampler_dram_input(device, input_width, auto_slice):
    """
    ConvTranspose1d 512 -> 256 (kernel 16, stride 8, padding 4) as conv_transpose2d at H=1, bfloat16 tiled DRAM
    input, default Conv2dConfig, prepared weights. Every length must run and match torch.

    The DRAM slicer picks the number of slices from an L1 estimate, so the estimate has to size what the op
    allocates. With a single slice the op shards the DRAM input for an unsliced conv (on Blackhole 64 channels per
    core on 8 columns). Sizing the halo output for a sliced input (40 channels per core on 13 columns) gives 26.8 KB
    per core instead of 42.9 KB; the single-slice plan then looks like it fits at lengths 384 and 396, and the op
    raises "Statically allocated circular buffers ... clash with L1 buffers".
    """
    in_channels, out_channels, kernel, stride, padding = 512, 256, 16, 8, 4
    torch.manual_seed(0)
    torch_input = torch.randn(1, input_width, in_channels).bfloat16().float()
    torch_weight = (torch.randn(in_channels, out_channels, kernel) / (in_channels * kernel) ** 0.5).bfloat16().float()
    torch_bias = (0.1 * torch.randn(out_channels)).bfloat16().float()
    expected = torch.nn.functional.conv_transpose1d(
        torch_input.permute(0, 2, 1), torch_weight, torch_bias, stride=stride, padding=padding
    )
    output_length = (input_width - 1) * stride - 2 * padding + kernel
    assert expected.shape == (1, out_channels, output_length)

    conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
    slice_config = ttnn.Conv2dSliceConfig(slice_type=SliceWidth, num_slices=0) if auto_slice else None
    tt_input = ttnn.reshape(
        ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        (1, 1, input_width, in_channels),
    )
    prepare_args = dict(
        input_memory_config=tt_input.memory_config(),
        input_layout=tt_input.layout,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=1,
        input_height=1,
        input_width=input_width,
        kernel_size=(1, kernel),
        stride=(1, stride),
        padding=(0, padding),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_config,
        dram_slice_config=slice_config,
    )
    tt_weight = ttnn.prepare_conv_transpose2d_weights(
        weight_tensor=ttnn.from_torch(torch_weight.unsqueeze(2), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        weights_format="IOHW",
        has_bias=True,
        **prepare_args,
    )
    tt_bias = ttnn.prepare_conv_transpose2d_bias(
        bias_tensor=ttnn.from_torch(torch_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        **prepare_args,
    )

    tt_output = ttnn.conv_transpose2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=1,
        input_height=1,
        input_width=input_width,
        kernel_size=(1, kernel),
        stride=(1, stride),
        padding=(0, padding),
        dilation=(1, 1),
        groups=1,
        conv_config=conv_config,
        dram_slice_config=slice_config,
        dtype=ttnn.bfloat16,
    )

    actual = ttnn.to_torch(tt_output).float()
    assert actual.numel() == output_length * out_channels
    actual = actual.reshape(1, output_length, out_channels).permute(0, 2, 1)
    passing, pcc_message = check_with_pcc_without_tensor_printout(expected, actual, pcc=0.999)
    logger.info(f"input_width={input_width} auto_slice={auto_slice}: {pcc_message}")
    assert passing, pcc_message
