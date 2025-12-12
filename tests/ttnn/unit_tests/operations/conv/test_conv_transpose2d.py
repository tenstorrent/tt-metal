# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.common.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
)
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn

torch.set_printoptions(linewidth=400, profile="full", sci_mode=False)
SliceHeight = ttnn.Conv2dDRAMSliceHeight
SliceWidth = ttnn.Conv2dDRAMSliceWidth
L1Full = ttnn.Conv2dL1Full


def run_conv_transpose2d(
    device,
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
    out_pad_h,
    out_pad_w,
    config_override=None,
    dilation=1,
    fp32_accum=False,
    packer_l1_acc=False,
    layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    groups=1,
    has_bias=True,
    shard_layout=None,
    auto_shard=False,
    mirror_kernel=True,
    enable_act_double_buffer=False,
    preprocess_weights_bias=False,
    config_tensors_in_dram=False,
    dram_slice_config=None,
):
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [input_channels, output_channels // groups, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()

    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv_transpose2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        output_padding=(out_pad_h, out_pad_w),
        dilation=(dilation, dilation),
        groups=groups,
    )

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    if not mirror_kernel:
        torch_flipped_weights = torch.flip(torch_weight_tensor, [2, 3])
        tt_weight_tensor = ttnn.from_torch(
            torch_flipped_weights, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, layout=layout, device=device)

    if auto_shard:
        shard_layout = None

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=enable_act_double_buffer,
        output_layout=layout,
        config_tensors_in_dram=config_tensors_in_dram,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override:
        conv_config.act_block_w_div = config_override["act_block_w_div"]
    if preprocess_weights_bias:
        tt_weight_tensor = ttnn.prepare_conv_transpose2d_weights(
            weight_tensor=tt_weight_tensor,
            input_memory_config=ttnn.L1_MEMORY_CONFIG,
            input_layout=layout,
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
            has_bias=has_bias,
            groups=groups,
            device=device,
            conv_config=conv_config,
            compute_config=compute_config,
            mirror_kernel=mirror_kernel,
            input_dtype=activations_dtype,
        )

        tt_bias_tensor = (
            ttnn.prepare_conv_transpose2d_bias(
                bias_tensor=tt_bias_tensor,
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
                # output_padding=(out_pad_h, out_pad_w),
                dilation=(dilation, dilation),
                groups=groups,
                conv_config=conv_config,
                compute_config=compute_config,
                input_dtype=activations_dtype,
            )
            if has_bias
            else None
        )
    [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv_transpose2d(
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
        dram_slice_config=dram_slice_config,
        groups=groups,
        mirror_kernel=mirror_kernel,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=activations_dtype,
    )
    logger.info(f"Conv2d Transpose Input = {(input_height, input_width)} Output = {out_height, out_width}")

    torch_output_tensor = ttnn.to_torch((tt_output_tensor_on_device).cpu())

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    out = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    if not fp32_accum:
        pcc = 0.99
    elif math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998

    ref = torch_out_golden_tensor
    passing, pcc_msg = check_with_pcc_without_tensor_printout(out, ref, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


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
        (1,  256, 256,  64, 64,  3, 3, 2, 2, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 4, SliceWidth ),
        (1,  256, 256,  64, 64,  3, 3, 2, 2, 1, 1, 1, 1, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 4, SliceWidth ),
        (1,   32,  32,  64, 64,  8, 8, 4, 4, 2, 2, 0, 0, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 2 , SliceWidth ),
        (1,   32,  32,  64, 64,  8, 8, 4, 4, 2, 2, 2, 2, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 2 , SliceWidth ),
        (16,  16,  16, 256, 128, 2, 2, 2, 2, 0, 0, 0, 0, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED,  2, SliceWidth ),
        (1, 512,  512, 512, 512, 3, 3, 1, 1, 1, 1, 0, 0, {'act_block_h' : 256}, ttnn.TensorMemoryLayout.BLOCK_SHARDED, 8, SliceWidth ),
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
        num_slices=num_slices,
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
