# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger
from models.common.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
)
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import math

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
    fast_compare=True,
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

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, activations_dtype, layout=layout, device=device)

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
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG if dram_slice_config is not None else ttnn.L1_MEMORY_CONFIG,
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
            dram_slice_config=dram_slice_config,
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
                dram_slice_config=dram_slice_config,
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
    torch.set_printoptions(precision=3, sci_mode=False)
    if fast_compare:
        if fp32_accum and activations_dtype != ttnn.bfloat8_b and weights_dtype != ttnn.bfloat8_b:
            threshold = 3e-1 + 5e-3 * math.log(input_channels * filter_height * filter_width, 2)
        else:
            threshold = 3e-1 + 1e-1 * math.log(input_channels * filter_height * filter_width, 2)
        logger.info(f"Threshold: {threshold}")
        diff = torch.abs(ref - out) / ref.abs().mean()
        assert torch.all(diff < threshold), f"Max diff: {diff.max()}, Threshold: {threshold} "
    else:
        passing, pcc_msg = check_with_pcc_without_tensor_printout(out, ref, pcc=pcc)
        logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
        assert passing


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
@pytest.mark.parametrize("auto_slice", [True, False])
@pytest.mark.parametrize("preprocess_weights", [True, False])
@pytest.mark.parametrize("mirror_kernel", [True, False])
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
    auto_slice,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 Grid for Wormhole_b0")
    dram_slice_config = ttnn.Conv2dSliceConfig(
        slice_type=slice_type,
        num_slices=0 if auto_slice else num_slices,
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv_transpose2d_fp32_accum_auto_default(device):
    """
    Test that FP32 accumulation is automatically enabled for conv_transpose2d when both input and weights are FP32.

    Runs conv_transpose2d three times with FP32 inputs and FP32 weights:
    1. Without compute_config (relies on auto-default)
    2. With explicit fp32_dest_acc_en=True
    3. With explicit fp32_dest_acc_en=False

    Verifies that auto-default matches explicit True (not False), proving FP32 accum is auto-enabled.
    """
    batch_size = 1
    out_channels = 64
    input_channels = 64
    input_height = 8
    input_width = 8
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1

    # Generate random FP32 inputs
    torch.manual_seed(42)
    torch_input_nchw = torch.rand(batch_size, input_channels, input_height, input_width, dtype=torch.float32)
    torch_weight = torch.rand(input_channels, out_channels, kernel_size, kernel_size, dtype=torch.float32)
    torch_bias = torch.rand(1, 1, 1, out_channels, dtype=torch.float32)

    # Convert input to NHWC for ttnn
    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))

    # Convert to ttnn tensors - all FP32
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.float32, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.float32)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.float32)

    # Run 1: WITHOUT explicit compute_config (auto-default behavior)
    # Default from get_conv_default_compute_kernel_config() is:
    # math_fidelity=HiFi4, math_approx_mode=true, fp32_dest_acc_en=true (for FP32xFP32), packer_l1_acc=false
    tt_output_auto = ttnn.conv_transpose2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=input_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        output_padding=(output_padding, output_padding),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        # No compute_config - uses get_conv_default_compute_kernel_config()
    )

    # Run 2: WITH explicit fp32_dest_acc_en=True (matching expected default)
    # Must match all default params: MathFidelity::HiFi4, math_approx_mode=true, packer_l1_acc=false
    compute_config_true = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    tt_output_explicit_true = ttnn.conv_transpose2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=input_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        output_padding=(output_padding, output_padding),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        compute_config=compute_config_true,
    )

    # Run 3: WITH explicit fp32_dest_acc_en=False (to verify difference)
    # Keep all other params same as default, only change fp32_dest_acc_en
    compute_config_false = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_output_explicit_false = ttnn.conv_transpose2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=input_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        output_padding=(output_padding, output_padding),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        compute_config=compute_config_false,
    )

    # Convert outputs to torch
    tt_output_auto_torch = ttnn.to_torch(tt_output_auto)
    tt_output_explicit_true_torch = ttnn.to_torch(tt_output_explicit_true)
    tt_output_explicit_false_torch = ttnn.to_torch(tt_output_explicit_false)

    # Auto-default should match explicit True (FP32 accum enabled)
    assert torch.equal(tt_output_auto_torch, tt_output_explicit_true_torch), (
        "Auto-default output does not match explicit fp32_dest_acc_en=True. "
        "FP32 accumulation was NOT automatically enabled for FP32 x FP32!"
    )

    # Auto-default should NOT match explicit False (verify they're different)
    assert not torch.equal(tt_output_auto_torch, tt_output_explicit_false_torch), (
        "Auto-default output matches explicit fp32_dest_acc_en=False. "
        "This suggests FP32 accumulation was NOT enabled (unexpected)."
    )
