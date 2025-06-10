# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    is_wormhole_b0,
    skip_for_grayskull,
)
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn

torch.set_printoptions(linewidth=400, profile="full", sci_mode=False)


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
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    groups=1,
    has_bias=True,
    shard_layout=None,
    auto_shard=False,
    mirror_kernel=True,
    enable_split_reader=False,
    enable_act_double_buffer=False,
    preprocess_weights_bias=False,
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

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    if shard_layout is None and not auto_shard:
        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_split_reader=enable_split_reader,
        enable_subblock_padding=False,
        output_layout=output_layout,
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
        tt_weight_tensor = ttnn.to_device(tt_weight_tensor, device)
        tt_weight_tensor = ttnn.prepare_conv_transpose2d_weights(
            weight_tensor=tt_weight_tensor,
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
            has_bias=has_bias,
            groups=groups,
            device=device,
            conv_config=conv_config,
            compute_config=compute_config,
            mirror_kernel=mirror_kernel,
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
        groups=groups,
        mirror_kernel=mirror_kernel,
        return_output_dim=True,
        return_weights_and_bias=True,
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


@skip_for_grayskull()
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
        # # (1, 16, 16, 32, 32, 3, 3, 2, 2, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED), # Issue with reading block sharded tensor
        # Vanilla Unet
        # Filter Size = 2 not supported in Block sharded
        # (1, 30, 40, 512, 256, 3, 3, 2, 2, 1, 1, 1, 1,  {"act_block_h": 64}, ttnn.TensorMemoryLayout.BLOCK_SHARDED), # Issue with reading block sharded tensor
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
    use_program_cache,
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


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
# fmt: off
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, config, enable_split_reader, enable_act_double_buffer, shard_layout",
    (
        (1, 64, 8, 64, 64, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True, True, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 128, 16, 128, 64, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True, True, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 256, 32, 128, 64, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True, True, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 512, 64, 128, 2, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True, True, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
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
    use_program_cache,
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
    enable_split_reader,
    enable_act_double_buffer,
    shard_layout,
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
        shard_layout=shard_layout,
        auto_shard=True,
        enable_split_reader=enable_split_reader,
        enable_act_double_buffer=enable_act_double_buffer,
        mirror_kernel=mirror_kernel,
    )
