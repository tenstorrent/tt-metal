# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, WS, BS
import ttnn
import torch


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (256, 256, 8, 8, WS, None),
        (128, 128, 32, 32, BS, None),
        (16, 16, 256, 256, HS, {"act_block_h": 32}),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [True, False],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [False],
)
@pytest.mark.parametrize(
    "filter, padding",
    [
        [3, (1, 2, 2, 3)],
        [1, 0],
        [5, (2, 4, 3, 5)],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv_features(
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
    shard_layout,
    config,
    filter,
    stride,
    padding,
    output_layout,
    fp32_accum,
    packer_l1_acc,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and shard_layout == WS:
        pytest.skip("Bug in Width Sharded Row Major Tensor Creation when height%32!=0. #19408")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat16 and packer_l1_acc and fp32_accum:
        pytest.skip("skipping due to pack_untilize_dst issue!")

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
        filter,
        filter,
        stride,
        stride,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=True,
        run_twice=True,
    )

SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, output_channels, input_height, input_width, slice_type, num_slices, weights_dtype, activations_dtype, kernel, stride, padding, dilation, input_channels_alignment, act_block_h_override,  math_fidelity",
    # fmt: off
    (
        (10,   64,   4096,  512,   SliceHeight,   4,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1), 16, 32 * 16, ttnn.MathFidelity.LoFi  ),
        (128,  128,  1024,  1024,  SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (128,  16,   1024,  1024,  SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (16,   512,  128,   128,   SliceWidth,    2,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (256,  128,  1024,  1024,  SliceWidth,   32,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 32 * 4,  ttnn.MathFidelity.LoFi  ),
        (256,  256,  1024,  1024,  SliceWidth,   32,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 32 * 8,  ttnn.MathFidelity.LoFi  ),
        (256,  256,  512,   512,   SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (512,  512,  128,   128,   SliceWidth,    2,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 32 * 8,  ttnn.MathFidelity.LoFi  ),
        (512,  512,  256,   256,   SliceWidth,    2,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (512,  256,  512,   512,   SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (512,  512,  512,   512,   SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (64,   64,   384,   64,    SliceHeight,   3,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (64,   64,   1024,  128,   SliceHeight,   2,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (64,   64,   512,   64,    SliceHeight,   2,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (4,    32,   1024,  1024,  SliceWidth,    4,  ttnn.bfloat8_b, ttnn.bfloat16, (5, 5), (1, 1), (0, 0), (1, 1), 16, 32,      ttnn.MathFidelity.LoFi  ),
        (32,   48,   1020,  1020,  SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (0, 0), (2, 2), 32, 32 * 2,  ttnn.MathFidelity.LoFi  ),
        (48,   56,   1016,  1016,  SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (0, 0), (4, 4), 32, 32 * 3,  ttnn.MathFidelity.LoFi  ),
        (56,   64,   1008,  256,   SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (0, 0), (8, 8), 32, 0,       ttnn.MathFidelity.LoFi  ),
        (64,   128,  992,   992,   SliceWidth,   64,  ttnn.bfloat8_b, ttnn.bfloat16, (2, 2), (1, 1), (0, 0), (1, 1), 32, 32 * 4,  ttnn.MathFidelity.LoFi  ),
    )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, False, False]],
)
def test_conv_dram(
    device,
    use_program_cache,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    activations_dtype,
    slice_type,
    num_slices,
    kernel,
    stride,
    padding,
    dilation,
    input_channels_alignment,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
):
    filter_height = kernel[0]
    filter_width = kernel[1]
    batch_size = 1
    groups = 1

    import time

    torch.manual_seed(time.time())
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]

    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor_nchw = torch_input_tensor_nchw.broadcast_to(conv_input_shape).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()

    tt_bias_tensor = None
    torch_bias_tensor = None
    if has_bias:
        torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() * 50
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
        torch_bias_tensor = torch_bias_tensor.reshape(-1)
    ref = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
    )

    reader_patterns_cache = {}
    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn.bfloat16)
    conv_slice_config = ttnn.Conv2dSliceConfig(
        slice_type=slice_type,
        num_slices=num_slices,
    )

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        act_block_h_override=act_block_h_override,
        input_channels_alignment=input_channels_alignment,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )

    [tt_output_tensor_on_device, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=stride,
        padding=padding,
        dilation=dilation,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache=reader_patterns_cache,
        groups=1,
        slice_config=conv_slice_config,
        return_output_dim=True,
    )

    out = ttnn.to_torch(tt_output_tensor_on_device)

    # out is in row major layout and NHWC shape
    # NHWC to NCHW
    # ref = torch.permute(ref, (0, 2, 3, 1))
    out = out.reshape(batch_size, out_height, out_width, output_channels)

    ref = torch.permute(ref, (0, 2, 3, 1))
    reader_patterns_cache.clear()

    pcc = 0.999
    diff = torch.abs(out - ref)
    # Sort the diff tensor and take the top 1% of the values.
    print("Sorting the diff tensor")
    diff, _ = torch.sort(diff.flatten(), descending=True)
    diff = diff[: diff.shape[0] // 100]

    abs_ref = ref.abs()
    abs_ref_mean = abs_ref.mean()
    scaled_diff_mean = diff.mean() / abs_ref_mean
    logger.info(f"Scaled diff mean = {scaled_diff_mean} ")
    if scaled_diff_mean > 0.01:
        passing, pcc_msg = check_with_pcc_without_tensor_printout(out, ref, pcc=pcc)
        logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
        if not passing:
            logger.error("Fails with PCC ", pcc_msg)
        assert passing



