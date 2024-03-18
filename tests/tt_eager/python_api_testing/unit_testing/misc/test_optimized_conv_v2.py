# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_allclose_and_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.demos.resnet.tt.metalResnetBlock50 import (
    compute_conv_output_shape,
    resnet50_1x1_conv_as_matmul,
    resnet50_optimized_conv,
    _nearest_32,
    _nearest_y,
    format_tensor,
)

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_composite_conv import (
    TTPyCompositeConv,
    SlidingWindowOpParams,
)


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, is_1d_systolic, bias, untilize_out, fuse_relu",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, True, False, False),
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, True, False, False),
        # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True, True, False, False),
        (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True, True, False, False),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True, True, False, False),
        # rn50 layer3
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False, True, False, False),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False, True, False, False),
        # rn50 layer4
        (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False, True, False, False),
        (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False, True, False, False),
        # sd conv
        (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, False, True, False, False),
        # Small conv
        (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, True, False, False, False),
        (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, True, False, False, True),
        # (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, False, True, False), # Asserts #5323
        # Untilize out
        (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, True, False, True, False),
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, False, True, False),
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, True, True, False),
        (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, False, True, True, False),
        (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, False, True, True, True),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["weights_BFLOAT16", "weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["activations_BFLOAT16", "activations_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "math_fidelity", [tt_lib.tensor.MathFidelity.HiFi4, tt_lib.tensor.MathFidelity.LoFi], ids=["HiFi4", "LoFi"]
)
def test_optimized_conv_v2(
    device,
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
    is_1d_systolic,
    bias,
    untilize_out,
    fuse_relu,
):
    if input_channels == 16:
        pytest.skip("These tests are hanging in interleaved_to_sharded after rebase. Issue: #4336")

    if math_fidelity != tt_lib.tensor.MathFidelity.LoFi:
        pytest.skip(
            "By default, only run tests with LoFi math for pipelines. For local unit testing, enable the other variants by uncommenting the skip here!"
        )

    if untilize_out and activations_dtype == tt_lib.tensor.DataType.BFLOAT8_B:
        pytest.skip("We can only untilize to non-bfp formats")

    if (
        activations_dtype == tt_lib.tensor.DataType.BFLOAT16
        and batch_size == 20
        and (
            output_channels == 64
            or (
                stride_h == 2
                and (
                    output_channels == 256
                    or (output_channels == 128 and weights_dtype == tt_lib.tensor.DataType.BFLOAT16)
                )
            )
        )
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    assert output_channels % 32 == 0
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
    conv_input_shape_nhwc = conv_input_pyt_nhwc.shape
    conv_weight_pyt = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    conv_bias_pyt = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if bias else None
    out_golden = torch.nn.functional.conv2d(
        conv_input_pyt,
        conv_weight_pyt,
        bias=conv_bias_pyt.reshape(-1) if bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
    )

    if fuse_relu:
        out_golden = torch.nn.ReLU()(out_golden)

    conv_params = [output_channels, input_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, 1, 1]
    conv_output_shape = compute_conv_output_shape(conv_params, conv_input_shape_nhwc)
    logger.info(f"Conv output shape - {conv_output_shape}")

    sliding_window_op_params = SlidingWindowOpParams(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        window_h=filter_height,
        window_w=filter_width,
        batch_size=batch_size,
        input_h=input_height,
        input_w=input_width,
    )

    reader_patterns_cache = {}

    tt_tensor_conv_weight = tt_lib.tensor.Tensor(
        conv_weight_pyt.reshape(-1).tolist(),
        conv_weight_pyt.shape,
        weights_dtype if weights_dtype != tt_lib.tensor.DataType.BFLOAT8_B else tt_lib.tensor.DataType.FLOAT32,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )
    tt_tensor_conv_bias = (
        tt_lib.tensor.Tensor(
            conv_bias_pyt.reshape(-1).tolist(),
            conv_bias_pyt.shape,
            weights_dtype if weights_dtype != tt_lib.tensor.DataType.BFLOAT8_B else tt_lib.tensor.DataType.FLOAT32,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        if bias
        else None
    )

    conv = TTPyCompositeConv(
        sliding_window_op_params,
        tt_tensor_conv_weight,
        output_channels,
        input_channels,
        device,
        is_1d_systolic,
        reader_patterns_cache,
        bias=tt_tensor_conv_bias,
        weights_dtype=weights_dtype,
        output_dtype=activations_dtype,
        math_fidelity=math_fidelity,
        deallocate_activation=True,
        output_layout=tt_lib.tensor.Layout.ROW_MAJOR if untilize_out else tt_lib.tensor.Layout.TILE,
        fuse_relu=fuse_relu,
    )

    conv_input = tt_lib.tensor.Tensor(
        conv_input_pyt_nhwc.reshape(-1).tolist(),
        conv_input_pyt_nhwc.shape,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )

    conv_input_on_device = conv.copy_input_to_device(conv_input)

    # Optimized conv v2
    output_on_device = conv(conv_input_on_device)

    # Copy sharded output on host
    # out is in row major layout and NHWC shape
    out = conv.copy_output_from_device(output_on_device)

    assert out.get_layout() == tt_lib.tensor.Layout.ROW_MAJOR

    out_result = out.to_torch()
    # NHWC to NCHW
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)

    if math_fidelity == tt_lib.tensor.MathFidelity.LoFi and activations_dtype == tt_lib.tensor.DataType.BFLOAT8_B:
        pcc = 0.998
    else:
        pcc = 0.999
    passing_pcc, info = comp_pcc(out_golden, out_result, pcc=pcc)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Info={info}")
    assert passing_pcc


def test_simple(
    device,
    use_program_cache,
):
    math_fidelity = tt_lib.tensor.MathFidelity.LoFi
    activations_dtype = tt_lib.tensor.DataType.BFLOAT16
    weights_dtype = tt_lib.tensor.DataType.BFLOAT8_B
    batch_size = 1
    output_channels = 128
    input_channels = 128
    input_height = 16
    input_width = 16
    filter_height = 3
    filter_width = 3
    stride_h = 2
    stride_w = 2
    pad_h = filter_height // 2
    pad_w = filter_width // 2
    is_1d_systolic = False
    untilize_out = False
    bias = False
    config = None  # {"act_reshard_num_cores_nhw": 2}
    debug = True
    fuse_relu = False

    assert output_channels % 32 == 0
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    if debug:
        volume = conv_input_shape[0] * conv_input_shape[1] * conv_input_shape[2] * conv_input_shape[3]
        conv_input_pyt = torch.arange(volume, dtype=torch.bfloat16).reshape(conv_input_shape)
        conv_weight_pyt = torch.zeros(conv_weight_shape, dtype=torch.bfloat16)
        conv_weight_pyt[0, 0, 0, 0] = 1.0
    else:
        conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16)
        conv_weight_pyt = torch.randn(conv_weight_shape, dtype=torch.bfloat16)
    conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
    conv_input_shape_nhwc = conv_input_pyt_nhwc.shape
    conv_bias_pyt = torch.randn(conv_bias_shape, dtype=torch.bfloat16) if bias else None
    out_golden = torch.nn.functional.conv2d(
        conv_input_pyt,
        conv_weight_pyt,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        bias=conv_bias_pyt.reshape(-1) if bias else None,
    )

    if fuse_relu:
        out_golden = torch.nn.ReLU()(out_golden)

    conv_params = [output_channels, input_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, 1, 1]
    conv_output_shape = compute_conv_output_shape(conv_params, conv_input_shape_nhwc)
    logger.info(f"Conv output shape - {conv_output_shape}")

    sliding_window_op_params = SlidingWindowOpParams(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        window_h=filter_height,
        window_w=filter_width,
        batch_size=batch_size,
        input_h=input_height,
        input_w=input_width,
    )

    reader_patterns_cache = {}

    tt_tensor_conv_weight = tt_lib.tensor.Tensor(
        conv_weight_pyt.reshape(-1).tolist(),
        conv_weight_pyt.shape,
        weights_dtype if weights_dtype != tt_lib.tensor.DataType.BFLOAT8_B else tt_lib.tensor.DataType.FLOAT32,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )
    tt_tensor_conv_bias = (
        tt_lib.tensor.Tensor(
            conv_bias_pyt.reshape(-1).tolist(),
            conv_bias_pyt.shape,
            weights_dtype if weights_dtype != tt_lib.tensor.DataType.BFLOAT8_B else tt_lib.tensor.DataType.FLOAT32,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        if bias
        else None
    )

    conv = TTPyCompositeConv(
        sliding_window_op_params,
        tt_tensor_conv_weight,
        output_channels,
        input_channels,
        device,
        is_1d_systolic,
        reader_patterns_cache,
        bias=tt_tensor_conv_bias,
        conv_blocking_and_parallelization_config_override=config,
        weights_dtype=weights_dtype,
        output_dtype=activations_dtype,
        math_fidelity=math_fidelity,
        deallocate_activation=True,
        output_layout=tt_lib.tensor.Layout.ROW_MAJOR if untilize_out else tt_lib.tensor.Layout.TILE,
        fuse_relu=fuse_relu,
    )

    conv_input_pyt_nhwc = conv_input_pyt_nhwc.reshape(
        (
            1,
            1,
            conv_input_pyt_nhwc.shape[0] * conv_input_pyt_nhwc.shape[1] * conv_input_pyt_nhwc.shape[2],
            conv_input_pyt_nhwc.shape[3],
        ),
    )
    conv_input = tt_lib.tensor.Tensor(
        conv_input_pyt_nhwc.reshape(-1).tolist(),
        conv_input_pyt_nhwc.shape,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )

    # Remove the else block when resolved (https://github.com/tenstorrent-metal/tt-metal/issues/6310):
    if is_1d_systolic:
        conv_input = conv_input.to(device, conv.input_sharded_memory_config)
    else:
        interleaved_mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
        )
        conv_input = conv_input.to(device, interleaved_mem_config)
        conv_input = tt_lib.tensor.interleaved_to_sharded(conv_input, conv.input_sharded_memory_config)

    # Optimized conv v2
    output_on_device = conv(conv_input)

    # Copy sharded output on host
    # out is in row major layout and NHWC shape
    out = conv.copy_output_from_device(output_on_device)

    assert out.get_layout() == tt_lib.tensor.Layout.ROW_MAJOR

    out_result = out.to_torch()
    # NHWC to NCHW
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)

    if math_fidelity == tt_lib.tensor.MathFidelity.LoFi and activations_dtype == tt_lib.tensor.DataType.BFLOAT8_B:
        pcc = 0.998
    else:
        pcc = 0.999
    passing_pcc, info = comp_pcc(out_golden, out_result, pcc=pcc)
    print("Passing=", passing_pcc)
    print("Info=", info)
    if debug:
        print("golden", out_golden[0, 0])
        print("result", out_result[0, 0])
    assert passing_pcc
