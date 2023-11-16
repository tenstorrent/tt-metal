# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
from .. import tensor, operations
from ..utils import _nearest_32, _nearest_y
import torch
import numpy


def compute_conv_output_shape(conv_params, x_shape):
    H = x_shape[1]
    W = x_shape[2]
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    OH = ((int)((H - R + 2 * P_H) / U)) + 1
    OW = ((int)((W - S + 2 * P_W) / V)) + 1
    return [x_shape[0], OH, OW, K]


def conv_op_trace(conv_params, input_nhwc_shape):
    assert len(conv_params) == 10
    output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups = [
        conv_params[i] for i in range(10)
    ]
    assert dilation == 1 and groups == 1
    assert len(input_nhwc_shape) == 4
    input_n, input_h, input_w, input_c = [input_nhwc_shape[i] for i in range(4)]
    # indices in the following arrays are channel/stick indices
    # data indices array contains the input indices corresponding to the top left corner of filter window
    data_indices = []

    # 2 lists -
    # data_start_size holds start stick index and size for contigious sticks in tensor
    # pad_start_size holds start stick index and size for contingous padding sticks
    # stick indices in data_start_size are after pad insertion
    # Example for input of 4x8 and pad = 1

    # image 1 data (example sequential integer data)
    # 1  2  3  4  5  6  7  8
    # 9  10 11 12 13 14 15 16
    # 17 18 19 20 21 22 23 24
    # 25 26 27 28 29 30 31 32

    # image 2 data (example sequential integer data)
    # 33 34 35 36 37 38 39 40
    # 41 42 43 44 45 46 47 48
    # 49 50 51 52 53 54 55 56
    # 57 58 59 60 61 62 63 64

    # Concatenated image data (example sequential integer data from above)
    # Inserted padding above and between and on the sides of the images (pad = 1)
    # 0  0  0  0  0  0  0  0  0 0
    # 0  1  2  3  4  5  6  7  8 0
    # 0  9 10 11 12 13 14 15 16 0
    # 0 17 18 19 20 21 22 23 24 0
    # 0 25 26 27 28 29 30 31 32 0
    # 0  0  0  0  0  0  0  0  0 0
    # 0 33 34 35 36 37 38 39 40 0
    # 0 41 42 43 44 45 46 47 48 0
    # 0 49 50 51 52 53 54 55 56 0
    # 0 57 58 59 60 61 62 63 64 0
    # 0  0  0  0  0  0  0  0  0 0

    # We encode the above shown padded tensor via 2 lists -
    # pad_start_size: [(0, 11), (20, 2), ... ]
    # data_start_size: [(11, 8), (22, 8), ... ]

    data_start_size = []
    pad_start_size = []

    # image padding
    padded_input_h = input_h + (2 * pad_h)
    padded_input_w = input_w + (2 * pad_w)

    def update_start_size_list_and_coalesce(start_size_list, new_start, new_size):
        if len(start_size_list) > 0 and start_size_list[-1][0] == new_start - 1:
            start_size_list[-1][1] += new_size
        else:
            start_size_list.append([new_start, new_size])

    # trace the image and collect padding and data start and sizes
    # also, populate pad_input - for every index in holistic padded tensor, specify pad is true or false
    pad_input = []
    channel_idx = 0
    for n in range(input_n):
        # top padding
        if pad_h > 0:
            if n == 0:
                # add top padding only for first image
                pad_start_size.append([channel_idx, pad_h * padded_input_w])
                pad_input.extend([True] * pad_h * padded_input_w)
                channel_idx += pad_h * padded_input_w
        for ih in range(pad_h, input_h + pad_h):
            if pad_w > 0:
                # left padding
                update_start_size_list_and_coalesce(pad_start_size, channel_idx, pad_w)
                pad_input.extend([True] * pad_w)
                channel_idx += pad_w
            update_start_size_list_and_coalesce(data_start_size, channel_idx, input_w)
            pad_input.extend([False] * input_w)
            channel_idx += input_w
            if pad_w > 0:
                # right padding
                update_start_size_list_and_coalesce(pad_start_size, channel_idx, pad_w)
                pad_input.extend([True] * pad_w)
                channel_idx += pad_w
        # bottom padding
        if pad_h > 0:
            update_start_size_list_and_coalesce(pad_start_size, channel_idx, pad_h * padded_input_w)
            pad_input.extend([True] * pad_h * padded_input_w)
            channel_idx += pad_h * padded_input_w
    assert len(pad_input) == channel_idx
    # output image size
    [output_n, output_h, output_w, output_c] = compute_conv_output_shape(conv_params, input_nhwc_shape)
    assert input_n == output_n

    # trace index of the top-left position of sliding window in the padded output tensor == channel_idx
    for n in range(input_n):
        for oh in range(output_h):
            for ow in range(output_w):
                ih = oh * stride_h
                iw = ow * stride_w
                channel_idx = (n * (input_h + pad_h) * padded_input_w) + (ih * padded_input_w) + iw
                data_indices.append(channel_idx)

    return data_indices, pad_input, data_start_size, pad_start_size


def conv(weight: List[Union[int, float]], conv_params, device, bias=None):
    """
    Returns a function that performs a Convolution.
    For bias, it calls bcast op with autoformatting
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    # Hardcode block sizes
    act_block_h = 4
    act_block_w = 4
    weight_block_h = act_block_w
    weight_block_w = 4
    out_subblock_h = 4
    out_subblock_w = 2
    if dilation != 1 or groups != 1:
        return None
    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, S]
    weight_untiled = tensor.Tensor(weight, weights_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0
    )
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(weight_untiled, weight_block_h, weight_block_w)
    weight_on_device = weight_tiled_.to(device)
    if bias is None:
        bias_on_device = None
    else:
        bias_shape = [1, 1, 1, K]
        bias_channels_padded_shape = [1, 1, 1, _nearest_32(K)]
        bias_ = tensor.Tensor(bias, bias_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR).pad(
            bias_channels_padded_shape, (0, 0, 0, 0), 0
        )
        bias_on_device = bias_.to(device)

    def conv_(activation):
        output = tensor.conv(
            activation,
            weight_on_device,
            None,
            [R, S, U, V, P_H, P_W],
            act_block_h,
            act_block_w,
            weight_block_w,
            out_subblock_h,
            out_subblock_w,
            K,
            False,
        )

        assert output.storage_type() == tensor.StorageType.DEVICE

        if bias_on_device is not None:
            output_plus_bias = tensor.bcast(output, bias_on_device, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            if output_plus_bias.layout() != tensor.Layout.ROW_MAJOR:
                assert output_plus_bias.layout() == tensor.Layout.TILE
                assert output_plus_bias.storage_type() == tensor.StorageType.DEVICE
                output_plus_bias = tensor.untilize(output_plus_bias, output_plus_bias.memory_config())
                assert output_plus_bias.layout() == tensor.Layout.ROW_MAJOR
            return output_plus_bias

        return output

    return conv_


def resnet50_1x1_conv_as_matmul(
    weight: List[Union[int, float]],
    conv_params,
    device,
    bias,
    matmul_config,
    fuse_relu=False,
    output_mem_config=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
):
    """
    Returns a function that performs a Convolution. Bias is fused with matmul.
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    assert R == S and R == 1 and P_H == P_W and P_H == 0 and U == V and U == 1
    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]

    assert C % 16 == 0
    assert K % 32 == 0

    weights_untiled_dtype = weights_dtype if weights_dtype != tensor.DataType.BFLOAT8_B else tensor.DataType.FLOAT32
    weight_untiled = tensor.Tensor(weight, weights_shape, weights_untiled_dtype, tensor.Layout.ROW_MAJOR)
    # weight for matmul op
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(weight_untiled, 1, 1, output_dtype=weights_dtype)

    weight_on_device = weight_tiled_.to(device)
    bias_shape = [1, 1, 1, K]
    bias_channels_padded_shape = [1, 1, 32, K]
    bias = torch.nn.functional.pad(torch.Tensor(bias).reshape(bias_shape), (0, 0, 0, 31)).flatten().tolist()
    bias_ = tensor.Tensor(bias, bias_channels_padded_shape, weights_dtype, tensor.Layout.ROW_MAJOR).to(
        tensor.Layout.TILE
    )
    bias_on_device = bias_.to(device)
    if isinstance(matmul_config, dict):
        matmul_program_config = operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=matmul_config["compute_with_storage_grid_size"],
            in0_block_w=matmul_config["in0_block_w"],
            out_subblock_h=matmul_config["out_subblock_h"],
            out_subblock_w=matmul_config["out_subblock_w"],
            per_core_M=matmul_config["per_core_M"],
            per_core_N=matmul_config["per_core_N"],
            transpose_mcast=False,
            fused_activation=tensor.FusibleActivationWithParam(tensor.FusibleActivation.RELU) if fuse_relu else None,
        )
    else:
        matmul_program_config = matmul_config
        if fuse_relu:
            matmul_program_config.fused_activation = tensor.FusibleActivationWithParam(tensor.FusibleActivation.RELU)

    def conv_(activation):
        # conv1x1 stride 1 padding 0, use matmul op
        output = operations.primary.matmul(
            activation,
            weight_on_device,
            bias=bias_on_device,
            program_config=matmul_program_config,
            output_mem_config=activation.memory_config() if output_mem_config is None else output_mem_config,
            output_dtype=output_dtype,
            math_fidelity=math_fidelity,
        )

        return output

    return conv_


def resnet50_optimized_conv(
    weight: List[Union[int, float]],
    conv_params,
    device,
    act_block_shape_hw,
    weight_block_shape_hw,
    outsubblock_shape_hw,
    out_block_shape_h,
    grid_size,
    per_core_out_matrix_h_ntiles,
    per_core_weight_matrix_w_ntiles,
    bias,
    fuse_relu=False,
    output_mem_config=None,
    input_tensor_shape=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
    act_c_num_blocks=1,
):
    """
    Returns a function that performs a Convolution. Bias is fused with conv.
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    assert len(act_block_shape_hw) == 2
    assert len(weight_block_shape_hw) == 2
    assert len(outsubblock_shape_hw) == 2
    assert act_block_shape_hw[1] == weight_block_shape_hw[0]
    assert act_block_shape_hw[0] % 32 == 0
    assert act_block_shape_hw[1] % 32 == 0

    out_block_h = (int)(out_block_shape_h / 32)
    act_block_h = (int)(act_block_shape_hw[0] / 32)
    act_block_w = (int)(act_block_shape_hw[1] / 32)
    weight_block_h = act_block_w
    weight_block_w = (int)(weight_block_shape_hw[1] / 32)
    out_subblock_h = (int)(outsubblock_shape_hw[0] / 32)
    out_subblock_w = (int)(outsubblock_shape_hw[1] / 32)
    assert out_subblock_h * out_subblock_w <= 8

    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, S]
    weights_untiled_dtype = weights_dtype if weights_dtype != tensor.DataType.BFLOAT8_B else tensor.DataType.FLOAT32
    weight_untiled = tensor.Tensor(weight, weights_shape, weights_untiled_dtype, tensor.Layout.ROW_MAJOR).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0
    )
    act_block_w_equals_input_channels_x_filter_width = act_block_shape_hw[1] == (C * S)
    # for conv op, pad the weights to block shape
    if act_block_w_equals_input_channels_x_filter_width:
        weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(
            weight_untiled,
            weight_block_h,
            weight_block_w,
            output_dtype=weights_dtype,
        )
    else:
        if R == 1 and S == 1:
            assert C % act_block_shape_hw[1] == 0
        else:
            assert act_block_shape_hw[1] == C
        weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(
            weight_untiled,
            weight_block_h,
            weight_block_w,
            output_dtype=weights_dtype,
        )
    weight_on_device = weight_tiled_.to(device)
    bias_shape = [1, 1, 1, K]
    assert K % (weight_block_w * 32) == 0
    bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
    bias = (
        torch.nn.functional.pad(torch.Tensor(bias).reshape(bias_shape), (0, _nearest_32(K) - K, 0, 31))
        .flatten()
        .tolist()
    )
    bias_ = tensor.Tensor(bias, bias_channels_padded_shape, weights_dtype, tensor.Layout.ROW_MAJOR).to(
        tensor.Layout.TILE
    )
    bias_on_device = bias_.to(device)

    def conv_(activation):
        # assert(activation.layout() == tensor.Layout.ROW_MAJOR)
        output = tensor.optimized_conv(
            activation,
            weight_on_device,
            bias_on_device,
            [R, S, U, V, P_H, P_W],
            K,
            False,
            True,
            fuse_relu,
            math_fidelity,
            tensor.OptimizedConvParallelizationConfig(
                grid_size=grid_size,
                per_core_out_matrix_height_ntiles=per_core_out_matrix_h_ntiles,
                per_core_weight_matrix_width_ntiles=per_core_weight_matrix_w_ntiles,
            ),
            tensor.OptimizedConvBlockConfig(
                act_block_h_ntiles=act_block_h,
                act_block_w_ntiles=act_block_w,
                act_c_num_blocks=act_c_num_blocks,
                weight_block_w_ntiles=weight_block_w,
                out_block_h_ntiles=out_block_h,
                out_subblock_h_ntiles=out_subblock_h,
                out_subblock_w_ntiles=out_subblock_w,
            ),
            0,
            output_mem_config=activation.memory_config() if output_mem_config is None else output_mem_config,
            output_dtype=output_dtype,
            input_tensor_shape=input_tensor_shape,
        )
        # assert(output.storage_type() == tensor.StorageType.DEVICE)
        return output

    return conv_


def resnet50_first_conv(
    weight: List[Union[int, float]],
    conv_params,
    device,
    act_block_shape_hw,
    weight_block_shape_hw,
    outsubblock_shape_hw,
    out_block_shape_h,
    grid_size,
    per_core_out_matrix_h_ntiles,
    bias,
    padded_filter_window_width,
    fuse_relu=False,
    out_mem_config=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
):
    """
    Returns a function that performs a Convolution. Bias is fused with conv.
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    extra_padding_for_32B_alignment = 25

    assert len(act_block_shape_hw) == 2
    assert len(weight_block_shape_hw) == 2
    assert len(outsubblock_shape_hw) == 2
    assert act_block_shape_hw[1] == weight_block_shape_hw[0]
    assert act_block_shape_hw[0] % 32 == 0
    assert act_block_shape_hw[1] % 32 == 0

    out_block_h = (int)(out_block_shape_h / 32)
    act_block_h = (int)(act_block_shape_hw[0] / 32)
    act_block_w = (int)(act_block_shape_hw[1] / 32)
    weight_block_h = act_block_w
    weight_block_w = (int)(weight_block_shape_hw[1] / 32)
    out_subblock_h = (int)(outsubblock_shape_hw[0] / 32)
    out_subblock_w = (int)(outsubblock_shape_hw[1] / 32)
    assert out_subblock_h * out_subblock_w <= 8

    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]
    assert padded_filter_window_width >= S
    weights_channels_padded_shape = [
        _nearest_32(K),
        _nearest_y(C, 4),
        R,
        padded_filter_window_width,
    ]  # first conv channel padded to 4 only
    weights_untiled_dtype = weights_dtype if weights_dtype != tensor.DataType.BFLOAT8_B else tensor.DataType.FLOAT32
    weight_untiled = tensor.Tensor(weight, weights_shape, weights_untiled_dtype, tensor.Layout.ROW_MAJOR).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0
    )
    per_core_weight_matrix_w_ntiles = (int)(weights_channels_padded_shape[0] / 32)
    # for conv op, pad the weights to block shape
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_special_padding_tiled_layout(
        weight_untiled, weight_block_h, weight_block_w, output_dtype=weights_dtype
    )
    weight_on_device = weight_tiled_.to(device)

    bias_shape = [1, 1, 1, K]
    assert K % (weight_block_w * 32) == 0
    bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
    bias = (
        torch.nn.functional.pad(torch.Tensor(bias).reshape(bias_shape), (0, _nearest_32(K) - K, 0, 31))
        .flatten()
        .tolist()
    )
    bias_ = tensor.Tensor(bias, bias_channels_padded_shape, weights_dtype, tensor.Layout.ROW_MAJOR).to(
        tensor.Layout.TILE
    )
    bias_on_device = bias_.to(device)

    # Resnet50 first conv is pre-padded on host
    P_H = 0
    P_W = 0

    def conv_(activation):
        # assert(activation.layout() == tensor.Layout.ROW_MAJOR)
        output_plus_bias = tensor.optimized_conv(
            activation,
            weight_on_device,
            bias_on_device,
            [R, padded_filter_window_width, U, V, P_H, P_W],
            K,
            False,
            True,
            fuse_relu,
            math_fidelity,
            tensor.OptimizedConvParallelizationConfig(
                grid_size=grid_size,
                per_core_out_matrix_height_ntiles=per_core_out_matrix_h_ntiles,
                per_core_weight_matrix_width_ntiles=per_core_weight_matrix_w_ntiles,
            ),
            tensor.OptimizedConvBlockConfig(
                act_block_h_ntiles=act_block_h,
                act_block_w_ntiles=act_block_w,
                weight_block_w_ntiles=weight_block_w,
                out_block_h_ntiles=out_block_h,
                out_subblock_h_ntiles=out_subblock_h,
                out_subblock_w_ntiles=out_subblock_w,
            ),
            extra_padding_for_32B_alignment,
            out_mem_config,
            output_dtype,
        )
        # assert(output.storage_type() == tensor.StorageType.DEVICE)
        # assert output.layout() == tensor.Layout.TILE
        return output_plus_bias

    return conv_


def resnet50_1x1_conv_s2_as_downsample_and_matmul(
    weight: List[Union[int, float]],
    conv_params,
    downsample_params,
    device,
    bias,
    matmul_config,
    out_sharded_mem_config,
    weights_dtype,
    output_dtype,
    math_fidelity,
):
    """
    Returns a function that performs a Convolution. Bias is fused with matmul.
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    assert R == S and R == 1 and P_H == P_W and P_H == 0 and U == V and U == 2
    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]

    assert C % 16 == 0
    assert K % 32 == 0

    weights_untiled_dtype = weights_dtype if weights_dtype != tensor.DataType.BFLOAT8_B else tensor.DataType.FLOAT32
    weight_untiled = tensor.Tensor(weight, weights_shape, weights_untiled_dtype, tensor.Layout.ROW_MAJOR)
    # weight for matmul op
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(weight_untiled, 1, 1, output_dtype=weights_dtype)

    weight_on_device = weight_tiled_.to(device)
    bias_shape = [1, 1, 1, K]
    bias_channels_padded_shape = [1, 1, 32, K]
    bias = torch.nn.functional.pad(torch.Tensor(bias).reshape(bias_shape), (0, 0, 0, 31)).flatten().tolist()
    bias_ = tensor.Tensor(bias, bias_channels_padded_shape, weights_dtype, tensor.Layout.ROW_MAJOR).to(
        tensor.Layout.TILE
    )
    bias_on_device = bias_.to(device)
    if isinstance(matmul_config, dict):
        matmul_program_config = operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=matmul_config["compute_with_storage_grid_size"],
            in0_block_w=matmul_config["in0_block_w"],
            out_subblock_h=matmul_config["out_subblock_h"],
            out_subblock_w=matmul_config["out_subblock_w"],
            per_core_M=matmul_config["per_core_M"],
            per_core_N=matmul_config["per_core_N"],
            transpose_mcast=False,
            fused_activation=None,
        )
    else:
        matmul_program_config = matmul_config

    def conv_(activation):
        # downsample op
        output = tensor.downsample(activation, downsample_params, output_dtype=output_dtype)
        output = operations.primary.matmul(
            output,
            weight_on_device,
            bias=bias_on_device,
            program_config=matmul_program_config,
            output_mem_config=out_sharded_mem_config,
            output_dtype=output_dtype,
            math_fidelity=math_fidelity,
        )

        return output

    return conv_
