"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

from typing import List, Union
from .. import tensor, operations
from ..utils import _nearest_32, _nearest_y


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
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C,  16), R, S]
    weight_untiled = tensor.Tensor(
        weight, weights_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
    ).pad(weights_channels_padded_shape, (0, 0, 0, 0), 0)
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(
        weight_untiled, weight_block_h, weight_block_w
    )
    weight_on_device = weight_tiled_.to(device)
    if bias is None:
        bias_on_device = None
    else:
        bias_shape = [1, 1, 1, K]
        bias_channels_padded_shape = [1, 1, 1, _nearest_32(K)]
        bias_ = tensor.Tensor(
            bias, bias_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
        ).pad(bias_channels_padded_shape, (0, 0, 0, 0), 0)
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
            False
        )

        assert output.storage_type() == tensor.StorageType.DEVICE

        if bias_on_device is not None:
            output_plus_bias = tensor.bcast(
                output, bias_on_device, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H
            )
            if output_plus_bias.layout() != tensor.Layout.ROW_MAJOR:
                assert output_plus_bias.layout() == tensor.Layout.TILE
                assert output_plus_bias.storage_type() == tensor.StorageType.DEVICE
                output_plus_bias = tensor.untilize(
                    output_plus_bias, output_plus_bias.memory_config()
                )
                assert output_plus_bias.layout() == tensor.Layout.ROW_MAJOR
            return output_plus_bias

        return output

    return conv_

def resnet50_1x1_conv_as_matmul(weight: List[Union[int, float]], conv_params, device, bias, matmul_config, fuse_relu=False):
    """
    Returns a function that performs a Convolution.
    For bias, it calls bcast op without autoformatting
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    assert R == S and R == 1 and P_H == P_W and P_H == 0 and U == V and U == 1
    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]

    assert C % 16 == 0
    assert K % 32 == 0

    weight_untiled = tensor.Tensor(
        weight, weights_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
    )
    # weight for matmul op
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(
        weight_untiled, 1, 1
    )

    weight_on_device = weight_tiled_.to(device)
    bias_shape = [1, 1, 1, K]
    bias_channels_padded_shape = [1, 1, 32, K]
    bias_ = (
        tensor.Tensor(
            bias, bias_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
        )
        .pad(bias_channels_padded_shape, (0, 0, 0, 0), 0)
        .to(tensor.Layout.TILE)
    )
    bias_on_device = bias_.to(device)
    matmul_program_config = operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                                    compute_with_storage_grid_size=matmul_config["compute_with_storage_grid_size"],
                                    in0_block_w=matmul_config["in0_block_w"],
                                    out_subblock_h=matmul_config["out_subblock_h"],
                                    out_subblock_w=matmul_config["out_subblock_w"],
                                    per_core_M=matmul_config["per_core_M"],
                                    per_core_N=matmul_config["per_core_N"],
                                    fused_activation=tensor.FusibleActivationWithParam(tensor.FusibleActivation.RELU) if fuse_relu else None)

    def conv_(activation):
        # conv1x1 stride 1 padding 0, use matmul op
        output = operations.primary.matmul(activation, weight_on_device, bias=bias_on_device, program_config=matmul_program_config,
                                            output_mem_config=activation.memory_config(),
                                            output_dtype=activation.dtype(),
                                            math_fidelity=tensor.MathFidelity.HiFi4)

        return output

    return conv_

def resnet50_optimized_conv(weight: List[Union[int, float]], conv_params, device, act_block_shape_hw, weight_block_shape_hw, outsubblock_shape_hw, bias, fuse_relu=False):
    """
    Returns a function that performs a Convolution.
    For bias, it calls bcast op without autoformatting
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    assert len(act_block_shape_hw) == 2
    assert len(weight_block_shape_hw) == 2
    assert len(outsubblock_shape_hw) == 2
    assert act_block_shape_hw[1] == weight_block_shape_hw[0]
    assert act_block_shape_hw[0] % 32 == 0
    assert act_block_shape_hw[1] % 32 == 0

    act_block_h = (int) (act_block_shape_hw[0]/32)
    act_block_w =(int) (act_block_shape_hw[1]/32)
    weight_block_h = act_block_w
    weight_block_w = (int) (weight_block_shape_hw[1]/32)
    out_subblock_h = (int) (outsubblock_shape_hw[0]/32)
    out_subblock_w = (int) (outsubblock_shape_hw[1]/32)
    assert out_subblock_h * out_subblock_w <= 8


    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, S]
    weight_untiled = tensor.Tensor(
        weight, weights_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
    ).pad(weights_channels_padded_shape, (0, 0, 0, 0), 0)

    # for conv op, pad the weights to block shape
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_special_padding_tiled_layout(
        weight_untiled, weight_block_h, weight_block_w
    )
    weight_on_device = weight_tiled_.to(device)

    bias_shape = [1, 1, 1, K]
    assert(K % (weight_block_w * 32) == 0)
    bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
    bias_ = (
        tensor.Tensor(
            bias, bias_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
        )
        .pad_to_tile(0)
        .to(tensor.Layout.TILE)
    )
    bias_on_device = bias_.to(device)
    def conv_(activation):
        #assert(activation.layout() == tensor.Layout.ROW_MAJOR)
        output = tensor.optimized_conv(activation, weight_on_device, bias_on_device, [R,S,U,V,P_H,P_W], act_block_h, act_block_w, weight_block_w, out_subblock_h, out_subblock_w, K, False, True, fuse_relu, tensor.MathFidelity.HiFi4)
        #assert(output.storage_type() == tensor.StorageType.DEVICE)
        return output

    return conv_

def resnet50_first_conv(weight: List[Union[int, float]], conv_params, device, act_block_shape_hw, weight_block_shape_hw, outsubblock_shape_hw, bias, padded_filter_window_width):
    """
    Returns a function that performs a Convolution.
    For bias, it calls bcast op without autoformatting
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]


    assert len(act_block_shape_hw) == 2
    assert len(weight_block_shape_hw) == 2
    assert len(outsubblock_shape_hw) == 2
    assert act_block_shape_hw[1] == weight_block_shape_hw[0]
    assert act_block_shape_hw[0] % 32 == 0
    assert act_block_shape_hw[1] % 32 == 0

    act_block_h = (int) (act_block_shape_hw[0]/32)
    act_block_w =(int) (act_block_shape_hw[1]/32)
    weight_block_h = act_block_w
    weight_block_w = (int) (weight_block_shape_hw[1]/32)
    out_subblock_h = (int) (outsubblock_shape_hw[0]/32)
    out_subblock_w = (int) (outsubblock_shape_hw[1]/32)
    assert out_subblock_h * out_subblock_w <= 8


    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]
    assert padded_filter_window_width >= S
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, padded_filter_window_width]
    weight_untiled = tensor.Tensor(
        weight, weights_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
    ).pad(weights_channels_padded_shape, (0, 0, 0, 0), 0)

    # for conv op, pad the weights to block shape
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_special_padding_tiled_layout(
        weight_untiled, weight_block_h, weight_block_w
    )
    weight_on_device = weight_tiled_.to(device)

    bias_shape = [1, 1, 1, K]
    assert(K % (weight_block_w * 32) == 0)
    bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
    bias_ = (
        tensor.Tensor(
            bias, bias_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
        )
        .pad_to_tile(0)
        .to(tensor.Layout.TILE)
    )
    bias_on_device = bias_.to(device)

    # Resnet50 first conv is pre-padded on host
    P_H = 0
    P_W = 0
    def conv_(activation):
        #assert(activation.layout() == tensor.Layout.ROW_MAJOR)
        output = tensor.optimized_conv(activation, weight_on_device, None, [R,padded_filter_window_width,U,V,P_H,P_W], act_block_h, act_block_w, weight_block_w, out_subblock_h, out_subblock_w, K, False, False, False, tensor.MathFidelity.HiFi4)
        #assert(output.storage_type() == tensor.StorageType.DEVICE)
        #assert output.layout() == tensor.Layout.TILE
        output_plus_bias = tensor.bcast_without_autoformat(output, bias_on_device, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H, output.memory_config())
        return output_plus_bias

    return conv_
