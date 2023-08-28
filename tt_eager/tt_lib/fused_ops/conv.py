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
            bias_on_device,
            [R, S, U, V, P_H, P_W],
            act_block_h,
            act_block_w,
            weight_block_w,
            out_subblock_h,
            out_subblock_w,
            K,
            bias != None
        )

        assert output.storage_type() == tensor.StorageType.DEVICE

        # if bias_on_device is not None:
        #     output_plus_bias = tensor.bcast(
        #         output, bias_on_device, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H
        #     )
        #     if output_plus_bias.layout() != tensor.Layout.ROW_MAJOR:
        #         assert output_plus_bias.layout() == tensor.Layout.TILE
        #         assert output_plus_bias.storage_type() == tensor.StorageType.DEVICE
        #         output_plus_bias = tensor.untilize(
        #             output_plus_bias, output_plus_bias.memory_config()
        #         )
        #         assert output_plus_bias.layout() == tensor.Layout.ROW_MAJOR
        #     return output_plus_bias

        return output

    return conv_

def resnet_1x1conv_as_mm(weight: List[Union[int, float]], conv_params, device, matmul_config, bias, fuse_relu=False):
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
                                            output_dtype=activation.dtype())

        return output

    return conv_

def resnet_conv(weight: List[Union[int, float]], conv_params, device, act_block_shape_hw, weight_block_shape_hw, outsubblock_shape_hw, bias=None, padded_filter_window_width=0, pre_pad_conv=False, matmul_config=None, fuse_relu=False, enable_fused_bias=True):
    """
    Returns a function that performs a Convolution.
    For bias, it calls bcast op without autoformatting
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    use_regular_matmul_op = False
    if R == S and R == 1 and P_H == P_W and P_H == 0 and U == V and U == 1:
        # use regular matmul op
        use_regular_matmul_op = True
        #assert matmul_config is not None

    if not use_regular_matmul_op:
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
    if padded_filter_window_width == 0:
        padded_filter_window_width = S
    assert padded_filter_window_width >= S
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, padded_filter_window_width]
    weight_untiled = tensor.Tensor(
        weight, weights_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
    ).pad(weights_channels_padded_shape, (0, 0, 0, 0), 0)

    if use_regular_matmul_op:
        # weight for matmul op
        weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(
            weight_untiled, 1, 1
        )
    else:
        # for conv op, pad the weights to block shape
        weight_tiled_ = tensor.convert_conv_weight_tensor_to_special_padding_tiled_layout(
            weight_untiled, weight_block_h, weight_block_w
        )
    weight_on_device = weight_tiled_.to(device)

    if bias is None:
        bias_on_device = None
        enable_bias = False
    else:
        bias_shape = [1, 1, 1, K]
        assert(use_regular_matmul_op or (not use_regular_matmul_op) and K % (weight_block_w * 32) == 0)
        bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
        bias_ = (
            tensor.Tensor(
                bias, bias_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
            )
            .pad_to_tile(0)
            .to(tensor.Layout.TILE)
        )
        bias_on_device = bias_.to(device)

    if not enable_fused_bias:
        bias_for_fused = None
    else:
        bias_for_fused = bias_on_device

    if pre_pad_conv:
        P_H = 0
        P_W = 0
    matmul_program_config = None
    if matmul_config is not None:
        matmul_program_config = operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                                    compute_with_storage_grid_size=matmul_config["compute_with_storage_grid_size"],
                                    in0_block_w=matmul_config["in0_block_w"],
                                    out_subblock_h=matmul_config["out_subblock_h"],
                                    out_subblock_w=matmul_config["out_subblock_w"],
                                    per_core_M=matmul_config["per_core_M"],
                                    per_core_N=matmul_config["per_core_N"],
                                    fused_activation=tensor.FusibleActivationWithParam(tensor.FusibleActivation.RELU) if fuse_relu else None)

    def conv_(activation):
        # if conv1x1 stride 1 padding 0, use matmul op
        if use_regular_matmul_op:
            assert(activation.layout() == tensor.Layout.TILE)
            if matmul_program_config is not None:
                output = operations.primary.matmul(activation, weight_on_device, bias=bias_on_device, program_config=matmul_program_config,
                                                   output_mem_config=activation.memory_config(),
                                                   output_dtype=activation.dtype())
            else:
                output = tensor.matmul(activation, weight_on_device, activation.memory_config())
        else:
            assert(activation.layout() == tensor.Layout.ROW_MAJOR)
            output = tensor.conv_with_fast_reader(activation, weight_on_device, bias_for_fused, [R,padded_filter_window_width,U,V,P_H,P_W], act_block_h, act_block_w, weight_block_w, out_subblock_h, out_subblock_w, K, False, bias_for_fused is not None)
        assert(output.storage_type() == tensor.StorageType.DEVICE)

        if (use_regular_matmul_op and matmul_program_config is None and bias_on_device is not None) or (not use_regular_matmul_op and not enable_fused_bias):
            assert output.layout() == tensor.Layout.TILE
            if output.layout() == tensor.Layout.ROW_MAJOR:
                # convert to tile layout
                output = output.reshape(1, 1, output.shape()[0] * output.shape()[1] * output.shape()[2], output.shape()[3])
                output_padded_shape = tensor.pad_to_tile_shape(output.shape(), False, False, True, True)
                output = tensor.format_input_tensor(output, device, output_padded_shape, 0.0, tensor.Layout.TILE)
            output_plus_bias = tensor.bcast_without_autoformat(output, bias_on_device, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H, output.memory_config())
            return output_plus_bias
        return output

    return conv_
