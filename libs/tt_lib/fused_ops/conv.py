from typing import List, Union
from .. import tensor
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
            [R, S, U, V, P_H, P_W],
            act_block_h,
            act_block_w,
            weight_block_w,
            out_subblock_h,
            out_subblock_w,
            K,
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


def resnet_conv(weight: List[Union[int, float]], conv_params, device, bias=None):
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
    use_fast_reader = True
    if (C >= 512):
        use_fast_reader = False
    # Hardcode block shapes for conv op
    act_block_h = 4
    act_block_w = (int)((_nearest_32(_nearest_y(C,16)*S))/32)
    if not use_fast_reader:
        act_block_w = 4
    weight_block_h = act_block_w
    weight_block_w = 4
    out_subblock_h = 4
    out_subblock_w = 2

    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, S]
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
        if use_fast_reader:
            weight_tiled_ = tensor.convert_conv_weight_tensor_to_special_padding_tiled_layout(
                weight_untiled, weight_block_h, weight_block_w
            )
        else:
            weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(
                weight_untiled, weight_block_h, weight_block_w
            )
    weight_on_device = weight_tiled_.to(device)

    if bias is None:
        bias_on_device = None
    else:
        bias_shape = [1, 1, 1, K]
        bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
        bias_ = (
            tensor.Tensor(
                bias, bias_shape, tensor.DataType.BFLOAT16, tensor.Layout.ROW_MAJOR
            )
            .pad(bias_channels_padded_shape, (0, 0, 0, 0), 0)
            .to(tensor.Layout.TILE)
        )
        bias_on_device = bias_.to(device)

    def conv_(activation):
        # if conv1x1 stride 1 padding 0, use matmul op
        if use_regular_matmul_op:
            # if(activation.layout() == tensor.Layout.ROW_MAJOR):
            #     activation = activation.reshape(1, 1, activation.shape()[0] * activation.shape()[1] * activation.shape()[2], activation.shape()[3])
            #     activation_padded_shape = tensor.pad_to_tile_shape(activation.shape(), False, False, True, True)
            #     activation = tensor.format_input_tensor(activation, device, activation_padded_shape, 0.0, tensor.Layout.TILE)
            assert(activation.layout() == tensor.Layout.TILE)
            output = tensor.matmul(activation, weight_on_device, activation.memory_config())
        else:
            assert(activation.layout() == tensor.Layout.ROW_MAJOR)
            if use_fast_reader:
                output = tensor.conv_with_fast_reader(activation, weight_on_device, [R,S,U,V,P_H,P_W], act_block_h, act_block_w, weight_block_w, out_subblock_h, out_subblock_w, K)
            else:
                output = tensor.conv(activation, weight_on_device, [R,S,U,V,P_H,P_W], act_block_h, act_block_w, weight_block_w, out_subblock_h, out_subblock_w, K)
            assert(output.layout() == tensor.Layout.ROW_MAJOR)
        assert(output.storage_type() == tensor.StorageType.DEVICE)

        if bias_on_device is not None:
            if output.layout() == tensor.Layout.ROW_MAJOR:
                # convert to tile layout
                output = output.reshape(1, 1, output.shape()[0] * output.shape()[1] * output.shape()[2], output.shape()[3])
                output_padded_shape = tensor.pad_to_tile_shape(output.shape(), False, False, True, True)
                output = tensor.format_input_tensor(output, device, output_padded_shape, 0.0, tensor.Layout.TILE)
            output_plus_bias = tensor.bcast_without_autoformat(output, bias_on_device, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H, output.memory_config())
            return output_plus_bias

        return output

    return conv_
