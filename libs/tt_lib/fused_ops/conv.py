from typing import List, Union
from .. import tensor
from libs.tt_lib.utils import _nearest_32

def conv(weight: List[Union[int, float]], conv_params, device, bias=None):
    """
    Returns a function that performs a Convolution.
    bias is optional. If provided, it must be in tiled layout
    """
    assert(len(conv_params) == 8)
    K, C, R, S, U, V, P_H, P_W = [conv_params[i] for i in range(8)]
    weights_shape = [K,C,R,S]
    weights_channels_padded_shape = [K,_nearest_32(C),R,S]
    weight_untiled = tensor.Tensor(
        weight,
        weights_shape,
        tensor.DataType.BFLOAT16,
        tensor.Layout.ROW_MAJOR
    ).pad(weights_channels_padded_shape, (0,0,0,0), 0)
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(weight_untiled)
    weight_on_device = weight_tiled_.to(device)
    if bias is None:
        bias = None
    else:
        bias = tensor.Tensor(
            bias,
            [1, 1, 1, K],
            tensor.DataType.BFLOAT16,
            tensor.Layout.ROW_MAJOR,
            device
        )

    def conv_(activation):
        [_,_,H,W] = activation.shape()
        OH = ((int) ((H - R + 2 * P_H) / U)) + 1
        OW = ((int) ((W - S + 2 * P_W) / V)) + 1
        conv_as_mm_output_shape = [1,1,_nearest_32(OH*OW),_nearest_32(K)]
        output = tensor.conv(activation, weight_on_device, [R,S,U,V,P_H,P_W], True)

        assert(output.shape() == conv_as_mm_output_shape)

        if bias is not None:
            assert False # unsupported
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return conv_
