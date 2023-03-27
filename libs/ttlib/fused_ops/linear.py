from typing import List, Union
from .. import tensor

def Linear(in_features: int, out_features: int, weight: List[Union[int, float]], bias, device):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be the weight as a tilized list of values.
    """
    weight = tensor.Tensor(
        weight,
        [1, 1, out_features, in_features],
        tensor.DataType.BFLOAT16,
        tensor.Layout.TILE,
        device
    )

    if bias is None:
        bias = None
    else:
        bias = tensor.Tensor(
            bias,
            [1, 1, 32, out_features],
            tensor.DataType.BFLOAT16,
            tensor.Layout.TILE,
            device
        )

    def linear_(activation):
        weight_T = tensor.transpose(weight)
        output = tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_
