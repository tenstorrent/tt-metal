from gpai import gpai

def Linear(in_features, out_features, weight, bias, device):

    weight = gpai.tensor.Tensor(
        weight,
        [1, 1, out_features, in_features],
        gpai.tensor.DataType.BFLOAT16,
        gpai.tensor.Layout.TILE,
        device
    )

    if bias is None:
        bias = None
    else:
        bias = gpai.tensor.Tensor(
            bias,
            [1, 1, 32, out_features],
            gpai.tensor.DataType.BFLOAT16,
            gpai.tensor.Layout.TILE,
            device
        )

    def linear_(activation):
        weight_T = gpai.tensor.transpose(weight)
        output = gpai.tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = gpai.tensor.bcast(output, bias, gpai.tensor.BcastOpMath.ADD, gpai.tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_
