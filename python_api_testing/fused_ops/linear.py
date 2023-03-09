from pymetal import ttmetal as ttm

def Linear(in_features, out_features, weight, bias, device):

    weight = ttm.tensor.Tensor(
        weight,
        [1, 1, out_features, in_features],
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device
    )

    if bias is None:
        bias = None
    else:
        bias = ttm.tensor.Tensor(
            bias,
            [1, 1, 32, out_features],
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.TILE,
            device
        )

    def linear_(activation):
        weight_T = ttm.tensor.transpose(weight)
        output = ttm.tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = ttm.tensor.bcast(output, bias, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_
