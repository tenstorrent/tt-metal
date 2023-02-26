from gpai import gpai

def Linear(out_features, in_features, weight, bias, device):

    weight = gpai.tensor.Tensor(
        weight,
        [1, 1, out_features, in_features],
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device
    )

    bias = gpai.tensor.Tensor(
        bias,
        [1, 1, 32, out_features],
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device
    )

    def linear_(activation):
        weight_T = gpai.tensor.transpose(weight)
        output = gpai.tensor.matmul(activation, weight_T)

        output_plus_bias = gpai.tensor.bcast(output, bias, gpai.tensor.BcastOpMath.ADD, gpai.tensor.BcastOpDim.H)
        return output_plus_bias

    return linear_
