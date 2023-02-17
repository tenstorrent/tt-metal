import ll_buda_bindings.ll_buda_bindings._C as _C

def Linear(out_features, in_features, weight, bias, device):

    weight = _C.tensor.Tensor(
        weight, 
        [1, 1, out_features, in_features], 
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device
    )
    
    bias = _C.tensor.Tensor(
        bias,
        [1, 1, 32, out_features],
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device
    )

    def linear_(activation):
        weight_T = _C.tensor.transpose(weight)
        output = _C.tensor.matmul(activation, weight_T)

        output_plus_bias = _C.tensor.bcast(output, bias, _C.tensor.BcastOpMath.ADD, _C.tensor.BcastOpDim.H)
        return output_plus_bias

    return linear_