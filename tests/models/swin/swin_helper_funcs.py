import tt_lib


def linear(
    x: tt_lib.tensor.Tensor,
    weight: tt_lib.tensor.Tensor,
    bias: tt_lib.tensor.Tensor = None,
) -> tt_lib.tensor.Tensor:
    weight = tt_lib.tensor.transpose(weight)
    x = tt_lib.tensor.matmul(x, weight)

    if bias is not None:
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
    return x
