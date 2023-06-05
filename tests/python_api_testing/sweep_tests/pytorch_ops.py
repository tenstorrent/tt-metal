import torch
from tt_lib.utils import (
    _nearest_32 as nearest_32,
    tilize as tilize_util,
    untilize as untilize_util,
)

################################################
################# Helper-Funcs #################
################################################

def linear(x, weight, bias=None, *args, **kwargs):
    while len(weight.shape) > 2:
        weight = weight.squeeze(0)
    while bias is not None and len(bias.shape) > 1:
        bias = bias.squeeze(0)

    return torch.nn.functional.linear(x, weight, bias)

################################################
#################### TT-DNN ####################
################################################


# Unary Ops
def power(x, *args, **kwargs):
    exponent = kwargs["exponent"]
    result = x**exponent
    return result


def polyval(x, *args, **kwargs):
    coeffs = kwargs["coeffs"]
    result = x * 0.0
    for coeff in coeffs:
        result = result * x + coeff
    return result


def relu_max(x, *args, **kwargs):
    upper_limit = kwargs["upper_limit"]
    return torch.min(torch.relu(x), torch.tensor(upper_limit))


def relu_min(x, *args, **kwargs):
    lower_limit = kwargs["lower_limit"]
    return torch.min(torch.relu(x), torch.tensor(lower_limit))


def abs(x, *args, **kwargs):
    return x.abs()


def ltz(x, *args, **kwargs):
    return (x < 0.0).to(x.dtype)


def gtz(x, *args, **kwargs):
    return (x > 0.0).to(x.dtype)


def lez(x, *args, **kwargs):
    return (x <= 0.0).to(x.dtype)


def gez(x, *args, **kwargs):
    return (x >= 0.0).to(x.dtype)


def eqz(x, *args, **kwargs):
    return (x == 0.0).to(x.dtype)


def nez(x, *args, **kwargs):
    return (x != 0.0).to(x.dtype)


def sign(x, *args, **kwargs):
    return torch.sign(x)


def datacopy(x, *args, **kwargs):
    return x


def neg(x, *args, **kwargs):
    return -x


def square(x, *args, **kwargs):
    return torch.square(x)


def log1p(x, *args, **kwargs):
    return torch.log1p(x)


def softplus(x, *args, **kwargs):
    return torch.nn.functional.softplus(x)


def add1(x, *args, **kwargs):
    return 1 + x


def mish(x, *args, **kwargs):
    return x * torch.tanh(softplus(x))


def recip(x, *args, **kwargs):
    return torch.reciprocal(x)


def exp(x, *args, **kwargs):
    return torch.exp(x)


def sqrt(x, *args, **kwargs):
    return torch.sqrt(x)


def gelu(x, *args, **kwargs):
    return torch.nn.functional.gelu(x)


def relu(x, *args, **kwargs):
    return torch.nn.functional.relu(x)


def sigmoid(x, *args, **kwargs):
    return torch.sigmoid(x)


def log(x, *args, **kwargs):
    return torch.log(x)


def log2(x, *args, **kwargs):
    return torch.log2(x)


def log10(x, *args, **kwargs):
    return torch.log10(x)


def tanh(x, *args, **kwargs):
    return torch.tanh(x)


def sin(x, *args, **kwargs):
    return torch.sin(x)


def cos(x, *args, **kwargs):
    return torch.cos(x)


def swish(x, *args, **kwargs):
    return torch.nn.functional.silu(x)


def silu(x, *args, **kwargs):
    return torch.nn.functional.silu(x)


## Trinary op
def mac(x, *args, **kwargs):
    if len(args) == 0:
        y = x
        z = x
    else:
        y = args[0]
        z = args[1]
    return x * y + z


## Binary Ops
def max(x, y, *args, **kwargs):
    return torch.max(x, y)


def min(x, y, *args, **kwargs):
    return torch.min(x, y)


def add(x, y, *args, **kwargs):
    return torch.add(x, y)


def sub(x, y, *args, **kwargs):
    return torch.sub(x, y)


def mul(x, y, *args, **kwargs):
    return torch.mul(x, y)


def matmul(x, y, *args, **kwargs):
    return torch.matmul(x, y)


def reduce_sum(x, dims=None, keepdim=True, *args, **kwargs):
    return torch.sum(x, dims, keepdim)


def reduce_max(x, dims=None, keepdim=True, *args, **kwargs):
    return torch.amax(x, dims, keepdim)


def flatten(x, *args, **kwargs):
    return torch.flatten(x)


def transpose(x, dim0=-2, dim1=-1, *args, **kwargs):
    return torch.transpose(x, dim0, dim1)


def permute(x, *args, **kwargs):
    assert "permute_dims" in kwargs
    permute_dims = kwargs["permute_dims"]
    return torch.permute(x, permute_dims)


def reshape(x, *args, **kwargs):
    assert "reshape_dims" in kwargs
    reshape_dims = kwargs["reshape_dims"]
    return torch.reshape(x, reshape_dims)


def tilize(x, *args, **kwargs):
    return tilize_util(x)


def untilize(x, *args, **kwargs):
    return untilize_util(x)


def tilize_with_zero_padding(x, *args, **kwargs):
    return tilize_util(
        torch.nn.functional.pad(x, (0, 0, 0, nearest_32(x.shape[-2]) - x.shape[-2]))
    )


def tilize_with_val_padding(
    x, output_tensor_shape, input_tensor_start, pad_value, *args, **kwargs
):
    pad = torch.nn.functional.pad(
        x,
        tuple(
            j
            for i in reversed(range(len(x.shape)))
            for j in (input_tensor_start[i], output_tensor_shape[i] - x.shape[i])
        ),
        value=pad_value,
    )
    tilized = tilize_util(pad)
    return tilized


def untilize_with_unpadding(x, output_tensor_start, output_tensor_end, *args, **kwargs):
    untilized = untilize_util(x)
    unpad = untilized[
        output_tensor_start[0] : output_tensor_end[0] + 1,
        output_tensor_start[1] : output_tensor_end[1] + 1,
        output_tensor_start[2] : output_tensor_end[2] + 1,
        output_tensor_start[3] : output_tensor_end[3] + 1,
    ]
    return unpad


################################################
#################### Tensor ####################
################################################
def pad(x, *args, **kwargs):
    assert "output_tensor_shape" in kwargs
    assert "input_tensor_start" in kwargs
    assert "pad_value" in kwargs

    output_tensor_shape = kwargs["output_tensor_shape"]
    input_tensor_start = kwargs["input_tensor_start"]
    pad_value = kwargs["pad_value"]

    input_tensor_shape = x.shape
    input_tensor_end = tuple(
        input_tensor_start[i] + input_tensor_shape[i]
        for i in range(len(input_tensor_shape))
    )
    out = torch.full(output_tensor_shape, pad_value, dtype=torch.bfloat16)
    out[
        input_tensor_start[0] : input_tensor_end[0],
        input_tensor_start[1] : input_tensor_end[1],
        input_tensor_start[2] : input_tensor_end[2],
        input_tensor_start[3] : input_tensor_end[3],
    ] = x

    return out


def unpad(x, *args, **kwargs):
    assert "output_tensor_start" in kwargs
    assert "output_tensor_end" in kwargs

    output_tensor_start = kwargs["output_tensor_start"]
    output_tensor_end = kwargs["output_tensor_end"]

    out = x[
        output_tensor_start[0] : output_tensor_end[0] + 1,
        output_tensor_start[1] : output_tensor_end[1] + 1,
        output_tensor_start[2] : output_tensor_end[2] + 1,
        output_tensor_start[3] : output_tensor_end[3] + 1,
    ]

    return out


def pad_to_tile(x, pad_value, *args, **kwargs):
    input_tensor_shape = x.shape
    output_tensor_shape = [
        *input_tensor_shape[:-2],
        nearest_32(input_tensor_shape[-2]),
        nearest_32(input_tensor_shape[-1]),
    ]
    out = torch.full(output_tensor_shape, pad_value, dtype=torch.bfloat16)
    out[
        0 : input_tensor_shape[0],
        0 : input_tensor_shape[1],
        0 : input_tensor_shape[2],
        0 : input_tensor_shape[3],
    ] = x

    return out


def unpad_from_tile(x, output_tensor_shape, *args, **kwargs):
    out = x[
        0 : output_tensor_shape[0],
        0 : output_tensor_shape[1],
        0 : output_tensor_shape[2],
        0 : output_tensor_shape[3],
    ]

    return out
