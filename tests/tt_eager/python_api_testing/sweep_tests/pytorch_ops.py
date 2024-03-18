# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
from tt_lib.utils import _nearest_32 as nearest_32, tilize as tilize_util, untilize as untilize_util
from tests.tt_eager.python_api_testing.sweep_tests.reference_optimizer import (
    lamb_optimizer_kernel,
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
#################### TT-LIB ####################
################################################
def copy(x, y, *args, **kwargs):
    return y.copy_(x)


def clone(x, *args, **kwargs):
    return torch.clone(x)


def typecast(x, pt_input_dtype, pt_output_dtype, *args, **kwargs):
    return x.to(pt_input_dtype[0]).to(pt_output_dtype[0])


def concat(x, y, *args, dim, **kwargs):
    return torch.concat([x, y], dim)


def move(x, *args, **kwargs):
    return x


# Stats Ops
def var_hw(x, *args, **kwargs):
    return torch.var(x, [2, 3], keepdim=True)


def std_hw(x, *args, **kwargs):
    return torch.std(x, [2, 3], keepdim=True)


def mean_hw(x, *args, **kwargs):
    return torch.mean(x, [2, 3], keepdim=True)


def normalize_hw(x, *args, **kwargs):
    mx = mean_hw(x)
    sx = std_hw(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j, :, :] = (x[i, j, :, :] - mx[i, j, :, :]) / sx[i, j, :, :]
    return x


def var_global(x, *args, **kwargs):
    return torch.var(x, [0, 1, 2, 3], keepdim=True)


def std_global(x, *args, **kwargs):
    return torch.std(x, [0, 1, 2, 3], keepdim=True)


def mean_global(x, *args, **kwargs):
    return torch.mean(x, [0, 1, 2, 3], keepdim=True)


def normalize_global(x, *args, **kwargs):
    mx = mean_global(x)
    sx = std_global(x)
    x = (x - mx) / sx
    return x


# Ternary Ops
def sum(x, *args, dim, **kwargs):
    return torch.sum(x, dim=dim, keepdim=True)


def where(x, y, z, *args, **kwargs):
    return torch.where(x > 0, y, z)


def where_bw(x, y, z, w, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data_1 = z
    other_data_2 = w

    in_data.requires_grad = True
    other_data_1.requires_grad = True
    other_data_2.requires_grad = True

    in_data.retain_grad()
    other_data_1.retain_grad()
    other_data_2.retain_grad()

    pyt_y = torch.where(in_data > 0, other_data_1, other_data_2)
    pyt_y.backward(gradient=grad_data)

    return [other_data_1.grad, other_data_2.grad]


def arange(x, *args, start, end, step=1, **kwargs):
    return torch.arange(start, end, step)


# Unary Ops
def hypot(x, y, *args, **kwargs):
    return torch.hypot(x, y)


def scatter(x, y, *args, **kwargs):
    y[:, :, : x.shape[-2], : x.shape[-1]] = x
    return y


def cbrt(x, *args, **kwargs):
    result = x.sign() * x.abs().pow(1.0 / 3.0)
    return result


def rad2deg(x, *args, **kwargs):
    result = torch.rad2deg(x)
    return result


def deg2rad(x, *args, **kwargs):
    result = torch.deg2rad(x)
    return result


def threshold(x, *args, threshold, value, **kwargs):
    result = torch.threshold(x, threshold, value)
    return result


def relu6(x, *args, **kwargs):
    result = torch.nn.functional.relu6(x)
    return result


def prelu(x, *args, **kwargs):
    t_weight = torch.Tensor((kwargs["weight"],))
    result = torch.nn.functional.prelu(x, t_weight)
    return result


def ttnn_prelu(x, *args, **kwargs):
    weight = kwargs.pop("scalar")
    t_weight = torch.ones([1], dtype=x.dtype) * weight
    result = torch.nn.functional.prelu(x, t_weight)
    return result


def softsign(x, *args, **kwargs):
    result = torch.nn.functional.softsign(x)
    return result


def leaky_relu(x, *args, negative_slope, **kwargs):
    result = torch.nn.functional.leaky_relu(x, negative_slope)
    return result


def softshrink(x, *args, _lambda, **kwargs):
    result = torch.nn.functional.softshrink(x, _lambda)
    return result


def hardshrink(x, *args, _lambda, **kwargs):
    result = torch.nn.functional.hardshrink(x, _lambda)
    return result


def bias_gelu_unary(x, *args, bias, **kwargs):
    result = torch.nn.functional.gelu(x + bias)
    return result


def hardtanh(x, *args, **kwargs):
    if "low" in kwargs and "high" in kwargs:
        low = kwargs.pop("low")
        high = kwargs.pop("high")
        result = torch.nn.functional.hardtanh(x, min_val=low, max_val=high)

    else:
        result = torch.nn.functional.hardtanh(x)

    return result


def clip(x, *args, low, high, **kwargs):
    result = torch.clip(x, low, high)
    return result


# Unary Ops and Composite Unary
def logical_not(x, *args, **kwargs):
    result = torch.logical_not(x).to(torch.int32)
    return result


def logical_not_unary(x, *args, **kwargs):
    result = torch.logical_not(x).to(torch.int32)
    return result


def i0(x, *args, **kwargs):
    result = torch.i0(x)
    return result


def cosh(x, *args, **kwargs):
    result = torch.cosh(x)
    return result


def sinh(x, *args, **kwargs):
    result = torch.sinh(x)
    return result


def power(x, *args, exponent, **kwargs):
    result = x**exponent
    return result


def power_bw(x, y, *args, exponent, **kwargs):
    grad_data = x
    in_data = y

    in_data.requires_grad = True
    in_data.retain_grad()

    pyt_y = in_data**exponent
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def bert_large_fused_qkv_matmul(x, y, z, *args, **kwargs):
    return torch.matmul(x, y) + z


def polyval(x, *args, coeffs, **kwargs):
    result = 0.0
    for coeff in coeffs:
        result = result * x + coeff
    return result


def relu_max(x, *args, upper_limit, **kwargs):
    return torch.relu(torch.min(x, torch.tensor(upper_limit)))


def relu_min(x, *args, lower_limit, **kwargs):
    return torch.max(x, torch.tensor(lower_limit))


def abs(x, *args, **kwargs):
    return x.abs()


def digamma(x, *args, **kwargs):
    return torch.digamma(x)


def isfinite(x, *args, **kwargs):
    return torch.isfinite(x)


def isinf(x, *args, **kwargs):
    return torch.isinf(x)


def isposinf(x, *args, **kwargs):
    return torch.isposinf(x)


def isneginf(x, *args, **kwargs):
    return torch.isneginf(x)


def isnan(x, *args, **kwargs):
    return torch.isnan(x)


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


def assign_unary(x, *args, **kwargs):
    return torch.clone(x)


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
    beta = kwargs.pop("beta")
    threshold = kwargs.pop("threshold")
    return torch.nn.functional.softplus(x, beta=beta, threshold=threshold)


def add1(x, *args, **kwargs):
    return 1 + x


def mish(x, *args, **kwargs):
    return x * torch.tanh(softplus(x, beta=1.0, threshold=20.0))


def recip(x, *args, **kwargs):
    return torch.reciprocal(x)


def exp(x, *args, **kwargs):
    return torch.exp(x)


def exp_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y

    in_data.requires_grad = True
    in_data.retain_grad()

    pyt_y = torch.exp(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def exp2(x, *args, **kwargs):
    return torch.exp2(x)


def expm1(x, *args, **kwargs):
    return torch.expm1(x)


def sqrt(x, *args, **kwargs):
    return torch.sqrt(x)


def gelu(x, *args, **kwargs):
    fast_and_approx = kwargs.pop("fast_and_approx")
    approximate = "tanh" if fast_and_approx else "none"
    return torch.nn.functional.gelu(x, approximate=approximate)


def softmax_in_place(x, *args, **kwargs):
    return torch.softmax(x, -1)


def ref_stable_softmax(x):
    torch.set_printoptions(precision=2, threshold=1000, sci_mode=False, edgeitems=8, linewidth=480)
    z = x  # - torch.max(x, dim=3, keepdim=True)[0]
    numerator = torch.exp(z)
    # print(x.shape)
    H = x.shape[-2]
    # print("H=", H)
    pw0, pw1 = 0, 1  # prints a tile slice with these tile coord range
    ph0, ph1 = 0, 1
    sh, sw = 16, 16  # stride inside the tile
    ow0, ow1 = 0, 0  # offset inside the tile
    oh0, oh1 = 0, 0
    # print(
    #     "Ref x=\n",
    #     x[
    #         0,
    #         0,
    #         ph0 * 32 + oh0 : ph1 * 32 + oh1 : sh,
    #         pw0 * 32 + ow0 : pw1 * 32 + ow1 : sw,
    #     ],
    # )
    # print("Ref exps=\n", numerator[0, 0, ph0*32 : ph1*32 : sh, pw0*32 : pw1*32 : sw])
    denominator = torch.sum(numerator, 3, keepdim=True)
    # print("denom shape=", denominator.shape)
    # print("Ref sumexp=\n", torch.reshape(denominator, (-1,H))[:, ph0*32:ph1*32])

    denom1 = torch.reciprocal(denominator)
    # print("ref 1/sumexp=\n", denom1[0, 0, 0:32:8, 0:64:8])
    softmax = numerator * denom1
    # print("softmaxManual=\n", softmax[0, 0, 0:32:8, 0:64:8])
    softmax = torch.nn.Softmax(3)(x)
    # print("softmaxTorch=\n", softmax[0, 0, 0:32:8, 0:64:8])

    return softmax


def layernorm(x, y, z, *args, **kwargs):
    y = y.squeeze(0)
    y = y.squeeze(0)
    y = y.squeeze(0)

    z = z.squeeze(0)
    z = z.squeeze(0)
    z = z.squeeze(0)

    return torch.nn.functional.layer_norm(input=x, normalized_shape=y.shape, weight=y, bias=z, eps=1e-05)


def layernorm_noweights(x, *args, **kwargs):
    last = x.shape[3]
    return torch.nn.functional.layer_norm(input=x, normalized_shape=(last,), weight=None, bias=None, eps=1e-05)


def groupnorm_noweights(x, *args, **kwargs):
    return torch.nn.functional.group_norm(input=x, num_groups=1, weight=None, bias=None, eps=1e-05)


def add_layernorm(x, y, z, w, *args, **kwargs):
    res = x + y

    w = w.squeeze(0)
    w = w.squeeze(0)
    w = w.squeeze(0)

    z = z.squeeze(0)
    z = z.squeeze(0)
    z = z.squeeze(0)

    return torch.nn.functional.layer_norm(input=res, normalized_shape=z.shape, weight=z, bias=w, eps=1e-05)


def add_layernorm_noweights(x, y, *args, **kwargs):
    res = x + y
    last = res.shape[3]

    return torch.nn.functional.layer_norm(input=res, normalized_shape=(last,), weight=None, bias=None, eps=1e-05)


def scale_mask_softmax_in_place(x, y, scale, *args, **kwargs):
    x1 = scale * x
    x2 = x1 + y
    retval = ref_stable_softmax(x2)
    return retval


def rsqrt(x, *args, **kwargs):
    return torch.rsqrt(x)


def logit(x, *args, eps, **kwargs):
    return torch.special.logit(x, eps=eps)


def polygamma(x, *args, k, **kwargs):
    return torch.special.polygamma(n=k, input=x)


def logical_xori(x, *args, **kwargs):
    value = kwargs.pop("immediate")
    result = torch.logical_xor(x, torch.tensor(value, dtype=torch.int32))
    return result


def relu(x, *args, **kwargs):
    return torch.nn.functional.relu(x)


def sigmoid(x, *args, **kwargs):
    return torch.sigmoid(x)


def log_sigmoid(x, *args, **kwargs):
    result = torch.nn.functional.logsigmoid(x)
    return result


def heaviside(x, *args, **kwargs):
    value = kwargs.pop("scalar")
    result = torch.heaviside(x, torch.tensor(value, dtype=x.dtype))
    return result


def erf(x, *args, **kwargs):
    return torch.erf(x)


def erfc(x, *args, **kwargs):
    return torch.erfc(x)


def erfinv(x, *args, **kwargs):
    return torch.erfinv(x)


def hardsigmoid(x, *args, **kwargs):
    return torch.nn.functional.hardsigmoid(x)


def hardswish(x, *args, **kwargs):
    return torch.nn.functional.hardswish(x)


def log(x, *args, **kwargs):
    return torch.log(x)


def log2(x, *args, **kwargs):
    return torch.log2(x)


def log10(x, *args, **kwargs):
    return torch.log10(x)


def tanh(x, *args, **kwargs):
    return torch.tanh(x)


def tanh_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y

    in_data.requires_grad = True
    in_data.retain_grad()

    pyt_y = torch.tanh(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def tanhshrink(x, *args, **kwargs):
    return torch.nn.functional.tanhshrink(x)


def signbit(x, *args, **kwargs):
    return torch.signbit(x)


def sin(x, *args, **kwargs):
    return torch.sin(x)


def asin(x, *args, **kwargs):
    return torch.asin(x)


def tan(x, *args, **kwargs):
    return torch.tan(x)


def tan_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y

    in_data.requires_grad = True

    pyt_y = torch.tan(in_data)
    in_data.retain_grad()
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def atan(x, *args, **kwargs):
    return torch.atan(x)


def atanh(x, *args, **kwargs):
    return torch.atanh(x)


def acos(x, *args, **kwargs):
    return torch.acos(x)


def cos(x, *args, **kwargs):
    return torch.cos(x)


def asinh(x, *args, **kwargs):
    return torch.asinh(x)


def acosh(x, *args, **kwargs):
    return torch.acosh(x)


def lgamma(x, *args, **kwargs):
    return torch.lgamma(x)


def multigammaln(x, *args, **kwargs):
    return torch.special.multigammaln(x, 4)


def logical_andi(x, *args, **kwargs):
    value = kwargs.pop("immediate")
    result = torch.logical_and(x, torch.tensor(value, dtype=torch.int32))
    return result


def elu(x, *args, alpha, **kwargs):
    return torch.nn.functional.elu(x, alpha)


def swish(x, *args, **kwargs):
    return torch.nn.functional.silu(x)


def silu(x, *args, **kwargs):
    return torch.nn.functional.silu(x)


def div_unary(x, *args, scalar, **kwargs):
    result = torch.div(x, scalar)
    return result


def mul_unary(x, *args, scalar, **kwargs):
    result = torch.mul(x, scalar)
    return result


def sub_unary(x, *args, scalar, **kwargs):
    result = torch.sub(x, scalar)
    return result


def sub_unary_bw(x, y, *args, scalar, **kwargs):
    grad_data = x
    in_data = y

    in_data.requires_grad = True
    in_data.retain_grad()

    pyt_y = torch.sub(in_data, scalar)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def add_unary(x, *args, scalar, **kwargs):
    result = torch.add(x, scalar)
    return result


def zeros_like(x, *args, **kwargs):
    result = torch.zeros_like(x)
    return result


def triu(x, *args, **kwargs):
    diag = kwargs.get("diag", 0)
    result = torch.triu(x, diag)
    return result


def tril(x, *args, **kwargs):
    diag = kwargs.get("diag", 0)
    result = torch.tril(x, diag)
    return result


def ones_like(x, *args, **kwargs):
    result = torch.ones_like(x)
    return result


def full_like(x, *args, scalar, **kwargs):
    result = torch.full_like(x, scalar)
    return result


def zeros(x, *args, **kwargs):
    result = torch.zeros(x.shape)
    return result


def empty(x, *args, **kwargs):
    result = torch.empty(x.shape)
    return result


def ones(x, *args, **kwargs):
    result = torch.ones(x.shape)
    return result


def full(x, *args, scalar, **kwargs):
    result = torch.full(x.shape, scalar)
    return result


def fill_rm(x, *args, **kwargs):
    hOnes = kwargs.pop("hOnes")
    wOnes = kwargs.pop("wOnes")

    val_hi = kwargs.pop("val_hi")
    val_lo = kwargs.pop("val_lo")

    y = x
    y[:, :, :, :] = val_lo
    y[:, :, 0:hOnes, 0:wOnes] = val_hi

    return y


def fill_bw(x, *args, **kwargs):
    grad_data = x.detach().clone()

    put_y = torch.zeros_like(grad_data)
    grad_sum = grad_data.sum()
    put_y.fill_(grad_sum)

    return put_y


def fill_zero_bw(x, *args, **kwargs):
    in_data = x
    put_y = torch.zeros_like(in_data)

    return put_y


def fill_ones_rm(x, *args, **kwargs):
    hOnes = kwargs.pop("hOnes")
    wOnes = kwargs.pop("wOnes")

    y = x
    y[:, :, :, :] = 0
    y[:, :, 0:hOnes, 0:wOnes] = 1

    return y


## Trinary op
def mac(x, y, z, *args, **kwargs):
    return x * y + z


def addcmul(x, y, z, *args, scalar, **kwargs):
    result = torch.addcmul(x, y, z, value=scalar)
    return result


def addcdiv(x, y, z, *args, scalar, **kwargs):
    result = torch.addcdiv(x, y, z, value=scalar)
    return result


def lerp_ternary(x, y, z, *args, **kwargs):
    return torch.lerp(x, y, z)


## Binary Ops
def atan2(x, y, *args, **kwargs):
    return torch.atan2(y, x)


def nextafter(x, y, *args, **kwargs):
    return torch.nextafter(x, y)


def logical_and(x, y, *args, **kwargs):
    result = torch.logical_and(x, y)
    return result


def logical_noti(x, *args, **kwargs):
    immediate = kwargs.pop("immediate")
    result = torch.logical_not(torch.full_like(x, immediate)).to(torch.int32)
    return result


def bias_gelu(x, y, *args, **kwargs):
    result = torch.nn.functional.gelu(torch.add(x, y))
    return result


def lerp_binary(x, y, *args, weight, **kwargs):
    return torch.lerp(x, y, weight)


def assign_binary(x, y, *args, **kwargs):
    y.copy_(x)
    return y


def isclose(x, y, *args, rtol, atol, equal_nan, **kwargs):
    return torch.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


def xlogy(x, y, *args, **kwargs):
    return torch.xlogy(x, y)


def ldexp(x, y, *args, **kwargs):
    return torch.ldexp(x, y)


def logical_xor(x, y, *args, **kwargs):
    return torch.logical_xor(x, y)


def subalpha(x, y, *args, alpha, **kwargs):
    return torch.sub(x, y, alpha=alpha)


def addalpha(x, y, *args, alpha, **kwargs):
    return torch.add(x, y, alpha=alpha)


def lamb_optimizer(x, y, z, w, *args, beta1, beta2, step_size, eps, weight_decay, **kwargs):
    exp_avg_out, exp_avg_sq_out, param = lamb_optimizer_kernel.lamb_kernel(
        x, y, z, w, beta1=beta1, beta2=beta2, step_size=step_size, eps=eps, weight_decay=weight_decay
    )

    return [exp_avg_out, exp_avg_sq_out, param]


def repeat_interleave(x, *args, repeat, dim, **kwargs):
    return torch.repeat_interleave(x, repeats=repeat, dim=dim)


def repeat(x, *args, repeat, **kwargs):
    return x.repeat(*repeat)


def lte(x, y, *args, **kwargs):
    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        return x <= scalar
    else:
        return x <= y


def lt(x, y, *args, **kwargs):
    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        return x < scalar
    else:
        return x < y


def gte(x, y, *args, **kwargs):
    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        return x >= scalar
    else:
        return x >= y


def gt(x, y, *args, **kwargs):
    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        return x > scalar
    else:
        return x > y


def eq(x, y, *args, **kwargs):
    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        return x == scalar
    else:
        return x == y


def ne(x, y, *args, **kwargs):
    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        return x != scalar
    else:
        return x != y


def max(x, y, *args, **kwargs):
    return torch.max(x, y)


def max_bw(x, y, z, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data = z

    in_data.requires_grad = True
    other_data.requires_grad = True

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.max(in_data, other_data)
    pyt_y.backward(gradient=grad_data)

    return [in_data.grad, other_data.grad]


def minimum(x, y, *args, **kwargs):
    return torch.minimum(x, y)


def min(x, y, *args, **kwargs):
    return torch.min(x, y)


def min_bw(x, y, z, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data = z

    in_data.requires_grad = True
    other_data.requires_grad = True

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.min(in_data, other_data)
    pyt_y.backward(gradient=grad_data)

    return [in_data.grad, other_data.grad]


def squared_difference(x, y, *args, **kwargs):
    t_diff = torch.sub(x, y)
    return torch.square(t_diff)


def add_and_apply_activation(x, y, *args, **kwargs):
    activation = kwargs.pop("activation")
    output = torch.add(x, y)

    if activation == "relu":
        output = torch.relu(output)
    elif activation == "gelu":
        output = torch.gelu(output)

    return output


def logaddexp(x, y, *args, **kwargs):
    return torch.logaddexp(x, y)


def logaddexp2(x, y, *args, **kwargs):
    return torch.logaddexp2(x, y)


def logical_or(x, y, *args, **kwargs):
    return torch.logical_or(x, y)


def logical_ori(x, *args, **kwargs):
    value = kwargs.pop("immediate")
    result = torch.logical_or(x, torch.tensor(value, dtype=torch.int32))
    return result


def add(x, y, *args, **kwargs):
    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        return torch.add(x, y, alpha=scalar)
    else:
        return torch.add(x, y)


def add_bw(x, y, z, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data = z

    in_data.requires_grad = True
    other_data.requires_grad = True

    in_data.retain_grad()
    other_data.retain_grad()

    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        pyt_y = torch.add(in_data, other_data, alpha=scalar)
        pyt_y.backward(gradient=grad_data)
    else:
        pyt_y = torch.add(in_data, other_data)
        pyt_y.backward(gradient=grad_data)

    return [in_data.grad, other_data.grad]


def sub(x, y, *args, **kwargs):
    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        return torch.sub(x, y, alpha=scalar)
    else:
        return torch.sub(x, y)


def sub_bw(x, y, z, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data = z

    in_data.requires_grad = True
    other_data.requires_grad = True

    in_data.retain_grad()
    other_data.retain_grad()

    if "scalar" in kwargs:
        scalar = kwargs.pop("scalar")
        pyt_y = torch.sub(in_data, other_data, alpha=scalar)
        pyt_y.backward(gradient=grad_data)
    else:
        pyt_y = torch.sub(in_data, other_data)
        pyt_y.backward(gradient=grad_data)

    return [in_data.grad, other_data.grad]


def mul(x, y, *args, **kwargs):
    return torch.mul(x, y)


def mul_bw(x, y, z, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data = z

    in_data.requires_grad = True
    other_data.requires_grad = True

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.mul(in_data, other_data)
    pyt_y.backward(gradient=grad_data)

    return [in_data.grad, other_data.grad]


def matmul(x, y, *args, **kwargs):
    return torch.matmul(x, y)


def outer(x, y, *args, **kwargs):
    return torch.outer(x.squeeze(), y.squeeze())


def reduce_sum(x, dims=None, keepdim=True, *args, **kwargs):
    return torch.sum(x, dims, keepdim)


def reduce_max(x, dims=None, keepdim=True, *args, **kwargs):
    return torch.amax(x, dims, keepdim)


def reduce_min(x, dims=None, keepdim=True, *args, **kwargs):
    return torch.amin(x, dims, keepdim)


def flatten(x, *args, **kwargs):
    return torch.flatten(x)


def transpose(x, dim0=-2, dim1=-1, *args, **kwargs):
    return torch.transpose(x, dim0, dim1)


def permute(x, *args, permute_dims, **kwargs):
    return torch.permute(x, permute_dims)


def reshape(x, *args, reshape_dims, **kwargs):
    return torch.reshape(x, reshape_dims)


def split_last_dim_two_chunks_tiled(x, *args, **kwargs):
    W = x.shape[-1]
    half = W // 2

    output0 = x[:, :, :, 0:half]
    output1 = x[:, :, :, half:W]

    return [output0, output1]


def tilize(x, *args, **kwargs):
    return tilize_util(x)


def untilize(x, *args, **kwargs):
    return untilize_util(x)


def tilize_with_zero_padding(x, *args, **kwargs):
    return tilize_util(
        torch.nn.functional.pad(x, (0, nearest_32(x.shape[-1]) - x.shape[-1], 0, nearest_32(x.shape[-2]) - x.shape[-2]))
    )


def tilize_with_val_padding(x, output_tensor_shape, input_tensor_start, pad_value, *args, **kwargs):
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
def pad(x, *args, output_tensor_shape, input_tensor_start, pad_value, **kwargs):
    print("PT")
    input_tensor_shape = x.shape
    input_tensor_end = tuple(input_tensor_start[i] + input_tensor_shape[i] for i in range(len(input_tensor_shape)))
    out = torch.full(output_tensor_shape, pad_value, dtype=torch.bfloat16)
    out[
        input_tensor_start[0] : input_tensor_end[0],
        input_tensor_start[1] : input_tensor_end[1],
        input_tensor_start[2] : input_tensor_end[2],
        input_tensor_start[3] : input_tensor_end[3],
    ] = x

    print("PT 2")

    return out


def unpad(x, *args, output_tensor_start, output_tensor_end, **kwargs):
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
    out[0 : input_tensor_shape[0], 0 : input_tensor_shape[1], 0 : input_tensor_shape[2], 0 : input_tensor_shape[3]] = x

    return out


def unpad_from_tile(x, output_tensor_shape, *args, **kwargs):
    out = x[
        0 : output_tensor_shape[0], 0 : output_tensor_shape[1], 0 : output_tensor_shape[2], 0 : output_tensor_shape[3]
    ]

    return out


def conv(x, y, *args, **kwargs):
    conv_params = kwargs.pop("conv_params")
    return torch.nn.functional.conv2d(
        x, y, bias=None, stride=(conv_params[2], conv_params[3]), padding=(conv_params[4], conv_params[5])
    )


def activation_glu(x, *args, **kwargs):
    dim = kwargs.get("dim", -1)
    return torch.nn.functional.glu(x, dim)


def activation_reglu(x, *args, **kwargs):
    dim = kwargs.get("dim", 3)
    a, b = torch.split(x, x.shape[dim] // 2, dim)
    return a * torch.nn.functional.relu(b)
    # return torch.matmul(a,torch.nn.functional.relu(b))


def activation_geglu(x, *args, **kwargs):
    dim = kwargs.get("dim", 3)
    a, b = torch.split(x, x.shape[dim] // 2, dim)
    return a * torch.nn.functional.gelu(b)
    # return torch.matmul(a,torch.nn.functional.gelu(b))


def activation_swiglu(x, *args, **kwargs):
    dim = kwargs.get("dim", 3)
    a, b = torch.split(x, x.shape[dim] // 2, dim)
    return a * torch.nn.functional.silu(b)
    # return torch.matmul(a,torch.nn.functional.silu(b))


def bert_large_pre_softmax_bmm(x, y, *args, **kwargs):
    ref_bmm = torch.matmul(x, y)
    return ref_bmm


def bert_large_post_softmax_bmm(x, y, *args, **kwargs):
    ref_bmm = torch.matmul(x.reshape([9, 16, 384, 384]), y)
    return ref_bmm


def bert_large_ff1_matmul(x, y, z, *args, **kwargs):
    ref_bmm = torch.matmul(x, y)
    ref_bmm = ref_bmm + z
    return ref_bmm


def bert_large_selfout_matmul(x, y, z, *args, **kwargs):
    ref_bmm = torch.matmul(x, y)
    ref_bmm = ref_bmm + z
    return ref_bmm


def bert_large_ff2_matmul(x, y, z, *args, **kwargs):
    ref_bmm = torch.matmul(x, y)
    ref_bmm = ref_bmm + z
    return ref_bmm


def eltwise_rpow(x, *args, **kwargs):
    dim = kwargs["factor"]
    return torch.pow(dim, x)


def eltwise_rsub(x, *args, **kwargs):
    dim = kwargs["factor"]
    return torch.sub(dim, x)


def eltwise_identity(x, *args, **kwargs):
    return x


def eltwise_rdiv(x, *args, **kwargs):
    dim = kwargs["factor"]
    return dim / x


def embeddings(x, y, *args, **kwargs):
    x = x.int()
    x_shape = x.shape
    y_shape = y.shape

    x_ref = x.detach().clone()
    y_ref = y.detach().clone()

    x_ref = torch.clamp(x_ref, min=0, max=y.shape[-2] - 1)

    batch_size = x_shape[0]
    num_rows = x_shape[3]
    num_embeddings = y_shape[2]
    embedding_dim = y_shape[3]

    z = torch.nn.functional.embedding(
        x_ref.reshape((batch_size, num_rows)), y_ref.reshape((num_embeddings, embedding_dim))
    ).reshape((batch_size, 1, num_rows, embedding_dim))
    return z


def pt_embedding_bw(x, y, z, *args, **kwargs):
    y_shape = y.shape
    z_shape = z.shape

    batch_size = y_shape[0]
    no_of_embeddings = y_shape[3]
    embedding_dim = z_shape[3]

    z_ref = z.detach().clone()

    input_tensor = torch.reshape(torch.arange(0, batch_size * no_of_embeddings), shape=y_shape)

    grad_data = x
    other_data = z_ref

    grad_data.requires_grad = True
    other_data.requires_grad = True

    grad_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.nn.functional.embedding(
        input_tensor.reshape((batch_size, no_of_embeddings)),
        other_data.reshape((batch_size * no_of_embeddings, embedding_dim)),
    ).reshape((1, 1, batch_size * no_of_embeddings, embedding_dim))

    pyt_y.backward(gradient=grad_data)

    return other_data.grad


def ttnn_embeddings(x, y, *args, **kwargs):
    x = x.int()
    x = torch.clamp(x, min=0, max=y.shape[0] - 1)
    z = torch.nn.functional.embedding(x, y)
    return z


def rmsnorm_noweights(x, *args, **kwargs):
    eps = 1e-5
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rmsnorm(x, y, z, *args, **kwargs):
    eps = 1e-5
    y = y.flatten()
    z = z.flatten()

    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * y + z


def groupnorm(x, y, z, *args, **kwargs):
    x_shape = x.shape
    num_channels = x_shape[1]

    weight = y.reshape(num_channels)
    bias = z.reshape(num_channels)

    res = torch.nn.functional.group_norm(input=x, num_groups=1, weight=weight, bias=bias, eps=1e-05)
    return res


def complex_real(x, *args, **kwargs):
    return torch.real(x)


def complex_recip(x, *args, **kwargs):
    result = torch.reciprocal(x)
    return result


def complex_div(x, y, *args, **kwargs):
    result = torch.div(x, y)
    return result


def complex_mul(x, y, *args, **kwargs):
    result = x * y
    return result


def complex_conj(x, *args, **kwargs):
    return torch.conj(x)


def complex_abs(x, *args, **kwargs):
    return torch.abs(x)


def complex_polar(x, y, *args, **kwargs):
    return torch.polar(x.to(torch.float32), y.to(torch.float32))


def complex_imag(x, *args, **kwargs):
    return torch.imag(x)


def unary_div_bw(x, y, scalar, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.div(in_data, torch.tensor(scalar))
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def unary_add_bw(x, y, scalar, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.add(in_data, torch.tensor(scalar))
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def unary_mul_bw(x, y, scalar, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.mul(in_data, torch.tensor(scalar))
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def unary_assign_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.clone(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def binary_assign_bw(x, y, z, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data = z

    in_data.requires_grad = True
    other_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.clone(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def div_bw(x, y, z, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data = z

    in_data.requires_grad = True
    other_data.requires_grad = True

    in_data.retain_grad()
    other_data.retain_grad()
    pyt_y = torch.div(in_data, other_data)
    pyt_y.backward(gradient=grad_data)

    return [in_data.grad, other_data.grad]


def rsqrt_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.rsqrt(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def addcdiv_bw(x, y, z, w, scalar, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data1 = z
    other_data2 = w

    in_data.requires_grad = True
    other_data1.requires_grad = True
    other_data2.requires_grad = True

    in_data.retain_grad()
    other_data1.retain_grad()
    other_data1.retain_grad()
    pyt_y = torch.addcdiv(in_data, other_data1, other_data2, value=scalar)
    pyt_y.backward(gradient=grad_data)

    return [in_data.grad, other_data1.grad, other_data2.grad]


def addcmul_bw(x, y, z, w, scalar, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data1 = z
    other_data2 = w

    in_data.requires_grad = True
    other_data1.requires_grad = True
    other_data2.requires_grad = True

    in_data.retain_grad()
    other_data1.retain_grad()
    other_data1.retain_grad()
    pyt_y = torch.addcmul(in_data, other_data1, other_data2, value=scalar)
    pyt_y.backward(gradient=grad_data)

    return [in_data.grad, other_data1.grad, other_data2.grad]


def addalpha_bw(x, y, z, scalar, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data1 = z

    in_data.requires_grad = True
    other_data1.requires_grad = True

    in_data.retain_grad()
    other_data1.retain_grad()
    other_data1.retain_grad()
    pyt_y = torch.add(in_data, other_data1, alpha=scalar)
    pyt_y.backward(gradient=grad_data)

    return [in_data.grad, other_data1.grad]


def ttnn_layernorm_weights_bias(x, y, z, *args, **kwargs):
    w = x.shape[1]
    torch_output_tensor = torch.nn.functional.layer_norm(x, normalized_shape=[w], weight=y, bias=z)
    return torch_output_tensor


def ttnn_layernorm_weights_bias_residual(x, y, z, w, *args, **kwargs):
    width = x.shape[1]
    torch_output_tensor = torch.nn.functional.layer_norm(x + y, normalized_shape=[width], weight=z, bias=w)
    return torch_output_tensor


def ttnn_layernorm_noweights(x, *args, **kwargs):
    w = x.shape[1]
    torch_output_tensor = torch.nn.functional.layer_norm(x, normalized_shape=[w])
    return torch_output_tensor


def abs_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.abs(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def sqrt_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.sqrt(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def relu_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.relu(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def neg_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.neg(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def log_bw(x, y, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.log(in_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def gt_bw(x, *args, **kwargs):
    grad_data = x

    pyt_y = torch.zeros_like(grad_data)

    golden_tensor = pyt_y

    return golden_tensor


def lt_bw(x, *args, **kwargs):
    grad_data = x

    pyt_y = torch.zeros_like(grad_data)

    golden_tensor = pyt_y

    return golden_tensor


def ne_bw(x, *args, **kwargs):
    grad_data = x

    pyt_y = torch.zeros_like(grad_data)

    golden_tensor = pyt_y

    return golden_tensor


def rsub_bw(x, y, z, *args, **kwargs):
    grad_data = x
    in_data = y
    other_data = z

    in_data.requires_grad = True
    other_data.requires_grad = True

    in_data.retain_grad()
    other_data.retain_grad()
    pyt_y = torch.rsub(in_data, other_data)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def binary_le_bw(x, y, *args, **kwargs):
    grad_data = x

    pyt_y = torch.zeros_like(grad_data)

    golden_tensor = pyt_y

    return golden_tensor


def clamp_max_bw(x, y, scalar, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.clamp(in_data, max=scalar)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def clamp_min_bw(x, y, scalar, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    in_data.retain_grad()
    pyt_y = torch.clamp(in_data, min=scalar)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def clamp_bw(x, y, scalar, *args, **kwargs):
    grad_data = x
    in_data = y
    in_data.requires_grad = True

    if scalar >= 0:
        max = scalar
        min = -scalar
    else:
        max = -scalar
        min = scalar

    in_data.retain_grad()
    pyt_y = torch.clamp(in_data, min=min, max=max)
    pyt_y.backward(gradient=grad_data)

    return in_data.grad


def attention_softmax_nomask(x, *args, **kwargs):
    torch_output_tensor = ttnn.transformer._torch_attention_softmax(
        x,
        head_size=None,
        attention_mask=None,
    )

    return torch_output_tensor


def attention_softmax(x, y, *args, scalar, **kwargs):
    y[y <= 0.50] = 0
    y[y > 0.50] = 1
    if scalar < 0:
        scalar = -scalar

    torch_output_tensor = ttnn.transformer._torch_attention_softmax(
        x,
        head_size=None,
        attention_mask=y,
    )

    return torch_output_tensor


def rms_norm(hidden_states, weight, *, epsilon=1e-6):
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        hidden_states = hidden_states.to(weight.dtype)

    return weight * hidden_states


def ttnn_rmsnorm(x, y, *args, **kwargs):
    torch_output_tensor = rms_norm(x, y)
    return torch_output_tensor


def transformer_concatenate_heads(x, *args, **kwargs):
    torch_output_tensor = ttnn.transformer._torch_concatenate_heads(x)

    return torch_output_tensor


def ttnn_groupnorm(x, y, z, *args, **kwargs):
    torch_output_tensor = torch.nn.functional.group_norm(x, num_groups=1, weight=y, bias=z)
    return torch_output_tensor


def global_avg_pool2d(x, *args, **kwargs):
    output_size = (1, 1)
    return torch.nn.functional.adaptive_avg_pool2d(x, output_size)


def upsample(x, *args, scale_factor, **kwargs):
    # return torch.nn.functional.upsample(x, scale_factor=2)

    tt_input = x.permute(0, 3, 1, 2)

    m = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = m(tt_input)
    torch_result = torch_result.permute(0, 2, 3, 1)

    return torch_result


def l1_loss(x, y, *args, **kwargs):
    # return torch.nn.functional.upsample(x, scale_factor=2)
    torch_output_tensor = torch.nn.L1Loss(reduction="none")(x, y)
    return torch_output_tensor


def l1_loss_sum(x, y, *args, **kwargs):
    # return torch.nn.functional.upsample(x, scale_factor=2)
    torch_output_tensor = torch.nn.L1Loss(reduction="sum")(x, y)

    return torch_output_tensor


def l1_loss_mean(x, y, *args, **kwargs):
    # return torch.nn.functional.upsample(x, scale_factor=2)
    torch_output_tensor = torch.nn.L1Loss(reduction="mean")(x, y)
    return torch_output_tensor


def mse_loss(x, y, *args, **kwargs):
    # return torch.nn.functional.upsample(x, scale_factor=2)
    torch_output_tensor = torch.nn.MSELoss(reduction="none")(x, y)
    return torch_output_tensor


def mse_loss_sum(x, y, *args, **kwargs):
    # return torch.nn.functional.upsample(x, scale_factor=2)
    torch_output_tensor = torch.nn.MSELoss(reduction="sum")(x, y)

    return torch_output_tensor


def mse_loss_mean(x, y, *args, **kwargs):
    # return torch.nn.functional.upsample(x, scale_factor=2)
    torch_output_tensor = torch.nn.L1Loss(reduction="mean")(x, y)
    return torch_output_tensor
