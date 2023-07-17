import torch
import tt_lib as ttl
from python_api_testing.models.helper_funcs import Linear as tt_Linear

from itertools import product


def setup_host_and_device(func):
    def wrap(*args, pcie_slot, **kwargs):
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
        ttl.device.InitializeDevice(device)
        ttl.device.SetDefaultDevice(device)
        try:
            output = func(*args, device=device, **kwargs)
        finally:
            ttl.device.CloseDevice(device)

        return output

    return wrap


################################################
################## Helper-Funcs ################
################################################
@setup_host_and_device
def linear(x, weight, bias=None, *args, device, dtype, layout, on_device, **kwargs):
    tt_bias = None
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    tt_weight = ttl.tensor.Tensor(
        weight.reshape(-1).tolist(),
        weight.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if bias is not None:
        tt_bias = ttl.tensor.Tensor(
            bias.reshape(-1).tolist(),
            bias.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )

    t0 = t0.to(layout)
    tt_weight = tt_weight.to(layout)

    if on_device:
        t0 = t0.to(device)
        tt_weight = tt_weight.to(device)
        if bias is not None:
            tt_bias = tt_bias.to(device)
    _, __, out_features, in_features = tt_weight.shape()
    tt_linear = tt_Linear(in_features, out_features, tt_weight, tt_bias)

    t1 = tt_linear(t0)
    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


################################################
#################### TT-DNN ####################
################################################
@setup_host_and_device
def move(
    x,
    *args,
    input_mem_config,
    output_mem_config,
    device,
    dtype,
    layout,
    on_device,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device, input_mem_config)

    t1 = ttl.tensor.move(t0, output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_bitwise_complement(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.bitwise_complement(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_logical_not(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.logical_not(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_cosh(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.cosh(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_sinh(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.sinh(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_threshold(
    x, *args, threshold, value, device, dtype, layout, on_device, **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.threshold(t0, threshold, value)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_leaky_relu(
    x, *args, negative_slope, device, dtype, layout, on_device, **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.leaky_relu(t0, negative_slope)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_hardshrink(x, *args, _lambda, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.hardshrink(t0, _lambda)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_softshrink(x, *args, _lambda, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.softshrink(t0, _lambda)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_softsign(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.softsign(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_relu6(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.relu6(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_elu(x, *args, alpha, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.elu(t0, alpha)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_hypot(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.hypot(t0, t1)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_cbrt(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.cbrt(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_rad2deg(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.rad2deg(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_deg2rad(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.deg2rad(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_sign(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.sign(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_abs(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.abs(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_exp(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.exp(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_expm1(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.expm1(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_neg(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.neg(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_exp2(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.exp2(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_recip(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.recip(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_sqrt(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.sqrt(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_gelu(x, *args, fast_and_appx, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.gelu(t0, fast_and_appx)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_rsqrt(x, *args, fast_and_appx, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.rsqrt(t0, fast_and_appx)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_relu(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.relu(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_relu_min(x, *args, lower_limit, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.relu_min(t0, lower_limit)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_relu_max(x, *args, upper_limit, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.relu_max(t0, upper_limit)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_polyval(x, *args, coeffs, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.polyval(t0, coeffs)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_mac(x, y, z, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout)
    if on_device:
        t2 = t2.to(device)

    t3 = ttl.tensor.mac(t0, t1, t2)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_addcmul(x, y, z, *args, value, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout)
    if on_device:
        t2 = t2.to(device)

    t3 = ttl.tensor.addcmul(t0, t1, t2, value)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_addcdiv(x, y, z, *args, value, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout)
    if on_device:
        t2 = t2.to(device)

    t3 = ttl.tensor.addcdiv(t0, t1, t2, value)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_lerp_binary(
    x, y, *args, weight, device, dtype, layout, on_device, **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.lerp(t0, t1, weight)

    output = torch.Tensor(t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def eltwise_lerp_ternary(x, y, z, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout)
    if on_device:
        t2 = t2.to(device)

    t3 = ttl.tensor.lerp(t0, t1, t2)

    output = torch.Tensor(t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t3.shape()
    )

    return output


@setup_host_and_device
def eltwise_sigmoid(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.sigmoid(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_log_sigmoid(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.log_sigmoid(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_heaviside(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    value = kwargs.pop("value")
    t1 = ttl.tensor.heaviside(t0, value)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


def eltwise_log_common(log_kind, x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = getattr(ttl.tensor, log_kind)(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_log(x, *args, device, dtype, layout, on_device, **kwargs):
    output = eltwise_log_common(
        "log",
        x,
        *args,
        device=device,
        dtype=dtype,
        layout=layout,
        on_device=on_device,
        **kwargs,
    )
    return output


@setup_host_and_device
def eltwise_log2(x, *args, device, dtype, layout, on_device, **kwargs):
    output = eltwise_log_common(
        "log2",
        x,
        *args,
        device=device,
        dtype=dtype,
        layout=layout,
        on_device=on_device,
        **kwargs,
    )
    return output


@setup_host_and_device
def eltwise_log10(x, *args, device, dtype, layout, on_device, **kwargs):
    output = eltwise_log_common(
        "log10",
        x,
        *args,
        device=device,
        dtype=dtype,
        layout=layout,
        on_device=on_device,
        **kwargs,
    )
    return output


@setup_host_and_device
def eltwise_tanh(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.tanh(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_signbit(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.signbit(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_tanhshrink(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.tanhshrink(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_sin(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.sin(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_cos(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.cos(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_asin(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.asin(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_atan(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.atan(t0)

    output = torch.Tensor(t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_acos(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.acos(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_swish(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.swish(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_add1(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.add1(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_log1p(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.log1p(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def zeros_like(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.zeros_like(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def ones_like(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.ones_like(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def full_like(x, *args, scalar, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.full_like(t0, scalar)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def ones(x, *args, device, dtype, layout, on_device, **kwargs):
    t1 = ttl.tensor.ones(x.shape)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def zeros(x, *args, device, dtype, layout, on_device, **kwargs):
    t1 = ttl.tensor.zeros(x.shape)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def full(x, *args, scalar, device, dtype, layout, on_device, **kwargs):
    t1 = ttl.tensor.full(x.shape, scalar)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def arange(x, *args, start, end, step=1, device, dtype, layout, on_device, **kwargs):
    t1 = ttl.tensor.arange(start, end, step)
    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def hardtanh(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.hardtanh(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def clip(x, *args, low, high, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.clip(t0, low, high)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def where(x, y, z, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    t1 = t1.to(layout)
    t2 = t2.to(layout)
    if on_device:
        t0 = t0.to(device)
        t1 = t1.to(device)
        t2 = t2.to(device)

    t3 = ttl.tensor.where(t0, t1, t2)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_div_unary(x, *args, scalar, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.div_unary(t0, scalar)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_mul_unary(x, *args, scalar, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.mul_unary(t0, scalar)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_sub_unary(x, *args, scalar, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.sub_unary(t0, scalar)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_add_unary(x, *args, scalar, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.add_unary(t0, scalar)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_softplus(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.softplus(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_mish(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.mish(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_hardswish(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.hardswish(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_hardsigmoid(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.hardsigmoid(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_square(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.square(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_silu(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.silu(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_ltz(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.ltz(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_gtz(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.gtz(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_lez(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.lez(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_gez(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.gez(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_nez(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.nez(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_eqz(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.eqz(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_min(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.min(t0, t1)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_max(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.max(t0, t1)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_squared_difference(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.squared_difference(t0, t1)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def matmul(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.matmul(t0, t1)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def outer(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.outer(t0, t1)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bmm(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bmm(t0, t1)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_add_h(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype)

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    if layout == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))
    t1 = ttl.tensor.Tensor(t1, dtype)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_add_w(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype)

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    if layout == ttl.tensor.Layout.TILE or (
        on_device and layout == ttl.tensor.Layout.ROW_MAJOR
    ):
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))
    t1 = ttl.tensor.Tensor(t1, dtype)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.W)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_add_hw(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype)

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    if layout == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3], 0, 32 - y.shape[2]))
    elif on_device and layout == ttl.tensor.Layout.ROW_MAJOR:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))
    t1 = ttl.tensor.Tensor(t1, dtype)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_sub_h(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype)

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    if layout == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))
    t1 = ttl.tensor.Tensor(t1, dtype)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.H)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_sub_w(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype)

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    if layout == ttl.tensor.Layout.TILE or (
        on_device and layout == ttl.tensor.Layout.ROW_MAJOR
    ):
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))
    t1 = ttl.tensor.Tensor(t1, dtype)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.W)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_sub_hw(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype)

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    if layout == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3], 0, 32 - y.shape[2]))
    elif on_device and layout == ttl.tensor.Layout.ROW_MAJOR:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))
    t1 = ttl.tensor.Tensor(t1, dtype)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.HW)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_mul_h(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype)

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)
    if layout == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))
    t1 = ttl.tensor.Tensor(t1, dtype)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_mul_w(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype)
    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    if layout == ttl.tensor.Layout.TILE or (
        on_device and layout == ttl.tensor.Layout.ROW_MAJOR
    ):
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))
    t1 = ttl.tensor.Tensor(t1, dtype)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_mul_hw(x, y, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype)

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    if layout == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3], 0, 32 - y.shape[2]))
    elif on_device and layout == ttl.tensor.Layout.ROW_MAJOR:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))
    t1 = ttl.tensor.Tensor(t1, dtype)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def reduce_sum_h(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.H, 1.0
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


@setup_host_and_device
def reduce_sum_w(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, 1.0
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :, :1]

    return output


@setup_host_and_device
def reduce_sum_hw(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.HW, 1.0
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


@setup_host_and_device
def reduce_max_h(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.H, 1.0
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


@setup_host_and_device
def reduce_max_w(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.W, 1.0
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1]

    return output


@setup_host_and_device
def reduce_max_hw(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.HW, 1.0
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


@setup_host_and_device
def transpose_wh(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.transpose(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def transpose_hc(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.transpose_hc(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def transpose_cn(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.transpose_cn(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def transpose_nh(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.transpose(t0, 0, 2)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def transpose_nw(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.transpose(t0, 0, 3)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def transpose_cw(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.transpose(t0, 1, 3)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def sum(x, *args, dim, device, dtype, layout, on_device, **kwargs):
    assert dim >= 0 and dim <= 3
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.sum(t0, dim)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    if dim == 2:
        output = output[:, :, :1, :]
    elif dim == 3:
        output = output[:, :, :, :1]
    return output


@setup_host_and_device
def permute(x, *args, device, dtype, layout, on_device, permute_dims, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.permute(t0, *permute_dims)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def reshape(x, *args, device, dtype, layout, on_device, reshape_dims, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.reshape(t0, *reshape_dims)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def tilize(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.tilize(t0)

    output = t1.cpu().to_torch()

    return output


@setup_host_and_device
def tilize_with_zero_padding(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.tilize_with_zero_padding(t0)

    output = t1.cpu().to_torch()

    return output


@setup_host_and_device
def untilize(x, *args, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.TILE,
    )

    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.untilize(t0)

    output = t1.cpu().to_torch()

    return output


@setup_host_and_device
def tilize_with_val_padding(
    x,
    *args,
    device,
    dtype,
    layout,
    on_device,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.tilize_with_val_padding(
        t0, output_tensor_shape, input_tensor_start, pad_value
    )

    output = t1.cpu().to_torch()

    return output


@setup_host_and_device
def untilize_with_unpadding(
    x,
    *args,
    device,
    dtype,
    layout,
    on_device,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.TILE,
    )

    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.untilize_with_unpadding(t0, output_tensor_start, output_tensor_end)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def pad(
    x,
    *args,
    device,
    dtype,
    layout,
    on_device,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.pad(t0, output_tensor_shape, input_tensor_start, pad_value)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def unpad(
    x,
    *args,
    device,
    dtype,
    layout,
    on_device,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.unpad(t0, output_tensor_start, output_tensor_end)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_power(x, *args, exponent, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.power(t0, exponent)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


def make_eltwise_binary_op(ttl_tensor_binop):
    @setup_host_and_device
    def eltwise_binary_op(
        x,
        y,
        *args,
        device,
        dtype,
        layout,
        on_device,
        input_mem_config=ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        output_mem_config=ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        **kwargs,
    ):
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )

        t0 = t0.to(layout)
        if on_device:
            t0 = t0.to(device, input_mem_config)

        t1 = ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )

        t1 = t1.to(layout)
        if on_device:
            t1 = t1.to(device, input_mem_config)

        t2 = ttl_tensor_binop(t0, t1, output_mem_config=output_mem_config)

        output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        return output

    return eltwise_binary_op


eltwise_add = make_eltwise_binary_op(ttl.tensor.add)
eltwise_sub = make_eltwise_binary_op(ttl.tensor.sub)
eltwise_mul = make_eltwise_binary_op(ttl.tensor.mul)
eltwise_ne = make_eltwise_binary_op(ttl.tensor.ne)
eltwise_eq = make_eltwise_binary_op(ttl.tensor.eq)
eltwise_gt = make_eltwise_binary_op(ttl.tensor.gt)
eltwise_lt = make_eltwise_binary_op(ttl.tensor.lt)
eltwise_gte = make_eltwise_binary_op(ttl.tensor.gte)
eltwise_lte = make_eltwise_binary_op(ttl.tensor.lte)


################################################
#################### Tensor ####################
################################################
def datacopy(
    x,
    pcie_slot,
    *args,
    dtype=ttl.tensor.DataType.BFLOAT16,
    memory_config=ttl.tensor.MemoryConfig(),
    **kwargs,
):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        device_tensor = (
            ttl.tensor.Tensor(x, dtype)
            .to(ttl.tensor.Layout.TILE)
            .to(device, memory_config)
        )

        host_tensor = device_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
        output = host_tensor.to_torch()
    finally:
        ttl.device.CloseDevice(device)

    return output


def tensor_pad(
    x, pcie_slot, *args, output_tensor_shape, input_tensor_start, pad_value, **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.pad(output_tensor_shape, input_tensor_start, pad_value)

    output = t1.to_torch()

    return output


def tensor_unpad(x, pcie_slot, *args, output_tensor_start, output_tensor_end, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.unpad(output_tensor_start, output_tensor_end)

    output = t1.to_torch()

    return output


def pad_to_tile(x, pcie_slot, *args, pad_value, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.pad_to_tile(pad_value)

    output = t1.to_torch()

    return output


def unpad_from_tile(x, pcie_slot, *args, output_tensor_shape, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.unpad_from_tile(output_tensor_shape)

    output = t1.to_torch()

    return output
