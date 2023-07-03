import torch
import tt_lib as ttl
from python_api_testing.models.helper_funcs import Linear as tt_Linear


def setup_host_and_device(func):
    def wrap(*args, pcie_slot, **kwargs):
        host = ttl.device.GetHost()
        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
        ttl.device.InitializeDevice(device)
        ttl.device.SetDefaultDevice(device)
        try:
            output = func(*args, host=host, device=device, **kwargs)
        finally:
            ttl.device.CloseDevice(device)

        return output

    return wrap


################################################
################## Helper-Funcs ################
################################################
@setup_host_and_device
def linear(x, weight, *args, host, device, dtype, layout, on_device, **kwargs):
    bias = None if len(args) == 0 else args[0]
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


################################################
#################### TT-DNN ####################
################################################
@setup_host_and_device
def eltwise_threshold(
    x, *args, threshold, value, host, device, dtype, layout, on_device, **kwargs
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_relu6(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    return output


@setup_host_and_device
def eltwise_hypot(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def eltwise_cbrt(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    return output


@setup_host_and_device
def eltwise_rad2deg(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_deg2rad(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_sign(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_abs(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    return output


@setup_host_and_device
def eltwise_exp(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_neg(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_recip(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_sqrt(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_gelu(x, *args, host, device, dtype, layout, on_device, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout)
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.gelu(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_relu(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_relu_min(
    x, *args, lower_limit, host, device, dtype, layout, on_device, **kwargs
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

    t1 = ttl.tensor.relu_min(t0, lower_limit)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_relu_max(
    x, *args, upper_limit, host, device, dtype, layout, on_device, **kwargs
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

    t1 = ttl.tensor.relu_max(t0, upper_limit)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_polyval(x, *args, coeffs, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_mac(x, y, z, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t3.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t3.shape()
    )

    return output


@setup_host_and_device
def eltwise_sigmoid(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


def eltwise_log_common(
    log_kind, x, *args, host, device, dtype, layout, on_device, **kwargs
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

    t1 = getattr(ttl.tensor, log_kind)(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_log(x, *args, host, device, dtype, layout, on_device, **kwargs):
    output = eltwise_log_common(
        "log",
        x,
        *args,
        host=host,
        device=device,
        dtype=dtype,
        layout=layout,
        on_device=on_device,
        **kwargs,
    )
    return output


@setup_host_and_device
def eltwise_log2(x, *args, host, device, dtype, layout, on_device, **kwargs):
    output = eltwise_log_common(
        "log2",
        x,
        *args,
        host=host,
        device=device,
        dtype=dtype,
        layout=layout,
        on_device=on_device,
        **kwargs,
    )
    return output


@setup_host_and_device
def eltwise_log10(x, *args, host, device, dtype, layout, on_device, **kwargs):
    output = eltwise_log_common(
        "log10",
        x,
        *args,
        host=host,
        device=device,
        dtype=dtype,
        layout=layout,
        on_device=on_device,
        **kwargs,
    )
    return output


@setup_host_and_device
def eltwise_tanh(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_sin(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_cos(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_swish(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_add1(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_log1p(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def zeros_like(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def ones_like(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def full_like(x, *args, scalar, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def ones(x, *args, host, device, dtype, layout, on_device, **kwargs):
    t1 = ttl.tensor.ones(x.shape)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def zeros(x, *args, host, device, dtype, layout, on_device, **kwargs):
    t1 = ttl.tensor.zeros(x.shape)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def full(x, *args, scalar, host, device, dtype, layout, on_device, **kwargs):
    t1 = ttl.tensor.full(x.shape, scalar)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def arange(
    x, *args, start, end, step=1, host, device, dtype, layout, on_device, **kwargs
):
    t1 = ttl.tensor.arange(start, end, step)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data())

    return output


@setup_host_and_device
def hardtanh(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def clip(x, *args, low, high, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def where(x, y, z, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t3.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t3.shape()
    )

    return output


@setup_host_and_device
def eltwise_div_unary(
    x, *args, scalar, host, device, dtype, layout, on_device, **kwargs
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

    t1 = ttl.tensor.div_unary(t0, scalar)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_mul_unary(
    x, *args, scalar, host, device, dtype, layout, on_device, **kwargs
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

    t1 = ttl.tensor.mul_unary(t0, scalar)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_sub_unary(
    x, *args, scalar, host, device, dtype, layout, on_device, **kwargs
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

    t1 = ttl.tensor.sub_unary(t0, scalar)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_add_unary(
    x, *args, scalar, host, device, dtype, layout, on_device, **kwargs
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

    t1 = ttl.tensor.add_unary(t0, scalar)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_softplus(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_mish(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_hardswish(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_hardsigmoid(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_square(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_silu(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_ltz(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_gtz(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_lez(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_gez(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_nez(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_eqz(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def eltwise_add(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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

    t2 = ttl.tensor.add(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def eltwise_sub(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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

    t2 = ttl.tensor.sub(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def eltwise_mul(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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

    t2 = ttl.tensor.mul(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def eltwise_min(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def eltwise_max(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def matmul(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def outer(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bmm(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bcast_add_h(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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
    if layout == ttl.tensor.Layout.TILE:
        pad_shape = list(y.shape)
        pad_shape[2] = 32
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bcast_add_w(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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
    if layout == ttl.tensor.Layout.TILE:
        pad_shape = list(y.shape)
        pad_shape[3] = 32
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.ROW_MAJOR:
        pad_shape = list(y.shape)
        pad_shape[3] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.CHANNELS_LAST:
        pad_shape = list(y.shape)
        pad_shape[1] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.W)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bcast_add_hw(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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
    if layout == ttl.tensor.Layout.TILE:
        pad_shape = list(y.shape)
        pad_shape[2] = 32
        pad_shape[3] = 32
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.ROW_MAJOR:
        pad_shape = list(y.shape)
        pad_shape[3] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.CHANNELS_LAST:
        pad_shape = list(y.shape)
        pad_shape[1] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bcast_sub_h(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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
    if layout == ttl.tensor.Layout.TILE:
        pad_shape = list(y.shape)
        pad_shape[2] = 32
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.H)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bcast_sub_w(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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
    if layout == ttl.tensor.Layout.TILE:
        pad_shape = list(y.shape)
        pad_shape[3] = 32
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.ROW_MAJOR:
        pad_shape = list(y.shape)
        pad_shape[3] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.CHANNELS_LAST:
        pad_shape = list(y.shape)
        pad_shape[1] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.W)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bcast_sub_hw(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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
    if layout == ttl.tensor.Layout.TILE:
        pad_shape = list(y.shape)
        pad_shape[2] = 32
        pad_shape[3] = 32
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.ROW_MAJOR:
        pad_shape = list(y.shape)
        pad_shape[3] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.CHANNELS_LAST:
        pad_shape = list(y.shape)
        pad_shape[1] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.HW)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bcast_mul_h(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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
    if layout == ttl.tensor.Layout.TILE:
        pad_shape = list(y.shape)
        pad_shape[2] = 32
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bcast_mul_w(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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
    if layout == ttl.tensor.Layout.TILE:
        pad_shape = list(y.shape)
        pad_shape[3] = 32
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.ROW_MAJOR:
        pad_shape = list(y.shape)
        pad_shape[3] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.CHANNELS_LAST:
        pad_shape = list(y.shape)
        pad_shape[1] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def bcast_mul_hw(x, y, *args, host, device, dtype, layout, on_device, **kwargs):
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
    if layout == ttl.tensor.Layout.TILE:
        pad_shape = list(y.shape)
        pad_shape[2] = 32
        pad_shape[3] = 32
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.ROW_MAJOR:
        pad_shape = list(y.shape)
        pad_shape[3] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    elif on_device and layout == ttl.tensor.Layout.CHANNELS_LAST:
        pad_shape = list(y.shape)
        pad_shape[1] = 2
        t1 = t1.pad(pad_shape, [0, 0, 0, 0], 0)
    t1 = t1.to(layout)
    if on_device:
        t1 = t1.to(device)

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )

    return output


@setup_host_and_device
def reduce_sum_h(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


@setup_host_and_device
def reduce_sum_w(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    # Slice out the 0 values from reduction
    output = output[..., :, :1]

    return output


@setup_host_and_device
def reduce_sum_hw(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


@setup_host_and_device
def reduce_max_h(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


@setup_host_and_device
def reduce_max_w(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    # Slice out the 0 values from reduction
    output = output[..., :1]

    return output


@setup_host_and_device
def reduce_max_hw(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


@setup_host_and_device
def transpose_wh(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def transpose_hc(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def transpose_cn(x, *args, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def permute(x, *args, host, device, dtype, layout, on_device, permute_dims, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def reshape(x, *args, host, device, dtype, layout, on_device, reshape_dims, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def tilize(x, *args, host, device, dtype, layout, on_device, **kwargs):
    assert dtype == ttl.tensor.DataType.BFLOAT16
    assert (
        layout == ttl.tensor.Layout.ROW_MAJOR
        or layout == ttl.tensor.Layout.CHANNELS_LAST
    )
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

    output = torch.Tensor(t1.to(host).data()).reshape(t1.shape())

    return output


@setup_host_and_device
def tilize_with_zero_padding(
    x, *args, host, device, dtype, layout, on_device, **kwargs
):
    assert dtype == ttl.tensor.DataType.BFLOAT16
    assert (
        layout == ttl.tensor.Layout.ROW_MAJOR
        or layout == ttl.tensor.Layout.CHANNELS_LAST
    )
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

    output = torch.Tensor(t1.to(host).data()).reshape(t1.shape())

    return output


@setup_host_and_device
def untilize(x, *args, host, device, dtype, layout, on_device, **kwargs):
    assert dtype == ttl.tensor.DataType.BFLOAT16
    assert layout == ttl.tensor.Layout.TILE
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.TILE,
    )

    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.untilize(t0)

    output = torch.Tensor(t1.to(host).data()).reshape(t1.shape())

    return output


@setup_host_and_device
def tilize_with_val_padding(
    x,
    *args,
    host,
    device,
    dtype,
    layout,
    on_device,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    assert dtype == ttl.tensor.DataType.BFLOAT16
    assert layout == ttl.tensor.Layout.ROW_MAJOR
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

    output = torch.Tensor(t1.to(host).data()).reshape(t1.shape())

    return output


@setup_host_and_device
def untilize_with_unpadding(
    x,
    *args,
    host,
    device,
    dtype,
    layout,
    on_device,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    assert dtype == ttl.tensor.DataType.BFLOAT16
    assert layout == ttl.tensor.Layout.TILE
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype,
        ttl.tensor.Layout.TILE,
    )

    if on_device:
        t0 = t0.to(device)

    t1 = ttl.tensor.untilize_with_unpadding(t0, output_tensor_start, output_tensor_end)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def pad(
    x,
    *args,
    host,
    device,
    dtype,
    layout,
    on_device,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    assert dtype == ttl.tensor.DataType.BFLOAT16
    assert layout in [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.TILE]
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


@setup_host_and_device
def unpad(
    x,
    *args,
    host,
    device,
    dtype,
    layout,
    on_device,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    assert dtype == ttl.tensor.DataType.BFLOAT16
    assert layout in [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.TILE]
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output


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
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        device_tensor = (
            ttl.tensor.Tensor(
                x.reshape(-1).tolist(),
                x.shape,
                dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, memory_config)
        )

        host_tensor = device_tensor.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
        output = torch.as_tensor(host_tensor.data()).reshape(host_tensor.shape())
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

    output = torch.Tensor(t1.data()).reshape(t1.shape())

    return output


def tensor_unpad(x, pcie_slot, *args, output_tensor_start, output_tensor_end, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.unpad(output_tensor_start, output_tensor_end)

    output = torch.Tensor(t1.data()).reshape(t1.shape())

    return output


def pad_to_tile(x, pcie_slot, *args, pad_value, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.pad_to_tile(pad_value)

    output = torch.Tensor(t1.data()).reshape(t1.shape())

    return output


def unpad_from_tile(x, pcie_slot, *args, output_tensor_shape, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.unpad_from_tile(output_tensor_shape)

    output = torch.Tensor(t1.data()).reshape(t1.shape())

    return output


@setup_host_and_device
def eltwise_power(x, *args, exponent, host, device, dtype, layout, on_device, **kwargs):
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

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )

    return output
