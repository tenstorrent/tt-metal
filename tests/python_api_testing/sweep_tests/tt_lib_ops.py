import torch
import tt_lib as ttl
from tt_lib.fallback_ops import fallback_ops


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
#################### TT-DNN ####################
################################################
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
        **kwargs
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
        **kwargs
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
        **kwargs
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
    **kwargs
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
    **kwargs
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
    **kwargs
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
    **kwargs
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
def datacopy(x, pcie_slot, *args, **kwargs):
    ttl_tensor_dtype = kwargs.get("dtype", ttl.tensor.DataType.BFLOAT16)
    ttl_tensor_memory_config = kwargs.get("memory_config", ttl.tensor.MemoryConfig())

    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        t0 = (
            ttl.tensor.Tensor(
                x.reshape(-1).tolist(),
                x.shape,
                ttl_tensor_dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device, ttl_tensor_memory_config)
        )

        output = torch.Tensor(
            t0.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t0.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def tensor_pad(x, pcie_slot, *args, **kwargs):
    assert "output_tensor_shape" in kwargs
    assert "input_tensor_start" in kwargs
    assert "pad_value" in kwargs

    output_tensor_shape = kwargs["output_tensor_shape"]
    input_tensor_start = kwargs["input_tensor_start"]
    pad_value = kwargs["pad_value"]

    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.pad(output_tensor_shape, input_tensor_start, pad_value)

    output = torch.Tensor(t1.data()).reshape(t1.shape())

    return output


def tensor_unpad(x, pcie_slot, *args, **kwargs):
    assert "output_tensor_start" in kwargs
    assert "output_tensor_end" in kwargs

    output_tensor_start = kwargs["output_tensor_start"]
    output_tensor_end = kwargs["output_tensor_end"]

    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.unpad(output_tensor_start, output_tensor_end)

    output = torch.Tensor(t1.data()).reshape(t1.shape())

    return output


def pad_to_tile(x, pcie_slot, *args, **kwargs):
    assert "pad_value" in kwargs

    pad_value = kwargs["pad_value"]

    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.pad_to_tile(pad_value)

    output = torch.Tensor(t1.data()).reshape(t1.shape())

    return output


def unpad_from_tile(x, pcie_slot, *args, **kwargs):
    assert "output_tensor_shape" in kwargs

    output_tensor_shape = kwargs["output_tensor_shape"]

    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t0.unpad_from_tile(output_tensor_shape)

    output = torch.Tensor(t1.data()).reshape(t1.shape())

    return output
