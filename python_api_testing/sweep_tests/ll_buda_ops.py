import torch
from gpai import gpai
from python_api_testing.models.utility_functions import (
    tilize_to_list,
    untilize,
    pad_weight,
)


def datacopy(pcie_slot, x):
    # TODO: Add actual datacopy once tensor op implementation is added

    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    output = untilize(torch.Tensor(t0.to(host).data()).reshape(t0.shape()))

    gpai.device.CloseDevice(device)

    return output


def eltwise_exp(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.exp(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    return output


def eltwise_recip(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.recip(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    return output


def eltwise_sqrt(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.sqrt(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    return output


def eltwise_gelu(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.gelu(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    return output


def eltwise_relu(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.relu(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    return output


def eltwise_add(pcie_slot, x, y):

    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.add(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def eltwise_sub(pcie_slot, x, y):

    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.sub(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def eltwise_mul(pcie_slot, x, y):

    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.mul(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def matmul(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.matmul(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def bcast_add_h(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.bcast(t0, t1, gpai.tensor.BcastOpMath.ADD, gpai.tensor.BcastOpDim.H)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def bcast_add_w(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.bcast(t0, t1, gpai.tensor.BcastOpMath.ADD, gpai.tensor.BcastOpDim.W)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def bcast_add_hw(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.bcast(t0, t1, gpai.tensor.BcastOpMath.ADD, gpai.tensor.BcastOpDim.HW)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def bcast_sub_h(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.bcast(t0, t1, gpai.tensor.BcastOpMath.SUB, gpai.tensor.BcastOpDim.H)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def bcast_sub_w(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.bcast(t0, t1, gpai.tensor.BcastOpMath.SUB, gpai.tensor.BcastOpDim.W)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def bcast_sub_hw(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.bcast(t0, t1, gpai.tensor.BcastOpMath.SUB, gpai.tensor.BcastOpDim.HW)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def bcast_mul_h(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.bcast(t0, t1, gpai.tensor.BcastOpMath.MUL, gpai.tensor.BcastOpDim.H)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def bcast_mul_w(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.bcast(t0, t1, gpai.tensor.BcastOpMath.MUL, gpai.tensor.BcastOpDim.W)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def bcast_mul_hw(pcie_slot, x, y):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )
    t1 = gpai.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t2 = gpai.tensor.bcast(t0, t1, gpai.tensor.BcastOpMath.MUL, gpai.tensor.BcastOpDim.HW)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    gpai.device.CloseDevice(device)

    return output


def reduce_sum_h(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.reduce(t0, gpai.tensor.ReduceOpMath.SUM, gpai.tensor.ReduceOpDim.H, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_sum_w(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.reduce(t0, gpai.tensor.ReduceOpMath.SUM, gpai.tensor.ReduceOpDim.W, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :, :1]

    return output


def reduce_sum_hw(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.reduce(t0, gpai.tensor.ReduceOpMath.SUM, gpai.tensor.ReduceOpDim.HW, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def reduce_max_h(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.reduce(t0, gpai.tensor.ReduceOpMath.MAX, gpai.tensor.ReduceOpDim.H, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_max_w(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.reduce(t0, gpai.tensor.ReduceOpMath.MAX, gpai.tensor.ReduceOpDim.W, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1]

    return output


def reduce_max_hw(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.reduce(t0, gpai.tensor.ReduceOpMath.MAX, gpai.tensor.ReduceOpDim.HW, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def transpose_wh(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.transpose(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    return output


def transpose_hc(pcie_slot, x):
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, pcie_slot)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    t0 = gpai.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device,
    )

    t1 = gpai.tensor.transpose_hc(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    gpai.device.CloseDevice(device)

    return output
