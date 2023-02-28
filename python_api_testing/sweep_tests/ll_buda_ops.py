import torch
import ll_buda_bindings.ll_buda_bindings._C as _C
from python_api_testing.models.utility_functions import (
    tilize_to_list,
    untilize,
    pad_weight,
)


def datacopy(pcie_slot, x):
    # TODO: Add actual datacopy once tensor op implementation is added

    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    output = untilize(torch.Tensor(t0.to(host).data()).reshape(t0.shape()))

    _C.device.CloseDevice(device)

    return output


def eltwise_exp(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.exp(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    return output


def eltwise_recip(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.recip(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    return output


def eltwise_sqrt(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.sqrt(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    return output


def eltwise_gelu(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.gelu(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    return output


def eltwise_relu(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.relu(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    return output


def eltwise_add(pcie_slot, x, y):

    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.add(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def eltwise_sub(pcie_slot, x, y):

    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.sub(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def eltwise_mul(pcie_slot, x, y):

    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.mul(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def matmul(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.matmul(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def bcast_add_h(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.ADD, _C.tensor.BcastOpDim.H)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def bcast_add_w(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.ADD, _C.tensor.BcastOpDim.W)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def bcast_add_hw(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.ADD, _C.tensor.BcastOpDim.HW)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def bcast_sub_h(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.SUB, _C.tensor.BcastOpDim.H)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def bcast_sub_w(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.SUB, _C.tensor.BcastOpDim.W)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def bcast_sub_hw(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.SUB, _C.tensor.BcastOpDim.HW)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def bcast_mul_h(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.MUL, _C.tensor.BcastOpDim.H)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def bcast_mul_w(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.MUL, _C.tensor.BcastOpDim.W)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def bcast_mul_hw(pcie_slot, x, y):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )
    t1 = _C.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.MUL, _C.tensor.BcastOpDim.HW)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    _C.device.CloseDevice(device)

    return output


def reduce_sum_h(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.reduce(t0, _C.tensor.ReduceOpMath.SUM, _C.tensor.ReduceOpDim.H, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_sum_w(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.reduce(t0, _C.tensor.ReduceOpMath.SUM, _C.tensor.ReduceOpDim.W, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :, :1]

    return output


def reduce_sum_hw(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.reduce(t0, _C.tensor.ReduceOpMath.SUM, _C.tensor.ReduceOpDim.HW, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def reduce_max_h(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.reduce(t0, _C.tensor.ReduceOpMath.MAX, _C.tensor.ReduceOpDim.H, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_max_w(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.reduce(t0, _C.tensor.ReduceOpMath.MAX, _C.tensor.ReduceOpDim.W, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1]

    return output


def reduce_max_hw(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.reduce(t0, _C.tensor.ReduceOpMath.MAX, _C.tensor.ReduceOpDim.HW, 1)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def transpose_wh(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.transpose(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    return output


def transpose_hc(pcie_slot, x):
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, pcie_slot)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()

    t0 = _C.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device,
    )

    t1 = _C.tensor.transpose_hc(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    _C.device.CloseDevice(device)

    return output
