import torch
from pymetal import tt_metal as ttm
from python_api_testing.models.utility_functions import (
    tilize_to_list,
    untilize,
    pad_weight,
)


def datacopy(pcie_slot, x):
    # TODO: Add actual datacopy once tensor op implementation is added

    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    output = untilize(torch.Tensor(t0.to(host).data()).reshape(t0.shape()))

    ttm.device.CloseDevice(device)

    return output


def eltwise_exp(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.exp(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_recip(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.recip(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_sqrt(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.sqrt(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_gelu(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.gelu(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_relu(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.relu(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_sigmoid(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.sigmoid(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_log(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.log(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_tanh(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.tanh(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_add(pcie_slot, x, y):

    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.add(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_sub(pcie_slot, x, y):

    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.sub(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def eltwise_mul(pcie_slot, x, y):

    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.mul(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def matmul(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.matmul(t0, t1)

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def bcast_add_h(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.bcast(
        t0, t1, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.H
    )

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def bcast_add_w(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.bcast(
        t0, t1, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.W
    )

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def bcast_add_hw(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.bcast(
        t0, t1, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.HW
    )

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def bcast_sub_h(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.bcast(
        t0, t1, ttm.tensor.BcastOpMath.SUB, ttm.tensor.BcastOpDim.H
    )

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def bcast_sub_w(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.bcast(
        t0, t1, ttm.tensor.BcastOpMath.SUB, ttm.tensor.BcastOpDim.W
    )

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def bcast_sub_hw(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.bcast(
        t0, t1, ttm.tensor.BcastOpMath.SUB, ttm.tensor.BcastOpDim.HW
    )

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def bcast_mul_h(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.bcast(
        t0, t1, ttm.tensor.BcastOpMath.MUL, ttm.tensor.BcastOpDim.H
    )

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def bcast_mul_w(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.bcast(
        t0, t1, ttm.tensor.BcastOpMath.MUL, ttm.tensor.BcastOpDim.W
    )

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def bcast_mul_hw(pcie_slot, x, y):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = pad_weight(y)
    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )
    t1 = ttm.tensor.Tensor(
        tilize_to_list(y),
        y.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t2 = ttm.tensor.bcast(
        t0, t1, ttm.tensor.BcastOpMath.MUL, ttm.tensor.BcastOpDim.HW
    )

    output = untilize(torch.Tensor(t2.to(host).data()).reshape(t2.shape()))
    ttm.device.CloseDevice(device)

    return output


def reduce_sum_h(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.reduce(
        t0, ttm.tensor.ReduceOpMath.SUM, ttm.tensor.ReduceOpDim.H, 1
    )

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_sum_w(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.reduce(
        t0, ttm.tensor.ReduceOpMath.SUM, ttm.tensor.ReduceOpDim.W, 1
    )

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :, :1]

    return output


def reduce_sum_hw(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.reduce(
        t0, ttm.tensor.ReduceOpMath.SUM, ttm.tensor.ReduceOpDim.HW, 1
    )

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def reduce_max_h(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.reduce(
        t0, ttm.tensor.ReduceOpMath.MAX, ttm.tensor.ReduceOpDim.H, 1
    )

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_max_w(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.reduce(
        t0, ttm.tensor.ReduceOpMath.MAX, ttm.tensor.ReduceOpDim.W, 1
    )

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1]

    return output


def reduce_max_hw(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.reduce(
        t0, ttm.tensor.ReduceOpMath.MAX, ttm.tensor.ReduceOpDim.HW, 1
    )

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def transpose_wh(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.transpose(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output


def transpose_hc(pcie_slot, x):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = ttm.tensor.Tensor(
        tilize_to_list(x),
        x.shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device,
    )

    t1 = ttm.tensor.transpose_hc(t0)

    output = untilize(torch.Tensor(t1.to(host).data()).reshape(t1.shape()))
    ttm.device.CloseDevice(device)

    return output
