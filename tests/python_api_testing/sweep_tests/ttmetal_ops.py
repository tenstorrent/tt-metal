import torch
from pymetal import ttmetal as ttm


def datacopy(x, pcie_slot, *args, **kwargs):
    # TODO: Add actual datacopy once tensor op implementation is added

    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    output = torch.Tensor(t0.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t0.shape()
    )

    ttm.device.CloseDevice(device)

    return output


def eltwise_exp(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.exp(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_recip(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.recip(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_sqrt(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.sqrt(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_gelu(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.gelu(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_relu(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.relu(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_sigmoid(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.sigmoid(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_log(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.log(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_tanh(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.tanh(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_add(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.add(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_sub(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.sub(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def eltwise_mul(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.mul(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def matmul(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.matmul(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bmm(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bmm(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bcast_add_h(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = ttm.utils.pad_weight(y)
    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bcast(t0, t1, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.H)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bcast_add_w(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = ttm.utils.pad_weight(y)
    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bcast(t0, t1, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.W)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bcast_add_hw(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = ttm.utils.pad_weight(y)
    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bcast(t0, t1, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.HW)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bcast_sub_h(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = ttm.utils.pad_weight(y)
    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bcast(t0, t1, ttm.tensor.BcastOpMath.SUB, ttm.tensor.BcastOpDim.H)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bcast_sub_w(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = ttm.utils.pad_weight(y)
    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bcast(t0, t1, ttm.tensor.BcastOpMath.SUB, ttm.tensor.BcastOpDim.W)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bcast_sub_hw(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = ttm.utils.pad_weight(y)
    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bcast(t0, t1, ttm.tensor.BcastOpMath.SUB, ttm.tensor.BcastOpDim.HW)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bcast_mul_h(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = ttm.utils.pad_weight(y)
    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bcast(t0, t1, ttm.tensor.BcastOpMath.MUL, ttm.tensor.BcastOpDim.H)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bcast_mul_w(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = ttm.utils.pad_weight(y)
    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bcast(t0, t1, ttm.tensor.BcastOpMath.MUL, ttm.tensor.BcastOpDim.W)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def bcast_mul_hw(x, y, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    # Pad bcast tensor
    y = ttm.utils.pad_weight(y)
    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttm.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttm.tensor.bcast(t0, t1, ttm.tensor.BcastOpMath.MUL, ttm.tensor.BcastOpDim.HW)

    output = torch.Tensor(t2.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def reduce_sum_h(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.reduce(t0, ttm.tensor.ReduceOpMath.SUM, ttm.tensor.ReduceOpDim.H, 1)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_sum_w(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.reduce(t0, ttm.tensor.ReduceOpMath.SUM, ttm.tensor.ReduceOpDim.W, 1)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :, :1]

    return output


def reduce_sum_hw(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.reduce(
        t0, ttm.tensor.ReduceOpMath.SUM, ttm.tensor.ReduceOpDim.HW, 1
    )

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def reduce_max_h(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.reduce(t0, ttm.tensor.ReduceOpMath.MAX, ttm.tensor.ReduceOpDim.H, 1)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_max_w(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.reduce(t0, ttm.tensor.ReduceOpMath.MAX, ttm.tensor.ReduceOpDim.W, 1)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1]

    return output


def reduce_max_hw(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.reduce(
        t0, ttm.tensor.ReduceOpMath.MAX, ttm.tensor.ReduceOpDim.HW, 1
    )

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def transpose_wh(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.transpose(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output


def transpose_hc(x, pcie_slot, *args, **kwargs):
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, pcie_slot)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    t0 = (
        ttm.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.ROW_MAJOR,
        )
        .to(ttm.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttm.tensor.transpose_hc(t0)

    output = torch.Tensor(t1.to(host).to(ttm.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttm.device.CloseDevice(device)

    return output
