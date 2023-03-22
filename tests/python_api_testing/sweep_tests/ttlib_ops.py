import torch
import ttlib as ttl


def datacopy(x, pcie_slot, *args, **kwargs):
    # TODO: Add actual datacopy once tensor op implementation is added

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    output = torch.Tensor(t0.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t0.shape()
    )

    ttl.device.CloseDevice(device)

    return output


def eltwise_exp(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.exp(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_recip(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.recip(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_sqrt(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.sqrt(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_gelu(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.gelu(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_relu(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.relu(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_sigmoid(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.sigmoid(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_log(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.log(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_tanh(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.tanh(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_add(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.add(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_sub(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.sub(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def eltwise_mul(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.mul(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def matmul(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.matmul(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bmm(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bmm(t0, t1)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bcast_add_h(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Pad bcast tensor
    y = ttl.utils.pad_weight(y)
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bcast_add_w(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Pad bcast tensor
    y = ttl.utils.pad_weight(y)
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.W)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bcast_add_hw(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Pad bcast tensor
    y = ttl.utils.pad_weight(y)
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bcast_sub_h(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Pad bcast tensor
    y = ttl.utils.pad_weight(y)
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.H)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bcast_sub_w(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Pad bcast tensor
    y = ttl.utils.pad_weight(y)
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.W)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bcast_sub_hw(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Pad bcast tensor
    y = ttl.utils.pad_weight(y)
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.HW)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bcast_mul_h(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Pad bcast tensor
    y = ttl.utils.pad_weight(y)
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bcast_mul_w(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Pad bcast tensor
    y = ttl.utils.pad_weight(y)
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def bcast_mul_hw(x, y, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    # Pad bcast tensor
    y = ttl.utils.pad_weight(y)
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    t1 = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW)

    output = torch.Tensor(t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t2.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def reduce_sum_h(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.reduce(t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.H, 1)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_sum_w(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.reduce(t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, 1)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :, :1]

    return output


def reduce_sum_hw(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.HW, 1
    )

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def reduce_max_h(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.reduce(t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.H, 1)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_max_w(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.reduce(t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.W, 1)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1]

    return output


def reduce_max_hw(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.HW, 1
    )

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def transpose_wh(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.transpose(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output


def transpose_hc(x, pcie_slot, *args, **kwargs):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    t1 = ttl.tensor.transpose_hc(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    ttl.device.CloseDevice(device)

    return output
