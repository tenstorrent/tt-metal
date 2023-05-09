import torch
import tt_lib as ttl
from tt_lib.utils import pad_weight


################################################
#################### TT-DNN ####################
################################################
def eltwise_exp(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_recip(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_sqrt(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_gelu(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_relu(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_sigmoid(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_log(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_tanh(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_add(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_sub(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def eltwise_mul(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def matmul(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bmm(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bcast_add_h(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        # Pad bcast tensor
        y = pad_weight(y)
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

        t2 = ttl.tensor.bcast(
            t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H
        )

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bcast_add_w(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        # Pad bcast tensor
        y = pad_weight(y)
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

        t2 = ttl.tensor.bcast(
            t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.W
        )

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bcast_add_hw(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        # Pad bcast tensor
        y = pad_weight(y)
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

        t2 = ttl.tensor.bcast(
            t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW
        )

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bcast_sub_h(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        # Pad bcast tensor
        y = pad_weight(y)
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

        t2 = ttl.tensor.bcast(
            t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.H
        )

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bcast_sub_w(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        # Pad bcast tensor
        y = pad_weight(y)
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

        t2 = ttl.tensor.bcast(
            t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.W
        )

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bcast_sub_hw(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        # Pad bcast tensor
        y = pad_weight(y)
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

        t2 = ttl.tensor.bcast(
            t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.HW
        )

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bcast_mul_h(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        # Pad bcast tensor
        y = pad_weight(y)
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

        t2 = ttl.tensor.bcast(
            t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H
        )

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bcast_mul_w(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        # Pad bcast tensor
        y = pad_weight(y)
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

        t2 = ttl.tensor.bcast(
            t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W
        )

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def bcast_mul_hw(x, y, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
        # Pad bcast tensor
        y = pad_weight(y)
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

        t2 = ttl.tensor.bcast(
            t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW
        )

        output = torch.Tensor(
            t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t2.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def reduce_sum_h(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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
            t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.H, 1
        )

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_sum_w(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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
            t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, 1
        )

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :, :1]

    return output


def reduce_sum_hw(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def reduce_max_h(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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
            t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.H, 1
        )

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :]

    return output


def reduce_max_w(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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
            t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.W, 1
        )

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1]

    return output


def reduce_max_hw(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]

    return output


def transpose_wh(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def transpose_hc(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def transpose_cn(x, pcie_slot, *args, **kwargs):
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        t1 = ttl.tensor.transpose_cn(t0)

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def permute(x, pcie_slot, *args, **kwargs):
    assert "permute_dims" in kwargs

    permute_dims = kwargs["permute_dims"]

    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        t1 = ttl.tensor.permute(t0, *permute_dims)

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def reshape(x, pcie_slot, *args, **kwargs):
    assert "reshape_dims" in kwargs

    reshape_dims = kwargs["reshape_dims"]

    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, pcie_slot)
    ttl.device.InitializeDevice(device)

    try:
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

        t1 = ttl.tensor.reshape(t0, *reshape_dims)

        output = torch.Tensor(
            t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t1.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


################################################
#################### Tensor ####################
################################################
def datacopy(x, pcie_slot, *args, **kwargs):
    ttl_tensor_dtype = kwargs.get("dtype", ttl.tensor.DataType.BFLOAT16)

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
            .to(device)
        )

        output = torch.Tensor(
            t0.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(t0.shape())
    finally:
        ttl.device.CloseDevice(device)

    return output


def pad(x, pcie_slot, *args, **kwargs):
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


def unpad(x, pcie_slot, *args, **kwargs):
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
