from libs import tt_lib as ttl
import torch


def test_tensor_alignment():
    a = torch.arange(1022).to(torch.bfloat16).reshape(1, 1, 1, 1022)
    b = torch.arange(1024).to(torch.bfloat16).reshape(1, 1, 32, 32)

    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    t0 = ttl.tensor.Tensor(
        a.reshape(-1).tolist(),
        a.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    t1 = ttl.tensor.Tensor(
        b.reshape(-1).tolist(),
        b.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
    )

    t0_d = t0.to(device)

    t1_d = t1.to(device)

    t0_h = t0_d.to(host)

    assert torch.equal(a, torch.Tensor(t0_h.data()).reshape(t0_h.shape()))

    t1_h = t1_d.to(host)

    assert torch.equal(b, torch.Tensor(t1_h.data()).reshape(t1_h.shape()))

    ttl.device.CloseDevice(device)
