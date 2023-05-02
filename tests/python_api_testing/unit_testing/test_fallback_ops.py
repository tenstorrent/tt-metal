import torch
import libs.tt_lib as ttl
from tests.python_api_testing.models.utility_functions import comp_allclose_and_pcc, comp_pcc
from libs.tt_lib.fallback_ops import fallback_ops
from loguru import logger

def test_reshape_fallback():
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.arange(start=0, end=1024, dtype=torch.bfloat16).reshape(1, 1, 32, 32)
    pt_out = torch.reshape(x, (1, 1, 1, 1024))

    # Test on host RM
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
    )

    t1 = fallback_ops.reshape(t0, 1, 1, 1, 1024)

    output = torch.Tensor(
        t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(t1.shape())
    comp_pass, _ = comp_pcc(pt_out, output)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert(comp_pass)

    # Test on device RM
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        ).to(device)
    )

    t1 = fallback_ops.reshape(t0, 1, 1, 1, 1024)

    output = torch.Tensor(
        t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(t1.shape())
    comp_pass, _ = comp_pcc(pt_out, output)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert(comp_pass)

    # Test pytorch
    t0 = x

    t1 = fallback_ops.reshape(t0, 1, 1, 1, 1024)

    output = torch.Tensor(
        t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(t1.shape())
    comp_pass, _ = comp_pcc(pt_out, output)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert(comp_pass)

    ttl.device.CloseDevice(device)

def test_chunk_fallback():
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.arange(start=0, end=32*32*32*32, dtype=torch.bfloat16).reshape(32, 32, 32, 32)
    pt_out1, pt_out2 = torch.chunk(x, 2, 0)

    # Test on host RM
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
    )

    t1, t2 = fallback_ops.chunk(t0, 2, 0)

    output1 = torch.Tensor(
        t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(t1.shape())
    output2 = torch.Tensor(
        t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(t2.shape())
    comp_pass, _ = comp_pcc(pt_out1, output1)
    _, comp_out = comp_allclose_and_pcc(pt_out1, output1)
    logger.info(comp_out)
    assert(comp_pass)
    comp_pass, _ = comp_pcc(pt_out2, output2)
    _, comp_out = comp_allclose_and_pcc(pt_out2, output2)
    logger.info(comp_out)
    assert(comp_pass)

    # Test on device RM
    t0 = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
    )

    t1, t2 = fallback_ops.chunk(t0, 2, 0)

    output1 = torch.Tensor(
        t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(t1.shape())
    output2 = torch.Tensor(
        t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(t2.shape())
    comp_pass, _ = comp_pcc(pt_out1, output1)
    _, comp_out = comp_allclose_and_pcc(pt_out1, output1)
    logger.info(comp_out)
    assert(comp_pass)
    comp_pass, _ = comp_pcc(pt_out2, output2)
    _, comp_out = comp_allclose_and_pcc(pt_out2, output2)
    logger.info(comp_out)
    assert(comp_pass)

    # Test pytorch
    t0 = x

    t1, t2 = fallback_ops.chunk(t0, 2, 0)

    output1 = torch.Tensor(
        t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(t1.shape())
    output2 = torch.Tensor(
        t2.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    ).reshape(t2.shape())
    comp_pass, _ = comp_pcc(pt_out1, output1)
    _, comp_out = comp_allclose_and_pcc(pt_out1, output1)
    logger.info(comp_out)
    assert(comp_pass)
    comp_pass, _ = comp_pcc(pt_out2, output2)
    _, comp_out = comp_allclose_and_pcc(pt_out2, output2)
    logger.info(comp_out)
    assert(comp_pass)

    ttl.device.CloseDevice(device)
