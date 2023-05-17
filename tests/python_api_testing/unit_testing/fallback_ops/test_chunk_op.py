import torch
import libs.tt_lib as ttl
from tests.python_api_testing.models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from libs.tt_lib.fallback_ops import fallback_ops
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "input_shape", [torch.Size([6, 12, 6, 24]), torch.Size([24, 30, 6, 6])]
)
@pytest.mark.parametrize("chunks", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("on_device", [False, True])
def test_chunk_fallback(input_shape, chunks, dim, on_device):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.randn(input_shape).to(torch.bfloat16)
    pt_out = torch.chunk(x, chunks, dim)

    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    tt_out = fallback_ops.chunk(x, chunks, dim)

    for i in range(len(pt_out)):
        pt_output = pt_out[i]
        tt_output = torch.Tensor(
            tt_out[i].to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
        ).reshape(tt_out[i].shape())
        comp_pass, _ = comp_pcc(pt_output, tt_output)
        _, comp_out = comp_allclose_and_pcc(pt_output, tt_output)
        logger.info(comp_out)
        assert comp_pass

    del tt_out

    ttl.device.CloseDevice(device)
