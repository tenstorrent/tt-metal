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
    "input_shape",
    [torch.Size([6, 12, 6, 24]), torch.Size([24, 30, 6, 6]), torch.Size([1, 2, 1, 2])],
)
@pytest.mark.parametrize("sizes", [[1, 2, 3, 4], [2, 2, 2, 2]])
@pytest.mark.parametrize("on_device", [False, True])
def test_repeat_fallback(input_shape, sizes, on_device):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)

    x = torch.randn(input_shape).bfloat16().float()
    pt_out = x.repeat(sizes)

    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    t1 = fallback_ops.repeat(t0, sizes)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass

    del t1

    ttl.device.CloseDevice(device)
