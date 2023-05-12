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
    "input_shape", [torch.Size([1, 3, 6, 4]), torch.Size([3, 2, 65, 10])]
)
@pytest.mark.parametrize("fill_value", [13.8, 5.5, 31, 0.1])
def test_full_fallback(input_shape, fill_value):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    pt_out = torch.full(input_shape, fill_value)

    t0 = fallback_ops.full(input_shape, fill_value)

    output = torch.Tensor(t0.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t0.shape()
    )
    comp_pass, _ = comp_pcc(pt_out, output)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass

    ttl.device.CloseDevice(device)
