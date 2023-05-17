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
    "input_shapes, dim, on_device",
    (
        (
            (
                torch.Size([1, 3, 6, 4]),
                torch.Size([1, 3, 6, 4]),
                torch.Size([1, 3, 6, 4]),
            ),
            1,
            True,
        ),
        (
            (
                torch.Size([1, 3, 6, 4]),
                torch.Size([1, 3, 6, 4]),
                torch.Size([1, 3, 6, 4]),
            ),
            1,
            False,
        ),
        ((torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 5, 20])), 3, True),
        ((torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 5, 20])), 3, False),
    ),
)
def test_concat_fallback(input_shapes, dim, on_device):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    xs = [torch.randn(input_shape).to(torch.bfloat16) for input_shape in input_shapes]
    pt_out = torch.concat(xs, dim)

    # Test on host RM
    t0s = []
    for x in xs:
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        if on_device:
            t0 = t0.to(device)
        t0s.append(t0)

    t1 = fallback_ops.concat(t0s, dim)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    comp_pass, _ = comp_pcc(pt_out, output)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass

    del t1

    ttl.device.CloseDevice(device)
