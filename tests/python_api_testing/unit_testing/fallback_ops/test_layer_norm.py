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
    "input_shape, weight_shape, bias_shape, normalized_shape, eps, on_device",
    (
        (
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 2, 3, 4]),
            (1, 2, 3, 4),
            1e-5,
            False,
        ),
        (
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 2, 3, 4]),
            (1, 2, 3, 4),
            1e-5,
            True,
        ),
        (
            torch.Size([2, 1, 3, 6]),
            torch.Size([1, 1, 3, 6]),
            torch.Size([1, 1, 3, 6]),
            (3, 6),
            1e-5,
            False,
        ),
        (
            torch.Size([2, 1, 3, 6]),
            torch.Size([1, 1, 3, 6]),
            torch.Size([1, 1, 3, 6]),
            (3, 6),
            1e-5,
            True,
        ),
    ),
)
def test_layer_norm_fallback(
    input_shape, weight_shape, bias_shape, normalized_shape, eps, on_device
):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.randn(input_shape).to(torch.bfloat16)
    w = torch.randn(weight_shape).to(torch.bfloat16)
    b = torch.randn(bias_shape).to(torch.bfloat16)
    pt_out = torch.nn.functional.layer_norm(
        x,
        normalized_shape,
        w.reshape(normalized_shape),
        b.reshape(normalized_shape),
        eps,
    )

    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    w0 = ttl.tensor.Tensor(
        w.reshape(-1).tolist(),
        w.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        w0 = w0.to(device)

    b0 = ttl.tensor.Tensor(
        b.reshape(-1).tolist(),
        b.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        b0 = b0.to(device)

    t1 = fallback_ops.layer_norm(t0, normalized_shape, w0, b0, eps)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    comp_pass, _ = comp_pcc(pt_out, output)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass

    del t1

    ttl.device.CloseDevice(device)


@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape, normalized_shape, eps, elementwise_affine, on_device",
    (
        (
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 2, 3, 4]),
            (1, 2, 3, 4),
            1e-5,
            True,
            False,
        ),
        (
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 2, 3, 4]),
            (1, 2, 3, 4),
            1e-5,
            True,
            True,
        ),
        (
            torch.Size([2, 1, 3, 6]),
            torch.Size([1, 1, 3, 6]),
            torch.Size([1, 1, 3, 6]),
            (3, 6),
            1e-5,
            True,
            False,
        ),
        (
            torch.Size([2, 1, 3, 6]),
            torch.Size([1, 1, 3, 6]),
            torch.Size([1, 1, 3, 6]),
            (3, 6),
            1e-5,
            True,
            True,
        ),
    ),
)
def test_LayerNorm_fallback(
    input_shape,
    weight_shape,
    bias_shape,
    normalized_shape,
    eps,
    elementwise_affine,
    on_device,
):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.randn(input_shape).to(torch.bfloat16)
    w = torch.randn(weight_shape).to(torch.bfloat16)
    b = torch.randn(bias_shape).to(torch.bfloat16)
    pt_nn = torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    pt_nn.weight = torch.nn.Parameter(w.reshape(normalized_shape))
    pt_nn.bias = torch.nn.Parameter(b.reshape(normalized_shape))
    pt_out = pt_nn(x)

    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    w0 = ttl.tensor.Tensor(
        w.reshape(-1).tolist(),
        w.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        w0 = w0.to(device)

    b0 = ttl.tensor.Tensor(
        b.reshape(-1).tolist(),
        b.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        b0 = b0.to(device)

    tt_nn = fallback_ops.LayerNorm(w0, b0, normalized_shape, eps, elementwise_affine)
    t1 = tt_nn(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    comp_pass, _ = comp_pcc(pt_out, output)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass

    del t1

    ttl.device.CloseDevice(device)
