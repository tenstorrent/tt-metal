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
    "input_shape, weight_shape, bias_shape, stride, padding, dilation, groups, on_device",
    (
        (
            torch.Size([1, 3, 6, 4]),
            torch.Size([3, 3, 6, 4]),
            torch.Size([1, 1, 1, 3]),
            1,
            0,
            1,
            1,
            False,
        ),
        (
            torch.Size([1, 4, 32, 16]),
            torch.Size([4, 1, 32, 16]),
            torch.Size([1, 1, 1, 4]),
            1,
            0,
            1,
            4,
            True,
        ),
    ),
)
def test_conv2d_fallback(
    input_shape, weight_shape, bias_shape, stride, padding, dilation, groups, on_device
):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.randn(input_shape).to(torch.bfloat16)
    w = torch.randn(weight_shape).to(torch.bfloat16)
    b = torch.randn(bias_shape).to(torch.bfloat16)
    pt_out = torch.conv2d(
        x, w, torch.reshape(b, (b.shape[-1],)), stride, padding, dilation, groups
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

    t1 = fallback_ops.conv2d(t0, w0, b0, stride, padding, dilation, groups)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    comp_pass, _ = comp_pcc(pt_out, output)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass

    ttl.device.CloseDevice(device)


@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,on_device",
    (
        (
            torch.Size([1, 3, 6, 4]),
            torch.Size([3, 3, 6, 4]),
            torch.Size([1, 1, 1, 3]),
            3,
            3,
            1,
            1,
            0,
            1,
            1,
            True,
            "zeros",
            False,
        ),
        (
            torch.Size([1, 4, 6, 4]),
            torch.Size([4, 1, 6, 4]),
            torch.Size([1, 1, 1, 4]),
            4,
            4,
            1,
            1,
            0,
            1,
            4,
            True,
            "zeros",
            True,
        ),
    ),
)
def test_Conv2d_fallback(
    input_shape,
    weight_shape,
    bias_shape,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
    padding_mode,
    on_device,
):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.randn(input_shape).to(torch.bfloat16)
    w = torch.randn(weight_shape).to(torch.bfloat16)
    b = torch.randn(bias_shape).to(torch.bfloat16)
    pt_nn = torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    )
    pt_nn.weight = torch.nn.Parameter(w)
    pt_nn.bias = torch.nn.Parameter(b.reshape((b.shape[-1])))
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

    tt_nn = fallback_ops.Conv2d(
        w0,
        b0,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
    )
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
