# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib.fallback_ops as fallback_ops

from models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
)
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
        (
            torch.Size([1, 3, 6, 4]),
            torch.Size([3, 3, 6, 4]),
            None,
            1,
            0,
            1,
            1,
            False,
        ),
        (
            torch.Size([1, 4, 32, 16]),
            torch.Size([4, 1, 32, 16]),
            None,
            1,
            0,
            1,
            4,
            True,
        ),
    ),
)
def test_conv2d_fallback(
    input_shape,
    weight_shape,
    bias_shape,
    stride,
    padding,
    dilation,
    groups,
    on_device,
    device,
):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    w = torch.randn(weight_shape).bfloat16().float()
    b = torch.randn(bias_shape).bfloat16().float() if bias_shape is not None else bias_shape
    pt_out = torch.conv2d(
        x,
        w,
        torch.reshape(b, (b.shape[-1],)) if b is not None else b,
        stride,
        padding,
        dilation,
        groups,
    )

    # Test on host RM
    t0 = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        t0 = t0.to(device)

    w0 = ttnn.Tensor(
        w.reshape(-1).tolist(),
        w.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        w0 = w0.to(device)

    if b is not None:
        b0 = ttnn.Tensor(
            b.reshape(-1).tolist(),
            b.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        if on_device:
            b0 = b0.to(device)
    else:
        b0 = b

    t1 = fallback_ops.conv2d(t0, w0, b0, stride, padding, dilation, groups)

    output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)


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
        (
            torch.Size([1, 3, 6, 4]),
            torch.Size([3, 3, 6, 4]),
            None,
            3,
            3,
            1,
            1,
            0,
            1,
            1,
            False,
            "zeros",
            False,
        ),
        (
            torch.Size([1, 4, 6, 4]),
            torch.Size([4, 1, 6, 4]),
            None,
            4,
            4,
            1,
            1,
            0,
            1,
            4,
            False,
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
    device,
):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    w = torch.randn(weight_shape).bfloat16().float()
    b = torch.randn(bias_shape).bfloat16().float() if bias_shape is not None else bias_shape
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
    if not bias and bias_shape is not None:
        logger.warning("Bias set to false but trying to set a bias tensor, Ignoring specified bias tensor")
    if bias:
        pt_nn.bias = torch.nn.Parameter(b.reshape((b.shape[-1]))) if b is not None else b

    pt_out = pt_nn(x)

    # Test on host RM
    t0 = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        t0 = t0.to(device)

    w0 = ttnn.Tensor(
        w.reshape(-1).tolist(),
        w.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        w0 = w0.to(device)

    if b is not None:
        b0 = ttnn.Tensor(
            b.reshape(-1).tolist(),
            b.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        if on_device:
            b0 = b0.to(device)
    else:
        b0 = None

    tt_nn = fallback_ops.Conv2d(
        w0,
        b0 if bias else None,
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

    output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
