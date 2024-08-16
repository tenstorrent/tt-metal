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
    "input_shape, weight_shape, bias_shape, num_groups, eps, on_device",
    (
        (
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 1, 1, 2]),
            torch.Size([1, 1, 1, 2]),
            1,
            1e-5,
            False,
        ),
        (
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 1, 1, 2]),
            torch.Size([1, 1, 1, 2]),
            1,
            1e-5,
            True,
        ),
        (
            torch.Size([2, 6, 12, 6]),
            torch.Size([1, 1, 1, 6]),
            torch.Size([1, 1, 1, 6]),
            6,
            1e-5,
            False,
        ),
        (
            torch.Size([2, 6, 12, 6]),
            torch.Size([1, 1, 1, 6]),
            torch.Size([1, 1, 1, 6]),
            6,
            1e-5,
            True,
        ),
        (
            torch.Size([1, 2, 3, 4]),
            None,
            None,
            1,
            1e-5,
            False,
        ),
        (
            torch.Size([1, 2, 3, 4]),
            None,
            None,
            1,
            1e-5,
            True,
        ),
        (
            torch.Size([2, 6, 12, 6]),
            None,
            None,
            6,
            1e-5,
            False,
        ),
        (
            torch.Size([2, 6, 12, 6]),
            None,
            None,
            6,
            1e-5,
            True,
        ),
    ),
)
def test_group_norm_fallback(input_shape, weight_shape, bias_shape, num_groups, eps, on_device, device):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    w = torch.randn(weight_shape).bfloat16().float() if weight_shape is not None else weight_shape
    b = torch.randn(bias_shape).bfloat16().float() if bias_shape is not None else bias_shape
    pt_out = torch.nn.functional.group_norm(
        x,
        num_groups,
        w.reshape(
            x.shape[1],
        )
        if w is not None
        else w,
        b.reshape(
            x.shape[1],
        )
        if b is not None
        else b,
        eps,
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
    if w is not None:
        w0 = ttnn.Tensor(
            w.reshape(-1).tolist(),
            w.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        if on_device:
            w0 = w0.to(device)
    else:
        w0 = None

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
    t1 = fallback_ops.group_norm(t0, num_groups, w0, b0, eps)

    output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape, num_groups, num_channels, eps, elementwise_affine, on_device",
    (
        (
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 1, 1, 2]),
            torch.Size([1, 1, 1, 2]),
            1,
            2,
            1e-5,
            True,
            False,
        ),
        (
            torch.Size([1, 2, 3, 4]),
            torch.Size([1, 1, 1, 2]),
            torch.Size([1, 1, 1, 2]),
            1,
            2,
            1e-5,
            True,
            True,
        ),
        (
            torch.Size([2, 6, 12, 6]),
            torch.Size([1, 1, 1, 6]),
            torch.Size([1, 1, 1, 6]),
            6,
            6,
            1e-5,
            True,
            False,
        ),
        (
            torch.Size([2, 6, 12, 6]),
            torch.Size([1, 1, 1, 6]),
            torch.Size([1, 1, 1, 6]),
            6,
            6,
            1e-5,
            True,
            True,
        ),
    ),
)
def test_GroupNorm_fallback(
    input_shape,
    weight_shape,
    bias_shape,
    num_groups,
    num_channels,
    eps,
    elementwise_affine,
    on_device,
    device,
):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    w = torch.randn(weight_shape).bfloat16().float()
    b = torch.randn(bias_shape).bfloat16().float()
    pt_nn = torch.nn.GroupNorm(num_groups, num_channels, eps, elementwise_affine)
    pt_nn.weight = torch.nn.Parameter(
        w.reshape(
            num_channels,
        )
    )
    pt_nn.bias = torch.nn.Parameter(
        b.reshape(
            num_channels,
        )
    )
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

    b0 = ttnn.Tensor(
        b.reshape(-1).tolist(),
        b.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        b0 = b0.to(device)

    tt_nn = fallback_ops.GroupNorm(w0, b0, num_groups, num_channels, eps, elementwise_affine)
    t1 = tt_nn(t0)

    output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
