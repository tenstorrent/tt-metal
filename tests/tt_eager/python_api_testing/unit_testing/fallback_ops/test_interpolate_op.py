# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib as ttl
import tt_lib.fallback_ops
from models.utility_functions import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "input_shape, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias",
    (
        (torch.Size([1, 3, 6, 4]), (5, 10), None, "nearest", None, False, False),
        (torch.Size([1, 3, 6, 4]), None, (2, 3), "bilinear", True, True, True),
        (torch.Size([2, 4, 32, 56]), 8, None, "nearest", None, None, False),
    ),
)
@pytest.mark.parametrize("on_device", [True, False])
def test_pad_fallback(
    input_shape,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
    on_device,
    device,
):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    pt_out = torch.nn.functional.interpolate(
        x, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias
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

    t1 = ttl.fallback_ops.interpolate(t0, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
