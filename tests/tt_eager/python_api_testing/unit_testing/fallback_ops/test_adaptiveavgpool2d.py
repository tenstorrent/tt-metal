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
    "input_shape, output_size, on_device",
    (
        (torch.Size([1, 2, 6, 8]), (2, 4), False),
        (torch.Size([1, 2, 6, 8]), (2, 4), True),
        (torch.Size([2, 1, 32, 64]), 6, False),
        (torch.Size([2, 1, 32, 64]), 6, False),
    ),
)
def test_AdaptiveAvgPool2d_fallback(
    input_shape,
    output_size,
    on_device,
    device,
):
    torch.manual_seed(1234)

    x = torch.randn(input_shape).bfloat16().float()
    pt_nn = torch.nn.AdaptiveAvgPool2d(output_size)
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

    tt_nn = fallback_ops.AdaptiveAvgPool2d(output_size)
    t1 = tt_nn(t0)

    output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
