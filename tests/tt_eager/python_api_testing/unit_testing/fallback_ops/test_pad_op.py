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
    "input_shape, pad, mode, value, on_device",
    (
        (torch.Size([1, 3, 6, 4]), (2, 2, 3, 1, 0, 1, 3, 2), "constant", 1.0, True),
        (torch.Size([1, 3, 6, 4]), (2, 2, 3, 1, 0, 1, 3, 2), "constant", 1.0, False),
        (torch.Size([2, 4, 31, 16]), (1, 3, 1, 0, 17, 3, 5, 7), "constant", 9.2, True),
        (torch.Size([2, 4, 31, 16]), (1, 3, 1, 0, 17, 3, 5, 7), "constant", 9.2, False),
    ),
)
def test_pad_fallback(input_shape, pad, mode, value, on_device, device):
    torch.manual_seed(1234)

    value = torch.Tensor([value]).bfloat16().float().item()
    x = torch.randn(input_shape).bfloat16().float()
    pt_out = torch.nn.functional.pad(x, pad, mode, value)

    # Test on host RM
    t0 = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    if on_device:
        t0 = t0.to(device)

    t1 = fallback_ops.pad(t0, pad, mode, value)

    output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
