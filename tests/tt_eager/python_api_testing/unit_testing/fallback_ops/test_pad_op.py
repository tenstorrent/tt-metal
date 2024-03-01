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
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    t1 = ttl.fallback_ops.pad(t0, pad, mode, value)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
