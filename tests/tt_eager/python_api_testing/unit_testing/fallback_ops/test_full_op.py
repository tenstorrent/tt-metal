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


@pytest.mark.parametrize("input_shape", [torch.Size([1, 3, 6, 4]), torch.Size([3, 2, 65, 10])])
@pytest.mark.parametrize("fill_value", [13.8, 5.5, 31, 0.1])
def test_full_fallback(input_shape, fill_value, device):
    torch.manual_seed(1234)

    fill_value = torch.Tensor([fill_value]).bfloat16().float().item()
    pt_out = torch.full(input_shape, fill_value)

    t0 = fallback_ops.full(input_shape, fill_value)

    output = t0.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
