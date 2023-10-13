# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib as ttl
from tt_lib.utils import (
    is_close,
)
from models.utility_functions import comp_equal

shapes = [[1, 1, 32, 32], [1, 1, 32, 128]]


@pytest.mark.parametrize("APPROX", [True, False])
@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 1, 32, 128]])
def test_expstuff(shape, device, APPROX):
    torch.manual_seed(1234)
    x = -88 * torch.ones(shape).bfloat16().float()
    pt_out = torch.zeros(shape).bfloat16().float()
    xt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttl.tensor.exp(xt, APPROX)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    passing, output = comp_equal(pt_out, tt_got_back)
    logger.info(output)
    assert passing
