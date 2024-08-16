# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib.fallback_ops as fallback_ops

from models.utility_functions import (
    comp_allclose_and_pcc,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "input_shape",
    [torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 320, 384]), torch.Size([1, 3, 320, 384])],
)
@pytest.mark.parametrize("on_device", [False, True])
class TestTruncOp:
    def test_trunc_fallbackop(self, input_shape, on_device, device):
        torch.manual_seed(1234)

        x = torch.randn(input_shape).bfloat16().float()
        pt_out = torch.trunc(x)

        # Test on host RM
        t0 = ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        if on_device:
            t0 = t0.to(device)

        t1 = fallback_ops.trunc(t0)

        output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        comp_pass, _ = comp_equal(pt_out, output)
        _, comp_out = comp_allclose_and_pcc(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass
