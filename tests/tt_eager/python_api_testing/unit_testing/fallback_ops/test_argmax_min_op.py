# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib.fallback_ops as fallback_ops

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from loguru import logger
import pytest


@pytest.mark.parametrize("on_device", [True, False])
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("arg_kind", ["max", "min"])
class TestArgMaxMinOps:
    @pytest.mark.parametrize("dim", [-2, -1, 0, 1, 2, 3])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_argmax_min_fallback(self, input_shapes, dim, keepdim, arg_kind, on_device, device):
        torch.manual_seed(1234)

        x = torch.randint(low=0, high=100, size=input_shapes)
        if keepdim == False:
            x = x.unsqueeze(0)
        # Test on host RM
        t0 = ttnn.Tensor(
            x,
            ttnn.uint32,
        )
        if on_device:
            t0 = t0.to(device)
        if arg_kind == "max":
            t1 = fallback_ops.torch_argmax(t0, dim, keepdim)
            pt_out = torch.argmax(x, dim, keepdim)
        elif arg_kind == "min":
            t1 = fallback_ops.torch_argmin(t0, dim, keepdim)
            pt_out = torch.argmin(x, dim, keepdim)

        output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        comp_pass, comp_out = comparison_funcs.comp_equal(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass
