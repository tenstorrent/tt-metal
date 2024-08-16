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


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("on_device", [True, False])
class TestBitwiseOps:
    @pytest.mark.parametrize("op_kind", ["or", "and", "not", "xor"])
    @pytest.mark.parametrize("other", [5, 10, -1])
    def test_unary_bitwise_ops_fallback(self, input_shapes, other, on_device, op_kind, device):
        torch.manual_seed(1234)

        # x = torch.randn(input_shape, dtype=torch.int8)
        x = torch.randint(low=0, high=100, size=input_shapes)
        # Test on host RM
        t0 = ttnn.Tensor(
            x,
            ttnn.uint32,
        )
        if on_device:
            t0 = t0.to(device)
        if op_kind == "or":
            t1 = fallback_ops.unary_bitwise_or(t0, other)
            pt_out = torch.bitwise_or(x, other)
        elif op_kind == "and":
            t1 = fallback_ops.unary_bitwise_and(t0, other)
            pt_out = torch.bitwise_and(x, other)
        elif op_kind == "xor":
            t1 = fallback_ops.unary_bitwise_xor(t0, other)
            pt_out = torch.bitwise_xor(x, other)
        elif op_kind == "not":
            t1 = fallback_ops.bitwise_not(t0)
            pt_out = torch.bitwise_not(x)

        output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        comp_pass, _ = comparison_funcs.comp_equal(pt_out, output)
        _, comp_out = comparison_funcs.comp_allclose_and_pcc(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass

    @pytest.mark.parametrize("op_kind", ["or", "and", "xor"])
    def test_binary_bitwise_ops_fallback(self, input_shapes, on_device, op_kind, device):
        torch.manual_seed(1234)

        x = torch.randint(low=0, high=100, size=input_shapes)
        y = torch.randint(low=0, high=200, size=input_shapes)
        # Test on host RM
        t0 = ttnn.Tensor(
            x,
            ttnn.uint32,
        )
        if on_device:
            t0 = t0.to(device)
        t1 = ttnn.Tensor(
            y,
            ttnn.uint32,
        )
        if on_device:
            t1 = t1.to(device)

        if op_kind == "or":
            tout = fallback_ops.binary_bitwise_or(t0, t1)
            pt_out = torch.bitwise_or(x, y)
        elif op_kind == "and":
            tout = fallback_ops.binary_bitwise_and(t0, t1)
            pt_out = torch.bitwise_and(x, y)
        elif op_kind == "xor":
            tout = fallback_ops.binary_bitwise_xor(t0, t1)
            pt_out = torch.bitwise_xor(x, y)

        output = tout.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        comp_pass, _ = comparison_funcs.comp_equal(pt_out, output)
        _, comp_out = comparison_funcs.comp_allclose_and_pcc(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass
