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
    "input_shape",
    [torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 320, 384]), torch.Size([1, 3, 320, 384])],
)
@pytest.mark.parametrize("on_device", [False, True])
class TestMathOps:
    def test_ceil_fallbackop(self, input_shape, on_device, device):
        torch.manual_seed(1234)

        x = torch.randn(input_shape).bfloat16().float()
        pt_out = torch.ceil(x)

        # Test on host RM
        t0 = ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        if on_device:
            t0 = t0.to(device)

        t1 = fallback_ops.ceil(t0)

        output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
        _, comp_out = comp_allclose_and_pcc(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass

    def test_floor_fallbackop(self, input_shape, on_device, device):
        torch.manual_seed(1234)

        x = torch.randn(input_shape).bfloat16().float()
        pt_out = torch.floor(x)

        # Test on host RM
        t0 = ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        if on_device:
            t0 = t0.to(device)

        t1 = fallback_ops.floor(t0)

        output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
        _, comp_out = comp_allclose_and_pcc(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass

    @pytest.mark.parametrize("other", [1.5, 2.0, 3.0])
    def test_unary_fmod_fallbackop(self, input_shape, other, on_device, device):
        torch.manual_seed(1234)

        x = torch.randn(input_shape).bfloat16().float()
        pt_out = torch.fmod(x, other)

        # Test on host RM
        t0 = ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        if on_device:
            t0 = t0.to(device)

        t1 = fallback_ops.unary_fmod(t0, other)

        output = t1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
        _, comp_out = comp_allclose_and_pcc(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass

    def test_binary_fmod_fallbackop(self, input_shape, on_device, device):
        torch.manual_seed(1234)

        x = torch.randn(input_shape).bfloat16().float()
        y = torch.randn(input_shape).bfloat16().float()

        pt_out = torch.fmod(x, y)

        # Test on host RM
        t0 = ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        t1 = ttnn.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        if on_device:
            t0 = t0.to(device)
            t1 = t1.to(device)

        tout = fallback_ops.binary_fmod(t0, t1)

        output = tout.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
        _, comp_out = comp_allclose_and_pcc(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass
