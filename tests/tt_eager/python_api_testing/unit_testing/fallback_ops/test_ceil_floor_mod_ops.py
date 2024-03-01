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
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        if on_device:
            t0 = t0.to(device)

        t1 = ttl.fallback_ops.ceil(t0)

        output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
        _, comp_out = comp_allclose_and_pcc(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass

    def test_floor_fallbackop(self, input_shape, on_device, device):
        torch.manual_seed(1234)

        x = torch.randn(input_shape).bfloat16().float()
        pt_out = torch.floor(x)

        # Test on host RM
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        if on_device:
            t0 = t0.to(device)

        t1 = ttl.fallback_ops.floor(t0)

        output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
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
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        if on_device:
            t0 = t0.to(device)

        t1 = ttl.fallback_ops.unary_fmod(t0, other)

        output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
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
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        t1 = ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        if on_device:
            t0 = t0.to(device)
            t1 = t1.to(device)

        tout = ttl.fallback_ops.binary_fmod(t0, t1)

        output = tout.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
        _, comp_out = comp_allclose_and_pcc(pt_out, output)
        logger.debug(comp_out)
        assert comp_pass
