# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from loguru import logger
import pytest


@pytest.mark.parametrize("op_kind", ["or", "and", "not", "xor"])
@pytest.mark.parametrize("other", [5, 10, -1])
@pytest.mark.parametrize("on_device", [True, False])
@pytest.mark.parametrize(
    "input_shape",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bitwise_ops_fallback(op_kind, input_shape, other, on_device, device):
    torch.manual_seed(1234)

    # x = torch.randn(input_shape, dtype=torch.int8)
    x = torch.randint(low=0, high=100, size=input_shape)
    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)
    if op_kind == "or":
        t1 = ttl.fallback_ops.bitwise_or(t0, other)
        pt_out = torch.bitwise_or(x, other)
    elif op_kind == "and":
        t1 = ttl.fallback_ops.bitwise_and(t0, other)
        pt_out = torch.bitwise_and(x, other)
    elif op_kind == "xor":
        t1 = ttl.fallback_ops.bitwise_xor(t0, other)
        pt_out = torch.bitwise_xor(x, other)
    elif op_kind == "not":
        t1 = ttl.fallback_ops.bitwise_not(t0)
        pt_out = torch.bitwise_not(x)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    comp_pass, _ = comparison_funcs.comp_equal(pt_out, output)
    _, comp_out = comparison_funcs.comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass
