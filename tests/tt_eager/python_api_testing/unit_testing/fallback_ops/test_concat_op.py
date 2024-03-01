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
    "input_shapes, dim, on_device",
    (
        (
            (
                torch.Size([1, 3, 6, 4]),
                torch.Size([1, 3, 6, 4]),
                torch.Size([1, 3, 6, 4]),
            ),
            1,
            True,
        ),
        (
            (
                torch.Size([1, 3, 6, 4]),
                torch.Size([1, 3, 6, 4]),
                torch.Size([1, 3, 6, 4]),
            ),
            1,
            False,
        ),
        ((torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 5, 20])), 3, True),
        ((torch.Size([2, 3, 5, 4]), torch.Size([2, 3, 5, 20])), 3, False),
    ),
)
def test_concat_fallback(input_shapes, dim, on_device, device):
    torch.manual_seed(1234)

    xs = [torch.randn(input_shape).bfloat16().float() for input_shape in input_shapes]
    pt_out = torch.concat(xs, dim)

    # Test on host RM
    t0s = []
    for x in xs:
        t0 = ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        if on_device:
            t0 = t0.to(device)
        t0s.append(t0)

    t1 = ttl.fallback_ops.concat(t0s, dim)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.debug(comp_out)
    assert comp_pass
