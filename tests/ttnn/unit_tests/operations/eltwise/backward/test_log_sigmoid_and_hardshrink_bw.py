# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)


def _t(v, dtype, device):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.from_torch(torch.full(SHAPE, v, dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("x, grad", [(-4.0, 2.0), (0.0, 2.0), (4.0, -1.5)])
def test_log_sigmoid_bw_matches_torch(device, dtype, x, grad):
    got = ttnn.to_torch(ttnn.log_sigmoid_bw(_t(grad, dtype, device), _t(x, dtype, device))[0]).float().flatten()[0].item()
    t = torch.tensor([x], dtype=torch.float32, requires_grad=True)
    torch.nn.functional.logsigmoid(t).backward(torch.tensor([grad], dtype=torch.float32))
    assert got == pytest.approx(t.grad.item(), rel=2e-2)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_log_sigmoid_bw_propagates_nan(device, dtype):
    got = ttnn.to_torch(ttnn.log_sigmoid_bw(_t(2.0, dtype, device), _t(float("nan"), dtype, device))[0]).float().flatten()[0].item()
    assert math.isnan(got)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "x, lambd, grad, want",
    [
        (0.0, 0.5, 3.0, 0.0),
        (1.0, 0.5, 3.0, 3.0),
        (-1.0, 0.5, 3.0, 3.0),
        (0.0, -0.5, 3.0, 3.0),
        (float("nan"), 0.5, 3.0, 0.0),
    ],
)
def test_hardshrink_bw_domain_and_nan(device, dtype, x, lambd, grad, want):
    got = ttnn.to_torch(ttnn.hardshrink_bw(_t(grad, dtype, device), _t(x, dtype, device), lambd=lambd)[0]).float().flatten()[0].item()
    assert got == pytest.approx(want, abs=1e-5)
