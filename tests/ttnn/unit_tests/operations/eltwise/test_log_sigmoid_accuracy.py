# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""log_sigmoid across the whole real line, gated on ULP.

The form this replaced split the domain at +-4 with no arm for x <= -4, so an
input there came back unchanged. The sweeps could not see it: they draw from
[-4, 10], which starts exactly at the missing boundary.
"""

import pytest
import torch
import ttnn

# The boundaries of the range split that is gone, and points either side of it.
NAMED = [-30.0, -10.0, -4.0001, -4.0, -3.9999, -1.0, 0.0, 1.0, 3.9999, 4.0, 4.0001, 10.0, 30.0]


def _ulp(got, want, torch_dtype):
    eps = torch.finfo(torch_dtype).eps
    return (got.double() - want).abs() / want.abs().clamp(min=1e-30) / eps


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_log_sigmoid_named_points(device, dtype):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    x = torch.tensor(NAMED + [0.0] * (1024 - len(NAMED)), dtype=torch.float32).reshape(1, 1, 32, 32).to(torch_dtype)
    ix = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.log_sigmoid(ix)).flatten()[: len(NAMED)]
    want = torch.nn.functional.logsigmoid(torch.tensor(NAMED, dtype=torch.float64))

    ulp = _ulp(got, want, torch_dtype)
    assert ulp.max() <= 2.0, f"max {ulp.max():.1f} ULP at x={NAMED[int(ulp.argmax())]}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_log_sigmoid_sweep(device, dtype):
    """[-30, 30], which the existing sweeps cannot reach below -4."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)

    x = ((torch.rand((1, 1, 32, 32), dtype=torch.float32) * 60) - 30).to(torch_dtype)
    ix = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.log_sigmoid(ix))
    want = torch.nn.functional.logsigmoid(x.double())

    ulp = _ulp(got, want, torch_dtype)
    assert ulp.max() <= 2.0, f"max {ulp.max():.1f} ULP"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_log_sigmoid_below_minus_four_is_not_the_input(device, dtype):
    """The arm that was missing: x <= -4 used to return x unchanged."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    x = torch.full((1, 1, 32, 32), -4.0, dtype=torch_dtype)
    ix = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.log_sigmoid(ix)).flatten()[0]

    assert got != x.flatten()[0], "log_sigmoid(-4) returned its input"
    want = torch.nn.functional.logsigmoid(torch.tensor(-4.0, dtype=torch.float64))
    # Bounded in ulp, not absolutely: one bfloat16 ulp at this magnitude is
    # 0.03125, so an absolute bound tighter than that fails a correctly
    # rounded answer.
    assert _ulp(got.reshape(1), want.reshape(1), torch_dtype).max() <= 2.0, f"got {got}, want {want}"
