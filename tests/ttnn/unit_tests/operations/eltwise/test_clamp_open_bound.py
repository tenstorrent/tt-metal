# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ttnn.clamp with an omitted bound.

`torch.clamp(x, min=m)` is documented as `torch.maximum(x, m)` — there is no upper bound.
Before this fix the host substituted `FLT_MAX` for the omitted side, so `clamp(+inf, min=0)`
returned `3.4028235e+38` where torch returns `+inf`. This asserts the fixed contract on
float32 for the three inputs that reveal the sentinel: `+inf`, `-inf`, and `FLT_MAX` itself.
"""

import pytest
import torch

import ttnn


def _tt(x, device):
    return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _one(x, device):
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded.view(-1)[0] = x
    tt = ttnn.from_torch(padded, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return tt


@pytest.mark.parametrize(
    "x, min_v, max_v, expected",
    [
        (float("inf"), 0.0, None, float("inf")),
        (float("inf"), -1.0, None, float("inf")),
        (float("-inf"), None, 0.0, float("-inf")),
        (float("-inf"), None, 1.0, float("-inf")),
        # FLT_MAX is the old sentinel; with the omitted upper bound now +inf,
        # clamp(FLT_MAX, min=0) should pass FLT_MAX through unchanged.
        (torch.finfo(torch.float32).max, 0.0, None, torch.finfo(torch.float32).max),
    ],
)
def test_clamp_open_bound_float32(device, x, min_v, max_v, expected):
    """clamp with one bound omitted must not clip ±inf or FLT_MAX."""
    got_scalar = ttnn.to_torch(ttnn.clamp(_one(x, device), min=min_v, max=max_v)).view(-1)[0].item()
    assert got_scalar == expected, f"clamp({x}, min={min_v}, max={max_v}) = {got_scalar}, expected {expected}"


def test_clamp_open_bound_matches_tensor_overload(device):
    """The scalar-bound and tensor-bound overloads of ttnn.clamp must agree on finite input and ±inf."""
    x = torch.tensor([[1.0, -2.0, float("inf"), float("-inf"), 3.0]] + [[0.0] * 5] * 31 * 32, dtype=torch.float32).view(
        1, 1, 32, 32
    )[:, :, :1, :5]
    # Pad to a full 32x32 tile of zeros with the interesting row at [0, 0, 0, :5]
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, :5] = x.view(-1)[:5]

    tx = ttnn.from_torch(padded, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    zero_tensor = ttnn.from_torch(torch.zeros_like(padded), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    got_scalar = ttnn.to_torch(ttnn.clamp(tx, min=0.0))
    got_tensor = ttnn.to_torch(ttnn.clamp(tx, min=zero_tensor))
    ref = torch.clamp(padded, min=0.0)

    # Both overloads must equal torch, exactly, on the five special values in row 0.
    for i, v in enumerate(padded[0, 0, 0, :5].tolist()):
        assert (
            got_scalar[0, 0, 0, i].item() == ref[0, 0, 0, i].item()
        ), f"scalar overload at {v}: {got_scalar[0, 0, 0, i].item()} != torch {ref[0, 0, 0, i].item()}"
        assert (
            got_tensor[0, 0, 0, i].item() == ref[0, 0, 0, i].item()
        ), f"tensor overload at {v}: {got_tensor[0, 0, 0, i].item()} != torch {ref[0, 0, 0, i].item()}"
