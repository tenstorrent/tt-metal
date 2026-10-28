# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verifier-added regression tests for gaps found in review (not part of the immutable spec)."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.clamp_square_practice import clamp_square_practice


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        pytest.param(-1e300, 1e300, id="beyond_fp32_both"),
        pytest.param(-1e300, 0.5, id="beyond_fp32_lo"),
        pytest.param(-0.5, 1e39, id="beyond_fp32_hi"),
    ],
)
def test_rejects_bounds_beyond_fp32_range(device, min_val, max_val, expect_error):
    """Finite Python floats past fp32 range used to escape as OverflowError from the host packer.

    torch.clamp refuses them as well, so the op rejects them in validate() with ValueError.
    """
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error((ValueError,), ""):
        clamp_square_practice(x, min_val, max_val)


def test_fp32_max_bounds_reduce_to_square(device):
    """Bounds at exactly +-FLT_MAX are legal and leave every finite bf16 input unclamped."""
    fmax = 3.4028234663852886e38
    shape = (1, 1, 64, 128)
    torch.manual_seed(1)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    expected = torch.clamp(x.to(torch.float32), -fmax, fmax) ** 2
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(clamp_square_practice(tt_x, -fmax, fmax)).to(torch.float32)
    assert_with_pcc(expected, actual, 0.9999)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "1.0", True])
def test_rejects_bad_bound_values(device, bad, expect_error):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), ""):
        clamp_square_practice(x, bad, 1.0)
