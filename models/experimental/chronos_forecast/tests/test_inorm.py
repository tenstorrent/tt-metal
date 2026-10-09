# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub Block 3: instance normalization on V, reuse loc/scale on W.

Oracle: reference/chronos_bolt_ops.InstanceNorm (CPU golden).
Math: per-row nanmean/nanstd, eps on zero-variance, optional arcsinh.
Inverse must round-trip.
"""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos_bolt_ops import InstanceNorm as RefInstanceNorm
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden


def test_block3_norm_golden():
    torch.manual_seed(0)
    x = torch.randn(2, 16)
    x[0, 3] = float("nan")
    norm = RefInstanceNorm(use_arcsinh=True).eval()
    y, (loc, scale) = norm(x)
    assert y.shape == x.shape
    assert loc.shape == (2, 1) and scale.shape == (2, 1)
    assert torch.equal(y.isnan(), x.isnan())
    log_golden("block3/normed", y)
    log_golden("block3/loc", loc)
    log_golden("block3/scale", scale)

    back = norm.inverse(y, (loc, scale))
    torch.testing.assert_close(back, x, equal_nan=True, atol=1e-5, rtol=1e-5)
    log_golden("block3/roundtrip", back)


def test_block3_zero_variance_uses_eps():
    norm = RefInstanceNorm().eval()
    y, (loc, scale) = norm(torch.ones(1, 4))
    torch.testing.assert_close(loc, torch.ones(1, 1))
    assert torch.isfinite(y).all()
    log_golden("block3/zero_var", y)
