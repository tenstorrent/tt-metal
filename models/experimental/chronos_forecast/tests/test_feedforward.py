# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: FeedForward (RMSNorm -> MLP -> residual). Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import FeedForward as RefFF
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden, tiny_config


def test_feedforward_golden():
    cfg = tiny_config()
    torch.manual_seed(0)
    ff = RefFF(cfg).eval()
    x = torch.randn(2, 8, cfg.d_model)
    y = ff(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    log_golden("feedforward/out", y)
