# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: MLP (Linear, no bias -> act -> dropout -> Linear, no bias). Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import MLP as RefMLP
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden, tiny_config


def test_mlp_golden():
    cfg = tiny_config()
    torch.manual_seed(0)
    mlp = RefMLP(cfg).eval()
    x = torch.randn(2, 8, cfg.d_model)
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    log_golden("mlp/out", y)
