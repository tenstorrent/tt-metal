# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: ResidualBlock (input/output patch embedding math). Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import ResidualBlock as RefRB
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden


def test_residual_golden():
    torch.manual_seed(0)
    block = RefRB(in_dim=48, h_dim=8, out_dim=6, act_fn_name="relu", dropout_p=0.0).eval()
    x = torch.randn(2, 4, 48)
    y = block(x)
    assert y.shape == (2, 4, 6)
    log_golden("residual/out", y)
