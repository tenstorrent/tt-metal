# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: RMS LayerNorm (no bias, no mean-sub). Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import Chronos2LayerNorm as RefLayerNorm
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden


def test_layernorm_golden():
    torch.manual_seed(0)
    layer = RefLayerNorm(6).eval()
    x = torch.randn(2, 8, 6)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    log_golden("layernorm/out", y)
    log_golden("layernorm/weight", layer.weight)
