# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: RoPE cos/sin tables. Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import Chronos2RotaryEmbedding as RefRotary
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden, tiny_config


def test_rope_golden():
    cfg = tiny_config()
    torch.manual_seed(0)
    rope = RefRotary(cfg).eval()
    x = torch.randn(2, cfg.num_heads, 8, cfg.d_kv)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    cos, sin = rope(x, pos)
    assert cos.shape == (2, 8, cfg.d_kv) and sin.shape == cos.shape
    log_golden("rope/cos", cos)
    log_golden("rope/sin", sin)
