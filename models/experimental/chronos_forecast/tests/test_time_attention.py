# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: TimeSelfAttention (Norm -> RoPE-MHA -> residual). Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import TimeSelfAttention as RefTSA
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden, tiny_config


def test_time_attention_golden():
    cfg = tiny_config()
    torch.manual_seed(0)
    x = torch.randn(2, 8, cfg.d_model)
    out = RefTSA(cfg).eval()(
        x,
        attention_mask=torch.zeros(2, cfg.num_heads, 8, 8),
        position_ids=torch.arange(8).unsqueeze(0).expand(2, -1),
    ).hidden_states
    assert out.shape == x.shape
    log_golden("time_attn/out", out)
