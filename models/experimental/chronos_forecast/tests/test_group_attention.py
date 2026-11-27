# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: GroupSelfAttention (attends over batch dim). Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import GroupSelfAttention as RefGSA
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden, tiny_config


def test_group_attention_golden():
    cfg = tiny_config()
    torch.manual_seed(0)
    x = torch.randn(2, 8, cfg.d_model)
    out = RefGSA(cfg).eval()(x, attention_mask=torch.zeros(8, 1, 2, 2)).hidden_states
    assert out.shape == x.shape
    log_golden("group_attn/out", out)
