# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: MHA with and without RoPE. Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import MHA as RefMHA
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden, tiny_config


def test_mha_golden():
    cfg = tiny_config()
    torch.manual_seed(0)
    x = torch.randn(2, 8, cfg.d_model)
    mask = torch.zeros(2, cfg.num_heads, 8, 8)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)

    y_plain = RefMHA(cfg, use_rope=False).eval()(x, mask=mask).hidden_states
    y_rope = RefMHA(cfg, use_rope=True).eval()(x, mask=mask, position_ids=pos).hidden_states
    assert y_plain.shape == x.shape and y_rope.shape == x.shape
    log_golden("mha/no_rope", y_plain)
    log_golden("mha/rope", y_rope)
