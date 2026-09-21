# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: full encoder block (time -> group -> FF). Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2EncoderBlock as RefBlock
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden, tiny_config


def test_encoder_block_golden():
    cfg = tiny_config()
    torch.manual_seed(0)
    x = torch.randn(2, 8, cfg.d_model)
    out = RefBlock(cfg).eval()(
        x,
        position_ids=torch.arange(8).unsqueeze(0).expand(2, -1),
        attention_mask=torch.zeros(2, cfg.num_heads, 8, 8),
        group_time_mask=torch.zeros(8, 1, 2, 2),
    ).hidden_states
    assert out.shape == x.shape
    log_golden("encoder_block/out", out)
