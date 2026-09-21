# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: full encoder (mask build + N blocks + final norm). Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Encoder as RefEncoder
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden, tiny_config


def test_encoder_golden():
    cfg = tiny_config()
    torch.manual_seed(0)
    x = torch.randn(2, 8, cfg.d_model)
    out = RefEncoder(cfg).eval()(
        inputs_embeds=x,
        group_ids=torch.arange(2),
        attention_mask=torch.ones(2, 8),
    ).last_hidden_state
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    log_golden("encoder/out", out)
