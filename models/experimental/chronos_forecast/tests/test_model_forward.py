# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: full-model forward -> quantile preds. Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH, log_golden


def test_model_forward_golden():
    assert (DUMMY_MODEL_PATH / "config.json").is_file()
    model = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    model.config._attn_implementation = "eager"
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(context=torch.randn(2, 32), num_output_patches=1)
    assert out.quantile_preds is not None
    assert torch.isfinite(out.quantile_preds).all()
    log_golden("model/quantile_preds", out.quantile_preds)
