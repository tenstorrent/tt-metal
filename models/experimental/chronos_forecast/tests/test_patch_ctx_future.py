# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: Patch + time-encoding + masks (bridges Blocks 1-3 to embedding).

Oracle: reference Chronos2Model._prepare_patched_context/_prepare_patched_future
on the dummy checkpoint. Locks patched shapes, attention masks, loc/scale.
"""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH, log_golden


def test_patch_ctx_future_golden():
    assert (DUMMY_MODEL_PATH / "config.json").is_file()
    model = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    torch.manual_seed(0)
    context = torch.randn(2, 32)
    patched_ctx, attn_mask, loc_scale = model._prepare_patched_context(context)

    p = model.chronos_config.input_patch_size
    assert patched_ctx.shape == (2, patched_ctx.shape[1], 3 * p)
    assert attn_mask.shape == (2, patched_ctx.shape[1])
    log_golden("patch/patched_context", patched_ctx)
    log_golden("patch/loc", loc_scale[0])
    log_golden("patch/scale", loc_scale[1])

    future = torch.randn(2, 16)
    future[0, 3] = float("nan")
    patched_fut, fut_mask = model._prepare_patched_future(
        future_covariates=future,
        future_covariates_mask=None,
        loc_scale=loc_scale,
        num_output_patches=1,
        batch_size=2,
    )
    op = model.chronos_config.output_patch_size
    assert patched_fut.shape == (2, 1, 3 * op)
    assert fut_mask.shape == (2, 1, op)
    log_golden("patch/patched_future", patched_fut)
