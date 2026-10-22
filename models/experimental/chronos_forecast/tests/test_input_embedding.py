# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub: input patch embedding incl. REG token concat. Oracle: reference/."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH, log_golden


def test_input_embedding_golden():
    assert (DUMMY_MODEL_PATH / "config.json").is_file()
    model = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    torch.manual_seed(0)
    context = torch.randn(2, 32)
    patched_ctx, attn_mask, loc_scale = model._prepare_patched_context(context)
    patched_fut, _ = model._prepare_patched_future(
        torch.randn(2, 16), None, loc_scale, num_output_patches=1, batch_size=2
    )

    ctx_emb = model.input_patch_embedding(patched_ctx)
    fut_emb = model.input_patch_embedding(patched_fut)
    assert ctx_emb.shape[-1] == model.config.d_model
    log_golden("in_embed/ctx", ctx_emb)
    log_golden("in_embed/fut", fut_emb)

    if model.chronos_config.use_reg_token:
        reg = model.shared.weight[model.config.reg_token_id]
        log_golden("in_embed/reg_token", reg)
