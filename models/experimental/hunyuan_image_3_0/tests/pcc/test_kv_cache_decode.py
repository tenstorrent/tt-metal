# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# KV-cache DECODE PCC — TTNN single-token decode (kv_cache) vs fp32 host reference.
#
# After a prefill at HY_MAX_ISL, run HY_DECODE_STEPS single-token decode steps that
# append to the per-layer K/V cache. Teacher-forced like tt_transformers
# test_model.py: the reference-greedy (argmax) token is fed as the next input to
# BOTH paths after each comparison, so the TT decode never drifts off the reference
# trajectory and each step's PCC reflects that step's numerics alone.
#
# Each decode step is scored against a fresh fp32 full-sequence forward at that
# length (the non-cached equivalent of the cached decode). Both the backbone hidden
# state (post-ln_f) and the LM-head logits are checked (PCC >= PCC_REQUIRED).
#
# HY_MAX_ISL (default 512) is the prefill ISL; it is clamped to the HF ceiling
# (max_position_embeddings = 22800 for HunyuanImage-3.0), the largest position the
# RoPE tables define. HY_DECODE_STEPS decode positions follow it.
#
# Run:
#   HY_NUM_LAYERS=2 HY_MAX_ISL=512 HY_DECODE_STEPS=8 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_kv_cache_decode.py -v -s --timeout=3600

from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tests.pcc.kv_cache_pcc_common import (
    DECODE_GENERATION_LENGTH,
    MAX_ISL,
    NUM_LAYERS,
    PCC_REQUIRED,
    build_context,
    _pad_ids_to,
)


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
@pytest.mark.timeout(3600)
def test_kv_cache_decode_pcc(device):
    ctx = build_context(device)
    logger.info(
        f"KV-cache decode PCC: layers={NUM_LAYERS} max_isl={MAX_ISL} "
        f"steps={DECODE_GENERATION_LENGTH} pcc>={PCC_REQUIRED}"
    )

    ids = _pad_ids_to(ctx.prompt_ids, MAX_ISL)  # prefill prefix at max ISL
    state = ctx.new_kv_state(max_cache_len=MAX_ISL + DECODE_GENERATION_LENGTH)

    # Prefill fills the cache; its last-token logit picks the first decode token.
    _, prefill_logits = ctx.prefill(state, ids)
    next_tok = int(torch.argmax(prefill_logits, dim=-1).item())

    failing = []
    for step in range(DECODE_GENERATION_LENGTH):
        ids = torch.cat([ids, torch.tensor([[next_tok]], dtype=ids.dtype)], dim=1)

        tt_hidden, tt_logits = ctx.decode(state, ids)
        ref_hidden, ref_logits = ctx.reference_forward(ids)

        hidden_pcc = h.pcc(ref_hidden, tt_hidden)
        logits_pcc = h.pcc(ref_logits, tt_logits)
        pos = int(ids.shape[1]) - 1
        logger.info(
            f"  step={step} pos={pos}: hidden_pcc={hidden_pcc:.6f}  " f"logits_pcc={logits_pcc:.6f} (>= {PCC_REQUIRED})"
        )
        if hidden_pcc < PCC_REQUIRED or logits_pcc < PCC_REQUIRED:
            failing.append((step, hidden_pcc, logits_pcc))

        # Teacher forcing: reference-greedy token feeds both paths next step.
        next_tok = int(torch.argmax(ref_logits, dim=-1).item())

    ctx.free_kv_state(state)

    assert not failing, "decode PCC below threshold: " + ", ".join(
        f"step={s} hidden={hp:.4f} logits={lp:.4f}" for s, hp, lp in failing
    )
