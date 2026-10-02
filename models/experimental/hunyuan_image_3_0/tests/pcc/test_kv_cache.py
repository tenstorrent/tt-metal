# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# KV-cache PCC — TTNN backbone (kv_cache) vs fp32 host reference, prefill and decode.
# Shared rig lives in ``kv_cache_pcc_common.py``; both phases score the backbone hidden
# state (post-ln_f) and the LM-head logits at PCC >= PCC_REQUIRED.
#
# PREFILL
#   One full-sequence forward per input sequence length (ISL) with use_cache=True: the
#   prefill that fills the per-layer K/V cache before decode. A cached prefill is
#   mathematically the non-cached full forward, so it is scored against the fp32 host
#   reference at the same ISL.
#
#   The max ISL is defined by the HF checkpoint (max_position_embeddings, == 22800 for
#   HunyuanImage-3.0): the sanity gate runs 128 + HY_MAX_ISL (default 512, clamped to
#   the HF ceiling); the @slow sweep climbs powers of two up to the HF ceiling.
#
# DECODE
#   After a prefill at HY_MAX_ISL, run HY_DECODE_STEPS single-token decode steps that
#   append to the per-layer K/V cache. Teacher-forced like tt_transformers
#   test_model.py: the reference-greedy (argmax) token is fed as the next input to
#   BOTH paths after each comparison, so the TT decode never drifts off the reference
#   trajectory and each step's PCC reflects that step's numerics alone. Each step is
#   scored against a fresh fp32 full-sequence forward at that length.
#
# Run:
#   HY_NUM_LAYERS=2 HY_MAX_ISL=512 HY_DECODE_STEPS=8 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_kv_cache.py -v -s --timeout=3600
#   # prefill sanity gate only:
#   ... -m pytest .../test_kv_cache.py -k "prefill and sanity" -v -s
#   # full HF-ceiling prefill sweep:
#   ... -m pytest .../test_kv_cache.py -k "prefill and sweep" -v -s

from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tests.pcc.kv_cache_pcc_common import (
    DECODE_GENERATION_LENGTH,
    HF_MAX_ISL,
    MAX_ISL,
    NUM_LAYERS,
    PCC_REQUIRED,
    PREFILL_SANITY_SEQ_LENGTHS,
    PREFILL_SWEEP_SEQ_LENGTHS,
    build_context,
    _pad_ids_to,
)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------
def _run_prefill_sweep(device, seq_lengths):
    ctx = build_context(device)
    logger.info(
        f"KV-cache prefill PCC: layers={NUM_LAYERS} ISLs={seq_lengths} " f"hf_max_isl={HF_MAX_ISL} pcc>={PCC_REQUIRED}"
    )

    failing = []
    for seq_len in seq_lengths:
        ids = _pad_ids_to(ctx.prompt_ids, seq_len)

        state = ctx.new_kv_state(max_cache_len=seq_len)
        tt_hidden, tt_logits = ctx.prefill(state, ids)
        ctx.free_kv_state(state)

        ref_hidden, ref_logits = ctx.reference_forward(ids)

        hidden_pcc = h.pcc(ref_hidden, tt_hidden)
        logits_pcc = h.pcc(ref_logits, tt_logits)
        logger.info(f"  ISL={seq_len:5d}: hidden_pcc={hidden_pcc:.6f}  logits_pcc={logits_pcc:.6f} (>= {PCC_REQUIRED})")
        if hidden_pcc < PCC_REQUIRED or logits_pcc < PCC_REQUIRED:
            failing.append((seq_len, hidden_pcc, logits_pcc))

    assert not failing, "prefill PCC below threshold: " + ", ".join(
        f"ISL={s} hidden={hp:.4f} logits={lp:.4f}" for s, hp, lp in failing
    )


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
@pytest.mark.timeout(3600)
def test_kv_cache_prefill_pcc_sanity(device):
    """Short-ISL gate (128 + HY_MAX_ISL): one cache-populating prefill per length."""
    _run_prefill_sweep(device, PREFILL_SANITY_SEQ_LENGTHS)


@pytest.mark.slow
@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
@pytest.mark.timeout(0)
def test_kv_cache_prefill_pcc_sweep(device):
    """Full ISL sweep up to the HF ceiling (max_position_embeddings = HF_MAX_ISL)."""
    _run_prefill_sweep(device, PREFILL_SWEEP_SEQ_LENGTHS)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
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
