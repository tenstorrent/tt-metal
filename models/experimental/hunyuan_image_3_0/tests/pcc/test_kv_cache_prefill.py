# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# KV-cache PREFILL PCC — TTNN backbone (kv_cache populated) vs fp32 host reference.
#
# One full-sequence forward per input sequence length (ISL) with use_cache=True: this
# is the prefill that fills the per-layer K/V cache before decode. The cached prefill
# is mathematically the non-cached full forward, so it is scored against the fp32
# host reference at the same ISL. Both the backbone hidden state (post-ln_f, the KV
# cache's own output) and the LM-head logits are checked (PCC >= PCC_REQUIRED).
#
# The max ISL is defined by the HF checkpoint (max_position_embeddings, == 22800 for
# HunyuanImage-3.0): the sanity gate runs 128 + HY_MAX_ISL (default 512, clamped to
# the HF ceiling); the @slow sweep climbs powers of two up to the HF ceiling.
#
# Run:
#   HY_NUM_LAYERS=2 HY_MAX_ISL=512 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_kv_cache_prefill.py -k sanity -v -s
#   # full HF-ceiling sweep:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_kv_cache_prefill.py -k sweep -v -s

from __future__ import annotations

import pytest
from loguru import logger

from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tests.pcc.kv_cache_pcc_common import (
    HF_MAX_ISL,
    NUM_LAYERS,
    PCC_REQUIRED,
    PREFILL_SANITY_SEQ_LENGTHS,
    PREFILL_SWEEP_SEQ_LENGTHS,
    build_context,
    _pad_ids_to,
)


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
