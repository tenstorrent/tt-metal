# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user (B > 1) DFlash2 speculative decoding: every user must get its own plain greedy story.

test_spec_batched.py proves the batched substrate with the MTP drafter; this file swaps in the
DFlash2 block drafter (DFlash2Decoder) and reuses that test's method and helpers verbatim: B users
with DISTINCT prompts of DISTINCT, unaligned lengths run one batched spec generation; then a
SEPARATE max_batch_size=1 model teacher-forces the plain decode path down each user's trajectory
and every committed token must be plain greedy's argmax unless that argmax was a bf16 near-tie.
On top of the substrate's row-mixing hazards, the block drafter adds its own: B blocks of 8 rows
share one draft forward (user-major rows), one block-diagonal in-block conv, one paged context KV
with a block range per user, and one set of verify tap rows sliced per user.

K per batch: the drafter's block caps K at 7 (block 8) and the verify at B*(K+1) <= 32 rows, so
B=2 -> K=7, B=4 -> K=7 (32 rows), B=8 -> K=3 (a 4-row block, the drafter's own block trimmed).

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_dflash2_batched.py -v -s
Needs the drafter matched to the served weights (DFLASH_WEIGHTS; incoai/Qwen3.8-27B-DFlash2 for
Qwen3.8-27B) and the full 64-layer model (the taps live at layers 5..61).
"""
import gc

import pytest
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_spec_batched import (
    _assert_lossless,
    _batch_prompts,
    _batched_kv,
    _build_model,
    _fresh_kv,
    _reference_model,
    _release,
)
from models.demos.blackhole.qwen36.tests.test_spec_lossless import MAX_NEW, _reference_greedy

DFLASH_K = {2: 7, 4: 7, 8: 3}


@run_for_blackhole()
@pytest.mark.timeout(3600)  # two full model loads (B users, then the B=1 reference) + B references
@pytest.mark.parametrize("batch", [2, 4, 8])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dflash2_batched_lossless(mesh_device, batch):
    """Every user's batched DFlash2 output is that user's own plain greedy story, token for token."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.dflash2_decode import DFlash2Decoder

    B, K = batch, DFLASH_K[batch]
    assert B * (K + 1) <= 32, f"B={B} K={K} needs {B * (K + 1)} verify rows (one 32-row tile)"
    device = mesh_device
    device.enable_program_cache()

    # --- spec run: B users, distinct prompts, one shared paged KV ----------------------------- #
    model = _build_model(device, B)
    assert (
        len(model.layers) >= 62
    ), "the DFlash2 taps live at layers 5..61: run the full model (unset QWEN36_SPEC_TEST_N_LAYERS)"
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(B, tokenizer)
    page_tables, kv_shape = _batched_kv(model, B, K, prompts)
    _fresh_kv(model, kv_shape, B)
    dec = DFlash2Decoder(model, page_tables, draft_len=K)
    rows = dec.generate(prompts, MAX_NEW)
    dec.log_stats(prefix=f"dflash2-batched B={B}")
    accept = dec.accept_rate()
    logger.info(
        f"[dflash2-batched] B={B} K={K}: ttft={dec.prefill_time:.2f}s "
        f"per-user decode={MAX_NEW / max(dec.decode_time, 1e-9):.2f} tok/s, aggregate={B * MAX_NEW / max(dec.decode_time, 1e-9):.1f} tok/s "
        f"over {dec.iters} iters"
    )
    for u in range(B):
        logger.info(f"[dflash2-batched] B={B} user {u} ({len(prompts[u])} prompt tokens): {rows[u]}")
        logger.info(f"[dflash2-batched] B={B} user {u} text: {tokenizer.decode(rows[u])!r}")
    assert len(rows) == B, f"expected {B} output rows, got {len(rows)}"
    for u in range(B):
        assert len(rows[u]) == MAX_NEW, f"user {u} produced {len(rows[u])} tokens, wanted {MAX_NEW}"
    assert accept > 0, f"accept_rate() == {accept}: no draft was accepted for any user at B={B}"

    _release(model)
    del dec, model
    gc.collect()

    # --- reference: plain B=1 decode, teacher-forced down each user's trajectory --------------- #
    ref_model, pt1, kv1 = _reference_model(device)
    try:
        for u in range(B):
            ref, gaps = _reference_greedy(ref_model, prompts[u], pt1, kv1, rows[u])
            _assert_lossless(rows[u], ref, gaps, tokenizer, f"dflash2 B={B} user {u}")
    finally:
        _release(ref_model)
    logger.info(f"[dflash2-batched] B={B}: all {B} users lossless, accept={accept:.2f}/{K}")
