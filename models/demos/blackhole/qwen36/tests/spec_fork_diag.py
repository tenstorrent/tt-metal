# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fork diagnostic (M2 stage 2): where does a spec trajectory leave the plain greedy one, and was that a near-tie?

Runs the demo's single-user spec loop (native MTP by default, K=11 -- the natreg configuration -- or
QWEN36_DRAFTER=dflash2) on the demo's ~PROMPT_LEN-token prompt, then teacher-forces the plain decode down the
spec trajectory (test_spec_lossless._reference_greedy) with BOTH plain GDN decode kernels:
  ring       = fused_recurrent_gated_delta_rule (set_gdn_fused_decode(True); the lossless tests' reference)
  composite  = the composite op chain (set_gdn_fused_decode(False); what QWEN36_SPEC=0 in the demo runs)
and logs, per mismatch position, the plain argmax and the plain top-2 logit gap. Log-only; nothing asserted
beyond shapes. Set QWEN36_GDN_SPEC_FUSED=0/1 to pick the verify path under test.

Run (natreg config): TT_CACHE_PATH=.../tt_cache_dflash QWEN36_GDN_SPEC_FUSED=1 MESH_DEVICE=P150x4 \
  pytest models/demos/blackhole/qwen36/tests/spec_fork_diag.py -q -s
Env: QWEN36_FORK_PROMPT_LEN (128), QWEN36_FORK_MAX_NEW (50), QWEN36_FORK_K (11), QWEN36_DRAFTER (mtp).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tests.test_spec_lossless import NEAR_TIE_GAP, NUM_BLOCKS, _reference_greedy
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PROMPT_LEN = int(os.environ.get("QWEN36_FORK_PROMPT_LEN", "128"))
MAX_NEW = int(os.environ.get("QWEN36_FORK_MAX_NEW", "50"))
DRAFT_LEN = int(os.environ.get("QWEN36_FORK_K", "11"))


def _report(tag, spec, ref, gaps, tokenizer):
    n = len(spec)
    mism = [i for i in range(n) if spec[i] != ref[i]]
    ties = [i for i in range(n) if gaps[i] < NEAR_TIE_GAP]
    conf = [i for i in mism if gaps[i] >= NEAR_TIE_GAP]
    logger.info(f"[fork:{tag}] plain argmax per spec position: {ref}")
    logger.info(f"[fork:{tag}] plain top-2 gap per position: {[round(g, 3) for g in gaps]}")
    logger.info(
        f"[fork:{tag}] near-tie positions (gap < {NEAR_TIE_GAP}): {len(ties)}/{n} {ties}, min gap {min(gaps):.4f}"
    )
    if not mism:
        logger.info(f"[fork:{tag}] spec == plain greedy at all {n} positions")
    for i in mism:
        logger.info(
            f"[fork:{tag}] position {i}: spec {spec[i]} ({tokenizer.decode([spec[i]])!r}) vs plain argmax {ref[i]} "
            f"({tokenizer.decode([ref[i]])!r}), plain top-2 gap = {gaps[i]:.4f} "
            f"{'NEAR-TIE' if gaps[i] < NEAR_TIE_GAP else 'CONFIDENT MISMATCH'}"
        )
    logger.info(
        f"[fork:{tag}] VERDICT: {n - len(mism)}/{n} identical, {len(mism)} mismatch(es), "
        f"{len(conf)} confident (gap >= {NEAR_TIE_GAP}) -> {'PASS (near-ties only)' if not conf else 'FAIL'}"
    )
    return conf


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_fork_diag(mesh_device):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    fused = os.environ.get("QWEN36_GDN_SPEC_FUSED") == "1"
    drafter = os.environ.get("QWEN36_DRAFTER", "mtp").lower()
    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    token_ids = _get_prompt(PROMPT_LEN, tokenizer)
    prompt_ids = token_ids[0].tolist()
    assert len(prompt_ids) == PROMPT_LEN, f"wanted a {PROMPT_LEN}-token prompt, got {len(prompt_ids)}"
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)

    # --- spec run, exactly as the demo's _run_tp_spec_generation constructs it ------------------------------ #
    model.set_gdn_fused_decode(True)
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    if drafter == "dflash2":
        from models.demos.blackhole.qwen36.tt.dflash2_decode import DFlash2Decoder, default_draft_len

        dec = DFlash2Decoder(model, pt, draft_len=default_draft_len())
    else:
        dec = SpeculativeDecoder(model, pt, draft_len=DRAFT_LEN)
    spec = dec.generate(prompt_ids, MAX_NEW)
    tag0 = f"{drafter} FUSED={int(fused)}"
    logger.info(f"[fork:{tag0}] spec {len(spec)} tokens: {spec}")
    logger.info(f"[fork:{tag0}] spec text: {tokenizer.decode(spec)!r}")
    dec.log_stats(prefix=f"fork:{tag0}")
    logger.info(
        f"[fork:{tag0}] accept={dec.accept_rate():.2f}/{dec.K} over {dec.iters} iters, "
        f"decode={len(spec) / max(dec.decode_time, 1e-9):.2f} tok/s"
    )
    model.free_kv_caches()
    assert len(spec) == MAX_NEW

    # --- plain reference, teacher-forced down the spec trajectory, both plain GDN kernels ------------------ #
    confident = {}
    for kern, enabled in (("ring", True), ("composite", False)):
        model.set_gdn_fused_decode(enabled)
        ref, gaps = _reference_greedy(model, prompt_ids, pt, kv_shape, spec)
        assert len(ref) == MAX_NEW
        confident[kern] = _report(f"{tag0} vs plain-{kern}", spec, ref, gaps, tokenizer)
    model.set_gdn_fused_decode(True)
    model.free_kv_caches()
    logger.info(
        f"[fork:{tag0}] SUMMARY confident mismatches: ring={confident['ring']} composite={confident['composite']}"
    )
