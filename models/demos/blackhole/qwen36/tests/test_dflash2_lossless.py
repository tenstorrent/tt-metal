# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 spec decode must be LOSSLESS: same bar and method as test_spec_lossless.py, with the DFlash2
block drafter in place of the MTP head. The plain decode path is TEACHER-FORCED down the DFlash2
trajectory (test_spec_lossless._reference_greedy) and every committed token must be the plain
argmax at its position unless that argmax was a bf16 near-tie (top-2 gap < NEAR_TIE_GAP). Acceptance
is logged, never asserted.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_dflash2_lossless.py -v -s
Env: see test_spec_lossless.py (QWEN36_LOSSLESS_PROMPT_LEN, QWEN36_SPEC_NEAR_TIE_GAP, ...).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tests.test_spec_lossless import (
    MAX_NEW,
    NEAR_TIE_GAP,
    NUM_BLOCKS,
    PROMPT_LEN,
    _reference_greedy,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

# Long-prompt runs: QWEN36_LOSSLESS_PROMPT_LEN=<n> asks _get_prompt for n tokens (the long corpora may
# return slightly fewer; the actual length is logged) and sizes the KV cache to fit prompt + MAX_NEW.
# Unset = the 130-token default, byte-identical to before.
_PROMPT_LEN = int(os.environ.get("QWEN36_LOSSLESS_PROMPT_LEN", str(PROMPT_LEN)))
_NUM_BLOCKS = NUM_BLOCKS
while _NUM_BLOCKS * BLOCK_SIZE < _PROMPT_LEN + MAX_NEW + 2 * BLOCK_SIZE:
    _NUM_BLOCKS *= 2


def check_lossless(spec, ref, gaps, tokenizer, tag, near_tie_gap=None):
    """The near-tie gate over EVERY position (the reference is teacher-forced, so no position is
    skipped): a mismatch fails only where plain greedy was CONFIDENT."""
    near_tie_gap = (
        float(os.environ.get("QWEN36_SPEC_NEAR_TIE_GAP", NEAR_TIE_GAP)) if near_tie_gap is None else near_tie_gap
    )
    n = len(spec)
    assert len(ref) == n, f"{tag}: {len(ref)} reference positions for {n} spec tokens"
    ties = [i for i in range(n) if gaps[i] < near_tie_gap]
    mismatches = [i for i in range(n) if spec[i] != ref[i]]
    confident = [i for i in mismatches if gaps[i] >= near_tie_gap]
    logger.info(
        f"[{tag}] near-tie positions (plain top-2 gap < {near_tie_gap}): {len(ties)}/{n} {ties}, min gap {min(gaps):.4f}"
    )
    assert not confident, (
        f"{tag}: DFlash2 committed a token plain greedy would NOT have chosen, where plain was CONFIDENT:\n"
        + "\n".join(
            f"  position {i}: spec {spec[i]} ({tokenizer.decode([spec[i]])!r}) vs plain argmax {ref[i]} "
            f"({tokenizer.decode([ref[i]])!r}), plain top-2 gap = {gaps[i]:.4f} >= {near_tie_gap}"
            for i in confident
        )
    )
    if not mismatches:
        logger.info(f"[{tag}] PASSED: reproduced plain greedy for all {n} tokens")
    else:
        logger.info(
            f"[{tag}] PASSED: {n - len(mismatches)}/{n} identical; near-tie flip(s) at "
            + ", ".join(f"{i} (gap {gaps[i]:.4f})" for i in mismatches)
        )


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dflash2_decode_is_lossless(mesh_device):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.dflash2_decode import DFlash2Decoder

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=_NUM_BLOCKS * BLOCK_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    token_ids = _get_prompt(_PROMPT_LEN, tokenizer)
    prompt_ids = token_ids[0].tolist()
    if _PROMPT_LEN == PROMPT_LEN:
        assert len(prompt_ids) == PROMPT_LEN
    else:
        assert 0 < len(prompt_ids) <= _PROMPT_LEN
    logger.info(f"[dflash2-lossless] prompt {len(prompt_ids)} tokens (asked {_PROMPT_LEN}), {_NUM_BLOCKS} KV blocks")
    kv_shape = [_NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(_NUM_BLOCKS, dtype=torch.int32).reshape(1, _NUM_BLOCKS)
    # Explicit and model-scoped: verify runs the fused GDN op, so the plain reference must use the
    # same math or greedy near-ties flip between the two paths.
    model.set_gdn_fused_decode(True)

    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    dec = DFlash2Decoder(model, pt)
    spec = dec.generate(prompt_ids, MAX_NEW)
    logger.info(f"[dflash2-lossless] dflash2 {len(spec)} tokens: {spec}")
    logger.info(f"[dflash2-lossless] dflash2 text: {tokenizer.decode(spec)!r}")
    dec.log_stats(prefix="dflash2-lossless")
    logger.info(
        f"[dflash2-lossless] ttft={dec.prefill_time:.2f}s decode={len(spec) / max(dec.decode_time, 1e-9):.2f} tok/s "
        f"over {dec.iters} iters"
    )
    assert len(spec) == MAX_NEW, f"expected {MAX_NEW} tokens, got {len(spec)}"
    assert dec.accept_rate() > 0, "no draft was accepted: the drafter is dead (losslessness alone cannot tell)"

    ref, gaps = _reference_greedy(model, prompt_ids, pt, kv_shape, spec)
    logger.info(f"[dflash2-lossless] plain argmax at each spec position: {ref}")
    model.free_kv_caches()
    check_lossless(spec, ref, gaps, tokenizer, "dflash2-lossless")
