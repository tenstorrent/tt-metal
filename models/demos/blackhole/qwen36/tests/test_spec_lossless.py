# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Spec decode must reproduce plain greedy except where the reference top-2 gap is a near-tie.
Prompt length 130 is not a multiple of the KV block or the 32-row tile."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

# Default deliberately not a multiple of BLOCK_SIZE (64) nor of the 32-row tile (see docstring).
PROMPT_LEN = int(os.environ.get("QWEN36_LOSSLESS_PROMPT_LEN", 130))
MAX_NEW = 48
NUM_BLOCKS = int(os.environ.get("QWEN36_LOSSLESS_NUM_BLOCKS", 64))
DEFAULT_NEAR_TIE_GAP = 2.0


def _top2_gap(row):
    """Top-1 minus top-2 logit: how confident this greedy argmax is."""
    top2 = torch.topk(row.float().reshape(-1), 2)
    return float(top2.values[0] - top2.values[1])


def _reference_greedy(model, prompt_ids, page_table, kv_shape, max_new, use_decode_step):
    """Plain greedy trajectory. Same prefill entry point as spec, so the difference is speculation."""
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    T = len(prompt_ids)
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.int32)
    logits_dev = model.prefill_for_spec(prompt, page_table, T, lambda hidden, chunk_start, valid_len: None)
    lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    ttnn.deallocate(logits_dev)
    # Logits are replicated; the first vocab_size entries are device 0.
    row = lt.reshape(-1)[: model.vocab_size].float()

    tok = int(row.argmax())  # the token at absolute position T
    tokens, gaps = [tok], [_top2_gap(row)]
    pos = T
    while len(tokens) < max_new:
        if use_decode_step:
            # The paged single-token decode: the production decode kernels, one token at a time.
            row, hidden = model.decode_step_paged(tok, pos, page_table)
        else:
            # verify_forward is the fallback reference (QWEN36_SPEC_REF_PATH). It returns logits and a hidden.
            vlogits, hidden = model.verify_forward([tok], pos, page_table, gdn_recurrent=True)
            row = vlogits[0]
        ttnn.deallocate(hidden)  # the MTP seed hidden; the reference has no drafter to feed
        tok = int(row.argmax())
        tokens.append(tok)
        gaps.append(_top2_gap(row))
        pos += 1

    model.free_kv_caches()
    return tokens, gaps


@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_decode_is_lossless(mesh_device):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    assert model.mtp is not None, "MTP head not built"
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    token_ids = _get_prompt(PROMPT_LEN, tokenizer)
    prompt_ids = token_ids[0].tolist()
    assert len(prompt_ids) == PROMPT_LEN, f"wanted a {PROMPT_LEN}-token prompt, got {len(prompt_ids)}"
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)
    near_tie_gap = float(os.environ.get("QWEN36_SPEC_NEAR_TIE_GAP", DEFAULT_NEAR_TIE_GAP))

    # Put the reference on the fused GDN op too, or near-ties flip between two kernels.
    if not int(os.environ.get("QWEN36_SPEC_LOSSLESS_STOCK_GDN", "0")):
        model.set_gdn_fused_decode(True)

    ref_path = os.environ.get("QWEN36_SPEC_REF_PATH", "decode")
    try:
        ref, gaps = _reference_greedy(model, prompt_ids, pt, kv_shape, MAX_NEW, use_decode_step=ref_path == "decode")
    except Exception as e:  # decode_step_paged is a debug entry point with no other callers
        if ref_path != "decode":
            raise
        logger.warning(
            f"[lossless] decode_step_paged reference failed ({type(e).__name__}: {e}) — "
            "falling back to the eager verify_forward single-token reference"
        )
        ref_path = "verify"
        ref, gaps = _reference_greedy(model, prompt_ids, pt, kv_shape, MAX_NEW, use_decode_step=False)
    logger.info(f"[lossless] reference ({ref_path}) {len(ref)} tokens: {ref}")
    logger.info(f"[lossless] reference text: {tokenizer.decode(ref)!r}")

    # Fresh KV caches and a new decoder; prefill re-zeroes GDN state.
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    dec = SpeculativeDecoder(model, pt)
    spec = dec.generate(prompt_ids, MAX_NEW)
    logger.info(f"[lossless] spec      {len(spec)} tokens: {spec}")
    logger.info(f"[lossless] spec text:      {tokenizer.decode(spec)!r}")
    # Logged, never asserted: acceptance is a speed property, and this test is about correctness.
    dec.log_stats(prefix="lossless")
    model.free_kv_caches()

    n = min(len(ref), len(spec))
    assert n == MAX_NEW, f"expected {MAX_NEW} tokens from both runs, got ref={len(ref)} spec={len(spec)}"
    ties = [i for i in range(n) if gaps[i] < near_tie_gap]
    logger.info(
        f"[lossless] near-tie positions (ref top-2 gap < {near_tie_gap}): {len(ties)}/{n} {ties}, "
        f"min gap {min(gaps[:n]):.4f}"
    )
    div = next((i for i in range(n) if spec[i] != ref[i]), None)

    if div is None:
        logger.info(f"[lossless] PASSED: spec decode reproduced plain greedy for all {n} tokens, token for token")
        return

    gap = gaps[div]
    assert gap < near_tie_gap, (
        f"spec decode diverged from plain greedy at token {div}, where the reference was CONFIDENT:\n"
        f"  expected {ref[div]} ({tokenizer.decode([ref[div]])!r}), "
        f"got {spec[div]} ({tokenizer.decode([spec[div]])!r})\n"
        f"  reference top-2 gap = {gap:.4f} >= QWEN36_SPEC_NEAR_TIE_GAP {near_tie_gap} — "
        "this is a real speculation bug, not bf16 noise\n"
        f"  ref [:{div + 1}] = {ref[: div + 1]}\n"
        f"  spec[:{div + 1}] = {spec[: div + 1]}"
    )
    # A near-tie flip ends the comparison: later tokens are different strings.
    logger.info(
        f"[lossless] PASSED: lossless up to {div}, near-tie flip (gap={gap:.4f} < {near_tie_gap}): "
        f"ref {ref[div]} ({tokenizer.decode([ref[div]])!r}) vs spec {spec[div]} "
        f"({tokenizer.decode([spec[div]])!r}); trajectories legitimately diverge after this point"
    )
