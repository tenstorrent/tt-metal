# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Spec decode must be LOSSLESS: it must reproduce the plain greedy trajectory, not merely a
plausible one.

Speculative decoding is only a speedup if the text it emits is the text the base model would have
emitted on its own. Acceptance rate does not prove that — a drafter whose proposals are verified
against a *subtly different* base forward (different GDN kernel, stale KV, a rolled-back state that
did not roll all the way back) still shows high acceptance while quietly writing a different story.
So this test runs the SAME prompt twice on the SAME weights:

  reference : prefill, then N eager single-token base decodes, argmax fed back (plain greedy)
  spec      : prefill, then SpeculativeDecoder.generate for the same N committed tokens

and walks the two id sequences to the first mismatch.

Exact equality is the wrong bar in bf16. The two paths reach the same position through different
op sequences (a batched masked-bucket verify vs one decode step), so their logits differ at ~1e-5;
wherever the top-2 logits are within that noise, greedy is a coin flip and EITHER token is a
faithful continuation. Once the trajectories take different tokens they are decoding different
strings, and everything after is incomparable. So, following the gemma4 methodology
(models/demos/gemma4/tests/unit/test_spec_decode.py::_assert_argmaxes_match_except_near_ties), the
reference run records the top-2 logit gap behind every one of its argmaxes, and a mismatch is:

  * a FAILURE if the reference was confident there (gap >= QWEN36_SPEC_NEAR_TIE_GAP, default 2.0)
    — spec decode changed a token the base model was sure about, which is a real bug;
  * a legitimate near-tie flip otherwise — reported as "lossless up to i", comparison stops there.

Prompt length is 130 on purpose: not a multiple of the 64-token KV block, not a multiple of the
32-row tile, and not a mask bucket (it lands in the 256 bucket with 126 padding rows). Off-by-one
frontier bugs — the first verify writing KV at the wrong offset, GDN masking one token too many —
survive a tile-aligned prompt and die here.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_spec_lossless.py -v -s

Env:
  QWEN36_SPEC_NEAR_TIE_GAP    top-1 minus top-2 logit below which a flip is a tie (default 2.0)
  QWEN36_SPEC_REF_PATH        "decode" (default, model.decode_step_paged) or "verify"
                              (model.verify_forward one token at a time)
  QWEN36_SPEC_LOSSLESS_STOCK_GDN=1  leave plain decode on the stock composite GDN kernel
  QWEN36_LOSSLESS_PROMPT_LEN  prompt length (default 130; see above for why that value)
  QWEN36_LOSSLESS_NUM_BLOCKS  KV blocks / max_seq_len budget (default 64); raise it with PROMPT_LEN
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

# Default deliberately not a multiple of BLOCK_SIZE (64) nor of the 32-row tile (see docstring).
PROMPT_LEN = int(os.environ.get("QWEN36_LOSSLESS_PROMPT_LEN", 130))
MAX_NEW = 48
NUM_BLOCKS = int(os.environ.get("QWEN36_LOSSLESS_NUM_BLOCKS", 64))
DEFAULT_NEAR_TIE_GAP = 2.0


def _top2_gap(row):
    """top-1 minus top-2 logit for one host row: how confident this greedy argmax is.

    A gap of ~0 means the two candidates are indistinguishable in bf16 and the argmax is decided by
    numerical noise; a large gap means the model committed to that token.
    """
    top2 = torch.topk(row.float().reshape(-1), 2)
    return float(top2.values[0] - top2.values[1])


def _reference_greedy(model, prompt_ids, page_table, kv_shape, max_new, use_decode_step):
    """Plain greedy decode: the trajectory spec decode must reproduce.

    Fresh KV caches + the SAME prefill entry point the spec loop uses (prefill_for_spec, minus the
    MTP warming callback), so the two runs start from an identical prompt state and any later
    difference is the speculation itself, not the prefill. prefill_masked_bucket re-zeroes the GDN
    recurrent + conv state at chunk_start==0, which is what makes this callable back-to-back with a
    spec run on the same model object.

    Returns (tokens, gaps): tokens[i] is the i-th generated id, gaps[i] the top-2 logit gap of the
    distribution that produced it (gaps[0] comes from the prefill logits).
    """
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    T = len(prompt_ids)
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.int32)
    logits_dev = model.prefill_for_spec(prompt, page_table, T, lambda hidden, chunk_start, valid_len: None)
    lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    ttnn.deallocate(logits_dev)
    # Logits are replicated across the TP mesh; the concat stacks the replicas, so the first
    # vocab_size entries are device 0's row (same slicing SpeculativeDecoder.generate uses).
    row = lt.reshape(-1)[: model.vocab_size].float()

    tok = int(row.argmax())  # the token at absolute position T
    tokens, gaps = [tok], [_top2_gap(row)]
    pos = T
    while len(tokens) < max_new:
        if use_decode_step:
            # The paged single-token decode: the production decode kernels, one token at a time.
            row, hidden = model.decode_step_paged(tok, pos, page_table)
        else:
            # Fallback reference step (see QWEN36_SPEC_REF_PATH). Same call the spec loop's own seed
            # makes (spec_decode.SpeculativeDecoder._seed): one eager recurrent verify over a
            # single candidate at absolute `pos`. A weaker statement than decode_step_paged — it
            # shares the verify machinery with the path under test — but it is the same base
            # forward and it always exists.
            # verify_forward returns (logits [K, vocab] host float, hidden [1,1,K,dim/tp] device).
            vlogits, hidden = model.verify_forward([tok], pos, page_table, gdn_recurrent=True)
            row = vlogits[0]
        ttnn.deallocate(hidden)  # the MTP seed hidden; the reference has no drafter to feed
        tok = int(row.argmax())
        tokens.append(tok)
        gaps.append(_top2_gap(row))
        pos += 1

    model.free_kv_caches()
    return tokens, gaps


@run_for_blackhole()
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

    # Spec verify advances GDN with the FUSED recurrent op (gdn/tp.py _forward_verify_recurrent_batched)
    # and SpeculativeDecoder puts plain decode on that same op (spec_decode.py, use_fused_recurrent_decode)
    # precisely because the composite decode kernel disagrees with it at ~1e-5, which flips greedy
    # near-ties. Put the reference on it too, so this test measures the speculation rather than the
    # difference between two GDN kernels (which test_fused_recurrent_gdn.py covers). B==max_batch_size==1
    # satisfies the fused path's full-batch assert. QWEN36_SPEC_LOSSLESS_STOCK_GDN=1 opts out.
    if not int(os.environ.get("QWEN36_SPEC_LOSSLESS_STOCK_GDN", "0")):
        for layer in model.layers:
            if not layer.is_full_attention:
                layer.attention.use_fused_recurrent_decode = True

    # --- reference: plain greedy ------------------------------------------------------------- #
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

    # --- spec run: same prompt, same fresh state ---------------------------------------------- #
    # Same reset recipe test_spec_determinism.py uses between runs: free + reallocate the paged KV
    # caches, build a fresh decoder. The GDN recurrent state is re-zeroed inside the prefill.
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    dec = SpeculativeDecoder(model, pt)
    spec = dec.generate(prompt_ids, MAX_NEW)
    logger.info(f"[lossless] spec      {len(spec)} tokens: {spec}")
    logger.info(f"[lossless] spec text:      {tokenizer.decode(spec)!r}")
    # Logged, never asserted: acceptance is a speed property, and this test is about correctness.
    dec.log_stats(prefix="lossless")
    model.free_kv_caches()

    # --- compare ------------------------------------------------------------------------------ #
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
    # A near-tie flip: both tokens are faithful continuations, but from here the two runs are
    # decoding different strings, so there is nothing left to compare.
    logger.info(
        f"[lossless] PASSED: lossless up to {div}, near-tie flip (gap={gap:.4f} < {near_tie_gap}): "
        f"ref {ref[div]} ({tokenizer.decode([ref[div]])!r}) vs spec {spec[div]} "
        f"({tokenizer.decode([spec[div]])!r}); trajectories legitimately diverge after this point"
    )
