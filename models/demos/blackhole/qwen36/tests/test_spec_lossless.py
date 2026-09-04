# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Spec decode must be lossless: reproduce the plain greedy trajectory, not merely a plausible one.

Acceptance rate does not prove that — a drafter verified against a subtly different base forward
(different GDN kernel, stale KV, incomplete rollback) still shows high acceptance while writing a
different story. Same prompt, same weights: reference (prefill + N eager single-token argmax
decodes) vs spec (prefill + SpeculativeDecoder.generate for the same N); walk to first mismatch.

Exact equality is the wrong bar in bf16: batched masked-bucket verify vs one decode step differ at
~1e-5, so a near top-2 gap is a coin flip and either token is faithful; after a flip the strings
diverge and later tokens are incomparable. Following gemma4
(models/demos/gemma4/tests/unit/test_spec_decode.py::_assert_argmaxes_match_except_near_ties), the
reference records the top-2 logit gap behind each argmax. A mismatch is a FAILURE if the reference
was confident (gap >= 2.0); otherwise a legitimate near-tie flip ("lossless up to i").

Prompt length 130 is not a multiple of the 64-token KV block, the 32-row tile, or a mask bucket
(it lands in the 256 bucket with 126 padding rows). Off-by-one frontier bugs (first verify writing
KV at the wrong offset, GDN masking one token too many) survive a tile-aligned prompt and die here.

Parametrized over acceptance mode. "greedy" is the argmax-prefix path. "topk1" runs the exact
speculative rejection sampler (tt/spec_sampling.py) at temp=1.0, top_k=1, top_p=1.0: top-k 1
collapses the sampled target to a delta at the argmax of the SAME bf16 verify logits the greedy
trace argmaxes on device, so "accept iff u < p(draft)" becomes "accept iff the draft IS that argmax"
and the run must reproduce greedy — up to bf16 near ties, where host torch.topk and device
ttnn.argmax may break an exact tie differently (the near-tie gate already arbitrates). That
checks sampling plumbing ([T, vocab] logits readback, which row belongs to which draft, the commit
index) rather than its math (test_spec_sampling_math.py).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

# 130 is not a multiple of BLOCK_SIZE (64) nor of the 32-row tile (see docstring).
PROMPT_LEN = 130
MAX_NEW = 48
NUM_BLOCKS = 64
NEAR_TIE_GAP = 2.0


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
            # Fallback if decode_step_paged fails: same call as SpeculativeDecoder._seed — one eager
            # recurrent verify over a single candidate at `pos`. Weaker (shares verify machinery with
            # the path under test) but the same base forward and always exists. Returns (logits [K, vocab] host float, hidden [1,1,K,dim/tp] device).
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
@pytest.mark.parametrize("sampling_mode", ["greedy", "topk1"])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_decode_is_lossless(mesh_device, sampling_mode):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder
    from models.demos.blackhole.qwen36.tt.spec_sampling import SpecSamplingParams

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

    # Spec verify advances GDN with the fused recurrent op; SpeculativeDecoder puts plain decode on
    # that same op because the composite decode kernel disagrees with it at ~1e-5, which flips greedy
    # near-ties. Put the reference on it too, so this test measures speculation rather than the
    # difference between two GDN kernels (test_fused_recurrent_gdn.py). B==max_batch_size==1
    # satisfies the fused path's full-batch assert.
    for layer in model.layers:
        if not layer.is_full_attention:
            layer.attention.use_fused_recurrent_decode = True

    # --- reference: plain greedy ------------------------------------------------------------- #
    ref_path = "decode"
    try:
        ref, gaps = _reference_greedy(model, prompt_ids, pt, kv_shape, MAX_NEW, use_decode_step=True)
    except Exception as e:  # decode_step_paged is a debug entry point with no other callers
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
    # top_k=1 makes the sampled target distribution a delta at the verify argmax, so this run
    # must land on the greedy trajectory too (see the docstring).
    sampling = None if sampling_mode == "greedy" else SpecSamplingParams(temperature=1.0, top_k=1, top_p=1.0, seed=0)
    dec = SpeculativeDecoder(model, pt, sampling=sampling)
    spec = dec.generate(prompt_ids, MAX_NEW)
    logger.info(f"[lossless] spec      {len(spec)} tokens: {spec}")
    logger.info(f"[lossless] spec text:      {tokenizer.decode(spec)!r}")
    # Logged, never asserted: acceptance is a speed property, and this test is about correctness.
    dec.log_stats(prefix="lossless")
    model.free_kv_caches()

    # --- compare ------------------------------------------------------------------------------ #
    n = min(len(ref), len(spec))
    assert n == MAX_NEW, f"expected {MAX_NEW} tokens from both runs, got ref={len(ref)} spec={len(spec)}"
    ties = [i for i in range(n) if gaps[i] < NEAR_TIE_GAP]
    logger.info(
        f"[lossless] near-tie positions (ref top-2 gap < {NEAR_TIE_GAP}): {len(ties)}/{n} {ties}, "
        f"min gap {min(gaps[:n]):.4f}"
    )
    div = next((i for i in range(n) if spec[i] != ref[i]), None)

    if div is None:
        logger.info(f"[lossless] PASSED: spec decode reproduced plain greedy for all {n} tokens, token for token")
        return

    gap = gaps[div]
    assert gap < NEAR_TIE_GAP, (
        f"spec decode diverged from plain greedy at token {div}, where the reference was CONFIDENT:\n"
        f"  expected {ref[div]} ({tokenizer.decode([ref[div]])!r}), "
        f"got {spec[div]} ({tokenizer.decode([spec[div]])!r})\n"
        f"  reference top-2 gap = {gap:.4f} >= NEAR_TIE_GAP {NEAR_TIE_GAP} — "
        "this is a real speculation bug, not bf16 noise\n"
        f"  ref [:{div + 1}] = {ref[: div + 1]}\n"
        f"  spec[:{div + 1}] = {spec[: div + 1]}"
    )
    # A near-tie flip: both tokens are faithful continuations, but from here the two runs are
    # decoding different strings, so there is nothing left to compare.
    logger.info(
        f"[lossless] PASSED: lossless up to {div}, near-tie flip (gap={gap:.4f} < {NEAR_TIE_GAP}): "
        f"ref {ref[div]} ({tokenizer.decode([ref[div]])!r}) vs spec {spec[div]} "
        f"({tokenizer.decode([spec[div]])!r}); trajectories legitimately diverge after this point"
    )
