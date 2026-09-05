# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Spec decode must match plain greedy decode token for token, up to bf16 near-ties.

The guarantee being tested: EVERY token spec decode commits is the plain decode path's argmax given
the same prefix. Acceptance rate does not prove that — a drafter verified against a subtly different
base forward (different GDN kernel, stale KV, incomplete rollback) still shows high acceptance while
writing a different story.

Method: run spec decode for N tokens, then TEACHER-FORCE the plain decode path down that exact
trajectory (prefill + N single-token decodes, each fed spec's token) and record the plain argmax and
its top-2 logit gap at every position. Because both paths see the identical prefix at every step,
the comparison never has to stop at a divergence: all N positions are checked.

Why not exact equality: the plain decode step and the batched verify bucket are different kernels
on different shapes and disagree at ~1e-5 in bf16. Where the plain argmax is CONFIDENT (top-2 gap
>= NEAR_TIE_GAP = 2.0) that noise cannot flip it, so any mismatch there is a real speculation bug
and fails the test. Where the top-2 gap is below the threshold the argmax is decided by numerical
noise and either token is a faithful greedy continuation; those mismatches are counted and logged,
not failed. The threshold follows gemma4
(models/demos/gemma4/tests/unit/test_spec_decode.py::_assert_argmaxes_match_except_near_ties).

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


def _reference_greedy(model, prompt_ids, page_table, kv_shape, force_tokens):
    """Plain decode, teacher-forced down ``force_tokens`` (the spec trajectory).

    Fresh KV caches + the SAME prefill entry point the spec loop uses (prefill_for_spec, minus the
    MTP warming callback), so the two runs start from an identical prompt state and any later
    difference is the speculation itself, not the prefill. prefill_masked_bucket re-zeroes the GDN
    recurrent + conv state at chunk_start==0, which is what makes this callable back-to-back with a
    spec run on the same model object.

    At each position i the plain path is fed force_tokens[:i] as the generated prefix, so its argmax
    at i is "what plain greedy would have emitted given exactly what spec emitted so far".

    Returns (argmaxes, gaps): argmaxes[i] is the plain argmax at generated position i, gaps[i] the
    top-2 logit gap of that distribution (position 0 comes from the prefill logits).
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

    argmaxes, gaps = [int(row.argmax())], [_top2_gap(row)]  # position 0: the token after the prompt
    pos = T
    for i in range(1, len(force_tokens)):
        # The paged single-token decode: the production decode kernels, one token at a time, fed the
        # token SPEC emitted at the previous position (teacher forcing). This is the reference the
        # test exists to compare against, so a failure here fails the test — it is deliberately NOT
        # caught and replaced by a verify-based stand-in (which would share the machinery under test).
        row, hidden = model.decode_step_paged(force_tokens[i - 1], pos, page_table)
        ttnn.deallocate(hidden)  # the MTP seed hidden; the reference has no drafter to feed
        argmaxes.append(int(row.argmax()))
        gaps.append(_top2_gap(row))
        pos += 1

    model.free_kv_caches()
    return argmaxes, gaps


@run_for_blackhole()
@pytest.mark.parametrize("sampling_mode", ["greedy", "topk1"])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_decode_matches_plain_greedy_up_to_near_ties(mesh_device, sampling_mode):
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

    # --- spec run: fresh state ---------------------------------------------------------------- #
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
    assert len(spec) == MAX_NEW, f"expected {MAX_NEW} tokens from spec decode, got {len(spec)}"

    # --- reference: plain decode, teacher-forced down the spec trajectory --------------------- #
    ref, gaps = _reference_greedy(model, prompt_ids, pt, kv_shape, spec)
    logger.info(f"[lossless] plain-decode argmax at each spec position: {ref}")
    assert len(ref) == MAX_NEW, f"expected {MAX_NEW} reference positions, got {len(ref)}"

    # --- compare: every position -------------------------------------------------------------- #
    n = MAX_NEW
    ties = [i for i in range(n) if gaps[i] < NEAR_TIE_GAP]
    logger.info(
        f"[lossless] near-tie positions (plain top-2 gap < {NEAR_TIE_GAP}): {len(ties)}/{n} {ties}, "
        f"min gap {min(gaps):.4f}"
    )
    mismatches = [i for i in range(n) if spec[i] != ref[i]]
    confident_mismatches = [i for i in mismatches if gaps[i] >= NEAR_TIE_GAP]

    assert not confident_mismatches, (
        f"spec decode committed a token plain greedy would NOT have chosen, where plain was CONFIDENT:\n"
        + "\n".join(
            f"  position {i}: spec {spec[i]} ({tokenizer.decode([spec[i]])!r}) vs plain argmax {ref[i]} "
            f"({tokenizer.decode([ref[i]])!r}), plain top-2 gap = {gaps[i]:.4f} >= NEAR_TIE_GAP {NEAR_TIE_GAP}"
            for i in confident_mismatches
        )
        + "\n  this is a real speculation bug, not bf16 noise"
    )

    if not mismatches:
        logger.info(f"[lossless] PASSED: spec decode matched plain greedy at all {n} positions, token for token")
        return
    # Every mismatch sits at a near-tie: both tokens are faithful greedy continuations and the two
    # kernels' ~1e-5 bf16 disagreement decided the coin flip.
    logger.info(
        f"[lossless] PASSED: {n - len(mismatches)}/{n} positions identical; {len(mismatches)} near-tie flip(s) at "
        + ", ".join(
            f"{i} (gap {gaps[i]:.4f}: spec {tokenizer.decode([spec[i]])!r} vs plain {tokenizer.decode([ref[i]])!r})"
            for i in mismatches
        )
    )
