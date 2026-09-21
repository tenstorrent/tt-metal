# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-user (B > 1) MTP speculative decoding: every user must get its own plain greedy story.

test_spec_lossless.py proves the B=1 guarantee (every committed token is the plain decode path's
argmax given the same prefix). Batching adds exactly one new way to be wrong: B users share ONE
32-row verify tile, one GDN state ring, one conv window and one paged KV cache, so user u can be
fed user v's state, KV block, position, page-table row or candidate row and STILL look healthy --
acceptance stays high (the drafter and the verify agree with each other on the wrong story) and the
text stays fluent. Only a per-user comparison against an independent single-user reference catches
that, so these tests deliberately give every user DIFFERENT content and DIFFERENT prompt lengths:
with identical prompts a row-mixing bug is invisible, and with equal lengths every user sits at the
same position and the per-user position/page-table plumbing is never exercised.

Method (mirrors test_spec_lossless.py, one user at a time):
  1. run batched spec decode for MAX_NEW tokens at B users;
  2. TEACHER-FORCE the plain decode path down each user's trajectory on a SEPARATE max_batch_size=1
     model and record its argmax + top-2 logit gap at every position (test_spec_lossless's
     _reference_greedy, imported, not copied);
  3. compare per user with the same near-tie gate: a mismatch is a failure unless the plain argmax
     there was a bf16 coin flip (top-2 gap < NEAR_TIE_GAP).
Because the reference is teacher-forced, all MAX_NEW positions are checked for every user; the
comparison never has to stop at the first fork.

Why a second model instance: the reference is a B=1 plain decode (decode_step_paged), the fused GDN
decode op only supports the full batch width (B == max_batch_size), and spec decode requires that
op. So the reference cannot run on the B-user model. Teacher forcing needs the spec trajectory
first, so the order is: build the B-user model, run spec, free it (del + gc, as
test_model_tp.py::...batched... does), then build the max_batch_size=1 reference model. Only one
model is resident at a time.

K per batch is the demo's auto-K cap table (verify decodes B*(K+1) rows in one 32-row tile and the
fused verify SDPA only has L1 plans for T = K+1 in {4, 8, 12}): B=2 -> K=11, B=4 -> K=7, B=8 -> K=3.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_spec_batched.py -v -s
"""

import gc
import json

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import (
    _MESH_SHAPE,
    _MULTI,
    BLOCK_SIZE,
    DEVICE_PARAMS,
    SHARED_PROMPTS_DIR,
    _get_prompt,
)

# The B=1 losslessness contract lives in test_spec_lossless.py; import its reference builder, its
# gate and its constants instead of restating them, so the two tests can never drift apart.
from models.demos.blackhole.qwen36.tests.test_spec_lossless import (
    _N_LAYERS,
    MAX_NEW,
    NEAR_TIE_GAP,
    NUM_BLOCKS,
    PROMPT_LEN,
    _reference_greedy,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

# Auto-K by batch: the demo's _SPEC_BATCH_K_CAP for the batches this file covers. Duplicated (not
# imported) so a demo-policy change cannot silently retune these correctness runs.
SPEC_K = {2: 11, 4: 7, 8: 3}

RUNS = 3  # determinism: how many identical generate() calls must agree

# 32 prompts, 28 of them distinct, each ~130-160 tokens. Entry 0 is skipped everywhere below: it
# opens with the same "What is your favorite condiment?" text that _get_prompt returns for
# seqlen <= 256, and user 0 already uses that one.
_PROMPTS_FILE = f"{SHARED_PROMPTS_DIR}/input_data_questions_prefill_256.json"


def _exact_len(ids, target):
    """Exactly ``target`` token ids from ``ids`` [1, n]: repeat, then clip (as _get_prompt and
    test_spec_determinism._prompt_of_len do). Exactness is what makes the documented prompt
    lengths -- and their block/tile misalignment -- real."""
    while ids.shape[1] < target:
        ids = torch.cat([ids, ids], dim=1)
    return ids[:, :target]


def _batch_prompts(B, tokenizer):
    """B prompts with DISTINCT content AND distinct lengths. Returns a list of B token-id lists.

    Recipe:
      * user 0 is test_spec_lossless's prompt verbatim -- _get_prompt(130) (the shared 110-token
        question prompt repeated and clipped to 130), so the B>1 runs cover exactly the prompt the
        B=1 losslessness test covers;
      * user u >= 1 is entries u and u+8 of input_data_questions_prefill_256.json joined by a blank
        line (~260-290 tokens of two different questions), clipped to 129 + 18*u tokens.

    Lengths are therefore 130, 147, 165, 183, 201, 219, 237, 255 for u = 0..7: all in [130, 255],
    all ragged, and none a multiple of 64 (the KV block) or 32 (the decode tile), so every user
    starts at a different, unaligned frontier and the per-user position / page-table / KV-offset
    plumbing is exercised rather than coincidentally aligned.
    """
    with open(_PROMPTS_FILE) as f:
        texts = [e["prompt"] for e in json.load(f)]
    assert len(texts) >= B + 8, f"{_PROMPTS_FILE} has {len(texts)} prompts, need {B + 8} for B={B}"

    first = _exact_len(_get_prompt(PROMPT_LEN, tokenizer), PROMPT_LEN)
    prompts = [first[0].tolist()]
    for u in range(1, B):
        ids = tokenizer(texts[u] + "\n\n" + texts[u + 8], return_tensors="pt")["input_ids"]
        prompts.append(_exact_len(ids, 129 + 18 * u)[0].tolist())

    assert len({tuple(p) for p in prompts}) == B, "per-user prompts must be distinct (see the docstring)"
    logger.info(f"[spec-batched] B={B} prompt lengths {[len(p) for p in prompts]} (distinct content)")
    return prompts


def _blocks_per_user(max_prompt_len, K, max_new=MAX_NEW):
    """Per-user page-table width, exactly as _run_tp_spec_generation_batched computes it: prompt +
    generation + the K+1 candidate slots verify writes past the committed position, rounded up to a
    multiple of 32 so every user's page-table row stays aligned for chunked-SDPA stick reads."""
    bpu = max(8, -(-(max_prompt_len + max_new + K + 1) // BLOCK_SIZE))
    return ((bpu + 31) // 32) * 32


def _build_model(device, B):
    """The B-user model. Built before the prompts exist, because the tokenizer comes from it."""
    model = Qwen36Model.from_pretrained(
        device, max_batch_size=B, max_seq_len=NUM_BLOCKS * BLOCK_SIZE, n_layers=_N_LAYERS
    )
    assert model.mtp is not None, "MTP head not built"
    # Explicit and model-scoped, as in every other spec test: verify runs the fused GDN op, so plain
    # decode must use the same math or greedy near-ties flip between the two paths.
    model.set_gdn_fused_decode(True)
    return model


def _batched_kv(model, B, K, prompts):
    """Page tables + KV shape for B users, mirroring the demo's batched spec path: each user owns a
    disjoint contiguous block range of one shared cache (the MTP cache adds its own scratch block
    on top, inside allocate_kv_caches). Returns (page_tables [B, bpu], kv_shape)."""
    bpu = _blocks_per_user(max(len(p) for p in prompts), K)
    page_tables = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    kv_shape = [B * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    logger.info(f"[spec-batched] B={B} K={K} rows={B * (K + 1)}/32, paged KV {bpu} blocks/user x {B}")
    return page_tables, kv_shape


def _fresh_kv(model, kv_shape, B):
    """Free + reallocate the paged KV caches: the reset recipe every spec test uses between runs
    (the GDN recurrent state is re-zeroed inside the prefill)."""
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)


def _release(model):
    """Release the paged caches. The CALLER must then ``del`` its own model (and decoder) names and
    gc.collect(): dropping the reference is what frees the weights, and a ``del`` inside this
    helper would only unbind the local (test_model_tp.py uses the same del + gc.collect() recipe
    before building the next model)."""
    model.free_kv_caches()


def _assert_lossless(spec, ref, gaps, tokenizer, tag):
    """test_spec_lossless's near-tie gate, per user.

    A mismatch fails only where the plain argmax was CONFIDENT (top-2 gap >= NEAR_TIE_GAP); below
    that threshold the two kernels' ~1e-5 bf16 disagreement decides the coin flip and either token
    is a faithful greedy continuation.
    """
    n = len(spec)
    assert len(ref) == n, f"{tag}: {len(ref)} reference positions for {n} spec tokens"
    ties = [i for i in range(n) if gaps[i] < NEAR_TIE_GAP]
    mismatches = [i for i in range(n) if spec[i] != ref[i]]
    confident = [i for i in mismatches if gaps[i] >= NEAR_TIE_GAP]
    logger.info(
        f"[spec-batched] {tag}: near-tie positions (plain top-2 gap < {NEAR_TIE_GAP}): "
        f"{len(ties)}/{n} {ties}, min gap {min(gaps):.4f}"
    )
    assert not confident, (
        f"{tag}: spec decode committed a token plain greedy would NOT have chosen, where plain was CONFIDENT:\n"
        + "\n".join(
            f"  position {i}: spec {spec[i]} ({tokenizer.decode([spec[i]])!r}) vs plain argmax {ref[i]} "
            f"({tokenizer.decode([ref[i]])!r}), plain top-2 gap = {gaps[i]:.4f} >= NEAR_TIE_GAP {NEAR_TIE_GAP}"
            for i in confident
        )
        + "\n  this is a real speculation bug, not bf16 noise (a batched one: check this user's rows, "
        "KV blocks, GDN slot and position against its neighbours)"
    )
    if not mismatches:
        logger.info(f"[spec-batched] {tag}: PASSED, matched plain greedy at all {n} positions, token for token")
        return
    logger.info(
        f"[spec-batched] {tag}: PASSED, {n - len(mismatches)}/{n} positions identical; "
        f"{len(mismatches)} near-tie flip(s) at "
        + ", ".join(
            f"{i} (gap {gaps[i]:.4f}: spec {tokenizer.decode([spec[i]])!r} vs plain {tokenizer.decode([ref[i]])!r})"
            for i in mismatches
        )
    )


def _reference_model(device):
    """The max_batch_size=1 model the plain greedy reference runs on (see the module docstring for
    why it cannot be the B-user model). Build it only after the B-user model is gone."""
    model = Qwen36Model.from_pretrained(
        device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE, n_layers=_N_LAYERS
    )
    model.set_gdn_fused_decode(True)
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    return model, pt, kv_shape


@run_for_blackhole()
@pytest.mark.timeout(3600)  # two full model loads (B users, then the B=1 reference) + B references
@pytest.mark.parametrize("batch", [2, 4, 8])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_batched_lossless(mesh_device, batch):
    """Every user's batched spec output is that user's own plain greedy story, token for token."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    B, K = batch, SPEC_K[batch]
    assert B * (K + 1) <= 32, f"B={B} K={K} needs {B * (K + 1)} verify rows (one 32-row tile)"
    device = mesh_device
    device.enable_program_cache()

    # --- spec run: B users, distinct prompts, one shared paged KV ----------------------------- #
    model = _build_model(device, B)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(B, tokenizer)
    page_tables, kv_shape = _batched_kv(model, B, K, prompts)
    _fresh_kv(model, kv_shape, B)
    dec = SpeculativeDecoder(model, page_tables, draft_len=K)
    rows = dec.generate(prompts, MAX_NEW)
    dec.log_stats(prefix=f"batched-lossless B={B}")
    accept = dec.accept_rate()
    for u in range(B):
        logger.info(f"[spec-batched] B={B} user {u} ({len(prompts[u])} prompt tokens): {rows[u]}")
        logger.info(f"[spec-batched] B={B} user {u} text: {tokenizer.decode(rows[u])!r}")

    assert len(rows) == B, f"expected {B} output rows, got {len(rows)}"
    for u in range(B):
        assert len(rows[u]) == MAX_NEW, f"user {u} produced {len(rows[u])} tokens, wanted {MAX_NEW}"
    # Zero acceptance would still emit MAX_NEW correct tokens (one per iteration), so losslessness
    # alone cannot tell a working drafter from a dead one at B > 1.
    assert accept > 0, f"accept_rate() == {accept}: no draft was accepted for any user at B={B}"

    # The B=1 reference model cannot coexist with this one; free it first (see module docstring).
    _release(model)
    del dec, model
    gc.collect()

    # --- reference: plain B=1 decode, teacher-forced down each user's trajectory --------------- #
    ref_model, pt1, kv1 = _reference_model(device)
    try:
        for u in range(B):
            ref, gaps = _reference_greedy(ref_model, prompts[u], pt1, kv1, rows[u])
            logger.info(f"[spec-batched] B={B} user {u} plain-decode argmax at each spec position: {ref}")
            _assert_lossless(rows[u], ref, gaps, tokenizer, f"B={B} user {u}")
    finally:
        _release(ref_model)
    logger.info(f"[spec-batched] B={B}: all {B} users lossless, accept={accept:.2f}/{K}")


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("batch", [2, 8])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_batched_determinism(mesh_device, batch):
    """Same prompts, same model, same tokens for every user, every run.

    The B=1 version of this caught a KV read-modify-write race between candidate rows sharing a
    block (test_spec_determinism.py). At B > 1 the same tile is shared by DIFFERENT users, so the
    race has more ways to fire; a run-to-run difference in any row fails.
    """
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    B, K = batch, SPEC_K[batch]
    device = mesh_device
    device.enable_program_cache()
    model = _build_model(device, B)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(B, tokenizer)
    page_tables, kv_shape = _batched_kv(model, B, K, prompts)

    outs, accepts = [], []
    for r in range(RUNS):
        # Same reset recipe as test_spec_determinism: free + reallocate the paged KV caches and
        # build a fresh decoder; the GDN recurrent state is re-zeroed inside the prefill.
        _fresh_kv(model, kv_shape, B)
        dec = SpeculativeDecoder(model, page_tables, draft_len=K)
        outs.append(dec.generate(prompts, MAX_NEW))
        accepts.append(dec.accept_rate())
        logger.info(f"[spec-batched-det] B={B} run {r}: accept={accepts[-1]:.3f}/{K}")
    _release(model)

    for r in range(RUNS):
        assert len(outs[r]) == B, f"run {r} returned {len(outs[r])} rows, wanted {B}"
        for u in range(B):
            assert len(outs[r][u]) == MAX_NEW, f"run {r} user {u}: {len(outs[r][u])} tokens, wanted {MAX_NEW}"
    for r in range(1, RUNS):
        for u in range(B):
            div = next((i for i in range(MAX_NEW) if outs[r][u][i] != outs[0][u][i]), None)
            assert div is None, (
                f"B={B} user {u}: run {r} diverged from run 0 at token {div} — batched spec decode is "
                f"nondeterministic.\nrun0={outs[0][u]}\nrun{r}={outs[r][u]}"
            )
    # Identical tokens with a different acceptance would mean the rejection landed at a different
    # depth and recovery merely agreed — the same tell test_spec_determinism checks for.
    assert len(set(f"{a:.6f}" for a in accepts)) == 1, f"acceptance varied across runs (B={B}): {accepts}"
    logger.info(f"[spec-batched-det] B={B}: {RUNS} runs identical for all {B} users, accept={accepts[0]:.3f}")


@run_for_blackhole()
@pytest.mark.timeout(3600)  # two full model loads (B=4 users, then the B=1 reference)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_batched_replicated_matches_single(mesh_device):
    """Four users, ONE prompt: the rows must be identical to each other and lossless.

    Identical prompts are the sharpest test of row independence -- any asymmetry between the four
    users (a row picking up a neighbour's state, KV block or position) shows up as rows that differ
    even though their inputs do not.

    The rows are NOT required to equal a B=1 spec run token for token: verify splits its SDPA cores
    per user, so a B=4 row and a B=1 row are the same math on different core counts and disagree at
    ~1e-5, which flips greedy near ties. They are held to the same standard the B=1 run is held to
    instead -- the near-tie gate against plain greedy decode.
    """
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    B, K = 4, SPEC_K[4]
    device = mesh_device
    device.enable_program_cache()
    model = _build_model(device, B)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt = _exact_len(_get_prompt(PROMPT_LEN, tokenizer), PROMPT_LEN)[0].tolist()
    assert len(prompt) == PROMPT_LEN, f"wanted a {PROMPT_LEN}-token prompt, got {len(prompt)}"
    prompts = [list(prompt) for _ in range(B)]

    page_tables, kv_shape = _batched_kv(model, B, K, prompts)
    _fresh_kv(model, kv_shape, B)
    dec = SpeculativeDecoder(model, page_tables, draft_len=K)
    rows = dec.generate(prompts, MAX_NEW)
    dec.log_stats(prefix=f"batched-replicated B={B}")
    accept = dec.accept_rate()
    # Free before the reference model is built: only one model resident at a time.
    _release(model)
    del dec, model
    gc.collect()

    assert len(rows) == B, f"expected {B} output rows, got {len(rows)}"
    for u in range(B):
        assert len(rows[u]) == MAX_NEW, f"user {u} produced {len(rows[u])} tokens, wanted {MAX_NEW}"
    for u in range(1, B):
        div = next((i for i in range(MAX_NEW) if rows[u][i] != rows[0][i]), None)
        assert div is None, (
            f"users 0 and {u} were given the SAME prompt but diverged at token {div} — the batched "
            f"verify is not row-independent.\nuser0={rows[0]}\nuser{u}={rows[u]}"
        )
    assert accept > 0, f"accept_rate() == {accept}: no draft was accepted at B={B}"

    ref_model, pt1, kv1 = _reference_model(device)
    try:
        ref, gaps = _reference_greedy(ref_model, prompt, pt1, kv1, rows[0])
        _assert_lossless(rows[0], ref, gaps, tokenizer, f"B={B} replicated")
    finally:
        _release(ref_model)
    logger.info(f"[spec-batched] B={B} replicated: all rows identical and lossless, accept={accept:.2f}/{K}")


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_batched_sampling_topk1_equals_greedy(mesh_device):
    """Batched spec SAMPLING at top_k=1 must reproduce the batched greedy output, row for row.

    top_k 1 collapses the sampled target to a delta at the argmax of the SAME bf16 verify logits the
    greedy path argmaxes on device, so "accept iff u < p(draft)" becomes "accept iff the draft IS
    that argmax". This is a plumbing smoke test for the batched sampling path -- which [B*T, vocab]
    logits row belongs to which user's which draft, and which row the commit index reads -- not a
    test of the sampler's math (test_spec_sampling_math.py). Exact equality is the bar; the one
    legitimate way it can fail is an EXACT bf16 tie broken differently by host torch.topk and device
    ttnn.argmax, which would show up as a single-token difference at one position.
    """
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder
    from models.demos.blackhole.qwen36.tt.spec_sampling import SpecSamplingParams

    B, K = 4, SPEC_K[4]
    device = mesh_device
    device.enable_program_cache()
    model = _build_model(device, B)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(B, tokenizer)
    page_tables, kv_shape = _batched_kv(model, B, K, prompts)

    _fresh_kv(model, kv_shape, B)
    greedy = SpeculativeDecoder(model, page_tables, draft_len=K).generate(prompts, MAX_NEW)

    _fresh_kv(model, kv_shape, B)
    sampling = SpecSamplingParams(temperature=1.0, top_k=1, top_p=1.0, seed=0)
    dec = SpeculativeDecoder(model, page_tables, draft_len=K, sampling=sampling)
    sampled = dec.generate(prompts, MAX_NEW)
    dec.log_stats(prefix=f"batched-topk1 B={B}")
    _release(model)

    assert len(greedy) == len(sampled) == B, f"expected {B} output rows, got {len(greedy)} / {len(sampled)}"
    for u in range(B):
        assert len(greedy[u]) == MAX_NEW, f"greedy user {u} produced {len(greedy[u])} tokens, wanted {MAX_NEW}"
        assert len(sampled[u]) == MAX_NEW, f"user {u} produced {len(sampled[u])} tokens, wanted {MAX_NEW}"
        div = next((i for i in range(MAX_NEW) if sampled[u][i] != greedy[u][i]), None)
        assert div is None, (
            f"user {u}: top_k=1 spec sampling diverged from batched greedy at token {div} "
            f"({sampled[u][div]} {tokenizer.decode([sampled[u][div]])!r} vs "
            f"{greedy[u][div]} {tokenizer.decode([greedy[u][div]])!r}) — top_k=1 makes the sampled "
            f"target a delta at the verify argmax, so the two runs must agree.\n"
            f"greedy={greedy[u]}\ntopk1 ={sampled[u]}"
        )
    logger.info(f"[spec-batched] B={B} top_k=1 sampling reproduced batched greedy for all {B} users")


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_batched_freeze_stop_token(mesh_device):
    """A user that hits a stop token FREEZES, and freezing must be invisible to everyone else.

    A frozen user keeps replaying its T verify rows (the row count is baked into the trace) with
    its pre-commit tokens and positions, but its accept result is discarded and neither its
    tokens, its statistics, nor -- the part that can actually break -- its neighbours' tokens may
    move. This is the batched failure mode the other tests in this file cannot see: they pass no
    stop tokens, so every user finishes on the same iteration and nothing is ever frozen.

    Method: run B=2 to MAX_NEW with no stop tokens (run A), then re-run with user 0's THIRD token
    as the only stop token (run B). User 0 must stop exactly there, and user 1 -- which never sees
    the stop token -- must reproduce run A token for token even though its partner spends most of
    the run frozen.
    """
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    B, K = 2, SPEC_K[2]
    device = mesh_device
    device.enable_program_cache()
    model = _build_model(device, B)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(B, tokenizer)
    page_tables, kv_shape = _batched_kv(model, B, K, prompts)

    # --- run A: no stop tokens, the reference trajectory for both users --------------------- #
    _fresh_kv(model, kv_shape, B)
    outs_a = SpeculativeDecoder(model, page_tables, draft_len=K).generate(prompts, MAX_NEW)
    for u in range(B):
        assert len(outs_a[u]) == MAX_NEW, f"run A user {u}: {len(outs_a[u])} tokens, wanted {MAX_NEW}"
    logger.info(f"[spec-freeze] run A user 0: {outs_a[0]}")
    logger.info(f"[spec-freeze] run A user 1: {outs_a[1]}")

    # User 0's third token. It may legitimately occur in user 1's story too (' and' does, at
    # index 39 of these fixed prompts), so the expectation for EVERY user is its own run-A
    # trajectory cut after the first occurrence of the stop token: user 0 at index 2, user 1
    # wherever the token first shows up (or not at all). What the test is really asserting is that
    # nothing else moves -- user 1 reproduces run A token for token over its whole output even
    # though its partner is frozen from the first iteration onwards.
    stop = outs_a[0][2]
    assert stop not in outs_a[0][:2], (
        f"token {stop} ({tokenizer.decode([stop])!r}) also appears at index "
        f"{outs_a[0].index(stop)} of user 0's run-A output, so a stop on it would fire before "
        f"index 2 and the expected output would not be outs_a[0][:3]"
    )

    def _cut(ids):
        """run-A trajectory truncated after the first stop token, i.e. what generate() must emit."""
        i = next((j for j, t in enumerate(ids) if t == stop), None)
        return ids if i is None else ids[: i + 1]

    expect = [_cut(outs_a[u]) for u in range(B)]
    assert expect[0] == outs_a[0][:3], "user 0's expected output must end exactly at its third token"
    assert len(expect[1]) > 3, (
        f"user 1's stop-truncated reference is only {len(expect[1])} tokens: the two users would "
        f"freeze at nearly the same iteration and the test would not exercise a long freeze"
    )
    logger.info(
        f"[spec-freeze] stop token = {stop} ({tokenizer.decode([stop])!r}); expected lengths "
        f"{[len(e) for e in expect]} of {MAX_NEW} (user 0 ends at index 2)"
    )

    # --- run B: same prompts, same K, user 0 stops after three tokens ----------------------- #
    # Fresh KV + a fresh decoder, the reset recipe test_spec_determinism uses between runs.
    _fresh_kv(model, kv_shape, B)
    dec = SpeculativeDecoder(model, page_tables, draft_len=K, stop_tokens={stop})
    outs_b = dec.generate(prompts, MAX_NEW)
    dec.log_stats(prefix="batched-freeze B=2")
    stats = dec.stats()
    _release(model)

    assert len(outs_b) == B, f"run B returned {len(outs_b)} rows, wanted {B}"
    assert outs_b[0] == expect[0], (
        f"user 0 did not stop exactly at its stop token {stop} ({tokenizer.decode([stop])!r}):\n"
        f"  got      {outs_b[0]}\n  expected {expect[0]}"
    )
    div = next((i for i in range(min(len(outs_b[1]), len(expect[1]))) if outs_b[1][i] != expect[1][i]), None)
    assert outs_b[1] == expect[1], (
        "user 1 changed when its NEIGHBOUR was frozen"
        + (f", first at token {div}" if div is not None else f" (length {len(outs_b[1])} vs {len(expect[1])})")
        + f":\n  frozen-neighbour run {outs_b[1]}\n  reference run        {expect[1]}\n"
        "  a frozen user's rows must not perturb a live user (check its KV blocks, GDN ring slot, "
        "conv window and reseed padding)"
    )

    mlu = stats.get("mean_live_users")
    if mlu is None:
        logger.info("[spec-freeze] stats() has no 'mean_live_users' key; skipping the live-user check")
    else:
        logger.info(f"[spec-freeze] mean_live_users={mlu:.3f} over {stats['iters']} iterations (B={B})")
        assert 1.0 <= mlu <= 2.0, (
            f"mean_live_users={mlu} outside [1, {B}]: user 0 freezes after three tokens while user 1 "
            f"runs to {MAX_NEW}, so the mean live-user count must sit between one and the full batch"
        )
    logger.info("[spec-freeze] user 0 stopped at its stop token; user 1 is unchanged by the freeze")
