# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-user (B > 1) MTP speculative decoding: every user must get its own plain greedy story.

test_spec_lossless.py proves the B=1 guarantee (every committed token is the plain decode path's
argmax given the same prefix). Batching adds exactly one new way to be wrong: B users share ONE
32-row verify tile, one GDN state ring, one conv window and one paged KV cache, so user u can be
fed user v's state, KV block, position, page-table row or candidate row and STILL look healthy --
acceptance stays high (the drafter and the verify agree with each other on the wrong story) and the
text stays fluent. Only a per-user comparison against an independent single-user reference catches
that, so the test deliberately gives every user DIFFERENT content and DIFFERENT prompt lengths:
with identical prompts a row-mixing bug is invisible, and with equal lengths every user sits at the
same position and the per-user position/page-table plumbing is never exercised.

Method (mirrors test_spec_lossless.py, one user at a time):
  1. run batched spec decode for MAX_NEW tokens at B users, plus the per-B variants that must
     reproduce that run (see test_spec_batched_lossless);
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
model is resident at a time. mesh_device is function-scoped, so all of it is one test body: the
B-user model is loaded once and every variant reuses it.

K per batch is the demo's auto-K policy (verify decodes B*(K+1) rows in one 32-row tile and the
fused verify SDPA only has L1 plans for T = K+1 in {4, 8, 12}): B=2 -> K=11, B=4 -> K=7, B=8 -> K=3.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_spec_batched.py -v -s
"""

import contextlib
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
    _spec_batch_draft_len,
)

# The B=1 losslessness contract lives in test_spec_lossless.py; import its reference builder, its
# gate and its constants instead of restating them, so the two tests can never drift apart.
from models.demos.blackhole.qwen36.tests.test_spec_lossless import (
    _N_LAYERS,
    MAX_NEW,
    NUM_BLOCKS,
    PROMPT_LEN,
    _assert_lossless_up_to_near_ties,
    _reference_greedy,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

# 32 prompts, 28 of them distinct, each ~130-160 tokens. Entry 0 is skipped everywhere below: it
# opens with the same "What is your favorite condiment?" text that _get_prompt returns for
# seqlen <= 256, and user 0 already uses that one.
_PROMPTS_FILE = f"{SHARED_PROMPTS_DIR}/input_data_questions_prefill_256.json"


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

    prompts = [_get_prompt(PROMPT_LEN, tokenizer)[0].tolist()]
    assert len(prompts[0]) == PROMPT_LEN, f"wanted a {PROMPT_LEN}-token prompt, got {len(prompts[0])}"
    for u in range(1, B):
        n = 129 + 18 * u
        ids = tokenizer(texts[u] + "\n\n" + texts[u + 8], return_tensors="pt")["input_ids"][0, :n].tolist()
        assert len(ids) == n, f"user {u}: the two questions tokenize to {len(ids)} < {n} tokens"
        prompts.append(ids)

    assert len({tuple(p) for p in prompts}) == B, "per-user prompts must be distinct (see the docstring)"
    logger.info(f"[spec-batched] B={B} prompt lengths {[len(p) for p in prompts]} (distinct content)")
    return prompts


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
    on top, inside allocate_kv_caches). Returns (page_tables [B, bpu], kv_shape).

    The per-user width is computed as _run_tp_spec_generation_batched does: prompt + generation + the
    K+1 candidate slots verify writes past the committed position, rounded up to a multiple of 32 so
    every user's page-table row stays aligned for chunked-SDPA stick reads."""
    bpu = max(8, -(-(max(len(p) for p in prompts) + MAX_NEW + K + 1) // BLOCK_SIZE))
    bpu = ((bpu + 31) // 32) * 32
    page_tables = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    kv_shape = [B * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    logger.info(f"[spec-batched] B={B} K={K} rows={B * (K + 1)}/32, paged KV {bpu} blocks/user x {B}")
    return page_tables, kv_shape


def _fresh_kv(model, kv_shape, B):
    """Free + reallocate the paged KV caches: the reset recipe every spec test uses between runs
    (the GDN recurrent state is re-zeroed inside the prefill)."""
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)


def _setup(mesh_device, B):
    """Common prelude: skip off the TP mesh, build the B-user model and its tokenizer, the distinct
    per-user prompts and the shared-KV page tables. Returns (model, tokenizer, prompts, K, page_tables, kv_shape)."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    K = _spec_batch_draft_len(B, PROMPT_LEN, None)[0]
    mesh_device.enable_program_cache()
    model = _build_model(mesh_device, B)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(B, tokenizer)
    page_tables, kv_shape = _batched_kv(model, B, K, prompts)
    return model, tokenizer, prompts, K, page_tables, kv_shape


def _spec_run(model, kv_shape, B, K, prompts, page_tables, tag, **dec_kwargs):
    """One fresh-KV batched spec generation: (per-user outputs, (accepted, iters, drafted), stats).

    The traced-draft/reseed env flags are read in SpeculativeDecoder.__init__, so every run builds a
    fresh decoder."""
    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    _fresh_kv(model, kv_shape, B)
    dec = SpeculativeDecoder(model, page_tables, draft_len=K, **dec_kwargs)
    outs = dec.generate(prompts, MAX_NEW)
    return outs, (dec.total_accepted, dec.iters, dec.total_drafted), dec.log_stats(prefix=f"batched {tag}")


def _assert_same_runs(a, b, B, tag, counters=True):
    """Per-user outputs token-identical and, unless ``counters`` is off, the acceptance counters equal.

    Equal counters on top of equal tokens catch a rejection landing at a different depth that
    recovery merely agreed with."""
    (a_outs, a_cnt, _), (b_outs, b_cnt, _) = a, b
    for outs in (a_outs, b_outs):
        assert len(outs) == B, f"{tag}: expected {B} output rows, got {len(outs)}"
        for u in range(B):
            assert len(outs[u]) == MAX_NEW, f"{tag}: user {u} produced {len(outs[u])} tokens, wanted {MAX_NEW}"
    for u in range(B):
        div = next((i for i in range(MAX_NEW) if a_outs[u][i] != b_outs[u][i]), None)
        assert div is None, f"{tag}: user {u} diverged at token {div}.\nrun A={a_outs[u]}\nrun B={b_outs[u]}"
    assert not counters or a_cnt == b_cnt, f"{tag}: (total_accepted, iters, total_drafted) {a_cnt} != {b_cnt}"


@contextlib.contextmanager
def _variant(failures, name):
    """Record a failed variant instead of aborting, so the other variants and the lossless leg
    still run and one run reports every divergence."""
    try:
        yield
    except AssertionError as e:
        logger.error(f"[spec-batched] variant {name!r} diverged from the base run: {e}")
        failures.append(f"{name}: {e}")


def _null_padded_tables(page_tables, prompts, K):
    """vLLM-style block tables: every column past the blocks a user can reach is the null block 0,
    so all rows share block 0 in their tails. The grouped verify KV write only needs distinct
    (block, 32-row tile) per user within each candidate call, which still holds (nothing writes the
    tails)."""
    padded = page_tables.clone()
    nb = padded.shape[1]
    needed = [-(-(len(p) + MAX_NEW + 2 * (K + 1) + 1) // BLOCK_SIZE) + 1 for p in prompts]
    for u, n in enumerate(needed):
        padded[u, min(n, nb) :] = 0
    assert nb > max(needed), "no tail to pad"
    assert (padded[:, -1] == 0).all(), "both rows must share the null block in their tails"
    return padded


def _check_freeze(run, tokenizer, outs_a):
    """A user that hits a stop token FREEZES, and freezing must be invisible to everyone else.

    A frozen user keeps replaying its T verify rows (the row count is baked into the trace) with
    its pre-commit tokens and positions, but its accept result is discarded and neither its
    tokens, its statistics, nor -- the part that can actually break -- its neighbours' tokens may
    move. Runs without stop tokens never freeze anyone, so only this variant sees it.

    ``outs_a`` is the no-stop-token run. Re-run with user 0's THIRD token as the only stop token:
    user 0 must stop exactly there, and user 1 -- which never sees the stop token -- must
    reproduce its run-A trajectory (cut after the first occurrence of the stop token, which may
    legitimately occur in its story too) even though its partner spends most of the run frozen.
    """
    B = len(outs_a)
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

    outs_b, _, stats = run("freeze", stop_tokens={stop})
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

    mlu = stats["mean_live_users"]
    logger.info(f"[spec-freeze] mean_live_users={mlu:.3f} over {stats['iters']} iterations (B={B})")
    assert 1.0 <= mlu <= B, (
        f"mean_live_users={mlu} outside [1, {B}]: user 0 freezes after three tokens while user 1 "
        f"runs to {MAX_NEW}, so the mean live-user count must sit between one and the full batch"
    )


@run_for_blackhole()
@pytest.mark.timeout(3600)  # B-user load + up to 7 spec runs, then the B=1 reference load + B references
@pytest.mark.parametrize("batch", [2, 4, 8])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_batched_lossless(mesh_device, batch, monkeypatch):
    """Every user's batched spec output is that user's own plain greedy story, token for token.

    Phase 1, on the one B-user model: a base greedy run (distinct prompts, one shared paged KV),
    then the variants that must reproduce it exactly (outputs and, where noted, the acceptance
    counters), each from fresh KV:
      B=2: two repeat runs (a run-to-run difference is a KV read-modify-write race between users
           sharing the tile); an eager run with the traced draft chain / reseed off; null-block-padded
           page tables; verify_traced re-staging the page tables on every replay (continuous
           batching); a stop-token run, where the frozen user must not move its neighbour.
      B=4: top_k=1 sampling (plumbing of which [B*T, vocab] row belongs to which user's draft;
           tokens only) and one prompt replicated across all rows, which must give identical rows
           (the sharpest row-independence test; not required to equal a B=1 spec run, whose SDPA
           core split differs at ~1e-5).
      B=8: two repeat runs and an eager run.
    Phase 2: free the B-user model, build the B=1 reference and run the near-tie gate per user.
    """
    B = batch
    model, tokenizer, prompts, K, page_tables, kv_shape = _setup(mesh_device, B)
    monkeypatch.setenv("QWEN36_TRACED_DRAFT", "1")
    monkeypatch.setenv("QWEN36_TRACED_RESEED", "1")

    def run(tag, prompts=prompts, tables=page_tables, **dec_kwargs):
        return _spec_run(model, kv_shape, B, K, prompts, tables, f"B={B} {tag}", **dec_kwargs)

    # --- phase 1: base run + variants --------------------------------------------------------- #
    base = run("base")
    rows, _, stats = base
    for u in range(B):
        logger.info(f"[spec-batched] B={B} user {u} ({len(prompts[u])} prompt tokens): {rows[u]}")
        logger.info(f"[spec-batched] B={B} user {u} text: {tokenizer.decode(rows[u])!r}")
    assert len(rows) == B, f"expected {B} output rows, got {len(rows)}"
    for u in range(B):
        assert len(rows[u]) == MAX_NEW, f"user {u} produced {len(rows[u])} tokens, wanted {MAX_NEW}"
    # Zero acceptance would still emit MAX_NEW correct tokens (one per iteration), so losslessness
    # alone cannot tell a working drafter from a dead one at B > 1.
    accept = stats["accept_rate"]
    assert accept > 0, f"accept_rate() == {accept}: no draft was accepted for any user at B={B}"

    failures = []
    if B != 4:
        with _variant(failures, "repeat runs"):
            for r in (1, 2):
                _assert_same_runs(
                    base, run(f"repeat {r}"), B, f"B={B} repeat {r} vs base (batched spec decode is nondeterministic)"
                )
        with _variant(failures, "traced vs eager"):
            with monkeypatch.context() as m:
                m.setenv("QWEN36_TRACED_DRAFT", "0")
                m.setenv("QWEN36_TRACED_RESEED", "0")
                eager = run("eager")
            _assert_same_runs(base, eager, B, f"B={B} eager draft/reseed vs traced")
    if B == 2:
        with _variant(failures, "null-block padding"):
            padded = _null_padded_tables(page_tables, prompts, K)
            _assert_same_runs(base, run("null-block", tables=padded), B, f"B={B} null-block padding")
        with _variant(failures, "restaged page tables"):
            calls = []
            orig = Qwen36Model.verify_traced

            def restaging(self, tokens, positions, mi_prev, read_logits=False, **_):
                calls.append(1)
                return orig(self, tokens, positions, mi_prev, read_logits=read_logits, page_tables=page_tables.clone())

            with monkeypatch.context() as m:
                m.setattr(Qwen36Model, "verify_traced", restaging)
                restaged = run("restage")
            assert calls, "the restaging verify_traced wrapper was never called"
            _assert_same_runs(base, restaged, B, f"B={B} restaged page tables")
        with _variant(failures, "stop-token freeze"):
            _check_freeze(run, tokenizer, rows)
    if B == 4:
        from models.demos.blackhole.qwen36.tt.spec_sampling import SpecSamplingParams

        with _variant(failures, "top_k=1 sampling"):
            sampling = SpecSamplingParams(temperature=1.0, top_k=1, top_p=1.0, seed=0)
            # Exact equality is the bar; the one legitimate failure is an EXACT bf16 tie broken
            # differently by host torch.topk and device ttnn.argmax.
            _assert_same_runs(
                base, run("topk1", sampling=sampling), B, f"B={B} top_k=1 sampling vs greedy", counters=False
            )
        with _variant(failures, "replicated prompt"):
            same, _, same_stats = run("replicated", prompts=[list(prompts[0]) for _ in range(B)])
            for u in range(B):
                assert len(same[u]) == MAX_NEW, f"replicated user {u} produced {len(same[u])} tokens, wanted {MAX_NEW}"
            for u in range(1, B):
                div = next((i for i in range(MAX_NEW) if same[u][i] != same[0][i]), None)
                assert div is None, (
                    f"users 0 and {u} were given the SAME prompt but diverged at token {div} -- the batched "
                    f"verify is not row-independent.\nuser0={same[0]}\nuser{u}={same[u]}"
                )
            assert same_stats["accept_rate"] > 0, f"replicated accept_rate == 0 at B={B}"

    # The B=1 reference model cannot coexist with this one; free it first (see module docstring).
    model.free_kv_caches()
    run = model = None  # the closure holds the model; the B=1 model must be the only one resident
    gc.collect()

    # --- phase 2: plain B=1 decode, teacher-forced down each user's trajectory ---------------- #
    ref_model = _build_model(mesh_device, 1)
    pt1 = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)
    kv1 = [NUM_BLOCKS, ref_model.args.n_local_kv_heads, BLOCK_SIZE, ref_model.args.head_dim]
    try:
        for u in range(B):
            ref, gaps = _reference_greedy(ref_model, prompts[u], pt1, kv1, rows[u])
            logger.info(f"[spec-batched] B={B} user {u} plain-decode argmax at each spec position: {ref}")
            _assert_lossless_up_to_near_ties(rows[u], ref, gaps, tokenizer, f"B={B} user {u}")
    finally:
        ref_model.free_kv_caches()
    assert not failures, "variants diverged from the base run:\n" + "\n".join(failures)
    logger.info(f"[spec-batched] B={B}: all {B} users lossless, accept={accept:.2f}/{K}")
