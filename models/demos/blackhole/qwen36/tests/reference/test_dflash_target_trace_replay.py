# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does REPLAYING the captured verify forward reproduce the eager one?

``test_dflash_target_trace_probe.py`` established that the masked verify forward can be captured
once every per-step host write is staged. A capture proves only that the writes are gone. This file
asks the question that decides whether any of it is usable:

    replay(inputs) == eager(inputs)?

on the two things a speculative step depends on:

* **hidden** — the verify forward's output, which the LM head turns into the logits the accept test
  compares against. Wrong here and speculation accepts or rejects the wrong tokens.
* **GDN recurrent state** — advanced in place by every forward. This is the one the model's own
  docstrings warn about ("a short prompt would pad through the GDN recurrence and corrupt the decode
  state"), and the reason masked buckets were kept off the traced path in the first place.

TWO RISKS THIS IS BUILT TO CATCH, both currently only *argued*:

1. The fixed-width chunk page table writes padding K/V to positions past ``valid_len``. That was
   argued harmless because they are FUTURE positions which the next speculative step rewrites —
   an argument, not a measurement. If it is wrong, hidden still matches (those rows are masked out
   of this forward) and the damage shows up a step later, so a single-step check CANNOT see it.
   The two-step case below is what exercises it.
2. Whether each replay reads the staged buffers at the right point. A trace bakes addresses; if a
   buffer is read before the DMA lands, replay 2 silently reuses replay 1's inputs. Staging
   DIFFERENT inputs for the second replay is what catches that — identical inputs would pass
   whether or not the DMA is respected.

MEASURED (T3K, full 27B, 16-token block in a 128 bucket):

    hidden A: replay vs eager pcc=1.0     <- bit-identical, as it must be: same ops, replayed
    hidden B: replay vs eager pcc=1.0
    GDN state A: all 48 layers match
    GDN state B: all 48 layers match
    replay honours per-step staging (A and B differ)

So the traced verify forward is CORRECT, GDN recurrence included -- the specific thing the model's
docstrings said could not survive a trace.

The other two tests in this file close the gaps a single-step check cannot see:

* ``test_fixed_width_page_table_matches_narrow`` -- two steps, step 2 attending over rows step 1
  padded, fixed-width page table vs the production narrow one: **pcc=1.0**. The padding K/V written
  past ``valid_len`` is harmless, MEASURED rather than argued.
* ``test_one_trace_serves_any_valid_len`` -- capture at 16, replay at 8: **pcc=1.0**. One capture
  per bucket serves every block length, so ``valid_len`` is confined to the staged mask's contents
  and the traced target needs no per-length trace cache.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_target_trace_replay.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 200_000_000
BLOCK = 16
PCC = 0.999  # replay vs eager is the SAME arithmetic; anything below this is a real defect


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


def _to_host(t, mesh_device):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _gdn_states(model, mesh_device):
    """Per-GDN-layer recurrent state, on host, for comparison."""
    out = []
    for layer in model.layers:
        if not layer.is_full_attention:
            rs = getattr(layer.attention, "rec_state", None)
            out.append(None if rs is None else _to_host(rs, mesh_device))
    return out


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_verify_trace_replay_matches_eager(mesh_device, device_params, reset_seeds, ensure_gc):
    """Replay the captured verify forward and compare it to the eager forward, twice over."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    bucket = model._mask_bucket_for(BLOCK)

    def _buf(seed):
        g = torch.Generator().manual_seed(seed)
        t = torch.zeros(1, bucket, dtype=torch.int32)
        t[:, :BLOCK] = torch.randint(1000, 2000, (1, BLOCK), generator=g, dtype=torch.int32)
        return t

    buf_a, buf_b = _buf(0), _buf(1)
    staged = model.alloc_verify_buffers(bucket, page_table)

    def eager(token_buf, chunk_start):
        """The reference: reset, stage, run WITHOUT the trace."""
        model._reset_gdn_state_for_new_sequence()
        model.stage_verify_inputs(token_buf, BLOCK, chunk_start, page_table, bucket)
        out = model._forward_prefill_chunk_masked_tp(
            token_buf, BLOCK, chunk_start, page_table, bucket, flex_sdpa=True, staged=staged
        )
        h = _to_host(out, mesh_device)
        st = _gdn_states(model, mesh_device)
        ttnn.deallocate(out)
        return h, st

    # ---- references, before any capture exists -------------------------------------------
    eager_a, eager_a_gdn = eager(buf_a, 0)
    eager_b, eager_b_gdn = eager(buf_b, 0)
    logger.info(f"eager references captured: hidden {tuple(eager_a.shape)}, {len(eager_a_gdn)} GDN layers")

    # ---- capture (the traced output tensor's address is baked; keep the handle) -----------
    model._reset_gdn_state_for_new_sequence()
    model.stage_verify_inputs(buf_a, BLOCK, 0, page_table, bucket)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    open_capture = True
    try:
        traced_out = model._forward_prefill_chunk_masked_tp(
            buf_a, BLOCK, 0, page_table, bucket, flex_sdpa=True, staged=staged
        )
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        open_capture = False
    finally:
        # Never leave the command queue in capture mode: the process then hangs in device teardown
        # and the next run blocks on a lock held by the corpse.
        if open_capture:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    logger.info("captured; replaying")

    def replay(token_buf, chunk_start):
        model._reset_gdn_state_for_new_sequence()
        model.stage_verify_inputs(token_buf, BLOCK, chunk_start, page_table, bucket)
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        # traced_out is the tensor the capture wrote into; the replay overwrites it in place.
        return _to_host(traced_out, mesh_device), _gdn_states(model, mesh_device)

    rep_a, rep_a_gdn = replay(buf_a, 0)
    rep_b, rep_b_gdn = replay(buf_b, 0)  # DIFFERENT tokens -- see risk 2 in the module docstring

    # ---- 1. the replay must reproduce the eager forward ------------------------------------
    for name, ref, got in (("A", eager_a, rep_a), ("B", eager_b, rep_b)):
        passing, pcc = comp_pcc(ref, got, PCC)
        logger.info(f"hidden {name}: replay vs eager pcc={pcc}")
        assert passing, (
            f"traced replay does not reproduce the eager verify forward for inputs {name} "
            f"(pcc={pcc}). The capture succeeding said only that the host writes were gone; this "
            "is the check that says the result is right."
        )

    # ---- 2. the GDN recurrent state must advance identically --------------------------------
    # This is the one the model's docstrings warn about: pad through the recurrence and the decode
    # state is silently corrupted. hidden can match while this does not.
    for name, ref_st, got_st in (("A", eager_a_gdn, rep_a_gdn), ("B", eager_b_gdn, rep_b_gdn)):
        for i, (r, g) in enumerate(zip(ref_st, got_st)):
            if r is None or g is None:
                continue
            passing, pcc = comp_pcc(r, g, PCC)
            assert passing, f"GDN layer {i} recurrent state diverged on replay {name} (pcc={pcc})"
        logger.info(f"GDN state {name}: all {sum(x is not None for x in ref_st)} layers match")

    # ---- 3. the replay must actually READ the staged inputs ---------------------------------
    # If a replay ignored the per-step DMA it would return A's result for B's tokens, and checks 1
    # and 2 would still pass for A. Different tokens must give a different hidden.
    assert not torch.allclose(rep_a, rep_b, atol=1e-3), (
        "replay returned the same hidden for different token inputs -- the trace is not reading "
        "the staged buffers, so every step after the first would silently reuse the first's inputs"
    )
    logger.info("replay honours per-step staging (A and B differ)")
    print("\n>>> TRACED VERIFY REPLAY == EAGER (hidden + GDN state), and staging is honoured\n")


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_fixed_width_page_table_matches_narrow(mesh_device, device_params, reset_seeds, ensure_gc):
    """Does the FIXED-WIDTH chunk page table change what a later step reads?

    The traced path replaces a chunk page table whose width tracked ``valid_len`` with a fixed
    ``bucket // block_size``. That makes ``paged_fill_cache`` write the whole bucket, so positions
    past ``valid_len`` receive PADDING K/V. The argument for why that is safe -- they are future
    positions and the next speculative step rewrites them -- is an argument, and the single-step
    replay test cannot see it: those rows are masked out of the forward that wrote them.

    So run TWO steps and compare the fixed-width path against the production narrow one:

        step 1 at chunk_start=0   writes real K/V for [0,16) and padding for [16,128)
        step 2 at chunk_start=64  ATTENDS OVER [0, 64+16) -- i.e. over rows step 1 padded

    If padding K/V leaks into a later step's attention, step 2's hidden diverges. No trace is
    involved; this isolates the page-table width alone.

    The two sequences use DISJOINT physical pages (0..31 vs 32..63) so neither can inherit the
    other's residue -- otherwise whichever ran second would read the first's padding and the
    comparison would be meaningless.
    """
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    bucket = model._mask_bucket_for(BLOCK)
    half = NUM_BLOCKS // 2
    pt_narrow = torch.arange(0, half, dtype=torch.int32).unsqueeze(0)
    pt_fixed = torch.arange(half, NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    g = torch.Generator().manual_seed(7)
    steps = []
    for cs in (0, PAGED_BLOCK_SIZE):
        t = torch.zeros(1, bucket, dtype=torch.int32)
        t[:, :BLOCK] = torch.randint(1000, 2000, (1, BLOCK), generator=g, dtype=torch.int32)
        steps.append((t, cs))

    def run(page_table, use_staged):
        """Two verify steps in sequence; returns the SECOND step's hidden."""
        staged = model.alloc_verify_buffers(bucket, page_table) if use_staged else None
        model._reset_gdn_state_for_new_sequence()
        h = None
        for token_buf, cs in steps:
            if use_staged:
                model.stage_verify_inputs(token_buf, BLOCK, cs, page_table, bucket)
            out = model._forward_prefill_chunk_masked_tp(
                token_buf, BLOCK, cs, page_table, bucket, flex_sdpa=True, staged=staged
            )
            h = _to_host(out, mesh_device)
            ttnn.deallocate(out)
        return h

    narrow = run(pt_narrow, use_staged=False)  # production: width tracks valid_len
    fixed = run(pt_fixed, use_staged=True)  # traced path: fixed width, pads the bucket

    passing, pcc = comp_pcc(narrow, fixed, PCC)
    logger.info(f"step-2 hidden, narrow vs fixed-width page table: pcc={pcc}")
    print(f"\n>>> two-step padding check: pcc={pcc}\n")
    assert passing, (
        f"a later step reads different values under the fixed-width page table (pcc={pcc}). The "
        "padding K/V written past valid_len is NOT harmless, so the traced verify would silently "
        "diverge from eager after the first block."
    )


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_one_trace_serves_any_valid_len(mesh_device, device_params, reset_seeds, ensure_gc):
    """Does ONE capture serve every block length, or is a trace needed per ``valid_len``?

    This decides the API. A speculative block is not always full: it shortens at a sequence end or
    a bucket boundary, so ``valid_len`` varies 1..block_size. If each value needs its own capture,
    the traced target needs a trace cache and a lot more trace memory; if one capture serves all,
    ``valid_len`` lives entirely in the staged mask's CONTENTS.

    The reasoning says one trace suffices -- every op is fixed-shape and only the mask values
    change -- with one caveat worth pinning: ``gdn/tp.py::_normalize_valid_len`` turns
    ``valid_len >= T`` into ``None``, which SKIPS the masking path and would compile different
    programs. That is why this stays strictly below the bucket. Capture at 16, replay at 8.
    """
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    bucket = model._mask_bucket_for(BLOCK)
    staged = model.alloc_verify_buffers(bucket, page_table)

    SHORT = 8
    g = torch.Generator().manual_seed(11)
    tok = torch.zeros(1, bucket, dtype=torch.int32)
    tok[:, :BLOCK] = torch.randint(1000, 2000, (1, BLOCK), generator=g, dtype=torch.int32)

    def eager(vl):
        model._reset_gdn_state_for_new_sequence()
        model.stage_verify_inputs(tok, vl, 0, page_table, bucket)
        out = model._forward_prefill_chunk_masked_tp(tok, vl, 0, page_table, bucket, flex_sdpa=True, staged=staged)
        h = _to_host(out, mesh_device)
        ttnn.deallocate(out)
        return h

    eager_short = eager(SHORT)

    # Capture at the FULL block length, then replay at the SHORT one.
    model._reset_gdn_state_for_new_sequence()
    model.stage_verify_inputs(tok, BLOCK, 0, page_table, bucket)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    open_capture = True
    try:
        traced_out = model._forward_prefill_chunk_masked_tp(
            tok, BLOCK, 0, page_table, bucket, flex_sdpa=True, staged=staged
        )
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        open_capture = False
    finally:
        if open_capture:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    model._reset_gdn_state_for_new_sequence()
    model.stage_verify_inputs(tok, SHORT, 0, page_table, bucket)  # only the MASK CONTENTS differ
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    replay_short = _to_host(traced_out, mesh_device)

    passing, pcc = comp_pcc(eager_short, replay_short, PCC)
    logger.info(f"captured at valid_len={BLOCK}, replayed at valid_len={SHORT}: pcc={pcc}")
    print(f"\n>>> one trace serves valid_len {BLOCK} -> {SHORT}: pcc={pcc}\n")
    assert passing, (
        f"a trace captured at valid_len={BLOCK} does not reproduce valid_len={SHORT} (pcc={pcc}), so "
        "valid_len is NOT confined to the staged mask and the traced target needs one capture per "
        "block length."
    )


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_verify_traced_api_matches_prefill_block_all_logits(mesh_device, device_params, reset_seeds, ensure_gc):
    """The public API: ``verify_traced`` must equal ``prefill_block_all_logits``, logit for logit.

    The earlier tests compared the forward's hidden. This compares what the speculative loop
    actually consumes -- all-row logits -- through the real entry points, so it also covers the
    norm and the LM head that the trace now contains.
    """
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    bucket = model._mask_bucket_for(BLOCK)

    g = torch.Generator().manual_seed(23)
    ids = torch.randint(1000, 2000, (1, BLOCK), generator=g, dtype=torch.int32)
    token_buf = torch.zeros(1, bucket, dtype=torch.int32)
    token_buf[:, :BLOCK] = ids

    # Reference through the eager entry point the loop uses today.
    model._reset_gdn_state_for_new_sequence()
    ref = model.prefill_block_all_logits(ids, page_table, actual_len=BLOCK, chunk_start=0)

    model.capture_verify_trace(page_table, bucket, warm_tokens=token_buf, valid_len=BLOCK)
    model._reset_gdn_state_for_new_sequence()
    got = model.verify_traced(token_buf, BLOCK, 0, page_table, bucket)

    assert got.shape == ref.shape, f"shape {tuple(got.shape)} != reference {tuple(ref.shape)}"
    passing, pcc = comp_pcc(ref, got, PCC)
    logger.info(f"verify_traced vs prefill_block_all_logits: pcc={pcc}, shape={tuple(got.shape)}")
    print(f"\n>>> verify_traced == prefill_block_all_logits: pcc={pcc}\n")
    model.release_verify_trace()
    assert passing, f"traced verify logits differ from the eager entry point (pcc={pcc})"
