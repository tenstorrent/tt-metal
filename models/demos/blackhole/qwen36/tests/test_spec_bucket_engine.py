# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Engine-level tests of the spec-step row buckets (QWEN36_SPEC_BUCKETS, work/notes/77 tasks T17, T18): greedy token streams of a
join / leave script that crosses every bucket size, with the buckets "1,2,4,8" against the same script with "8" (today's engine).

Script (SCRIPT_JOINS / SCRIPT_LEAVES, step -> request ids): 1 user, then 2, 3, 5, 8, then the newest users leave down to 4, 3, 2, 1,
then 2, 3, 4 users join again and leave again down to 1. Leavers are the newest requests, so the live engine rows stay 0..n-1 when the
engine places a new request on the lowest free row. The "8" run puts every new request on the lowest free runner slot; the bucket run
puts it on the HIGHEST free runner slot, so the engine must swap the slot map (row placement, ``row_swaps``) to reach the same engine
row: the same request then runs on the same engine row in both runs.

Pinned (QWEN36_VERIFY_SDPA_MAX_CORES=13, QWEN36_MTP_SDPA_MAX_CORES=13): the SDPA core split no longer depends on the row count, so the
streams must be bit-identical. Unpinned: the number of mismatching tokens is logged and every first mismatch must be a near-tie of
plain greedy (top-2 gap < NEAR_TIE_GAP, teacher-forced B=1 reference, as in test_spec_lossless).

One node id per device run:
  pytest models/demos/blackhole/qwen36/tests/test_spec_bucket_engine.py::test_greedy_streams_match -sv
"""

import gc
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.spec_helpers import _build
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import (
    BIG,
    NB_BIG,
    ChainDriver,
    _env,
    _marks,
    _setup,
    _streams,
)
from models.demos.blackhole.qwen36.tests.test_spec_lossless import NEAR_TIE_GAP, _reference_greedy

STEPS = 80
SCRIPT_JOINS = {0: [0], 6: [1], 12: [2], 18: [3, 4], 24: [5, 6, 7], 56: [8, 9], 62: [10]}
SCRIPT_LEAVES = {30: [7], 32: [6], 34: [5], 36: [4], 42: [3], 44: [2], 50: [1], 68: [10, 9], 74: [8]}
PINS = {"QWEN36_VERIFY_SDPA_MAX_CORES": "13", "QWEN36_MTP_SDPA_MAX_CORES": "13"}


def _prompt_of(rid):
    return rid % 8


def _flags(monkeypatch, flag, pinned):
    _env(monkeypatch)
    for k, v in {
        "QWEN36_SPEC_CHAIN_SAMPLED": "1",
        "QWEN36_SPEC_FAST_PREFILL": os.environ.get("QWEN36_BUCKET_TEST_FAST_PREFILL", "1"),
        "QWEN36_SPEC_K": "3",
        "QWEN36_SPEC_JOIN_TRACE": "1",
        "QWEN36_SPEC_BUCKETS": flag,
    }.items():
        monkeypatch.setenv(k, v)
    for k, v in PINS.items():
        if pinned:
            monkeypatch.setenv(k, v)
        else:
            monkeypatch.delenv(k, raising=False)


def _join(drv, slots, rid, prompts, high):
    free = [r for r in range(tc.B) if drv.reqs[r] is None]
    s = max(free) if high else min(free)
    drv.admit(s, rid, prompts[_prompt_of(rid)], max_new=BIG)
    slots[rid] = s


def _run_sync(eng, prompts, high):
    drv = ChainDriver(eng, seed=17, sampled=lambda s: False)
    slots = {}
    for i in range(STEPS):
        for rid in SCRIPT_LEAVES.get(i, []):
            drv._finish(slots[rid])
        for rid in SCRIPT_JOINS.get(i, []):
            _join(drv, slots, rid, prompts, high)
        drv.step()
    streams = _streams(drv)
    for r in drv.live():
        drv._finish(r)
    return streams


def _run_async(eng, prompts, high):
    """Depth-1 deferred pipeline; events run with one step in flight (the drained join order of
    test_spec_async_engine._prefill_over_pending_run): a join reads the pending step first, then prefills."""
    drv = AsyncDriver(eng, seed=17, sampled=lambda s: False)
    slots = {}
    for rid in SCRIPT_JOINS[0]:
        _join(drv, slots, rid, prompts, high)
    for i in range(STEPS):
        drv.enq()
        if len(drv.pend) == 2:
            drv.resolve()
        if i == 0:
            continue
        for rid in SCRIPT_LEAVES.get(i, []):
            drv._finish(slots[rid])
        for rid in SCRIPT_JOINS.get(i, []):
            if drv.pend:
                drv.resolve()
            _join(drv, slots, rid, prompts, high)
            drv.host_rows.add(slots[rid])
            drv.chain_on_tip = True
    drv.drain()
    streams = _streams(drv)
    for r in drv.live():
        drv._finish(r)
    return streams


def _stats(eng):
    return {
        "bucket_steps": dict(eng.__dict__.get("bucket_steps", {})),
        "bucket_fallbacks": eng.bucket_fallbacks,
        "row_swaps": eng.row_swaps,
        "downs_hold": getattr(eng, "bucket_downs_hold", None),
        "downs_free": getattr(eng, "bucket_downs_free", None),
        "free_switch": getattr(eng, "bucket_free_switch", None),
        "chain_steps": eng.chain_steps,
        "chain_audits": eng.chain_audits,
        "chain_breaks": dict(eng.chain_breaks),
        "parks": dict(getattr(eng, "park_stats", {}) or {}),
    }


def _one_run(mesh_device, monkeypatch, flag, pinned, runner):
    """Build the model + engine under ``flag`` (engine rebuilt per call, one device), run the script, tear down."""
    _flags(monkeypatch, flag, pinned)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    try:
        streams = runner(eng, prompts, high=(flag != "8"))
        st = _stats(eng)
    finally:
        eng.shutdown()
        model.free_kv_caches()
    del eng, model
    gc.collect()
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"[bucket_engine] flag={flag} pinned={pinned} stats={st} lens={ {k: len(v) for k, v in sorted(streams.items())} }"
    )
    return streams, st, tokenizer, prompts


def _mismatches(a, b):
    """({rid: (n_mismatching_tokens, first_index or None)}, total) over the common prefix of the two streams per request."""
    out, total = {}, 0
    for rid in sorted(a):
        n = min(len(a[rid]), len(b[rid]))
        bad = [i for i in range(n) if a[rid][i] != b[rid][i]]
        out[rid] = (len(bad), bad[0] if bad else None)
        total += len(bad)
    return out, total


def _assert_buckets_used(st):
    bs = st["bucket_steps"]
    assert all(bs.get(b, 0) > 0 for b in (1, 2, 4, 8)), f"bucket steps {bs}"
    assert st["row_swaps"] > 0, "no row swap"
    assert st["bucket_fallbacks"] == 0, st


@_marks
def test_greedy_streams_match(mesh_device, monkeypatch):
    base, st0, *_ = _one_run(mesh_device, monkeypatch, "8", True, _run_sync)
    got, st1, *_ = _one_run(mesh_device, monkeypatch, "1,2,4,8", True, _run_sync)
    logger.info(
        f"[bucket_engine] sync pinned: bucket steps {st1['bucket_steps']} row swaps {st1['row_swaps']} parks {st1['parks']}"
    )
    assert not st1["parks"], st1["parks"]
    _assert_buckets_used(st1)
    assert sorted(base) == sorted(got) == list(range(11))
    mm, total = _mismatches(base, got)
    assert total == 0 and all(len(base[r]) == len(got[r]) for r in base), f"pinned streams differ: {mm}"
    assert st0["bucket_steps"] == {}, st0


@_marks
def test_greedy_streams_unpinned(mesh_device, monkeypatch):
    base, st0, tokenizer, prompts = _one_run(mesh_device, monkeypatch, "8", False, _run_sync)
    got, st1, *_ = _one_run(mesh_device, monkeypatch, "1,2,4,8", False, _run_sync)
    _assert_buckets_used(st1)
    mm, total = _mismatches(base, got)
    logger.info(f"[bucket_engine] unpinned: token mismatches {total} (per request: count, first index) {mm}")
    firsts = {rid: i for rid, (n, i) in mm.items() if n}
    if firsts:
        ref_model, _, _ = _build(mesh_device, NB_BIG + 1)
        kv1 = [NB_BIG + 1, ref_model.args.n_local_kv_heads, BLOCK_SIZE, ref_model.args.head_dim]
        pt1 = torch.arange(1, NB_BIG + 1, dtype=torch.int32).reshape(1, NB_BIG)
        bad = []
        for rid, i in sorted(firsts.items()):
            ref, gaps = _reference_greedy(ref_model, prompts[_prompt_of(rid)], pt1, kv1, base[rid][: i + 1])
            near = gaps[i] < NEAR_TIE_GAP
            logger.info(
                f"[bucket_engine] request {rid} first mismatch at token {i}: baseline {base[rid][i]} bucket {got[rid][i]} "
                f"plain argmax {ref[i]} top-2 gap {gaps[i]:.4f} -> {'near-tie' if near else 'NOT a near-tie'}"
            )
            if not near:
                bad.append(rid)
        ref_model.free_kv_caches()
        assert not bad, f"first mismatch is not a near-tie for requests {bad}"


@_marks
def test_async_chain_with_bucket_changes(mesh_device, monkeypatch):
    base, st0, *_ = _one_run(mesh_device, monkeypatch, "8", True, _run_async)
    got, st1, *_ = _one_run(mesh_device, monkeypatch, "1,2,4,8", True, _run_async)
    cb = st1["chain_breaks"]
    logger.info(
        f"[bucket_engine] async: chain breaks base {st0['chain_breaks']} buckets {cb}; audits {st1['chain_audits']} steps {st1['chain_steps']}"
    )
    from models.demos.blackhole.qwen36.tt.tp_common import spec_bucket_hold

    if spec_bucket_hold() == 0:
        assert cb.get("bucket_change", 0) >= 3, cb
    else:  # the down-shift hold defers/absorbs most bucket changes: only require that some switch happened
        n_sw = cb.get("bucket_change", 0) + (st1["downs_free"] or 0)
        logger.info(
            f"[bucket_engine] hold>0: bucket_change {cb.get('bucket_change', 0)} downs_free {st1['downs_free']}"
        )
        assert n_sw >= 1, (cb, st1["downs_free"])
    extra = set(cb) - set(st0["chain_breaks"])
    assert extra <= {"bucket_change"}, f"new break reasons {extra}: {cb} vs {st0['chain_breaks']}"
    assert (
        st1["chain_audits"] > 0 and st1["chain_audits"] == st1["chain_steps"]
    ), st1  # every chained step audited (a mismatch raises)
    _assert_buckets_used(st1)
    mm, total = _mismatches(base, got)
    assert total == 0 and all(len(base[r]) == len(got[r]) for r in base), f"pinned async streams differ: {mm}"


def _dram_bytes(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(v.total_bytes_allocated_per_bank)


@_marks
def test_no_late_allocations_buckets(mesh_device, monkeypatch):
    """T19: B=8 engine with QWEN36_SPEC_BUCKETS=1,2,4,8 (run it with TT_METAL_TRACE_ALLOC_TRACKING=1 too: the tracker raises at
    execute_trace when a buffer allocated after a capture is still alive at a replay). After capture the script (steps through
    every bucket, joins and leaves) runs with program-cache misses forbidden: the program-cache entries, the DRAM bytes allocated and
    the verify capture count do not change. (test_spec_engine_no_late_allocations builds a B=1 engine, which cannot take buckets.)
    """
    # fast prefill is off by default here: with the allocation tracker on, the short-prefill trace replay of ANY engine (also flag "8")
    # trips it on 12 buffers allocated after its capture (control run, work/m2/t19ctl.log)
    monkeypatch.setenv("QWEN36_BUCKET_TEST_FAST_PREFILL", os.environ.get("QWEN36_BUCKET_TEST_FAST_PREFILL", "0"))
    flag = os.environ.get("QWEN36_BUCKET_TEST_FLAG", "1,2,4,8")  # "8": the same test on today's engine (control)
    _flags(monkeypatch, flag, True)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    n_prog, dram = mesh_device.num_program_cache_entries(), _dram_bytes(mesh_device)
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        streams = _run_sync(eng, prompts, high=flag != "8")
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    st = _stats(eng)
    logger.info(
        f"[bucket_engine] no_late_alloc: stats {st} program cache {n_prog} -> {mesh_device.num_program_cache_entries()} dram/bank {dram} -> {_dram_bytes(mesh_device)}"
    )
    if flag != "8":
        _assert_buckets_used(st)
    assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    assert _dram_bytes(mesh_device) == dram, "DRAM allocation changed over the run"
    assert model._vfy_capture_count == 1
    assert sorted(streams) == list(range(11))
