# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for MTPSpecEngine / SpeculativeDecoder (no vLLM): the plugin-runner call pattern
(FakeRunner) and engine reuse across requests. Lossless vs plain greedy, exact up to the first bf16
near-tie flip (top-2 gap < NEAR_TIE_GAP); one trace capture, no late compiles or memory growth.
"""

import time

import pytest
import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tests.spec_helpers import (
    FakeRunner,
    _assert_matches_ref,
    _build,
    _fresh_kv,
    _mem_state,
    _perm_table,
    _plain_greedy,
    _prompt_pool,
)
from models.demos.blackhole.qwen36.tests.test_spec_lossless import NEAR_TIE_GAP

K = 3
NUM_BLOCKS = 72  # pool size and page-table width nb
CONTRACT_MAX_NEW = 96
MAX_NEW = 48
NEAR_CAP_MAX_NEW = 200


def _check_spec_memory(eng, model, nb):
    """Formula (config + tp only) must equal the bytes actually allocated, leaf by leaf."""
    from models.demos.blackhole.qwen36.tt.spec_engine import spec_memory_costs

    tc = model.args.hf_config.get_text_config()
    want = spec_memory_costs(
        tc, model.num_devices, eng.B, eng.K, block_size=eng.block_size, nb=nb, num_layers=len(model.layers)
    )
    got = eng.measured_spec_bytes()
    mib = lambda d: {k: round(v / (1 << 20), 3) for k, v in d.items()}
    for grp in ("per_seq", "fixed", "per_token"):
        logger.info(f"[spec_memory] {grp} MiB/device formula={mib(want[grp])} measured={mib(got[grp])}")
    for grp in want:
        for leaf, v in want[grp].items():
            assert got[grp][leaf] == v, f"spec memory {grp}.{leaf}: formula {v} != measured {got[grp][leaf]}"


def _log(name, out, stats):
    st = stats["steps"]
    steps = len(st)
    committed = sum(s["m"] + 1 for s in st)
    zeros = sum(1 for s in st if s["n"] == 0)
    declines = sum(1 for s in st if s["num_valid"] == 0)
    logger.info(
        f"[contract] {name}: tokens={len(out)} steps={steps} mean_committed/step={committed / max(steps, 1):.2f} "
        f"n==0 steps={zeros} declines={declines} stop={stats['stop']}"
    )
    return declines


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_contract_fake_runner(mesh_device):
    """Plugin-runner call pattern on one engine: lossless vs plain greedy, one capture, no late compiles."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from models.demos.blackhole.qwen36.tt.spec_engine import get_spec_engine

    nb = NUM_BLOCKS
    cap = nb * BLOCK_SIZE
    model, tokenizer, kv_shape = _build(mesh_device, NUM_BLOCKS)
    pool = _prompt_pool(tokenizer, cap)
    prompt = pool[:300]
    near_cap_prompt = pool[: cap - 64]

    # --- plain references first (before the engine exists) ------------------------------------ #
    ref_pt = _perm_table(NUM_BLOCKS, seed=5)
    ref, gaps = _plain_greedy(model, kv_shape, prompt, ref_pt, CONTRACT_MAX_NEW)
    # Runner stops at <= 64 new tokens here (capacity rule); reference must stay inside the table.
    cref, cgaps = _plain_greedy(model, kv_shape, near_cap_prompt, ref_pt, 64)

    # --- engine: prepare on a disposable table, capture once ---------------------------------- #
    _fresh_kv(model, kv_shape)
    eng = get_spec_engine(model, 1, K)
    eng.prepare(torch.arange(nb, dtype=torch.int32).reshape(1, nb))
    eng.capture()
    assert model._vfy_capture_count == 1
    _check_spec_memory(eng, model, nb)
    n_prog = mesh_device.num_program_cache_entries()

    rng = torch.Generator().manual_seed(1234)

    def random_policy(step):
        return int(torch.randint(0, K + 1, (1,), generator=rng))

    outs = {}
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        # 1. full drafts
        out, st = FakeRunner(eng, NUM_BLOCKS, nb, seed=1).run(prompt, CONTRACT_MAX_NEW, lambda s: K)
        _log("full", out, st)
        assert len(out) == CONTRACT_MAX_NEW
        _assert_matches_ref("full", out, ref, gaps)
        outs["full"] = out

        # 2. random n in [0, K]
        out, st = FakeRunner(eng, NUM_BLOCKS, nb, seed=2).run(prompt, CONTRACT_MAX_NEW, random_policy)
        _log("random", out, st)
        assert len(out) == CONTRACT_MAX_NEW
        assert any(0 < s["n"] < K for s in st["steps"]), "random: no step with 0 < n < K"
        _assert_matches_ref("random", out, ref, gaps)

        # 3. zero stretch: n = 0 for steps 1..8, then K
        out, st = FakeRunner(eng, NUM_BLOCKS, nb, seed=3).run(
            prompt, CONTRACT_MAX_NEW, lambda s: 0 if 1 <= s <= 8 else K
        )
        _log("zero_stretch", out, st)
        assert len(out) == CONTRACT_MAX_NEW
        assert any(s["n"] == 0 for s in st["steps"][1:9]), "zero_stretch: no n == 0 step"
        _assert_matches_ref("zero_stretch", out, ref, gaps)

        # 4. repeat of scenario 1 on a different shuffled block pool
        out, st = FakeRunner(eng, NUM_BLOCKS, nb, seed=4).run(prompt, CONTRACT_MAX_NEW, lambda s: K)
        _log("repeat_full", out, st)
        assert out == outs["full"], "repeat on a different block pool differs from scenario 1"

        # 5. near capacity: length stop / decline path
        out, st = FakeRunner(eng, NUM_BLOCKS, nb, seed=5).run(near_cap_prompt, NEAR_CAP_MAX_NEW, lambda s: K)
        declines = _log("near_capacity", out, st)
        logger.info(f"[contract] near_capacity: propose declined (num_valid 0) in {declines} steps")
        assert st["stop"] == "capacity", f"near_capacity: stopped by {st['stop']}, expected capacity rule"
        assert len(out) < NEAR_CAP_MAX_NEW
        _assert_matches_ref("near_capacity", out, cref, cgaps)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)

    assert model._vfy_capture_count == 1, "verify trace was re-captured"
    assert mesh_device.num_program_cache_entries() == n_prog, "program cache grew"


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_recorded_drafter_matches_eager(mesh_device, monkeypatch):
    """Recorded (traced) drafter vs eager drafter: identical tokens and (n, m, num_valid) per step, on different block pools."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from models.demos.blackhole.qwen36.tt.spec_engine import get_spec_engine

    nb = NUM_BLOCKS
    cap = nb * BLOCK_SIZE
    model, tokenizer, kv_shape = _build(mesh_device, NUM_BLOCKS)
    pool = _prompt_pool(tokenizer, cap)
    cases = [("short", pool[:300], 96), ("long", pool[:2200], 64)]  # long = one recorded prefill chunk + tail

    _fresh_kv(model, kv_shape)
    eng = get_spec_engine(model, 1, K)
    eng.prepare(torch.arange(nb, dtype=torch.int32).reshape(1, nb))
    eng.capture()
    assert model._vfy_capture_count == 1
    n_prog = mesh_device.num_program_cache_entries()

    # Count trace replays per run (no dedicated engine counter): verify (+ prefill chunks) replay identically in
    # both runs, so recorded - eager == drafter replays.
    calls = {"n": 0}
    real_execute_trace = ttnn.execute_trace

    def counting_execute_trace(*a, **kw):
        calls["n"] += 1
        return real_execute_trace(*a, **kw)

    monkeypatch.setattr(ttnn, "execute_trace", counting_execute_trace)
    assert getattr(eng, "recorded_drafter", False), "engine.recorded_drafter must default to True"

    def run(prompt, max_new, recorded, seed, policy_name):
        gen = torch.Generator().manual_seed(4321)  # identical policy stream for both runs
        policy = (
            (lambda s: K) if policy_name == "full" else (lambda s: int(torch.randint(0, K + 1, (1,), generator=gen)))
        )
        eng.recorded_drafter = recorded
        calls["n"] = 0
        t0 = time.perf_counter()
        try:
            out, st = FakeRunner(eng, NUM_BLOCKS, nb, seed=seed).run(prompt, max_new, policy)
        finally:
            eng.recorded_drafter = True
        dt = time.perf_counter() - t0
        return out, st, calls["n"], dt

    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for cname, prompt, max_new in cases:
            for policy_name in ("random", "full"):
                tag = f"{cname}/{policy_name}"
                rec_out, rec_st, rec_calls, rec_dt = run(prompt, max_new, True, 11, policy_name)
                eag_out, eag_st, eag_calls, eag_dt = run(prompt, max_new, False, 22, policy_name)
                _log(f"{tag} recorded", rec_out, rec_st)
                _log(f"{tag} eager", eag_out, eag_st)
                assert len(rec_out) == max_new and len(eag_out) == max_new
                assert rec_out == eag_out, f"{tag}: recorded-drafter tokens differ from eager"
                key = lambda st: [(s["n"], s["m"], s["num_valid"]) for s in st["steps"]]
                assert key(rec_st) == key(eag_st), f"{tag}: per-step (n, m, num_valid) differ"
                steps = len(rec_st["steps"])
                assert steps == len(eag_st["steps"])
                if policy_name == "random":
                    assert any(0 < s["n"] < K for s in rec_st["steps"]), f"{tag}: no partial-draft step"
                assert (
                    rec_calls - eag_calls >= steps - 1
                ), f"{tag}: recorded run replayed the drafter trace {rec_calls - eag_calls} times over {steps} steps"
                logger.info(
                    f"[drafter] {tag}: recorded {rec_dt / steps * 1e3:.1f} ms/step vs "
                    f"eager {eag_dt / steps * 1e3:.1f} ms/step (trace replays: recorded={rec_calls} eager={eag_calls})"
                )
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
        eng.recorded_drafter = True

    assert model._vfy_capture_count == 1, "verify trace was re-captured"
    assert mesh_device.num_program_cache_entries() == n_prog, "program cache grew"


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_engine_serves_many_requests_with_one_capture(mesh_device):
    """Catches a re-capture, late compile, leaked buffer or stale page table across requests on one engine."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    num_blocks = 72  # 4096 + MAX_NEW + K + margin = 65 blocks, rounded up
    model, tokenizer, kv_shape = _build(mesh_device, num_blocks)
    pool = _prompt_pool(tokenizer, 4096 + 256)
    short1 = _get_prompt(130, tokenizer)[0].tolist()
    # name, prompt; distinct text via offsets into one real-text pool
    reqs = [
        ("short", short1),
        ("2148", pool[200 : 200 + 2148]),
        ("4096", pool[:4096]),
        ("stop", pool[300 : 300 + 77]),
        ("short2", pool[50 : 50 + 100]),
        ("repeat", short1),
    ]
    assert len(reqs[1][1]) == 2148 and len(reqs[2][1]) == 4096
    tables = [_perm_table(num_blocks, seed=11 + i) for i in range(len(reqs))]
    tables[5] = tables[0].roll(5, dims=1)  # same prompt, different block assignment

    # --- plain references first, each on its own page table ----------------------------------- #
    refs = {}
    for i, (name, prompt) in enumerate(reqs):
        if name == "repeat":
            refs[name] = refs["short"]
            continue
        refs[name] = _plain_greedy(model, kv_shape, prompt, tables[i], MAX_NEW)
    # Stop token: first-occurrence index >= 2 where the plain path is confident, so it must stop there.
    sref, sgaps = refs["stop"]
    cand = [i for i in range(2, MAX_NEW) if sref[i] not in sref[:i] and min(sgaps[: i + 1]) >= NEAR_TIE_GAP]
    stop_idx = cand[0] if cand else 2
    stop_tok = sref[stop_idx]
    stop_idx = sref.index(stop_tok)
    logger.info(f"[reuse] stop token {stop_tok} expected at index {stop_idx}")

    # --- engine: built once, then every request through a NEW decoder ------------------------- #
    _fresh_kv(model, kv_shape)
    dec0 = SpeculativeDecoder(model, tables[0], draft_len=K)
    n_prog = mesh_device.num_program_cache_entries()
    n_cap = model._vfy_capture_count
    assert n_cap == 1, f"expected exactly one capture after construction, got {n_cap}"

    outs, mem = {}, {}
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for i, (name, prompt) in enumerate(reqs):
            stops = [stop_tok] if name == "stop" else None
            dec = SpeculativeDecoder(model, tables[i], draft_len=K, stop_tokens=stops)
            assert dec.engine is dec0.engine, f"{name}: a new engine was built"
            out = dec.generate(prompt, MAX_NEW)
            outs[name] = out
            assert mesh_device.num_program_cache_entries() == n_prog, f"{name}: program cache grew"
            assert model._vfy_capture_count == n_cap, f"{name}: verify trace was re-captured"
            ref, gaps = refs[name]
            if name == "stop":
                assert out[-1] == stop_tok and stop_tok not in out[:-1], f"stop: bad end {out}"
                assert len(out) < MAX_NEW
                _assert_matches_ref(name, out, ref, gaps)
            else:
                assert len(out) == MAX_NEW, f"{name}: got {len(out)} tokens"
                _assert_matches_ref(name, out, ref, gaps)
            if name == "short":
                mem["first"] = _mem_state(mesh_device)
        mem["last"] = _mem_state(mesh_device)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)

    assert outs["repeat"] == outs["short"], "repeat of request 1 differs token for token"
    assert mem["last"] == mem["first"], f"memory grew across requests: {mem['first']} -> {mem['last']}"
    assert model._vfy_capture_count == 1


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_restaging_protects_other_requests_blocks(mesh_device):
    """Catches the verify trace or drafter replaying a stale page table: request B must not write A's blocks."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    width, num_blocks = 16, 32
    SA, SB = list(range(width)), list(range(width, num_blocks))
    ta = _perm_table(num_blocks, 1, SA)
    tb = _perm_table(num_blocks, 2, SB)
    model, tokenizer, kv_shape = _build(mesh_device, num_blocks)
    prompt_a = _get_prompt(130, tokenizer)[0].tolist()
    pool = _prompt_pool(tokenizer, 256)
    prompt_b = pool[40:170]

    _fresh_kv(model, kv_shape)

    def snap(blocks):
        caches = {
            "base_first": model._paged_kv_caches[0][0],
            "base_last": model._paged_kv_caches[-1][0],
            "mtp": model.mtp.attention.paged_k,
        }
        return {
            (k, d): ttnn.to_torch(t)[blocks].clone()
            for k, c in caches.items()
            for d, t in enumerate(ttnn.get_device_tensors(c))
        }

    SpeculativeDecoder(model, ta, draft_len=K).generate(prompt_a, 32)
    a_before, b_before = snap(SA), snap(SB)
    SpeculativeDecoder(model, tb, draft_len=K).generate(prompt_b, 32)
    a_after, b_after = snap(SA), snap(SB)

    for key, ref in a_before.items():
        assert torch.equal(a_after[key], ref), f"request B modified request A's blocks in {key}"
    assert any(
        not torch.equal(b_after[key], b_before[key]) for key in b_before
    ), "request B wrote none of its own blocks (test is vacuous)"
    assert model._vfy_capture_count == 1


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_engine_no_late_allocations(mesh_device):
    """Catches a buffer allocated after the verify capture still alive at replay (tracker raises at execute_trace)."""
    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        pytest.skip("needs TT_METAL_TRACE_ALLOC_TRACKING=1")
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    num_blocks = 32
    model, tokenizer, kv_shape = _build(mesh_device, num_blocks)
    pool = _prompt_pool(tokenizer, 256)
    prompts = [_get_prompt(130, tokenizer)[0].tolist(), pool[30:130], pool[60:200]]
    _fresh_kv(model, kv_shape)
    for i, prompt in enumerate(prompts):
        out = SpeculativeDecoder(model, _perm_table(num_blocks, 100 + i), draft_len=K).generate(prompt, 16)
        assert len(out) == 16
    assert model._vfy_capture_count == 1


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_eager_reseed_path(mesh_device):
    """Forces the eager per-slot reseed (eager_reseed_len = 0): no late compile, no re-capture, output
    equal to plain greedy up to a near-tie."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    num_blocks = 32
    model, tokenizer, kv_shape = _build(mesh_device, num_blocks)
    pool = _prompt_pool(tokenizer, 4096)  # sizes with a long_<n> sample file, as in test 1
    reqs = [("p130", _get_prompt(130, tokenizer)[0].tolist()), ("p300", pool[100:400])]
    tables = [_perm_table(num_blocks, 21 + i) for i in range(len(reqs))]
    refs = [_plain_greedy(model, kv_shape, prompt, tables[i], MAX_NEW) for i, (_, prompt) in enumerate(reqs)]

    _fresh_kv(model, kv_shape)
    dec0 = SpeculativeDecoder(model, tables[0], draft_len=K)
    engine = dec0.engine
    engine.eager_reseed_len = 0
    eager_calls = []
    orig_row = engine._eager_row
    engine._eager_row = lambda feed, r: (eager_calls.append(r), orig_row(feed, r))[1]
    n_prog = mesh_device.num_program_cache_entries()
    n_cap = model._vfy_capture_count
    assert n_cap == 1
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for i, (name, prompt) in enumerate(reqs):
            dec = SpeculativeDecoder(model, tables[i], draft_len=K)
            assert dec.engine is engine
            n_eager = len(eager_calls)
            out = dec.generate(prompt, MAX_NEW)
            assert len(out) == MAX_NEW, f"{name}: got {len(out)} tokens"
            assert len(eager_calls) > n_eager, f"{name}: eager reseed path never ran"
            assert mesh_device.num_program_cache_entries() == n_prog, f"{name}: program cache grew"
            assert model._vfy_capture_count == n_cap, f"{name}: verify trace was re-captured"
            _assert_matches_ref(name, out, *refs[i])
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
