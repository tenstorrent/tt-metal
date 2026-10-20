# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""QWEN36_SPEC_FUSED=1 engine tests (device): spec_mode "fused_sample" steps the plugin way.

A. test_fused_step_equivalence[b1|b8]: every fused step's cached drafts equal the NORMAL propose path run on the same
   commit (cache bypassed by a second propose_draft_tokens call); final streams equal plain greedy up to near-ties.
B. test_fused_continuous: joins / finishes / parked rows; streams equal the argmax_ids + host walk run of the same scenario.
C. test_fused_one_capture_no_recompile: one prepare + capture, 20 fused steps, no program compiled after capture.
D. (inside A[b8]) median step wall time, fused vs argmax_ids + host walk + propose.
"""

import gc
import statistics
import time

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.spec_helpers import PAD, _build
from models.demos.blackhole.qwen36.tests.test_spec_lossless import _assert_lossless_up_to_near_ties, _reference_greedy
from models.demos.blackhole.qwen36.tt.spec_fused import fused_greedy_walk

MAX_NEW_A = 130  # ~48 steps at ~2.7 committed tokens per step
TIME_STEPS = 30


class FusedDriver(tc.BatchDriver):
    """tc.BatchDriver with a selectable verify mode. mode "fused": spec_mode fused_sample, commit from the device walk,
    propose with page_table=None (the plugin's call, so the engine's fused cache can hit). mode "argmax": argmax_ids +
    host greedy walk + propose with the explicit table (the unfused path)."""

    def __init__(self, eng, seed, mode, check=False, **kw):
        super().__init__(eng, seed, **kw)
        self.mode, self.check = mode, check
        self.wall_ms = []
        self.fb_steps = []  # per step: None (fused ran) or the fallback reason
        self.cmp_steps = 0  # steps whose fused drafts were compared with the normal propose

    def step(self, skip=frozenset()):
        B, K, eng = tc.B, tc.K, self.eng
        tokens = torch.zeros((B, 1 + K), dtype=torch.int32)
        start = torch.full((B, 1 + K), PAD, dtype=torch.int32)
        nvd = torch.zeros(B, dtype=torch.int32)
        acc = torch.ones(B, dtype=torch.int32)
        pt = torch.zeros((B, tc.NB), dtype=torch.int32)
        live = [r for r in self.live() if self.reqs[r]["rid"] not in skip]
        pos0, drafts, ns = {}, {}, {}
        for r in live:
            q = self.reqs[r]
            pos0[r] = q["P"] + len(q["out"]) - 1
            drafts[r] = list(q["proposal"]) if self.drafts else []
            n = ns[r] = len(drafts[r])
            tokens[r] = PAD
            tokens[r, : 1 + n] = torch.tensor([q["out"][-1]] + drafts[r], dtype=torch.int32)
            start[r, : 1 + n] = torch.arange(pos0[r], pos0[r] + 1 + n, dtype=torch.int32)
            nvd[r] = n
            acc[r] = q["acc"]
            pt[r] = self._blocks_for(q, pos0[r])
        fresh = bool(eng._fresh)
        fb_before = dict(getattr(eng, "fused_fallbacks", {}))
        t0 = time.perf_counter()
        fused = self.mode == "fused"
        res = eng.decode_forward(
            tokens,
            start,
            num_valid_drafts=nvd,
            accepted_counts=acc,
            page_table=pt,
            spec_mode="fused_sample" if fused else "argmax_ids",
        )
        self.step_ms.append((1e3 * (time.perf_counter() - t0), self.remapped, fresh))
        self.remapped = False
        ids_all = res.argmax_ids
        assert tuple(ids_all.shape) == (B, 1 + K)
        if fused:
            fb_after = getattr(eng, "fused_fallbacks", {})
            reason = next((k for k, v in fb_after.items() if v > fb_before.get(k, 0)), None)
            self.fb_steps.append(reason)
            assert res.accepted_token_ids is not None and res.accepted_counts is not None
            for r in live:  # the device commit is a valid greedy walk of the ids (the engine's own check is stricter)
                c = int(res.accepted_counts[r])
                assert 1 <= c <= 1 + K, f"row {r}: count {c}"
        committed = torch.full((B, 1 + K), PAD, dtype=torch.int32)
        cpos = torch.full((B, 1 + K), PAD, dtype=torch.int32)
        acc_vec = torch.ones(B, dtype=torch.int32)
        pt2 = torch.zeros((B, tc.NB), dtype=torch.int32)
        ms = {}
        if not fused:
            walk_c, walk_n = fused_greedy_walk(
                torch.where(ids_all == PAD, torch.zeros_like(ids_all), ids_all), tokens, nvd, K
            )
        for r in live:
            q, n = self.reqs[r], ns[r]
            if fused:
                cnt = int(res.accepted_counts[r])
                full = res.accepted_token_ids[r, :cnt].tolist()
            else:
                ids = ids_all[r].tolist()
                assert all(i != PAD for i in ids[: n + 1]), f"row {r}: -1 inside valid ids {ids} (n={n})"
                m = 0
                while m < n and drafts[r][m] == ids[m]:
                    m += 1
                full = drafts[r][:m] + [ids[m]]
                assert full == walk_c[r, : m + 1].tolist() and m + 1 == int(walk_n[r])
            ms[r] = len(full) - 1
            new = full[: q["max_new"] - len(q["out"])]
            q["out"].extend(new)
            q["acc"] = len(new)
            committed[r, : len(new)] = torch.tensor(new, dtype=torch.int32)
            cpos[r, : len(new)] = torch.arange(pos0[r] + 1, pos0[r] + 1 + len(new), dtype=torch.int32)
            acc_vec[r] = len(new)
            pt2[r] = self._blocks_for(q, pos0[r] + len(new))
        if fused:
            cached = eng._fused_cache is not None
            d = eng.propose_draft_tokens(
                K, committed, cpos, acc_vec, None
            )  # the plugin's call: cache hit when fused ran
            self.wall_ms.append(1e3 * (time.perf_counter() - t0))
            if self.check and cached:
                assert eng._fused_cache is None, "fused cache not consumed by the matching propose"
                d2 = eng.propose_draft_tokens(K, committed, cpos, acc_vec, None)  # cache empty: the NORMAL path
                assert torch.equal(d.num_valid, d2.num_valid), f"num_valid fused {d.num_valid} vs normal {d2.num_valid}"
                for r in range(B):
                    nv = int(d.num_valid[r])
                    assert torch.equal(
                        d.draft_token_ids[r, :nv], d2.draft_token_ids[r, :nv]
                    ), f"step {self.steps} row {r}: fused drafts {d.draft_token_ids[r].tolist()} vs normal {d2.draft_token_ids[r].tolist()}"
                self.cmp_steps += 1
        else:
            d = eng.propose_draft_tokens(K, committed, cpos, acc_vec, pt2)
            self.wall_ms.append(1e3 * (time.perf_counter() - t0))
        assert tuple(d.draft_token_ids.shape) == (B, K) and tuple(d.num_valid.shape) == (B,)
        for r in range(B):
            nv = int(d.num_valid[r])
            if r in pos0:
                assert 0 <= nv <= K
                self.reqs[r]["proposal"] = d.draft_token_ids[r, :nv].tolist()
            else:
                assert nv == 0, f"padding row {r} proposed {nv} drafts"
        for r in live:
            if len(self.reqs[r]["out"]) >= self.reqs[r]["max_new"]:
                self._finish(r)
        self.stats.append({"live": live, "n": ns, "m": ms, "cross": [], "no_draft": sorted(eng._no_draft)})
        self.steps += 1


def _fb_report(eng, drv, tag):
    fb = dict(getattr(eng, "fused_fallbacks", {}))
    logger.info(
        f"[fused:{tag}] steps={drv.steps} fused_steps(total)={eng.fused_steps} fallbacks={fb} "
        f"compared_with_normal_propose={drv.cmp_steps} mean_m(committed/row-step)={tc._accept_rate(drv):.2f}"
    )
    return fb


def _admit_all(drv, prompts, n, max_new):
    for u in range(n):
        drv.admit(u, u, prompts[u], max_new=max_new)


def _timed(eng, prompts, mode, seed):
    drv = FusedDriver(eng, seed, mode)
    _admit_all(drv, prompts, tc.B, 100)
    for _ in range(TIME_STEPS):
        drv.step()
    for r in drv.live():
        drv._finish(r)
    return statistics.median(drv.wall_ms), drv.wall_ms


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("nb", [1, 8], ids=["b1", "b8"])
def test_fused_step_equivalence(mesh_device, nb, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    monkeypatch.setattr(tc, "B", nb)
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._fa is not None and eng._fa_trace is not None, "fused accept not prepared / captured"
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = FusedDriver(eng, seed=1, mode="fused", check=True)
        for u in range(nb):
            drv.admit(u, u, prompts[u], max_new=MAX_NEW_A)
        while drv.live():
            drv.step()
        outs = drv.outs
        fb = _fb_report(eng, drv, f"b{nb}")
        assert drv.cmp_steps > 0, "no fused step was compared with the normal propose"
        assert tc._accept_rate(drv) > 1.0, "no drafts accepted"
        for u in range(nb):
            assert len(outs[u]) == MAX_NEW_A, f"user {u}: {len(outs[u])} tokens"
        if nb == 8:  # D: step time, fused vs argmax_ids + host walk + propose
            med_f, _ = _timed(eng, prompts, "fused", seed=11)
            med_a, _ = _timed(eng, prompts, "argmax", seed=12)
            logger.info(
                f"[fused:time] b8 median wall ms/step over {TIME_STEPS}: fused={med_f:.2f} argmax+walk+propose={med_a:.2f} "
                f"delta={med_a - med_f:.2f}"
            )
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)

    # plain-greedy reference: teacher-forced on a B=1 model (the B-user model cannot coexist with it)
    model.free_kv_caches()
    if nb > 1:
        model = eng = drv = None
        gc.collect()
        model, _, _ = _build(mesh_device, 17)
    kv1 = [17, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt1 = torch.arange(1, 17, dtype=torch.int32).reshape(1, 16)
    for u in range(nb):
        ref, gaps = _reference_greedy(model, prompts[u], pt1, kv1, outs[u])
        _assert_lossless_up_to_near_ties(outs[u], ref, gaps, tokenizer, f"fused b{nb} user {u}")
    model.free_kv_caches()


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fused_continuous(mesh_device, monkeypatch):
    """Joins, finishes (request 100 reuses a finished row) and parked rows: fused streams == argmax_ids + host walk."""
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._fa is not None and eng._fa_trace is not None
    skips = {4: frozenset({0}), 9: frozenset({5, 2}), 10: frozenset({5})}
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        ref_drv = FusedDriver(eng, seed=4, mode="argmax")
        ref = ref_drv.run(tc._join_schedule(prompts), skips=skips)
        drv = FusedDriver(eng, seed=4, mode="fused", check=True)
        got = drv.run(tc._join_schedule(prompts), skips=skips)
        fb = _fb_report(eng, drv, "continuous")
        logger.info(f"[fused:continuous] argmax pass steps={ref_drv.steps} rate={tc._accept_rate(ref_drv):.2f}")
        assert set(got) == set(ref)
        for rid in sorted(ref):
            tc._assert_same(f"request {rid} fused vs argmax_ids", got[rid], ref[rid])
        assert drv.cmp_steps > 0
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fused_one_capture_no_recompile(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    model, eng, tokenizer, prompts = tc._setup(mesh_device)  # prepare + capture once
    assert eng._fa is not None, "fused accept not allocated"
    assert eng._fa_trace is not None, "fused accept trace not captured"
    assert eng._dr_trace is not None and model._vfy_capture_count == 1
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = FusedDriver(eng, seed=7, mode="fused")
        _admit_all(drv, prompts, tc.B, 100)
        for _ in range(20):
            drv.step()
        fb = _fb_report(eng, drv, "one_capture")
        eligible = eng.fused_steps - sum(fb.values())
        assert eligible >= 1, f"no step ran fused: fallbacks {fb}"
        assert mesh_device.num_program_cache_entries() == n_prog, "program cache grew after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    assert eng._fa_trace is not None and model._vfy_capture_count == 1, "trace re-captured"
