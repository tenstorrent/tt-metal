# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the logprob payload of the fused MTP step (design note 67 sections 4 and 8.2, step S4). One Galaxy slot
(TP mesh, MESH_DEVICE=P150x8), one pytest node per process. They need the stats traces (QWEN36_SPEC_DEVICE_LOGPROBS default 1).
They drive MTPSpecEngine.decode_forward directly (no plugin, no capability): ``spec_sampling["num_logprobs"]`` selects the rows.

D2   test_d2_sync_payload_vs_host[greedy|sampled]: sync steps; for every live row and every committed column the payload (logprob of the
     committed token, rank, top-N) equals what log_softmax of the SAME step's full logits gives (fetch_verify_logits). N cycles in
     {0, 5, 20}. Gates: |logprob diff| <= 1e-4, rank equal (or a flagged lower bound), 0 missing ids.
D2c  test_d2_chain_payload_vs_sync_greedy: the deferred, chained pipeline (QWEN36_SPEC_CHAIN_SAMPLED=1, greedy rows with logprobs, so
     S2 steps) against the same scenario run synchronously: same committed tokens, logprobs equal within 5e-2 (batch windows differ),
     plus per-payload sanity (sorted, <= 0, sum exp(top) <= 1 + 1e-3). Sampled chained rows: sanity only.
D5l  test_d5_logprobs_overflow_redo: top_k = 63 (= M - 1) makes the tie rule fire often (rank 63 and 64 tie in bf16); an overflow step is
     redone on the host and its payload comes from the full logits: exact vs log_softmax; the step count of redos is reported.
D8   test_d8_fallback_payload[mixed_rope_delta|eager_drafter]: the engine fallback (the reason is forced: _fused_ineligible, or the
     eager reseed limit for B = 1) gives the payload from the full logits; exact vs log_softmax of the same logits, and equal (within
     5e-2) to the fused payload of the same greedy scenario.
D9   test_d9_chain_soak: 1000 deferred chained steps, one logprobs row among 6 normal sampled rows, audits every 8 steps: no
     RuntimeError, no program compiled, miss / lower-bound counters reported (miss must be 0).

Run (one node per process): pytest -o addopts= --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_device_logprobs.py::<test> (needs a Galaxy slot; set P*_OUT_DIR to keep the JSON output)
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import BIG, _env
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine import _admit_all
from models.demos.blackhole.qwen36.tests.test_spec_fused_sampled_engine import SampledDriver
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine
from models.demos.blackhole.qwen36.tt.spec_logprobs import gather_logprobs_from_candidates

OUT_DIR = None  # set by the module fixture below: P2_OUT_DIR, else a pytest tmp dir


@pytest.fixture(scope="module", autouse=True)
def _out_dir(tmp_path_factory):
    global OUT_DIR
    OUT_DIR = os.environ.get("P2_OUT_DIR") or str(tmp_path_factory.mktemp("p2_out"))
    os.makedirs(OUT_DIR, exist_ok=True)
    yield


K = tc.K
T = K + 1
NLP_CYCLE = (5, 0, 20)  # per-row num_logprobs of the logprobs rows
LP_TOL = 1e-4


def _nlp_vec(B, rows, base):
    """int32 [B]: -2 everywhere except ``rows`` (cycled N values)."""
    v = torch.full((B,), -2, dtype=torch.int32)
    for i, r in enumerate(rows):
        v[r] = NLP_CYCLE[(i + base) % 3]
    return v


class Stats:
    """Running maxima / counters over the compared payloads."""

    def __init__(self):
        self.n = self.n_rows = self.rank_lower = self.rank_bad = self.missing = self.top_bad = 0
        self.max_lp = self.max_top = self.max_lse = 0.0

    def as_dict(self):
        return dict(self.__dict__)


def _sanity(cand_ids, cand_vals, lse, tok=None):
    """Per-payload checks that need no reference: finite, sorted (value desc), ids unique, logprob <= 0, sum exp(top-64 logprobs) <= 1."""
    assert bool(torch.isfinite(cand_vals).all()) and bool(torch.isfinite(lse).all()), "non-finite payload"
    assert bool((cand_vals[1:] <= cand_vals[:-1]).all()), "candidates not sorted"
    assert len(set(cand_ids.tolist())) == cand_ids.numel(), "duplicate candidate ids"
    lp = cand_vals - lse
    assert float(lp.max()) <= 1e-4, f"logprob above 0: {float(lp.max())}"
    assert float(lp.exp().sum()) <= 1.0 + 1e-3, f"sum of top-64 probabilities {float(lp.exp().sum())}"


def _compare_row(st, cand, s, j, row_logits, chosen, n_lp):
    """The payload of (slot s, column j) against log_softmax of the row's full logits (float32)."""
    lp = row_logits.float().log_softmax(-1)
    ref_lp = float(lp[chosen])
    ref_rank = int((lp >= lp[chosen]).sum())
    tok, got_lp, rank, lower, missing = gather_logprobs_from_candidates(
        cand.ids[s, j][None], cand.values[s, j][None], cand.lse[s, j][None], torch.tensor([chosen]), n_lp
    )
    st.n += 1
    st.missing += int(missing[0])
    st.max_lp = max(st.max_lp, abs(float(got_lp[0, 0]) - ref_lp))
    if bool(lower[0]):
        st.rank_lower += 1
        if int(rank[0]) > ref_rank:
            st.rank_bad += 1
    elif int(rank[0]) != ref_rank:
        st.rank_bad += 1
    if n_lp > 0:
        tv, ti = lp.topk(n_lp)
        st.max_top = max(st.max_top, float((got_lp[0, 1:] - tv).abs().max()))
        # ids must agree unless the value is tied with a neighbour (torch.topk picks arbitrarily among ties)
        for c, (a, b) in enumerate(zip(tok[0, 1:].tolist(), ti.tolist())):
            if a != b and not (
                float(tv[c]) == float(tv[min(c + 1, n_lp - 1)]) or float(tv[c]) == float(tv[max(c - 1, 0)])
            ):
                st.top_bad += 1
    _sanity(cand.ids[s, j], cand.values[s, j], cand.lse[s, j])
    st.max_lse = max(st.max_lse, abs(float(cand.lse[s, j]) - float(torch.logsumexp(row_logits.double(), -1))))


class LpSyncDriver(SampledDriver):
    """SampledDriver whose spec_sampling carries per-row T / greedy / num_logprobs; after every sync step the payload is compared
    with the full logits of the SAME step (fetch_verify_logits, before anything else runs on the device)."""

    def __init__(self, eng, seed, model, mode, lp_rows=(0, 1, 2), top_k=20, **kw):
        super().__init__(eng, seed, **kw)
        self.model, self.lp_mode, self.lp_rows, self.top_k = model, mode, tuple(lp_rows), top_k
        self.st = Stats()
        self.rec = {}  # (rid, output index) -> (token, ids, values, lse)
        self.redo_steps = 0
        self.fallback_steps = {}

    def _spec_sampling(self):
        B = tc.B
        s = super()._spec_sampling()
        s["top_k"] = torch.full((B,), self.top_k, dtype=torch.int32)
        if self.lp_mode == "greedy":
            s["greedy"] = torch.ones(B, dtype=torch.bool)
        s["num_logprobs"] = _nlp_vec(B, self.lp_rows, self.steps)
        self.last = s
        return s

    def step(self, skip=frozenset()):
        eng, drv = self.eng, self

        def checking(*a, **kw):
            fb0 = dict(getattr(eng, "fused_fallbacks", {}))
            res = MTPSpecEngine.decode_forward(eng, *a, **kw)
            fb1 = dict(getattr(eng, "fused_fallbacks", {}))
            for k, v in fb1.items():
                if v > fb0.get(k, 0):
                    drv.fallback_steps[k] = drv.fallback_steps.get(k, 0) + 1
            ss = kw["spec_sampling"]
            assert res.candidates is not None, "a step with num_logprobs >= 0 returned no payload"
            logits = drv.model.fetch_verify_logits()  # engine rows [B*T, V]
            phys = None if eng._phys_perm is None else eng._phys_perm[0]
            for s in range(tc.B):
                if int(ss["num_logprobs"][s]) < 0 or int(a[1][s, 0]) < 0:
                    continue
                e = s if phys is None else int(phys[s])
                n_lp = int(ss["num_logprobs"][s])
                for j in range(int(res.accepted_counts[s])):
                    chosen = int(res.accepted_token_ids[s, j])
                    _compare_row(drv.st, res.candidates, s, j, logits[e * T + j], chosen, n_lp)
                    q = drv.reqs[s]
                    drv.rec[(q["rid"], len(q["out"]) + j)] = (
                        chosen,
                        res.candidates.ids[s, j].clone(),
                        res.candidates.values[s, j].clone(),
                        float(res.candidates.lse[s, j]),
                    )
            drv.st.n_rows += 1
            return res

        eng.decode_forward = checking
        try:
            super().step(skip)
        finally:
            eng.__dict__.pop("decode_forward", None)


def _gate(st, tag, out_name):
    d = st.as_dict()
    logger.info(f"[logprobs {tag}] {d}")
    with open(os.path.join(OUT_DIR, f"{out_name}.json"), "w") as f:
        json.dump(d, f, indent=1)
    assert st.n > 0, "nothing compared"
    assert st.missing == 0, f"{st.missing} chosen ids missing from the 64 candidates"
    assert st.max_lp <= LP_TOL, f"logprob diff {st.max_lp}"
    assert st.max_lse <= LP_TOL, f"lse diff {st.max_lse}"
    assert st.max_top <= LP_TOL, f"top-N diff {st.max_top}"
    assert st.rank_bad == 0, f"{st.rank_bad} wrong ranks"
    assert st.top_bad == 0, f"{st.top_bad} top-N id differences without a tie"


def _setup(mesh_device, monkeypatch, env=None, nb=None):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    if nb is not None:
        monkeypatch.setattr(tc, "B", nb)
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._fa_trace is not None and eng._fs_trace is not None, "fused traces missing"
    assert eng._stats_ok(), "stats trace missing or QWEN36_SPEC_DEVICE_LOGPROBS=0"
    return model, eng, prompts


_MARKS = (
    run_for_blackhole(),
    pytest.mark.timeout(7200),
    pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True),
    pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True),
)


def _marks(f):
    for m in reversed(_MARKS):
        f = m(f)
    return f


# --------------------------------------------------------------------------- #
@_marks
@pytest.mark.parametrize("mode", ["greedy", "sampled"])
def test_d2_sync_payload_vs_host(mesh_device, mode, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = LpSyncDriver(eng, seed=11, model=model, mode=mode, lp_rows=(0, 1, 2, 5))
        _admit_all(drv, prompts, 8, 400)
        t0 = time.perf_counter()
        for _ in range(int(os.environ.get("P2_STEPS", "200"))):
            if not drv.live():
                break
            drv.step()
        logger.info(
            f"[logprobs D2:{mode}] {drv.steps} steps, {time.perf_counter() - t0:.0f} s, fallbacks {dict(getattr(eng, 'fused_fallbacks', {}))}"
        )
        for r in drv.live():
            drv._finish(r)
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    _gate(drv.st, f"D2:{mode}", f"d2_sync_{mode}")


@_marks
def test_d5_logprobs_overflow_redo(mesh_device, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    drv = LpSyncDriver(eng, seed=12, model=model, mode="sampled", lp_rows=(0, 1, 2, 3, 4, 5, 6, 7), top_k=63)
    _admit_all(drv, prompts, 8, 400)
    for _ in range(int(os.environ.get("P2_STEPS", "60"))):
        if not drv.live():
            break
        drv.step()
    fb = dict(getattr(eng, "fused_fallbacks", {}))
    logger.info(f"[logprobs D5l] fallbacks {fb}, redo steps {fb.get('overflow_redo', 0)} of {drv.steps}")
    for r in drv.live():
        drv._finish(r)
    with open(os.path.join(OUT_DIR, "d5l_fallbacks.json"), "w") as f:
        json.dump(fb, f)
    _gate(drv.st, "D5l", "d5l")
    assert (
        fb.get("overflow_redo", 0) > 0
    ), "no overflow redo happened: raise P2_STEPS or check the tie rule (the redo payload is untested)"


@_marks
@pytest.mark.parametrize("reason", ["mixed_rope_delta", "eager_drafter"])
def test_d8_fallback_payload(mesh_device, reason, monkeypatch):
    nb = 1 if reason == "eager_drafter" else None
    model, eng, prompts = _setup(mesh_device, monkeypatch, nb=nb)
    rows = (0,) if nb == 1 else (0, 1, 2)
    n_admit = 1 if nb == 1 else 4
    results = {}
    for variant in ("fused", "fallback"):
        saved = (eng._fused_ineligible, eng.eager_reseed_len)
        if variant == "fallback":
            if reason == "mixed_rope_delta":
                eng._fused_ineligible = lambda *a, **k: ("mixed_rope_delta", None)
            else:
                eng.eager_reseed_len = (
                    0  # B = 1 and start_pos + K > 0: the eager drafter would be chosen, the step falls back
                )
        try:
            drv = LpSyncDriver(eng, seed=13, model=model, mode="greedy", lp_rows=rows)
            _admit_all(drv, prompts, n_admit, 200)
            for _ in range(int(os.environ.get("P2_STEPS", "40"))):
                if not drv.live():
                    break
                drv.step()
            fb = dict(getattr(eng, "fused_fallbacks", {}))
            for r in drv.live():
                drv._finish(r)
        finally:
            eng._fused_ineligible, eng.eager_reseed_len = saved
        results[variant] = (drv, fb)
        _gate(drv.st, f"D8:{reason}:{variant}", f"d8_{reason}_{variant}")
    assert (
        results["fallback"][0].fallback_steps.get(reason, 0) > 0
    ), f"the {reason} fallback never ran: {results['fallback'][0].fallback_steps}"
    assert not results["fused"][0].fallback_steps.get(reason), "the fused run fell back"
    # the same greedy scenario: tokens equal, logprobs close (windows differ when drafts differ)
    a, b = results["fused"][0].rec, results["fallback"][0].rec
    common = sorted(set(a) & set(b))
    assert len(common) > 20
    worst = 0.0
    for k in common:
        ta, ia, va, la = a[k]
        tb, ib, vb, lb = b[k]
        assert ta == tb, f"{k}: committed token differs fused vs fallback"
        worst = max(worst, abs((float(va[0]) - la) - (float(vb[0]) - lb)))
    logger.info(
        f"[logprobs D8:{reason}] compared {len(common)} positions, max |top-1 logprob diff| fused vs fallback {worst:.4g}"
    )
    assert worst <= 5e-2


# --------------------------------------------------------------------------- #
class LpAsyncDriver(AsyncDriver):
    """AsyncDriver (deferred depth-1 pipeline) whose spec_sampling carries num_logprobs; records the payload of every resolved step
    by (request id, output index)."""

    def __init__(self, eng, seed, mode, lp_rows, **kw):
        super().__init__(eng, seed, sampled=lambda s: True, **kw)
        self.lp_mode, self.lp_rows = mode, tuple(lp_rows)
        self.rec = {}
        self.payloads = 0
        self.cnt = Stats()

    def _spec_sampling(self):
        B = tc.B
        s = super()._spec_sampling()
        if self.lp_mode == "greedy":
            s["greedy"] = torch.ones(B, dtype=torch.bool)
        s["num_logprobs"] = _nlp_vec(B, self.lp_rows, 0)
        return s

    def resolve(self):
        h, rows, rids = self.pend[0]
        ids, counts, p0 = h.wait()  # cached for the parent's call
        cand = h.candidates()
        assert cand is not None, "a deferred step with a logprobs row has no payload"
        starts = {
            r: len(self.reqs[r]["out"]) for r in rows if self.reqs[r] is not None and self.reqs[r]["rid"] == rids[r]
        }
        for r in rows:
            if r in starts and r in self.lp_rows:
                for j in range(int(counts[r])):
                    _sanity(cand.ids[r, j], cand.values[r, j], cand.lse[r, j])
                    self.rec[(rids[r], starts[r] + j)] = (
                        int(ids[r, j]),
                        cand.ids[r, j].clone(),
                        cand.values[r, j].clone(),
                        float(cand.lse[r, j]),
                    )
                    n_lp = int(_nlp_vec(tc.B, self.lp_rows, 0)[r])
                    _, _, _, lower, missing = gather_logprobs_from_candidates(
                        cand.ids[r, j][None],
                        cand.values[r, j][None],
                        cand.lse[r, j][None],
                        torch.tensor([int(ids[r, j])]),
                        n_lp,
                    )
                    self.cnt.missing += int(missing[0])
                    self.cnt.rank_lower += int(lower[0])
                    self.cnt.n += 1
        self.payloads += 1
        super().resolve()


@_marks
def test_d2_chain_payload_vs_sync_greedy(mesh_device, monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_SAMPLED", "1")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    steps = int(os.environ.get("P2_STEPS", "120"))
    lp_rows = (0, 1, 2)
    # sync reference run, greedy rows with logprobs, exact vs logits
    sync = LpSyncDriver(eng, seed=21, model=model, mode="greedy", lp_rows=lp_rows)
    _admit_all(sync, prompts, 8, BIG)
    for _ in range(steps):
        sync.step()
    for r in sync.live():
        sync._finish(r)
    _gate(sync.st, "D2c:sync", "d2c_sync")
    # deferred chained run of the same scenario
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = LpAsyncDriver(eng, 21, "greedy", lp_rows)
        _admit_all(drv, prompts, 8, BIG)
        drv.pipeline(steps)
        for r in drv.live():
            drv._finish(r)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    logger.info(
        f"[logprobs D2c] chain_steps {eng.chain_steps} payload steps {drv.payloads} counters {drv.cnt.as_dict()}"
    )
    assert eng.chain_steps > 0 and drv.cnt.missing == 0
    common = sorted(set(sync.rec) & set(drv.rec))
    assert len(common) > 50, f"only {len(common)} common positions"
    worst, bad_tok = 0.0, 0
    for k in common:
        ta, _, va, la = sync.rec[k]
        tb, _, vb, lb = drv.rec[k]
        if ta != tb:
            bad_tok += 1
            continue
        worst = max(worst, abs((float(va[0]) - la) - (float(vb[0]) - lb)))
    logger.info(
        f"[logprobs D2c] {len(common)} common positions, token differences {bad_tok}, max top-1 logprob diff sync vs chain {worst:.4g}"
    )
    assert bad_tok == 0 and worst <= 5e-2


@_marks
def test_d9_chain_soak(mesh_device, monkeypatch):
    _env(monkeypatch)
    # 1000 steps x up to 4 tokens + a 600-token prompt do not fit the 48-block rows of _env: widen the page table
    monkeypatch.setattr(tc, "NB", 72)
    monkeypatch.setattr(tc, "NUM_BLOCKS", 8 * 72 + 40)
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_SAMPLED", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "8")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT_EVERY", "8")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    err = None
    try:
        drv = LpAsyncDriver(eng, 31, "sampled", lp_rows=(4,))  # one logprobs row (N = 20) among 7 sampled rows
        _admit_all(drv, prompts, 7, BIG)
        try:
            drv.pipeline(int(os.environ.get("P2_STEPS", "1000")))
        except RuntimeError as e:
            err = e
            logger.error(f"[logprobs D9] RuntimeError: {e}")
        logger.info(
            f"[logprobs D9] steps {drv.steps} payload steps {drv.payloads} chain_steps {eng.chain_steps} chain_breaks {dict(eng.chain_breaks)} "
            f"audits chain {eng.chain_audits} sampled {eng.fused_audited_steps} fallbacks {dict(getattr(eng, 'fused_fallbacks', {}))} "
            f"counters {drv.cnt.as_dict()}"
        )
        if err is not None:
            raise err
        for r in drv.live():
            drv._finish(r)
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    with open(os.path.join(OUT_DIR, "d9.json"), "w") as f:
        json.dump(dict(steps=drv.steps, counters=drv.cnt.as_dict(), chain_steps=eng.chain_steps), f)
    assert drv.cnt.missing == 0, "logprobs_miss counter must be 0"
    assert eng.chain_steps > 0.5 * drv.steps, "the soak did not chain"
