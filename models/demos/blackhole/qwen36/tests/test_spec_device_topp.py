# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the top-p certificate of the fused MTP step (design note 67 sections 5 and 8.2, step S3). One Galaxy slot (TP mesh,
MESH_DEVICE=P150x8), one pytest node per process. They need the stats traces and the certificate accept trace (QWEN36_SPEC_DEVICE_TOPP
default 1). They drive MTPSpecEngine.decode_forward directly (no plugin, no capability): a certificate row is a sampled row with
``top_k = 0`` and ``top_p < 1`` in ``spec_sampling``. Certificate steps run synchronously (defer_read=False).

D3   test_d3_plain_vs_cert_trace_bitwise: sampled steps with NORMAL rows only (top_k 20). After every step the plain accept trace and the
     certificate variant are replayed on the same staged buffers: result_out, pack_out, rs_pt_out must be bitwise equal. Also: the shared
     buffer addresses are unchanged, the engine replayed the plain trace on every one of these steps (no cert step counted).
D4   test_d4_cert_rows_vs_host: rows 0..3 are certificate rows, the others top-k rows; (T, p) cycle over T in {0.3, 0.6, 1.0, 1.5} and p in
     {0.8, 0.9, 0.95, 0.99}. The engine audit runs on every sync step (QWEN36_SPEC_FUSED_AUDIT_EVERY=1: device == fused_sampled_reference
     with the device L_T, except explained float32 boundaries; it raises otherwise). Every step is also compared with host_redo_walk on the
     SAME step's full logits and uniforms (exact float64). Gates: unexplained differing users <= 1 per 100 user-steps, at least 20 steps
     passed the certificate and at least one failed (else the sweep did not cover both).
D5   test_d5_forced_failure_redo_is_exact: p = 0.999 and T = 1.5 on rows 0..3 (the certificate fails), rows 4..7 normal top-k rows: a step
     with a failed certificate is redone on the host for the flagged users only; the merged result equals the full float64 host walk
     of the same logits and uniforms (boundary tolerance as D4); the next step proposes and runs normally (the driver keeps stepping);
     the host redo was called with a user subset in at least one step; step times of failing steps are logged. Also: a deferred step with a
     certificate row raises ValueError before anything is enqueued.

Run (one node per process): pytest -o addopts= --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_device_topp.py::<test> (needs a Galaxy slot; set P*_OUT_DIR to keep the JSON output)
"""

import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine import _admit_all
from models.demos.blackhole.qwen36.tests.test_spec_fused_sampled_engine import SampledDriver
from models.demos.blackhole.qwen36.tt import spec_engine as SE
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import host_redo_walk

OUT_DIR = None  # set by the module fixture below: P2_OUT_DIR, else a pytest tmp dir


@pytest.fixture(scope="module", autouse=True)
def _out_dir(tmp_path_factory):
    global OUT_DIR
    OUT_DIR = os.environ.get("P2_OUT_DIR") or str(tmp_path_factory.mktemp("p2_out"))
    os.makedirs(OUT_DIR, exist_ok=True)
    yield


K = tc.K
T = K + 1
SWEEP_T = (0.3, 0.6, 1.0, 1.5)
SWEEP_P = (0.8, 0.9, 0.95, 0.99)
BAD_PER_100 = 1.0  # unexplained differing users per 100 user-steps (float32 boundaries are rarer than this)


def _setup(mesh_device, monkeypatch, env=None):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._fa_trace is not None and eng._fs_trace is not None, "fused traces missing"
    assert (
        eng._fs_trace_c is not None and eng._cert_ok()
    ), "certificate trace missing (QWEN36_SPEC_DEVICE_TOPP=0 or no stats trace)"
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


class CertDriver(SampledDriver):
    """SampledDriver whose spec_sampling has ``cert_rows`` certificate rows (top_k 0) with (T, p) from ``tp(step)``; the other rows are
    top-k 20 rows at T 0.6 / top_p 0.95. ``after(drv, eng, res, ss, args, kw)`` runs after every sync step, before anything else touches the
    device."""

    def __init__(self, eng, seed, model, cert_rows=(0, 1, 2, 3), tp=None, after=None, **kw):
        super().__init__(eng, seed, **kw)
        self.model, self.cert_rows, self.tp, self.after = model, tuple(cert_rows), tp or (lambda s: (0.6, 0.95)), after
        self.cert_step_ms = []
        self.redo_users = []  # per host redo: the users it walked

    def _spec_sampling(self):
        B = tc.B
        s = super()._spec_sampling()
        temp, p = self.tp(self.steps)
        for r in self.cert_rows:
            s["top_k"][r] = 0
            s["top_p"][r] = p
            s["temperature"][r] = temp
        self.last = s
        return s

    def step(self, skip=frozenset()):
        eng, drv = self.eng, self
        real = SE.host_redo_walk

        def spy(*a, **kw):
            drv.redo_users.append(None if kw.get("users") is None else sorted(kw["users"]))
            return real(*a, **kw)

        def checking(*a, **kw):
            t0 = time.perf_counter()
            res = MTPSpecEngine.decode_forward(eng, *a, **kw)
            drv.cert_step_ms.append(1e3 * (time.perf_counter() - t0))
            if drv.after is not None:
                drv.after(drv, eng, res, kw["spec_sampling"], a, kw)
            return res

        eng.decode_forward = checking
        SE.host_redo_walk = spy
        try:
            super().step(skip)
        finally:
            eng.__dict__.pop("decode_forward", None)
            SE.host_redo_walk = real


def _read(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


# --------------------------------------------------------------------------- #
@_marks
def test_d3_plain_vs_cert_trace_bitwise(mesh_device, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    snap = eng._shared_addrs()
    assert eng._fs_trace != eng._fs_trace_c
    fs = eng._fs
    n_cmp = [0]

    def after(drv, eng, res, ss, args, kw):
        ttnn.synchronize_device(mesh_device)
        got = []
        for tid in (eng._fs_trace, eng._fs_trace_c):  # the same staged inputs through both traces
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            got.append([_read(t) for t in (fs.result_out, fs.pack_out, fs.rs_pt_out)])
        for name, a, b in zip(("result_out", "pack_out", "rs_pt_out"), *got):
            assert torch.equal(a, b), f"step {drv.steps}: {name} differs between the plain and the certificate trace"
        n_cmp[0] += 1

    drv = CertDriver(eng, seed=31, model=model, cert_rows=(), after=after)
    _admit_all(drv, prompts, 8, 400)
    for _ in range(int(os.environ.get("P2_STEPS", "120"))):
        if not drv.live():
            break
        drv.step()
    for r in drv.live():
        drv._finish(r)
    eng._check_shared_addrs(snap, "after the D3 steps")
    logger.info(
        f"[topp D3] compared {n_cmp[0]} steps, cert steps {eng.cert_steps}, fallbacks {dict(getattr(eng, 'fused_fallbacks', {}))}"
    )
    assert n_cmp[0] > 20, "too few sync sampled steps compared"
    assert eng.cert_steps == 0, "a normal step counted as a certificate step"
    with open(os.path.join(OUT_DIR, "d3.json"), "w") as f:
        json.dump(dict(compared=n_cmp[0]), f)


def _vs_host(drv, eng, res, ss, args, kw, st):
    """Compare a sync step with the exact host walk of the same step (logits of the step, the step's uniforms): one user at a time, from
    the user's own rows (a user's walk does not depend on the others)."""
    logits = drv.model.fetch_verify_logits()  # engine rows [B*T, V]
    phys = None if eng._phys_perm is None else eng._phys_perm[0]
    toks, nvd = torch.as_tensor(args[0]), torch.as_tensor(kw["num_valid_drafts"])
    for s in range(tc.B):
        if int(args[1][s, 0]) < 0:
            continue
        e = s if phys is None else int(phys[s])
        p = {k: torch.as_tensor(ss[k])[s : s + 1] for k in ("temperature", "top_k", "top_p", "greedy")}
        hc, hn = host_redo_walk(
            logits[e * T : (e + 1) * T],
            toks[s : s + 1],
            p,
            torch.as_tensor(ss["uniforms"])[s : s + 1],
            B=1,
            K=K,
            num_valid=nvd[s : s + 1],
        )
        n = int(res.accepted_counts[s])
        same = n == int(hn[0]) and torch.equal(res.accepted_token_ids[s, :n].to(torch.int32), hc[0, :n])
        st["n"] += 1
        st["bad"] += int(not same)


@_marks
def test_d4_cert_rows_vs_host(mesh_device, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch, env={"QWEN36_SPEC_FUSED_AUDIT_EVERY": "1"})
    st = dict(n=0, bad=0)
    drv = CertDriver(
        eng, seed=41, model=model, cert_rows=(0, 1, 2, 3),
        tp=lambda s: (SWEEP_T[s % 4], SWEEP_P[(s // 4) % 4]),
        after=lambda d, e, r, ss, a, kw: _vs_host(d, e, r, ss, a, kw, st),
    )  # fmt: skip
    _admit_all(drv, prompts, 8, 400)
    for _ in range(int(os.environ.get("P2_STEPS", "160"))):
        if not drv.live():
            break
        drv.step()
    for r in drv.live():
        drv._finish(r)
    fb = dict(getattr(eng, "fused_fallbacks", {}))
    passed = eng.cert_steps - eng.cert_failed_steps
    rep = dict(steps=drv.steps, cert_steps=eng.cert_steps, cert_failed_steps=eng.cert_failed_steps, user_steps=st["n"], bad=st["bad"], fallbacks=fb,
               boundary_diffs=eng.fused_boundary_diffs)  # fmt: skip
    logger.info(f"[topp D4] {rep}")
    with open(os.path.join(OUT_DIR, "d4.json"), "w") as f:
        json.dump(rep, f)
    assert eng.cert_steps >= 20 and passed >= 20, "too few steps passed the certificate"
    assert eng.cert_failed_steps >= 1, "no certificate failure in the sweep (raise P2_STEPS)"
    assert st["bad"] * 100 <= BAD_PER_100 * st["n"], f"{st['bad']} of {st['n']} user-steps differ from the host walk"


@_marks
def test_d5_forced_failure_redo_is_exact(mesh_device, monkeypatch, expect_error):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    st = dict(n=0, bad=0)
    drv = CertDriver(
        eng, seed=51, model=model, cert_rows=(0, 1, 2, 3), tp=lambda s: (1.5, 0.999),
        after=lambda d, e, r, ss, a, kw: _vs_host(d, e, r, ss, a, kw, st),
    )  # fmt: skip
    _admit_all(drv, prompts, 8, 400)
    # a deferred step with a certificate row is refused before anything is enqueued
    ss = drv._spec_sampling()
    B = tc.B
    with expect_error(ValueError, "certificate row"):
        eng.decode_forward(
            torch.zeros(B, T, dtype=torch.int32), torch.full((B, T), -1, dtype=torch.int32),
            num_valid_drafts=torch.zeros(B, dtype=torch.int32), accepted_counts=torch.ones(B, dtype=torch.int32),
            page_table=torch.zeros(B, tc.NB, dtype=torch.int32), spec_mode="fused_sample", spec_sampling=ss, defer_read=True,
        )  # fmt: skip
    for _ in range(int(os.environ.get("P2_STEPS", "60"))):
        if not drv.live():
            break
        drv.step()
    live_rows = len(drv.cert_rows)
    for r in drv.live():
        drv._finish(r)
    fb = dict(getattr(eng, "fused_fallbacks", {}))
    redo = fb.get("overflow_redo", 0)
    subset = [u for u in drv.redo_users if u is not None and len(u) < tc.B]
    fail_ms = statistics.median(drv.cert_step_ms) if drv.cert_step_ms else 0.0
    rep = dict(steps=drv.steps, cert_steps=eng.cert_steps, cert_failed_steps=eng.cert_failed_steps, redo_steps=redo, subset_redos=len(subset),
               user_steps=st["n"], bad=st["bad"], median_step_ms=fail_ms, fallbacks=fb)  # fmt: skip
    logger.info(f"[topp D5] {rep}")
    with open(os.path.join(OUT_DIR, "d5.json"), "w") as f:
        json.dump(rep, f)
    assert eng.cert_failed_steps > 0 and redo > 0, "no certificate failure happened (p 0.999 at T 1.5 must fail)"
    assert subset, "the host redo always walked every user: the subset redo did not run"
    assert (
        st["bad"] * 100 <= BAD_PER_100 * st["n"]
    ), f"{st['bad']} of {st['n']} user-steps differ from the full host walk"
    assert live_rows == 4
