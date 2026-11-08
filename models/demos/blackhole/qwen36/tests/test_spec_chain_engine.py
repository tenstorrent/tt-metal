# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""QWEN36_SPEC_CHAIN=1 (stage S3c chain mode) on the engine, device tests. Flags are set before the model / engine is built.
The in-engine chain audit (QWEN36_SPEC_CHAIN_AUDIT, every step) raises when the device-built next-verify tokens / aux differ
from the host's. The per-step drafts check against the normal propose path breaks the chain by design (the normal propose
drops it, reason "propose"), so it runs every CHECK_EVERY-th step only.
"""

import gc
import os
import statistics

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.spec_helpers import _build
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine import _admit_all
from models.demos.blackhole.qwen36.tests.test_spec_fused_sampled_engine import SampledDriver
from models.demos.blackhole.qwen36.tests.test_spec_lossless import _assert_lossless_up_to_near_ties, _reference_greedy

NB_BIG = 48  # blocks per row: 3072 tokens (600-token prompt + 200 steps of up to 4 tokens)
CHECK_EVERY = 25
BIG = 5000


class ChainDriver(SampledDriver):
    def __init__(self, eng, seed, check_every=0, **kw):
        super().__init__(eng, seed, check=False, **kw)
        self.check_every = check_every

    def step(self, skip=frozenset()):
        self.check = bool(self.check_every) and self.steps % self.check_every == self.check_every - 1
        super().step(skip)


def _env(monkeypatch, nb=8):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT_EVERY", "1")
    monkeypatch.setattr(tc, "B", nb)
    monkeypatch.setattr(tc, "NB", NB_BIG)
    monkeypatch.setattr(tc, "NUM_BLOCKS", nb * NB_BIG + 40)


def _setup(mesh_device):
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._ch_g is not None and eng._ch_trace_g is not None, "chain greedy trace missing"
    assert eng._fa_trace is not None
    return model, eng, tokenizer, prompts


def _report(eng, drv, tag, before=(0, 0, {})):
    fb = dict(getattr(eng, "fused_fallbacks", {}))
    cb = dict(eng.chain_breaks)
    logger.info(
        f"[chain:{tag}] steps={drv.steps} sampled={drv.n_sampled} greedy={drv.n_greedy} chain_steps={eng.chain_steps} "
        f"chain_audits={eng.chain_audits} chain_breaks={cb} fused_fallbacks={fb} "
        f"overflow_redo={fb.get('overflow_redo', 0)} drafts_compared={drv.cmp_steps} "
        f"mean_count/row-step={tc._accept_rate(drv):.2f}"
    )
    return cb, fb


def _streams(drv):
    out = {}
    for r in drv.live():
        q = drv.reqs[r]
        out[q["rid"]] = list(q["out"])
    for rid, o in drv.outs.items():
        out.setdefault(rid, o)
    return out


def _greedy_refs(mesh_device, tokenizer, prompts, streams, tag):
    """Plain greedy, teacher-forced down each stream, on a B=1 model (the B-user model cannot coexist with it)."""
    gc.collect()
    ref_model, _, _ = _build(mesh_device, NB_BIG + 1)
    kv1 = [NB_BIG + 1, ref_model.args.n_local_kv_heads, BLOCK_SIZE, ref_model.args.head_dim]
    pt1 = torch.arange(1, NB_BIG + 1, dtype=torch.int32).reshape(1, NB_BIG)
    for rid, out in sorted(streams.items()):
        ref, gaps = _reference_greedy(ref_model, prompts[rid], pt1, kv1, out)
        _assert_lossless_up_to_near_ties(out, ref, gaps, tokenizer, f"{tag} user {rid}")
    ref_model.free_kv_caches()


def _time_chain(eng, prompts):
    """Median wall ms per step over 30 greedy fused steps, B=8: chained vs unchained (chain devices detached), audit off."""
    os.environ["QWEN36_SPEC_CHAIN_AUDIT"] = "0"
    res = {}
    try:
        for name in ("chained", "unchained"):
            saved = (eng._ch_g, eng._ch_s)
            if name == "unchained":
                eng._ch_g = eng._ch_s = None
            try:
                d = ChainDriver(eng, seed=31, sampled=lambda s: False)
                _admit_all(d, prompts, tc.B, 150)
                cs0 = eng.chain_steps
                for _ in range(30):
                    d.step()
                res[name] = (statistics.median(d.wall_ms[1:]), eng.chain_steps - cs0)
                for r in d.live():
                    d._finish(r)
            finally:
                eng._ch_g, eng._ch_s = saved
                eng._chain_prev = eng._chain_in = eng._chain_cand = None
    finally:
        os.environ["QWEN36_SPEC_CHAIN_AUDIT"] = "1"
    logger.info(
        f"[chain:time] b8 greedy median wall ms/step over 29 steps (audit off): chained={res['chained'][0]:.2f} "
        f"(chain steps {res['chained'][1]}) unchained={res['unchained'][0]:.2f} (chain steps {res['unchained'][1]})"
    )


MARKS = [
    run_for_blackhole(),
    pytest.mark.timeout(7200),
    pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True),
    pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True),
]


def _marks(f):
    for m in reversed(MARKS):
        f = m(f)
    return f


@_marks
def test_chain_greedy_long(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = ChainDriver(eng, seed=1, check_every=CHECK_EVERY, sampled=lambda s: False)
        _admit_all(drv, prompts, 8, BIG)
        for _ in range(200):
            drv.step()
        cb, fb = _report(eng, drv, "greedy_long")
        assert eng.chain_steps >= 100 and eng.chain_audits == eng.chain_steps
        assert drv.cmp_steps >= 1
        streams = _streams(drv)
        for r in drv.live():
            drv._finish(r)
        _time_chain(eng, prompts)
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    model.free_kv_caches()
    model = eng = drv = None
    _greedy_refs(mesh_device, tokenizer, prompts, streams, "chain greedy")


@_marks
def test_chain_sampled_long(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    assert eng._ch_trace_s is not None, "chain sampled trace missing"
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = ChainDriver(eng, seed=2, check_every=CHECK_EVERY, sampled=lambda s: True)
        _admit_all(drv, prompts, 8, BIG)
        for _ in range(200):
            drv.step()
        cb, fb = _report(eng, drv, "sampled_long")
        assert eng.chain_steps >= 50 and eng.chain_audits == eng.chain_steps
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@_marks
def test_chain_finishes_overlay(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = ChainDriver(eng, seed=3, sampled=lambda s: False)
        lens = [BIG] * 5 + [40, 80, 120]  # rows 5, 6, 7 finish mid-run
        for u in range(8):
            drv.admit(u, u, prompts[u], max_new=lens[u])
        for _ in range(60):
            if not drv.live():
                break
            drv.step()
        cb, fb = _report(eng, drv, "finishes")
        assert {5, 6, 7} <= set(drv.outs), f"finished {sorted(drv.outs)}"
        assert "row_set" not in cb, f"a finish broke the chain as row_set: {cb}"
        assert eng.chain_steps >= drv.steps - 8 and eng.chain_audits == eng.chain_steps
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@_marks
def test_chain_breaks_parity(mesh_device, monkeypatch):
    """10 chained steps then a forced break (one request parked for a step), 120 steps; every chain step audited; the streams
    equal plain greedy."""
    _env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = ChainDriver(eng, seed=4, sampled=lambda s: False)
        _admit_all(drv, prompts, 8, BIG)
        for s in range(120):
            skip = frozenset({(s // 11) % 8}) if s % 11 == 10 else frozenset()
            drv.step(skip)
        cb, fb = _report(eng, drv, "breaks_parity")
        assert sum(cb.values()) >= 10, f"expected forced breaks, got {cb}"
        assert eng.chain_steps >= 80 and eng.chain_audits == eng.chain_steps
        streams = _streams(drv)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    model.free_kv_caches()
    model = eng = drv = None
    _greedy_refs(mesh_device, tokenizer, prompts, streams, "chain breaks")


@_marks
def test_chain_no_recompile(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    assert eng._ch_trace_g is not None and eng._ch_trace_s is not None
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = ChainDriver(eng, seed=5, sampled=lambda s: s % 2 == 1)
        _admit_all(drv, prompts, 8, BIG)
        for _ in range(40):
            drv.step()
        cb, fb = _report(eng, drv, "no_recompile")
        assert eng.chain_steps >= 20
        assert mesh_device.num_program_cache_entries() == n_prog, "program cache grew after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    assert model._vfy_capture_count == 1
