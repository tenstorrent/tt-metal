# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""QWEN36_SPEC_FUSED=1 sampled fused step (stage S2, decode_forward spec_sampling) on the engine, device tests.

a/b. test_fused_sampled_equivalence[b8|b1]: temperature 0.6 / top_k 20 / top_p 0.95 on every row, seeded uniforms; the
     in-engine audit (QWEN36_SPEC_FUSED_AUDIT default 1) must pass on each step, and each step's cached drafts must equal the
     normal propose path's. Streams are sampled, so only ranges are asserted (token ids in [0, vocab), counts in [1, K+1]).
c.   test_fused_mixed_no_recompile: greedy-only and sampled steps alternate, with joins, no compile after capture.
"""

import statistics

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine import FusedDriver, _admit_all

MAX_NEW_A = 130
TIME_STEPS = 30


class SampledDriver(FusedDriver):
    """FusedDriver whose fused steps carry spec_sampling when ``sampled(step)`` is true (else the S1 greedy step)."""

    def __init__(self, eng, seed, sampled=lambda s: True, **kw):
        super().__init__(eng, seed, "fused", **kw)
        self.sampled = sampled
        self.gen = torch.Generator().manual_seed(1000 + seed)
        self.n_sampled = self.n_greedy = 0
        self.vocab = int(eng.vocab)

    def _spec_sampling(self):
        B, K = tc.B, tc.K
        return {
            "temperature": torch.full((B,), 0.6, dtype=torch.float32),
            "top_k": torch.full((B,), 20, dtype=torch.int32),
            "top_p": torch.full((B,), 0.95, dtype=torch.float32),
            "greedy": torch.zeros(B, dtype=torch.bool),
            "uniforms": torch.rand((B, K + 2), generator=self.gen, dtype=torch.float32),
        }

    def step(self, skip=frozenset()):
        eng = self.eng
        use = bool(self.sampled(self.steps))
        orig = eng.decode_forward

        def wrapped(*a, **kw):
            if use:
                kw["spec_sampling"] = self._spec_sampling()
            res = orig(*a, **kw)
            c = res.accepted_counts
            assert int(c.min()) >= 1 and int(c.max()) <= tc.K + 1, f"counts {c.tolist()}"
            for r in range(c.shape[0]):
                t = res.accepted_token_ids[r, : int(c[r])]
                assert int(t.min()) >= 0 and int(t.max()) < self.vocab, f"row {r}: token out of range {t.tolist()}"
            return res

        eng.decode_forward = wrapped
        try:
            super().step(skip)
        finally:
            del eng.decode_forward  # back to the class method
        if use:
            self.n_sampled += 1
        else:
            self.n_greedy += 1


def _report(eng, drv, tag, live_user_steps):
    fb = dict(getattr(eng, "fused_fallbacks", {}))
    redo = fb.get("overflow_redo", 0)
    logger.info(
        f"[fused-sampled:{tag}] steps={drv.steps} sampled={drv.n_sampled} greedy={drv.n_greedy} fused_steps={eng.fused_steps} "
        f"fallbacks={fb} overflow_redo={redo} rate/user-step={redo / max(live_user_steps, 1):.4f} "
        f"boundary_diffs={eng.fused_boundary_diffs} compared={drv.cmp_steps} mean_count/row-step={tc._accept_rate(drv):.2f}"
    )
    return fb


def _user_steps(drv):
    return sum(len(s["live"]) for s in drv.stats)


def _equiv(mesh_device, nb, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    monkeypatch.setattr(tc, "B", nb)
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._fa_trace is not None and eng._fs_trace is not None, "fused traces missing"
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = SampledDriver(eng, seed=1, check=True)
        _admit_all(drv, prompts, nb, MAX_NEW_A)
        while drv.live() and drv.steps < 48:
            drv.step()
        fb = _report(eng, drv, f"b{nb}", _user_steps(drv))
        assert drv.steps == 48 or not drv.live()
        assert drv.cmp_steps > 0, "no step compared with the normal propose"
        if nb == 8:
            for r in drv.live():
                drv._finish(r)
            import os

            os.environ["QWEN36_SPEC_FUSED_AUDIT"] = "0"
            try:
                res = {}
                for name, sm in (("sampled", lambda s: True), ("greedy_S1", lambda s: False)):
                    d = SampledDriver(eng, seed=21, sampled=sm)
                    _admit_all(d, prompts, nb, 100)
                    for _ in range(TIME_STEPS):
                        d.step()
                    res[name] = statistics.median(d.wall_ms)
                    for r in d.live():
                        d._finish(r)
            finally:
                os.environ["QWEN36_SPEC_FUSED_AUDIT"] = "1"
            logger.info(
                f"[fused-sampled:time] b8 median wall ms/step over {TIME_STEPS} (audit off): "
                f"fused sampled={res['sampled']:.2f} fused greedy S1={res['greedy_S1']:.2f}"
            )
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("nb", [8, 1], ids=["b8", "b1"])
def test_fused_sampled_equivalence(mesh_device, nb, monkeypatch):
    _equiv(mesh_device, nb, monkeypatch)


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fused_mixed_no_recompile(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._fa_trace is not None and eng._fs_trace is not None, "fused traces missing"
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = SampledDriver(eng, seed=5, sampled=lambda s: s % 2 == 1, check=True)
        for u in range(5):
            drv.admit(u, u, prompts[u], max_new=150)
        drv.admit(5, 5, prompts[5], max_new=14)  # finishes after a few steps
        joins = {10: (6, 6), 20: (7, 7)}
        for step in range(40):
            if step in joins:
                row, rid = joins[step]
                drv.admit(row, rid, prompts[row], max_new=150)
            if not drv.live():
                break
            drv.step()
        fb = _report(eng, drv, "mixed", _user_steps(drv))
        logger.info(
            f"[fused-sampled:mixed] sampled steps (sampled trace) {drv.n_sampled}, greedy steps (S1 trace) {drv.n_greedy}, "
            f"finished requests {sorted(drv.outs)}"
        )
        assert drv.n_sampled > 0 and drv.n_greedy > 0 and 5 in drv.outs
        assert eng._fa_trace is not None and eng._fs_trace is not None
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
