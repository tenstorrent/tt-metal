# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test: a token mask and penalties in the SAME fused step (merged tree wt/p3-all; design note 86 section 7).

test_mask_pen_same_step: ~100 sync fused steps, 8 users. User 0: structure mask (2-8 allowed tokens per column). User 1: presence 0.5 +
repetition 1.1. User 2: both. Users 3-7: plain sampled (T 0.6, top_k 20, top_p 0.95). Gates:
  G1 committed tokens of masked rows are always allowed by their mask;
  G2 device commit (tokens, counts) = host reference for ALL users every step: raw logits (logits trace) -> penalties of the device
     model (bf16) -> mask -> host_redo_walk with the same uniforms;
  G3 device pen counts = host counts after every step;
  G4 plain rows 3-7: same committed streams as a control run (special rows plain, same slots; bucket 8 in both, so the check applies);
  G5 trace order observed: main, mask, pen, topk (first replay of each in a step);
  G6 no program compiled after capture.
Run (one node per process): pytest -o addopts= --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_device_mask_pen.py::<test> (needs a Galaxy slot; set P*_OUT_DIR to keep the JSON output)
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_device_mask import MaskDriver, _case_rows
from models.demos.blackhole.qwen36.tests.test_spec_device_penalties import PenMixin, _assert_counts
from models.demos.blackhole.qwen36.tt import spec_masks as SM
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import host_redo_walk
from models.demos.blackhole.qwen36.tt.spec_penalties import (
    apply_penalties_device_model,
    column_history_counts,
    counts_from_ids,
    mask_from_ids,
)

STEPS = int(os.environ.get("MP_STEPS", "100"))
MASKED = (0, 2)
PENROWS = (1, 2)
PEN = (1.1, 0.0, 0.5, False)  # repetition, frequency, presence, greedy


def _fixed_mask_fn(seed):
    g = torch.Generator().manual_seed(seed)

    def fn(step, B):
        T = tc.K + 1
        m = torch.full((B, T, SM.MASK_WORDS), -1, dtype=torch.int32)
        for u in MASKED:
            for j in range(T):
                a = _case_rows("structure", torch.randn(1, SM.VOCAB, generator=g), g)
                m[u, j] = SM.pack_allowed(a)[0]
        return m

    return fn


class MPDriver(PenMixin, MaskDriver):
    """Mask + penalties driver. ``special`` False = control run (no mask, no penalties). Checks G1, G2 on every step."""

    def __init__(self, eng, seed, special=True, **kw):
        super().__init__(eng, seed, mask_fn=_fixed_mask_fn(7) if special else None, **kw)
        self._pen_init()
        self.special, self.bad, self.n_ref = special, [], 0
        self.committed_log = []

    def step(self, skip=frozenset()):
        eng = self.eng
        inner = eng.decode_forward
        outs_before = {r: list(self.reqs[r]["out"]) for r in self.live()}

        def hooked(tokens, start_pos, **kw):
            res = inner(tokens, start_pos, **kw)
            if self.special:
                self._check(res, tokens, kw, outs_before)
            return res

        eng.decode_forward = hooked
        try:
            super().step(skip)
        finally:
            eng.__dict__.pop("decode_forward", None)  # SampledDriver.step removed its own wrapper

    def _check(self, res, tokens, kw, outs_before):
        eng, B, K = self.eng, tc.B, tc.K
        T, V = K + 1, int(eng.vocab)
        s = kw["spec_sampling"]
        nv = torch.as_tensor(kw["num_valid_drafts"]).reshape(B)
        raw = torch.as_tensor(eng.model.fetch_verify_logits())[: B * T, :V].to(torch.bfloat16)
        lg = raw.clone()
        for u in PENROWS:
            q = self.reqs[u]
            cnt = counts_from_ids(outs_before[u], V)
            hist = column_history_counts(cnt, torch.as_tensor(tokens)[u, 1:], int(nv[u]), T, device_rule=True)
            msk = mask_from_ids(q["prompt"], V).reshape(1, -1).expand(T, -1)
            rep, fq, ps = PEN[0], PEN[1], PEN[2]
            ones = torch.ones(T)
            lg[u * T : (u + 1) * T] = apply_penalties_device_model(
                raw[u * T : (u + 1) * T], hist, msk, ones * rep, ones * fq, ones * ps
            )
        toks = torch.as_tensor(tokens).clamp_min(0)
        ref_c, ref_n = host_redo_walk(lg, toks, s, s["uniforms"], B=B, K=K, num_valid=nv, mask=s.get("mask"))
        self.n_ref += 1
        flat = s["mask"].reshape(-1, SM.MASK_WORDS)
        for u in range(B):
            n = int(res.accepted_counts[u])
            dev = res.accepted_token_ids[u, :n].tolist()
            ref = ref_c[u, : int(ref_n[u])].tolist()
            if dev != ref:
                self.bad.append(("ref", self.steps, u, dev, ref))
            if u in MASKED:
                for j, t in enumerate(dev):
                    if not bool(SM.unpack_allowed(flat[u * T + j : u * T + j + 1], V)[0, t]):
                        self.bad.append(("disallowed", self.steps, u, j, t))

    def _spec_sampling(self):
        s = super()._spec_sampling()
        if not self.special:
            s.pop("penalties", None)
        return s


def _run(eng, prompts, special, steps):
    drv = MPDriver(eng, 77, special=special, sampled=lambda s: True, drafts=True)
    for u in range(tc.B):
        pen = PEN if (special and u in PENROWS) else None
        drv.admit_pen(u, u, prompts[u], 10**6, pen)
    cnt_checked = 0
    try:
        for _ in range(steps):
            drv.step()
            if special:
                cnt_checked += _assert_counts(eng, drv, f"step {drv.steps}")
        plain = {u: list(drv.reqs[u]["out"]) for u in range(tc.B)}
    finally:
        for r in drv.live():
            drv._finish(r)
    return drv, plain, cnt_checked


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_mask_pen_same_step(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "1")
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT_EVERY", "1")
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._fa_trace is not None and eng._fs_trace is not None, "fused traces missing"
    assert eng._pen_ok() and eng._mask_ok(), "pen / mask traces missing"
    # trace order hook: name of each replayed trace id
    names = {}
    for nm in ("trace", "mask_trace", "pen_trace", "topk_trace", "stats_trace", "logits_trace"):
        v = getattr(model, f"_vfy_{nm}_id", None)
        if v is not None:
            names[str(v)] = nm
    for nm in ("_fs_trace", "_fs_trace_c", "_pen_upd_trace", "_dr_trace"):
        v = getattr(eng, nm, None)
        if v is not None:
            names[str(v)] = nm
    order, orig_exec = [], ttnn.execute_trace

    def spy(dev, tid, *a, **k):
        order.append(names.get(str(tid), f"?{tid}"))
        return orig_exec(dev, tid, *a, **k)

    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    monkeypatch.setattr(ttnn, "execute_trace", spy)
    try:
        drv, special_out, n_cnt = _run(eng, prompts, True, STEPS)
        monkeypatch.setattr(ttnn, "execute_trace", orig_exec)
        ctrl, ctrl_out, _ = _run(eng, prompts, False, STEPS)
    finally:
        monkeypatch.setattr(ttnn, "execute_trace", orig_exec)
        mesh_device.set_program_cache_misses_allowed(True)
    compiled = mesh_device.num_program_cache_entries() - n_prog
    fb = dict(getattr(eng, "fused_fallbacks", {}))
    first = order[: order.index("_dr_trace") + 1] if "_dr_trace" in order else order[:12]
    plain_same = {u: special_out[u] == ctrl_out[u] for u in range(3, tc.B)}
    logger.info(
        f"[mask_pen] steps={drv.steps} ref_checked={drv.n_ref} bad={len(drv.bad)} {drv.bad[:3]} count_rows_checked={n_cnt} "
        f"mask_steps={eng.mask_steps} pen_steps={eng.pen_steps} audited={eng.fused_audited_steps} fallbacks={fb} "
        f"plain_rows_bit_identical_to_control={plain_same} compiled_after_capture={compiled}"
    )
    logger.info(f"[mask_pen] trace order of the first sync step: {first}")
    idx = {n: first.index(n) for n in ("trace", "mask_trace", "pen_trace", "topk_trace") if n in first}
    assert not drv.bad, f"G1/G2 failed: {drv.bad[:5]}"
    assert eng.mask_steps > 0 and eng.pen_steps > 0
    assert set(idx) == {"trace", "mask_trace", "pen_trace", "topk_trace"} and (
        idx["trace"] < idx["mask_trace"] < idx["pen_trace"] < idx["topk_trace"]
    ), f"trace order {first}"
    assert compiled == 0, f"{compiled} programs compiled after capture"
    assert all(plain_same.values()), f"plain rows differ from the control run: {plain_same}"
