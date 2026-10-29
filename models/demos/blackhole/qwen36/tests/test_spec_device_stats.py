# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of the verify STATS trace (work/research_1026/67 sections 3 and 8.2; QWEN36_SPEC_DEVICE_LOGPROBS /
QWEN36_SPEC_DEVICE_TOPP, default on). One Galaxy slot (TP mesh, MESH_DEVICE=P150x8), one pytest node per process.

D1 + D7  test_d1_d7_stats_numerics_all_buckets: on real sampled fused steps, for the row buckets 1 / 2 / 4 / 8, the stats
         buffer (L_raw, L_T per engine row) equals the float64 values computed from the logits trace's rows (|lse diff| <= 1e-4,
         L_T relative <= 1e-5 for per-user T in {0.3, 0.6, 1, 1.5} and a greedy row), and the topk union equals the global top-64
         of the same logits (ids, values, ties). Also D6: no program compiled after capture while every bucket runs with and
         without the stats replay.
D6       test_d6_trace_memory_and_streams[on|off]: trace-region bytes after every capture (the stats traces' cost is the
         difference of the two runs) and a scripted sampled run whose committed streams must be identical on / off
         (normal rows bit-identical: the stats replay, alternating on and off, changes no committed token).
D10      test_d10_stats_timing: device time of the stats trace per bucket (replays), the topk trace for scale, and the median
         sampled fused step wall time with and without the stats replay.

Run (one node per process; see work/scratch/phase2/run_dev.sh):
  pytest models/demos/blackhole/qwen36/tests/test_spec_device_stats.py::test_d1_d7_stats_numerics_all_buckets
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
from models.demos.blackhole.qwen36.tt.spec_logprobs import _stable_topm_row, candidates_from_union, stats_from_logits

OUT_DIR = os.environ.get("P2_OUT_DIR", "/home/ttuser/atupe/qwen-or/work/scratch/phase2")
TEMPS = (0.3, 0.6, 1.0, 1.5)
STEPS_PER_CASE = int(os.environ.get("P2_STEPS", "14"))
M = 64


class StatsDriver(SampledDriver):
    """SampledDriver with per-user temperatures (TEMPS cycled, user 3 greedy when ``greedy_row``), a record of the sampling
    parameters of the last step, and a switch that makes every verify replay also run the stats trace."""

    def __init__(self, eng, seed, model, stats_on=lambda s: True, greedy_row=True, **kw):
        super().__init__(eng, seed, **kw)
        self.model, self.stats_on, self.greedy_row = model, stats_on, greedy_row
        self.last_samp = None
        self.stats_ran = False
        self.buckets_seen = []
        self._orig = model.verify_stage_and_enqueue
        drv = self

        def wrapped(*a, **kw2):
            want = bool(drv.stats_on(drv.steps)) and bool(kw2.get("execute_topk")) and drv._has_stats(kw2.get("bucket"))
            drv.stats_ran = want
            if want:
                kw2["execute_stats"] = True
            h = drv._orig(*a, **kw2)
            drv.buckets_seen.append(None if h.bucket is None else int(h.bucket))
            return h

        model.verify_stage_and_enqueue = wrapped

    def close(self):
        if "verify_stage_and_enqueue" in self.model.__dict__:
            del self.model.verify_stage_and_enqueue

    def _has_stats(self, bucket):
        m = self.model
        if bucket is None or int(bucket) == m._vfy_B:
            return getattr(m, "_vfy_stats_trace_id", None) is not None
        return m._vfy_bk[int(bucket)].stats_trace_id is not None

    def _spec_sampling(self):
        B = tc.B
        s = super()._spec_sampling()
        s["temperature"] = torch.tensor([TEMPS[u % 4] for u in range(B)], dtype=torch.float32)
        g = torch.zeros(B, dtype=torch.bool)
        if self.greedy_row:
            g[3] = True
        s["greedy"] = g
        self.last_samp = s
        return s

    def step(self, skip=frozenset()):
        self.stats_ran = False
        super().step(skip)


def _row_temps(samp, rows):
    t = samp["temperature"].clone()
    t = torch.where(samp["greedy"] | (t <= 0), torch.ones_like(t), t)
    return t.repeat_interleave(tc.K + 1)[:rows]


def _check_stats(model, drv, tag):
    """Compare the stats buffer / union of the LAST step with float64 values from the same step's logits trace rows."""
    B, T = tc.B, tc.K + 1
    bucket = drv.buckets_seen[-1]
    rows = (bucket or B) * T
    l_raw, l_t = model.fetch_verify_stats()
    logits = model.fetch_verify_logits()[:rows].float()
    temps = _row_temps(drv.last_samp, rows)
    m64, lr64, lt64 = stats_from_logits(logits, temps)
    lse64 = m64 + lr64.log()
    lse_dev = m64 + l_raw[:rows].double().log()  # the device max is exactly the global max (a bf16 logit)
    err_lse = float((lse_dev - lse64).abs().max())
    err_lt = float(((l_t[:rows].double() - lt64).abs() / lt64).max())
    err_lr = float(((l_raw[:rows].double() - lr64).abs() / lr64).max())
    assert bool(torch.isfinite(l_raw[:rows]).all() and torch.isfinite(l_t[:rows]).all()), f"{tag}: non-finite stats"
    pad_finite = bool(torch.isfinite(l_raw[rows:]).all() and torch.isfinite(l_t[rows:]).all())
    # union == global top-64 of the same logits
    width = model.num_devices * model._vfy_topk_m
    uv = ttnn.to_torch(ttnn.get_device_tensors(model._vfy_topk_vals_out)[0]).reshape(-1, width)[:rows]
    ui = ttnn.to_torch(ttnn.get_device_tensors(model._vfy_topk_ids_out)[0]).reshape(-1, width)[:rows]
    cv, ci = candidates_from_union(uv, ui, model.vocab_size // model.num_devices, M)
    bad_rows = []
    for r in range(rows):
        wv, wi = _stable_topm_row(logits[r], M)
        if not (torch.equal(cv[r], wv) and torch.equal(ci[r], wi)):
            bad_rows.append(r)
    return dict(
        bucket=bucket,
        rows=rows,
        err_lse=err_lse,
        err_lt=err_lt,
        err_lr=err_lr,
        union_bad_rows=bad_rows,
        pad_finite=pad_finite,
    )


def _setup(mesh_device, monkeypatch, env=None):
    monkeypatch.setenv("QWEN36_SPEC_FUSED", "1")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    model, eng, tokenizer, prompts = tc._setup(mesh_device)
    assert eng._fa_trace is not None and eng._fs_trace is not None, "fused traces missing"
    return model, eng, prompts


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d1_d7_stats_numerics_all_buckets(mesh_device, monkeypatch):
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    assert model._vfy_stats_trace_id is not None, "B-row stats trace not captured"
    buckets = sorted(model._vfy_bk)
    assert buckets == [1, 2, 4], f"buckets {buckets}"
    assert all(model._vfy_bk[b].stats_trace_id is not None for b in buckets), "bucket stats trace missing"
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    results, seen = [], set()
    try:
        for n in (1, 2, 4, 8):  # ascending: a bucket switches up at once but down only after a hold
            drv = StatsDriver(eng, seed=30 + n, model=model, stats_on=lambda s: s % 3 != 2, greedy_row=n >= 4)
            _admit_all(drv, prompts, n, 200)
            try:
                for i in range(STEPS_PER_CASE):
                    if not drv.live():
                        break
                    drv.step()
                    b = drv.buckets_seen[-1]
                    if not drv.stats_ran:
                        continue  # a step without the stats replay (the on/off alternation) or a fallback step
                    r = _check_stats(model, drv, f"n={n} step={i}")
                    r.update(n=n, step=i)
                    results.append(r)
                    seen.add(r["bucket"] or 8)
            finally:
                drv.close()
                for row in drv.live():
                    drv._finish(row)
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    by_bucket = {}
    for r in results:
        by_bucket.setdefault(r["bucket"] or 8, []).append(r)
    summary = {}
    for b, rs in sorted(by_bucket.items()):
        summary[b] = dict(
            steps=len(rs),
            max_err_lse=max(x["err_lse"] for x in rs),
            max_relerr_L_raw=max(x["err_lr"] for x in rs),
            max_relerr_L_T=max(x["err_lt"] for x in rs),
            union_bad_rows=sum(len(x["union_bad_rows"]) for x in rs),
            pad_rows_finite=all(x["pad_finite"] for x in rs),
        )
        logger.info(f"[stats D1] bucket {b}: {summary[b]}")
    with open(os.path.join(OUT_DIR, "d1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    assert seen == {1, 2, 4, 8}, f"buckets covered {sorted(seen)}"
    for b, s in summary.items():
        assert s["max_err_lse"] <= 1e-4, f"bucket {b}: lse error {s['max_err_lse']}"
        assert s["max_relerr_L_T"] <= 1e-5, f"bucket {b}: L_T relative error {s['max_relerr_L_T']}"
        assert s["max_relerr_L_raw"] <= 1e-5, f"bucket {b}: L_raw relative error {s['max_relerr_L_raw']}"
        assert s["union_bad_rows"] == 0, f"bucket {b}: union differs from the global top-64"


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("stats", ["on", "off"])
def test_d6_trace_memory_and_streams(mesh_device, stats, monkeypatch):
    env = {} if stats == "on" else {"QWEN36_SPEC_DEVICE_LOGPROBS": "0", "QWEN36_SPEC_DEVICE_TOPP": "0"}
    model, eng, prompts = _setup(mesh_device, monkeypatch, env)
    on = stats == "on"
    assert (model._vfy_stats_trace_id is not None) == on
    mem = dict(getattr(model, "_vfy_mem_log", {}))
    mem.update({k: v for k, v in getattr(eng, "_mem_log", {}).items()})
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    streams = {}
    try:
        for n in (8, 3):
            drv = StatsDriver(
                eng, seed=7 + n, model=model, stats_on=(lambda s: s % 2 == 0) if on else (lambda s: False)
            )
            _admit_all(drv, prompts, n, 60)
            try:
                for _ in range(30):
                    if not drv.live():
                        break
                    drv.step()
            finally:
                drv.close()
                for row in drv.live():
                    drv._finish(row)
            streams[str(n)] = {str(k): v for k, v in drv.outs.items()}
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    with open(os.path.join(OUT_DIR, f"d6_{stats}.json"), "w") as f:
        json.dump({"mem_log_trace_bytes_dram_bytes": mem, "streams": streams}, f)
    logger.info(f"[stats D6:{stats}] trace/DRAM bytes after each capture: {mem}")


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_d10_stats_timing(mesh_device, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_FUSED_AUDIT", "0")
    model, eng, prompts = _setup(mesh_device, monkeypatch)
    dev = model.device
    res = {"trace_ms": {}}
    reps = 100

    def replay_ms(tid):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / reps * 1e3

    # run one sampled step first so the traces read real (not zero) data
    drv = StatsDriver(eng, seed=3, model=model)
    _admit_all(drv, prompts, 8, 200)
    drv.step()
    drv.step()
    traces = {8: (model._vfy_stats_trace_id, model._vfy_topk_trace_id)}
    for b, bk in sorted(model._vfy_bk.items()):
        traces[b] = (bk.stats_trace_id, bk.topk_trace_id)
    for b, (st, tk) in sorted(traces.items()):
        res["trace_ms"][b] = {"stats": replay_ms(st), "topk": replay_ms(tk)}
        logger.info(
            f"[stats D10] bucket {b}: stats trace {res['trace_ms'][b]['stats']:.3f} ms, topk trace {res['trace_ms'][b]['topk']:.3f} ms"
        )
    # whole sampled fused step (8 live rows, B-row traces), stats replay on / off, interleaved blocks, wall ms per decode_forward
    wall = {"on": [], "off": []}
    for blk in range(4):
        for mode in ("off", "on"):
            drv.close()
            drv = StatsDriver(
                eng, seed=40 + blk, model=model, stats_on=(lambda s: True) if mode == "on" else (lambda s: False)
            )
            _admit_all(drv, prompts, 8, 100)
            for _ in range(12):
                if not drv.live():
                    break
                drv.step()
            wall[mode] += [ms for ms, rm, fr in drv.step_ms[3:] if not fr]
            for row in drv.live():
                drv._finish(row)
    drv.close()
    res["step_wall_ms_median"] = {k: statistics.median(v) for k, v in wall.items()}
    res["step_wall_ms_n"] = {k: len(v) for k, v in wall.items()}
    logger.info(
        f"[stats D10] sampled fused step wall ms (B=8, sync, audit off): {res['step_wall_ms_median']} n={res['step_wall_ms_n']}"
    )
    with open(os.path.join(OUT_DIR, "d10_timing.json"), "w") as f:
        json.dump(res, f, indent=1)
