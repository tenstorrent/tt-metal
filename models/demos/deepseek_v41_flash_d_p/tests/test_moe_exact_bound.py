# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.55a (owner 2026-11-13, MoE exact replicas, STEP 1, the go/no-go; no new kernel, no model change): how much device
time a chunk the exact split can save at most, and how much of it survives the copy deadline.

Device part (one process, the shipped settings, MOE_REPLICAS=off, MOE_COUNTS=on read back without files): chunk 10
(tokens [51200, 56320), the quick rung's chunk) of three workloads, each profiled with the program real-time profiler
(no device profiler, no per-op mode):
  novel   = the ladder's prompt, chunks 0-9 prefilled on the device first (the device prefix, as P.45's count files);
  code    = P.45's code prompt (tests/test_moe_counts._ids), the same way;
  randids = the position profile's chunk at 51,200 (testing/profile.PositionGolden: random ids, zero prefix; DS41-228:
            its routing is not a prompt's).
Each workload's chunk 10 runs PROFILED_RUNS times profiled (novel / code: the real chunk 10 first, then re-runs on the
profile's steady-state path; randids: one compile run first). Per run and layer: the router's counts (every slot's
tokens, from the head's counts buffer) and the device time of the layer's routed_experts_combine program (experts +
combine, one program a chip), sparse_dispatch_2d and fused_head, per chip (the ops wrapped to know their program ids).
The experts+combine time used is the shortest chip's (the chip that started last: DS41-248, the others' extra is the
wait for the late chips' host dispatch at the start barrier); the slowest chip's is recorded too.

CPU part (tests/moe_exact_model.py, its docstring defines every term): a per-chip expert-time model fitted on all the
profiled layers (A = one expert's weights at 512 GB/s; B, C0 least squares), every chip's own-expert finish per layer,
the exact split planned per layer with free copies (max) and with timed copies (deadline: store-and-forward trees at
P.47's link rate, FIFO per link). Metrics per workload (suffix _<workload>_p51200, the median over the runs):
  exact_save_ms_max_*       device ms a chunk the exact split saves at most (copies free);
  exact_save_ms_deadline_*  what survives the copy deadline (one link pair a copy, as measured);
  exact_lever_ms_2path_*    the same with each copy split over two paths (the brief's lever);
  exact_ceiling_ms_*        every chip at the mean (no slot limit, no copy);
the per-layer table in results/P.55a_exact_bound.json. The gate passes when the numbers are measured, whatever they
are; under ~5 ms a chunk is the NO-GO answer.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_exact_bound.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free
from models.demos.common.bringup.testing.profile import PositionGolden
from models.demos.deepseek_v41_flash_d_p.tests import moe_exact_model as M

S = spec()
SEQ, POS = 56320, 51200
PROFILED_RUNS = 3
WORKLOADS = ("novel", "code", "randids")
OPS = ("fused_head", "sparse_dispatch_2d", "routed_experts_combine")
DRAM_GBPS = 512.0  # (the owner's page: 16 Gbps GDDR6, 512 GB/s a chip)
NO_GO_MS = 5.0


class _Spy:
    """The MoE block's three ops, wrapped: each call's device-operation id range, under the current layer."""

    def __init__(self, monkeypatch):
        mb = ttnn.bringup.moe_block_main
        self.calls, self.layer = [], None
        for name in OPS:
            orig = getattr(mb, name)

            def wrap(*a, _o=orig, _n=name, **k):
                lo = ttnn._ttnn.get_device_operation_id()
                out = _o(*a, **k)
                self.calls.append((self.layer, _n, lo, ttnn._ttnn.get_device_operation_id()))
                return out

            monkeypatch.setattr(mb, name, wrap)


def _times(calls, recs: dict, d2c: dict) -> dict:
    """{layer: {op: {"max": ms, "min": ms, "chips": [ms a chip]}}} from the real-time records of the calls' ids."""
    out = {}
    for layer, name, lo, hi in calls:
        per = {}
        for i in range(lo, hi):
            for d, e in recs.get(i, {}).items():
                c = d2c.get(int(d), int(d))
                per[c] = per.get(c, 0.0) + e[0]
        if not per:
            continue
        slot = out.setdefault(layer, {}).setdefault(name, {"chips": {}})
        for c, ns in per.items():
            slot["chips"][c] = slot["chips"].get(c, 0.0) + ns / 1e6
    for layer in out.values():
        for v in layer.values():
            chips = [v["chips"][c] for c in sorted(v["chips"])]
            v.update(max=max(chips), min=min(chips), chips=[round(x, 4) for x in chips])
    return out


def _chunk(model, mesh, layers, t, st, state, spy, prof: bool):
    """One chunk at st; prof: under the real-time profiler. -> (the counts record, the MoE ops' times or None)."""
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A

    spy.calls.clear()
    if prof:
        profiler.enable(mesh, rt=True)
    try:
        begin_chunk(model, t, st)
        h = model.embed(t)
        for i in layers:
            spy.layer = i
            h2 = model.layer(i, h, st, state)
            model.free(h)
            h = h2
        spy.layer = None
        if layers[-1] == S.num_layers - 1:
            h2 = model.final_norm(h)
            model.free(h)
            h = h2
        model.free(h)
        timing = None
        if prof:
            profiler.signpost("end")
            profiler.finish_rt()
            timing = _times(spy.calls, profiler._P["rt_records"], profiler._P["dev_to_chip"])
        else:
            ttnn.synchronize_device(mesh)
    finally:
        if prof:
            profiler.disable()
    rec = A.flush_counts(mesh, st, int(S.get("target.chunk")))
    return rec, timing


def _measure(model, mesh, layers, tokens: torch.Tensor, wl: str, spy) -> list:
    """The workload's PROFILED_RUNS profiled chunk-10 runs: [(counts record, times)]."""
    chunk = int(S.get("target.chunk"))
    state = model.new_state(SEQ)
    steady = getattr(model, "steady_state", None)
    runs = []
    try:
        if wl == "randids":  # (the position profile: a fresh state at 51,200; one compile run first)
            t = tokens[POS : POS + chunk]
            _chunk(model, mesh, layers, t, POS, state, spy, prof=False)
        else:
            for c in range(POS // chunk):
                t0 = time.time()
                _chunk(model, mesh, layers, tokens[c * chunk : (c + 1) * chunk], c * chunk, state, spy, prof=False)
                logger.info(f"P.55a {wl} prefix chunk {c} {time.time() - t0:.1f} s")
        t = tokens[POS : POS + chunk]
        for r in range(PROFILED_RUNS):
            if r > 0 or wl == "randids":  # (a re-run of the same chunk: the profile's timed path, F236)
                if steady is not None:
                    steady(POS)
            runs.append(_chunk(model, mesh, layers, t, POS, state, spy, prof=True))
            rec, tm = runs[-1]
            tot = sum(v["routed_experts_combine"]["max"] for v in tm.values() if "routed_experts_combine" in v)
            logger.info(f"P.55a {wl} profiled run {r}: routed_experts_combine {tot:.2f} ms (slowest chip, summed)")
    finally:
        if steady is not None:
            steady(None)
        _free(state)
    return runs


def _row(v) -> str:
    return " ".join(str(x) for x in v)


def _ids(wl: str) -> torch.Tensor:
    if wl == "randids":
        return PositionGolden(S, POS, int(S.get("target.chunk"))).tokens()
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_counts import _ids as prompt_ids

    return torch.tensor(prompt_ids(wl))


@mesh_parametrize(timeout_s=2 * 3600)
def test_moe_exact_bound(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_COUNTS", "on")
    monkeypatch.delenv("DS41_MOE_COUNTS_DIR", raising=False)  # (read back only, no files)
    settings.thaw()
    assert settings.get("MOE_COUNTS") == "on"
    assert settings.get("MOE_REPLICAS") == "off", "the bound is the off path's imbalance (MOE_REPLICAS=off)"
    chunk = int(S.get("target.chunk"))
    assert POS % chunk == 0 and POS + chunk <= SEQ, (POS, chunk, SEQ)
    layers = S.layers()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    cfg = model.cfg
    topk, n_exp = int(cfg.topk_experts), int(cfg.n_experts)
    shape = tuple(mesh_device.shape)
    n_chips = shape[0] * shape[1]
    spy = _Spy(monkeypatch)

    data, fails = {}, []
    for wl in WORKLOADS:
        t0 = time.time()
        data[wl] = _measure(model, mesh_device, layers, _ids(wl), wl, spy)
        logger.info(f"P.55a {wl}: measured in {time.time() - t0:.0f} s")

    # ------------------------------------------------------------------------------------------ the samples
    samples = {}  # (wl, run) -> {layer: (rec min ms, rec max ms, own counts, dispatch max, head max)}
    for wl, runs in data.items():
        for r, (rec, tm) in enumerate(runs):
            if rec is None:
                fails.append(f"{wl} run {r}: no counts")
                continue
            spc = int(rec["slots_per_chip"])
            per = {}
            for L in layers:
                lc = rec["layers"].get(str(L))
                t = tm.get(L, {}).get("routed_experts_combine")
                if lc is None or t is None:
                    fails.append(f"{wl} run {r} layer {L}: counts {lc is not None}, time {t is not None}")
                    continue
                own = M.own_counts(lc["chips"], spc)
                if int(own.sum()) != chunk * topk:
                    fails.append(f"{wl} run {r} layer {L}: own counts sum {int(own.sum())} != {chunk * topk}")
                per[L] = (
                    t["min"],
                    t["max"],
                    own,
                    tm[L].get("sparse_dispatch_2d", {}).get("max", 0.0),
                    tm[L].get("fused_head", {}).get("max", 0.0),
                )
            samples[(wl, r)] = per
    for f in fails:
        logger.error(f"P.55a: {f}")
    assert not fails, f"{len(fails)} missing measurements: {fails[:5]}"

    # ------------------------------------------------------------------------------------------ the time model
    w_bytes = 3 * int(cfg.dim) * int(cfg.inter) * 0.5625  # (bfp4: 4 bits + a shared exponent byte per 16 values)
    A_ms = w_bytes / (DRAM_GBPS * 1e9) * 1e3
    flat = [(v[0], v[2]) for per in samples.values() for v in per.values()]
    fit = M.fit(flat, A_ms)
    sens = {f"{a / A_ms:.1f}xA": round(M.fit(flat, a)["rmse_ms"], 4) for a in (A_ms, 1.5 * A_ms, 2 * A_ms)}
    B_ms, C0 = fit["B_ms"], fit["C0_ms"]
    logger.info(
        f"P.55a time model: expert weights {w_bytes / 1e6:.1f} MB -> A {A_ms * 1e3:.1f} us (512 GB/s); fitted B "
        f"{B_ms * 1e3:.2f} us a row tile, C0 {C0:.3f} ms; R^2 {fit['r2']:.3f}, rmse {fit['rmse_ms']:.3f} ms over "
        f"{fit['samples']} layer calls; rmse at other A: {sens}"
    )

    # ------------------------------------------------------------------------------------------ the plans
    tasks = [
        ((wl, r, L, paths), per[L][2], shape, A_ms, B_ms, paths)
        for (wl, r), per in samples.items()
        for L in per
        for paths in (1, 2)
    ]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(48, max(1, (os.cpu_count() or 2) - 8))) as pool:
        res = dict(pool.map(M.layer_task, tasks, chunksize=4))
    logger.info(f"P.55a planner: {len(tasks)} layer plans in {time.time() - t0:.0f} s")

    table, out = {}, {}
    for wl in WORKLOADS:
        tag = f"{wl}_p{POS}"
        per_run = []
        for r in range(PROFILED_RUNS):
            per = samples[(wl, r)]
            b1 = {L: res[(wl, r, L, 1)] for L in per}
            b2 = {L: res[(wl, r, L, 2)] for L in per}
            per_run.append(
                {
                    "measured_rec_ms": sum(v[1] for v in per.values()),
                    "measured_rec_min_ms": sum(v[0] for v in per.values()),
                    "measured_dispatch_ms": sum(v[3] for v in per.values()),
                    "model_rec_ms": sum(C0 + b["before_ms"] for b in b1.values()),
                    "save_max": sum(b["save_max_ms"] for b in b1.values()),
                    "save_deadline": sum(b["save_deadline_ms"] for b in b1.values()),
                    "save_2path": sum(b["save_deadline_ms"] for b in b2.values()),
                    "ceiling": sum(b["before_ms"] - b["mean_ms"] for b in b1.values()),
                    "replicas": sum(b["n_rep_deadline"] for b in b1.values()),
                    "on_time": sum(b["on_time"] for b in b1.values()),
                    "src_short": sum(b["src_window_short"] for b in b1.values()),
                    "busy_over_mean": float(np.median([b["before_ms"] / b["mean_ms"] for b in b1.values()])),
                }
            )
            if r == 0:  # (the per-layer table: the first profiled run, the real chunk 10 for the prompts)
                table[tag] = [
                    {
                        "layer": L,
                        "measured_rec_ms": round(per[L][1], 4),
                        "measured_rec_min_ms": round(per[L][0], 4),
                        "model_rec_ms": round(C0 + b1[L]["before_ms"], 4),
                        "busiest_ms": round(b1[L]["before_ms"], 4),
                        "mean_ms": round(b1[L]["mean_ms"], 4),
                        "exact_max_ms": round(b1[L]["max_ms"], 4),
                        "exact_deadline_ms": round(b1[L]["deadline_ms"], 4),
                        "exact_2path_ms": round(b2[L]["deadline_ms"], 4),
                        "save_max_ms": round(b1[L]["save_max_ms"], 4),
                        "save_deadline_ms": round(b1[L]["save_deadline_ms"], 4),
                        "replicas": b1[L]["n_rep_deadline"],
                        "on_time": b1[L]["on_time"],
                        "src_window_short": b1[L]["src_window_short"],
                        # (lists as space-separated strings: indent=1 puts a number a line, > the 500 KB file limit)
                        "reach_ms": _row(b1[L]["reach_ms"]),  # (every chip's own-expert finish, the model; chip order)
                        "tokens_chip": _row(b1[L]["tokens_chip"]),
                        "tokens_slot": [_row(r) for r in per[L][2].tolist()],  # (every expert's tokens: a chip's slots)
                        "copy_hops_max": max((x["hops"] for x in b1[L]["replicas_deadline"]), default=0),
                        "copy_land_ms_min": min((x["t_land_ms"] for x in b1[L]["replicas_deadline"]), default=0.0),
                    }
                    for L in sorted(per)
                ]
        med = {k: float(np.median([p[k] for p in per_run])) for k in per_run[0]}
        out[tag] = {"median": med, "runs": per_run}
        metrics.record(f"exact_save_ms_max_{tag}", round(med["save_max"], 2))
        metrics.record(f"exact_save_ms_deadline_{tag}", round(med["save_deadline"], 2))
        metrics.record(f"exact_lever_ms_2path_{tag}", round(med["save_2path"], 2))
        metrics.record(f"exact_ceiling_ms_{tag}", round(med["ceiling"], 2))
        metrics.record(f"exact_measured_rec_ms_{tag}", round(med["measured_rec_ms"], 2))
        metrics.record(f"exact_model_rec_ms_{tag}", round(med["model_rec_ms"], 2))
        metrics.record(f"exact_replicas_{tag}", med["replicas"])
        metrics.record(f"exact_on_time_{tag}", med["on_time"])
        metrics.record(f"exact_busy_over_mean_{tag}", round(med["busy_over_mean"], 3))
    metrics.record("exact_fit_B_us", round(B_ms * 1e3, 3))
    metrics.record("exact_fit_C0_ms", round(C0, 4))
    metrics.record("exact_fit_A_us", round(A_ms * 1e3, 2))
    metrics.record("exact_fit_r2", round(fit["r2"], 4))
    metrics.record("exact_fit_rmse_ms", round(fit["rmse_ms"], 4))
    metrics.record_json(
        "exact_bound",
        {
            "fit": {k: v for k, v in fit.items() if k != "pred_ms"},
            "fit_rmse_at_A": sens,
            "copy": {"t1_ms": M.T1_MS, "hop_ms": M.HOP_MS, "sub_tiles": M.SUB_TILES},
            "mesh": list(shape),
            "chips": n_chips,
            "experts": n_exp,
            "summary": out,
            "layers": table,
        },
    )

    # ------------------------------------------------------------------------------------------ the report
    for tag, v in out.items():
        m = v["median"]
        print(
            f"P.55a {tag}: measured experts+combine {m['measured_rec_ms']:.1f} ms a chunk (model {m['model_rec_ms']:.1f}), "
            f"busiest/mean {m['busy_over_mean']:.2f}; exact split saves at most {m['save_max']:.1f} ms (ceiling "
            f"{m['ceiling']:.1f}), {m['save_deadline']:.1f} ms survive the copy deadline ({m['on_time']:.0f} of "
            f"{m['replicas']:.0f} copies on time), {m['save_2path']:.1f} ms with two paths a copy"
        )
        print(
            "  layer | meas ms | model | busiest | mean | max | deadline | 2path | reps on-time"
            + "".join(
                f"\n  {row['layer']:5d} | {row['measured_rec_ms']:7.3f} | {row['model_rec_ms']:5.3f} | "
                f"{row['busiest_ms']:7.3f} | {row['mean_ms']:.3f} | {row['exact_max_ms']:.3f} | "
                f"{row['exact_deadline_ms']:8.3f} | {row['exact_2path_ms']:.3f} | {row['replicas']:4d} {row['on_time']:3d}"
                for row in table[tag]
            )
        )
    for tag, v in out.items():  # (the owner's line: under ~5 ms a chunk surviving the deadline is a NO-GO)
        d = v["median"]["save_deadline"]
        print(f"P.55a verdict {tag}: {d:.1f} ms a chunk survive the deadline: " + ("GO" if d >= NO_GO_MS else "NO-GO"))
