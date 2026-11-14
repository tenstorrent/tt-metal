# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.55c (owner 2026-11-13, MoE exact replicas, STEP 3): the exact split in the whole model, replicas off vs exact, on
the real prompts' chunk 10 (OV-40: the decision is read from these numbers, not from the A/B's random-token profile).

Terms (tt/expert_exact.py and tests/moe_exact_model.py define the rest):
* plan: per MoE layer, from THIS chunk's router counts (MOE_COUNTS), the hot experts, their receivers (one replica a
  receiver: its spare 13th slot) and how many tokens each holder takes (moe_exact_model.Plan; A, B from P.55a's fit).
* split: the fused head sends each (token, expert) to the expert's home slot or a replica by the token's hash, in the
  planner's shares (router_rows.hpp slot_of, the uneven split: a copy word slot | lower bound << 9).
* fill (the brief's fallback c'): every receiver's spare slot filled BEFORE the timed chunk by P.47's serial
  expert_slot_copy (outside the profiled window): the split's gain with free copies.
* exact (the design, streamed): the copy runs inside the layer's own routed_experts_combine (P.55b, REC_FX), the
  receivers' flat expert computes the replica last and reads its weights once the chip's waiter port has seen every
  block land (fx_wait.hpp; the receivers' spare slots are spoiled first, so a replica read early shows as a wrong
  output). The plan is deadline mode at P.55b's measured pair copy (COPY_PATH_MS), the receiver waiting for the whole
  copy (moe_exact_model.SUB_TILES 0). While a stream plan is set every MoE call runs the exact program (empty jobs
  where a layer has none).
The plan is computed on the host from a counting run of the same chunk (the router's counts of the off run; the inputs
are the same, so a device planner fed the same counts would plan the same): the planner's own device cost and the
counts' gather are NOT in these numbers (see BREADCRUMBS P.55c).

Per workload (novel = the ladder's prompt, code = P.45's code prompt), one process, the shipped settings with
MOE_REPLICAS=exact (13 slots a chip) and MOE_COUNTS=on (read back, no files): the device's own prefill of chunks 0-9,
the real chunk 10, then re-runs of chunk 10 on the profile's steady path (model.steady_state: every re-run sees the same
inputs): off (no plan: the empty map, the off path's routing bit for bit, P.52) and exact (the plan). Each side: one
accuracy run (every MoE layer's output streams compared on the device with the off run's: max |diff|), RUNS profiled
runs (program real-time profiler, no device profiler, no op mode): device ms a chunk (each program's slowest chip,
summed), the experts phase (routed_experts_combine, experts + combine: one program a chip, slowest chip summed over the
layers), dispatch, head; WALL_RUNS unprofiled runs (the wall); then a second accuracy run (exact twice: the repeat
must be bit-identical).

The steady path's first re-run is not a baseline (its window carry is the real chunk 10's; P.55c attempt 1 measured a
0.97 difference at layer 0 with no plan): one warm re-run first, then the off reference, then an off repeat (must be
bit-identical: exact_repeat_maxdiff_off).

Metrics (suffix _<wl>): exact_device_ms_{off,fill,exact}, exact_experts_ms_{off,fill,exact}, exact_wall_ms_*,
exact_maxdiff_{fill,exact} (vs off, every layer), exact_repeat_maxdiff_{off,fill,exact}, exact_copies; the side record
results/P.55c_exact_replicas.json (per layer: off / exact experts ms, the plan).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_exact_replicas.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free
from models.demos.deepseek_v41_flash_d_p.tests import moe_exact_model as M

S = spec()
SEQ, POS = 56320, 51200
RUNS = 3
WALL_RUNS = 5  # (unprofiled: the wall from the chunk's first host call to its sync, median; MOE_COUNTS' readback after)
WORKLOADS = ("novel", "code")
OPS = ("fused_head", "sparse_dispatch_2d", "routed_experts_combine", "expert_slot_copy")
COPY_PATH_MS = 0.89  # (P.55b: one expert copy over both links of one hop, inside the layer's combine)


class _Spy:
    """The MoE block's ops, wrapped: each call's device-operation id range, under the current layer."""

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
    """{layer: {op: slowest chip ms}} plus "_device": every program's slowest chip, summed (the chunk's device ms)."""
    out: dict = {}
    for layer, name, lo, hi in calls:
        per: dict = {}
        for i in range(lo, hi):
            for d, e in recs.get(i, {}).items():
                c = d2c.get(int(d), int(d))
                per[c] = per.get(c, 0.0) + e[0] / 1e6
        if per:
            slot = out.setdefault(layer, {})
            slot[name] = slot.get(name, 0.0) + max(per.values())
    out["_device"] = sum(max(e[0] for e in chips.values()) / 1e6 for chips in recs.values() if chips)
    return out


def _sum(tm: dict, op: str) -> float:
    return sum(v.get(op, 0.0) for k, v in tm.items() if k != "_device" and k is not None)


def _maxdiff(a: ttnn.Tensor, b: ttnn.Tensor) -> float:
    d = ttnn.max(ttnn.abs(ttnn.sub(a, b)))
    v = max(float(ttnn.to_torch(t).float().abs().max()) for t in ttnn.get_device_tensors(d))
    ttnn.deallocate(d)
    return v


def _chunk(model, mesh, layers, t, st, state, spy, prof=False, keep=None, ref=None):
    """One chunk at st. prof: under the real-time profiler -> times. keep (dict): every MoE layer's output streams
    cloned into it; ref (dict): each compared with ref's (max |diff| a layer). -> (counts record, times, diffs)."""
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A
    from models.demos.deepseek_v41_flash_d_p.tt.model import DevH

    spy.calls.clear()
    diffs = {}
    if prof:
        profiler.enable(mesh, rt=True)
    else:
        ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    try:
        begin_chunk(model, t, st)
        h = model.embed(t)
        for i in layers:
            spy.layer = i
            h2 = model.layer(i, h, st, state)
            model.free(h)
            h = h2
            if isinstance(h, DevH) and (keep is not None or ref is not None):
                if ref is not None and i in ref:
                    diffs[i] = _maxdiff(h.streams, ref[i])
                if keep is not None:
                    keep[i] = ttnn.clone(h.streams)
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
            timing = {"_wall": (time.perf_counter() - t0) * 1e3}  # (no profiler: the wall to the sync)
    finally:
        if prof:
            profiler.disable()
    rec = A.flush_counts(mesh, st, int(S.get("target.chunk")))
    return rec, timing, diffs


def _ids(wl: str) -> torch.Tensor:
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_counts import _ids as prompt_ids

    return torch.tensor(prompt_ids(wl))


def _fit() -> tuple[float, float]:
    """A, B (ms) of P.55a's fitted per-chip expert time (its record in the gate's results dir, else the ledger)."""
    name = "P.55a_exact_bound.json"
    R, C = S.mesh
    for d in (
        metrics.results_dir(),
        Path(__file__).parents[1] / "bringup" / "ledgers" / f"galaxy_{R}x{C}-full" / "results",
    ):
        if (d / name).exists():
            f = json.loads((d / name).read_text())["fit"]
            return float(f["A_ms"]), float(f["B_ms"])
    raise FileNotFoundError(f"{name}: P.55a's record")


def _plan(rec: dict, shape: tuple, epc: int, A: float, B: float, copies: bool, paths: int = 2) -> dict:
    """{layer: [(home, slot, kept, [(receiver, tokens)])]} from a counts record (the off run's: spare slots empty)."""
    spc = int(rec["slots_per_chip"])
    out = {}
    for L, lc in rec["layers"].items():
        own = M.own_counts(lc["chips"], spc)[:, :epc]
        p = M.Plan(own, shape, A, B, copies=copies, paths=paths).run()
        hot: dict = {}
        for r, (k, j, x) in sorted(p.rep.items()):
            hot.setdefault((int(k), int(j)), []).append((int(r), int(x)))
        out[int(L)] = [(k, j, int(p.nh[k, j]), rcv) for (k, j), rcv in sorted(hot.items())]
    return out


def _measure(model, mesh, layers, tokens, wl, spy, plans_for, modes) -> dict:
    """One workload: prefill, the real chunk 10, then per side (off, then each mode) an accuracy run, RUNS profiled
    runs and a repeat accuracy run on the steady path."""
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A

    chunk = int(S.get("target.chunk"))
    state = model.new_state(SEQ)
    main = {m.layer: m for m in A._MAIN_LAYERS}
    out: dict = {"sides": {}}
    ref: dict = {}
    try:
        t0 = time.time()
        for c in range(POS // chunk):
            _chunk(model, mesh, layers, tokens[c * chunk : (c + 1) * chunk], c * chunk, state, spy)
        logger.info(f"P.55c {wl}: prefix chunks 0-{POS // chunk - 1} in {time.time() - t0:.0f} s")
        t = tokens[POS : POS + chunk]
        _chunk(model, mesh, layers, t, POS, state, spy)  # (the real chunk 10)
        model.steady_state(POS)
        _chunk(model, mesh, layers, t, POS, state, spy)  # (the steady path's first re-run: its carry is the real one's)
        rec_off, _, _ = _chunk(model, mesh, layers, t, POS, state, spy, keep=ref)  # (off: the reference outputs)
        out["counts_off"] = rec_off
        out["off_repeat"] = _chunk(model, mesh, layers, t, POS, state, spy, ref=ref)[2]  # (off vs off: must be 0)
        for side in ("off",) + tuple(modes):
            plans = {}
            if side != "off":
                plans, info = plans_for(side, rec_off)
                out.setdefault("plans", {})[side] = {str(L): p for L, p in plans.items()}
                out.setdefault("plan_info", {})[side] = info
                t0 = time.time()
                for L, m in main.items():
                    m.set_exact_mode(side, plans.get(L, []))
                ttnn.synchronize_device(mesh)
                logger.info(f"P.55c {wl} {side}: plans set in {time.time() - t0:.1f} s")
                _, _, d1 = _chunk(model, mesh, layers, t, POS, state, spy, ref=ref)
            else:
                d1 = {}
            runs = []
            for r in range(RUNS):
                _, tm, _ = _chunk(model, mesh, layers, t, POS, state, spy, prof=True)
                runs.append(tm)
                logger.info(
                    f"P.55c {wl} {side} run {r}: device {tm['_device']:.2f} ms, experts+combine "
                    f"{_sum(tm, 'routed_experts_combine'):.2f}, dispatch {_sum(tm, 'sparse_dispatch_2d'):.2f}, head "
                    f"{_sum(tm, 'fused_head'):.2f}, slot copies {_sum(tm, 'expert_slot_copy'):.2f}"
                )
            walls = [_chunk(model, mesh, layers, t, POS, state, spy)[1]["_wall"] for _ in range(WALL_RUNS)]
            d2 = _chunk(model, mesh, layers, t, POS, state, spy, ref=ref)[2] if side != "off" else {}
            out["sides"][side] = {"runs": runs, "walls": walls, "diff": d1, "diff_repeat": d2}
            if side != "off":
                for m in main.values():
                    m.set_exact_mode(side, [])
                ttnn.synchronize_device(mesh)
    finally:
        for v in ref.values():
            ttnn.deallocate(v)
        model.steady_state(None)
        _free(state)
    return out


@mesh_parametrize(timeout_s=3 * 3600)
def test_moe_exact_replicas(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_REPLICAS", "exact")
    monkeypatch.setenv("DS41_MOE_COUNTS", "on")
    monkeypatch.delenv("DS41_MOE_COUNTS_DIR", raising=False)
    settings.thaw()
    assert settings.get("MOE_REPLICAS") == "exact" and settings.get("MOE_COUNTS") == "on"
    chunk = int(S.get("target.chunk"))
    layers = S.layers()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    shape = tuple(mesh_device.shape)
    n = shape[0] * shape[1]
    epc = int(model.cfg.n_experts) // n
    A_ms, B_ms = _fit()
    spy = _Spy(monkeypatch)
    modes = ("fill", "exact")

    def plans_for(side: str, rec: dict):
        t0 = time.time()
        if side == "fill":
            p = _plan(rec, shape, epc, A_ms, B_ms, copies=False)
            mode = "max (copies free)"
        else:  # (deadline mode at P.55b's measured pair copy, both links; the receiver waits for the whole copy)
            old = (M.T1_MS, M.SUB_TILES)
            M.T1_MS, M.SUB_TILES = 2.0 * COPY_PATH_MS, 0
            try:
                p = _plan(rec, shape, epc, A_ms, B_ms, copies=True, paths=2)
            finally:
                M.T1_MS, M.SUB_TILES = old
            mode = f"deadline (a copy edge {COPY_PATH_MS} ms, whole-copy wait)"
        reps = sum(len(rcv) for e in p.values() for *_, rcv in e)
        return p, {"replicas": reps, "plan_s": round(time.time() - t0, 1), "mode": mode}

    data, fails = {}, []
    for wl in WORKLOADS:
        t0 = time.time()
        data[wl] = _measure(model, mesh_device, layers, _ids(wl), wl, spy, plans_for, modes)
        logger.info(f"P.55c {wl}: measured in {time.time() - t0:.0f} s")

    table = {}
    for wl, d in data.items():
        sides = d["sides"]
        orr = max(d["off_repeat"].values(), default=float("nan"))
        metrics.record(f"exact_repeat_maxdiff_off_{wl}", orr)
        if orr != 0.0:
            fails.append(f"{wl}: two off runs of the same chunk differ ({orr}): no baseline")
        med = {}
        for side, v in sides.items():
            runs = v["runs"]
            med[side] = {
                "device_ms": statistics.median(r["_device"] for r in runs),
                "experts_ms": statistics.median(_sum(r, "routed_experts_combine") for r in runs),
                "dispatch_ms": statistics.median(_sum(r, "sparse_dispatch_2d") for r in runs),
                "head_ms": statistics.median(_sum(r, "fused_head") for r in runs),
                "copy_ms": statistics.median(_sum(r, "expert_slot_copy") for r in runs),
                "device_ms_runs": [round(r["_device"], 2) for r in runs],
                "wall_ms": statistics.median(v["walls"]),
                "wall_ms_runs": [round(w, 1) for w in v["walls"]],
            }
            metrics.record(f"exact_wall_ms_{side}_{wl}", round(med[side]["wall_ms"], 1))
            metrics.record(f"exact_device_ms_{side}_{wl}", round(med[side]["device_ms"], 2))
            metrics.record(f"exact_experts_ms_{side}_{wl}", round(med[side]["experts_ms"], 2))
            metrics.record(f"exact_dispatch_ms_{side}_{wl}", round(med[side]["dispatch_ms"], 2))
            if side != "off":
                md = max(v["diff"].values(), default=float("nan"))
                mr = max(v["diff_repeat"].values(), default=float("nan"))
                metrics.record(f"exact_maxdiff_{side}_{wl}", md)
                metrics.record(f"exact_repeat_maxdiff_{side}_{wl}", mr)
                metrics.record(f"exact_copies_{side}_{wl}", d["plan_info"][side]["replicas"])
                if not v["diff"] or md != md:
                    fails.append(f"{wl} {side}: no layer compared")
                if mr != md:
                    fails.append(f"{wl} {side}: the repeat differs from the first run ({mr} vs {md})")
        per_layer = []
        r_off = sides["off"]["runs"][0]
        for L in sorted(k for k in r_off if k != "_device" and k is not None):
            row = {"layer": L, "off_experts_ms": round(r_off[L].get("routed_experts_combine", 0.0), 4)}
            for side in modes:
                rs = sides[side]["runs"][0]
                row[f"{side}_experts_ms"] = round(rs.get(L, {}).get("routed_experts_combine", 0.0), 4)
                row[f"{side}_maxdiff"] = sides[side]["diff"].get(L)
                row[f"{side}_plan"] = json.dumps(d["plans"][side].get(str(L), []))
            per_layer.append(row)
        table[wl] = {"median": med, "layers": per_layer, "plan_info": d.get("plan_info", {})}
        for side in modes:
            o, e = med["off"], med[side]
            print(
                f"P.55c {wl} {side}: device {o['device_ms']:.2f} -> {e['device_ms']:.2f} ms a chunk "
                f"({e['device_ms'] - o['device_ms']:+.2f}); wall {o['wall_ms']:.1f} -> {e['wall_ms']:.1f}; experts+combine {o['experts_ms']:.2f} -> "
                f"{e['experts_ms']:.2f}; dispatch {o['dispatch_ms']:.2f} -> {e['dispatch_ms']:.2f}; head "
                f"{o['head_ms']:.2f} -> {e['head_ms']:.2f}; copies in the window {e['copy_ms']:.2f} ms; "
                f"max |diff| vs off {max(sides[side]['diff'].values(), default=float('nan'))}"
            )
    metrics.record_json("exact_replicas", {"mesh": list(shape), "A_ms": A_ms, "B_ms": B_ms, "workloads": table})
    for f in fails:
        logger.error(f"P.55c: {f}")
    assert not fails, f"{len(fails)} P.55c checks failed: {fails[:5]}"
