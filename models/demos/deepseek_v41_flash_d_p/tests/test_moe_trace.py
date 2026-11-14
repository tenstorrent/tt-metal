# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.61 (overseer 2026-11-14, owner away): MOE_TRACE=moe, the MoE half of every layer replayed from a Metal trace
(tt/moe_trace.py), against the same model untraced. One process, one model load with MOE_TRACE=moe; untraced = the
blocks' traces switched off in the loaded model (``_traced(model, False)``: every MoE half eager, attn_residual still
writing the same buffers; ``SegmentTrace.enabled``), so both sides run the same programs on the same buffers.

Terms:
  host_ms   first host call of a chunk (begin_chunk) to the last layer's enqueue (before the closing sync).
  wall_ms   first host call to the end of the closing sync (the clean wall: nothing profiled).
  device_ms the program real-time profiler, a separate pass: each program's time on its slowest chip, summed over the
            chunk's programs (traced programs report under their capture-time ids: every record of the window is
            counted, not only the window's ids).
  c1-10     the mean of chunks 1..10 of a fresh request (chunks 0..10 of the prompt, 56,320 tokens, back to back).
  moe_host_ms  host time inside the MoE halves (the segment's call: eager steps, or guard + replay + bookkeeping).
  maxdiff   the 16-bit words of each chunk's last-layer output streams that differ, traced vs untraced (0 = bit-identical).
Per workload (novel = the ladder's prompt, code = P.45's) and side (off = MOE_REPLICAS=prev switched off in the
loaded model, prev = P.56 on): untraced warm + clean + profiled passes, then traced warm (eager calls, then a capture
a layer) + clean + profiled passes. At 51,200 (random ids, fresh state, the profile's steady_state): warm until no
new program, CLEAN clean runs a side, one profiled run a side, prev on. Also: the DRAM a trace keeps a chip (the
buffers its capture allocated; the command buffer apart), the capture time a layer, the replays.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_trace.py
P61_QUICK=1 (iterating only): 4 layers, chunks 0..3, novel, prev on and off, no 51,200. P61_ONLY=<workload>:<side>
(diagnostic): that pass only, no 51,200.
"""

from __future__ import annotations

import os
import statistics
import time

import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free

S = spec()
QUICK = os.environ.get("P61_QUICK") == "1"
SEQ = 56320
CHUNKS = 4 if QUICK else 11
WORKLOADS = ("novel",) if QUICK else ("novel", "code")
SIDES = ("prev", "off")
if os.environ.get("P61_ONLY"):  # (diagnostic: one workload:side, e.g. novel:prev, and no 51,200)
    WORKLOADS, SIDES = (os.environ["P61_ONLY"].split(":")[0],), (os.environ["P61_ONLY"].split(":")[1],)
POS = 51200
CLEAN = 5


def _traced(model, on: bool) -> None:
    """Every block's MoE-half trace switched on / off in the loaded model (its captured traces kept; off runs the same
    segment wrapper eager, so the MoE half's host time is measured alike on both sides)."""
    if not hasattr(model, "_p61_segs"):
        model._p61_segs = {i: b.seg for i, b in model.blocks.items() if b.seg is not None}
    for seg in model._p61_segs.values():
        seg.enabled = on


_SEG_S = [0.0]  # host seconds inside the MoE halves (SegmentTrace.__call__) since the last reset


def _time_segments(monkeypatch) -> None:
    from models.demos.deepseek_v41_flash_d_p.tt import moe_trace as MT

    orig = MT.SegmentTrace.__call__

    def call(self, *a, **k):
        t0 = time.perf_counter()
        out = orig(self, *a, **k)
        _SEG_S[0] += time.perf_counter() - t0
        return out

    monkeypatch.setattr(MT.SegmentTrace, "__call__", call)


def _rt_begin(mesh):
    profiler.enable(mesh, rt=True)


def _trace_rids(model) -> set:
    """The 16-bit runtime ids of every captured MoE-half program (their replays report these)."""
    out = set()
    for seg in getattr(model, "_p61_segs", {}).values():
        for r in seg.traces.values():
            out.update(i % profiler.RT_ID_SPAN for i in range(*r.ids))
    return out


def _rt_end(mesh, model=None) -> dict:
    """Every real-time record since _rt_begin. device_ms: each runtime id's slowest chip, summed. Per chip (the mean
    over chips): span_ms = first program start to last program end; idle_ms = span - the chip's program time; the idle
    gaps by what they sit between (programs in start order, a gap = a start after every earlier program ended):
    gap_entry_ms (eager -> a replayed trace's program), gap_exit_ms (trace -> eager), gap_eager_ms, gap_trace_ms."""
    ttnn.synchronize_device(mesh)
    time.sleep(1.0)
    recs = profiler._P["rt_records"]
    if profiler._P.get("rt_handle") is not None:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(profiler._P["rt_handle"])
        profiler._P["rt_handle"] = None
    drop = profiler._P.get("rt_dropped", [0])[0]
    profiler.disable()
    assert not drop, f"the real-time profiler dropped {drop} records"
    out = {"device_ms": sum(max(e[0] for e in chips.values()) / 1e6 for chips in recs.values() if chips)}
    tr = _trace_rids(model) if model is not None else set()
    per: dict = {}
    for rid, chips in recs.items():
        for c, e in chips.items():
            per.setdefault(c, []).append((e[1], e[2], e[0], int(rid) in tr))
    acc = {k: [] for k in ("span_ms", "idle_ms", "gap_entry_ms", "gap_exit_ms", "gap_eager_ms", "gap_trace_ms")}
    for progs in per.values():
        progs.sort()
        g = {"entry": 0.0, "exit": 0.0, "eager": 0.0, "trace": 0.0}
        end, prev_tr = progs[0][1], progs[0][3]
        for st, en, _, is_tr in progs[1:]:
            if st > end:
                kind = ("trace" if is_tr else "exit") if prev_tr else ("entry" if is_tr else "eager")
                g[kind] += st - end
            end, prev_tr = max(end, en), is_tr
        span = max(e for _, e, _, _ in progs) - progs[0][0]
        acc["span_ms"].append(span / 1e6)
        acc["idle_ms"].append((span - sum(p[2] for p in progs)) / 1e6)
        for k, v in g.items():
            acc[f"gap_{k}_ms"].append(v / 1e6)
    for k, v in acc.items():
        out[k] = statistics.mean(v) if v else 0.0
    return out


def _chunk(model, layers, t, st, state):
    begin_chunk(model, t, st)
    h = model.embed(t)
    for i in layers:
        h2 = model.layer(i, h, st, state)
        model.free(h)
        h = h2
    return h


def _host(h) -> torch.Tensor:
    return torch.cat([ttnn.to_torch(d).contiguous().view(torch.int16).reshape(-1) for d in ttnn.get_device_tensors(h)])


def _pass(model, mesh, layers, tokens, kind: str, ref: dict | None = None) -> list:
    """Chunks 0..CHUNKS-1 of a fresh request. kind: warm | clean (host / wall) | prof (device). ref: {chunk: host
    bytes of the last layer's streams} filled (None in it) or compared."""
    chunk = int(S.get("target.chunk"))
    state = model.new_state(SEQ)
    rows = []
    try:
        for c in range(CHUNKS):
            st = c * chunk
            t = tokens[st : st + chunk]
            row: dict = {"chunk": c}
            if kind == "prof":
                _rt_begin(mesh)
            else:
                ttnn.synchronize_device(mesh)
            _SEG_S[0] = 0.0
            t0 = time.perf_counter()
            h = _chunk(model, layers, t, st, state)
            t1 = time.perf_counter()
            row["moe_host_ms"] = _SEG_S[0] * 1e3
            if kind == "prof":
                row.update(_rt_end(mesh, model))
            else:
                ttnn.synchronize_device(mesh)
                row["host_ms"] = (t1 - t0) * 1e3
                row["wall_ms"] = (time.perf_counter() - t0) * 1e3
            if ref is not None and kind == "clean":
                got = _host(h.streams)
                if c not in ref:
                    ref[c] = got
                else:
                    row["maxdiff"] = int((got != ref[c]).sum())  # (differing 16-bit words: 0 = bit-identical)
            model.free(h)
            rows.append(row)
            logger.info(f"P.61 {kind} chunk {c}: {row}")
    finally:
        _free(state)
    return rows


def _mean(rows, k):
    vals = [r[k] for r in rows if r["chunk"] >= 1 and k in r]
    return statistics.mean(vals) if vals else float("nan")


def _position(model, mesh, layers) -> dict:
    """One warm chunk at POS (random ids, fresh state, steady_state), untraced and traced, prev on."""
    from models.demos.common.bringup.testing import profile as P

    chunk = int(S.get("target.chunk"))
    vocab = int(S.get("checkpoint.config.vocab_size", 32000) or 32000)
    tokens = torch.randint(0, vocab, (POS + chunk,), generator=torch.Generator().manual_seed(0))[POS:]
    state = model.new_state(POS + chunk)
    out: dict = {}
    ref = None
    try:
        for side in ("untraced", "traced"):
            _traced(model, side == "traced")
            n, quiet = P.cache_entries(mesh), 0
            for k in range(20):  # (no new program twice in a row; the traced side: its warm calls and captures)
                model.steady_state(POS)
                h = _chunk(model, layers, tokens, POS, state)
                model.free(h)
                ttnn.synchronize_device(mesh)
                m = P.cache_entries(mesh)
                quiet = quiet + 1 if m == n else 0
                n = m
                if quiet >= 2 and k >= 4:
                    break
            hosts, walls = [], []
            for _ in range(CLEAN):
                model.steady_state(POS)
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                h = _chunk(model, layers, tokens, POS, state)
                t1 = time.perf_counter()
                ttnn.synchronize_device(mesh)
                walls.append((time.perf_counter() - t0) * 1e3)
                hosts.append((t1 - t0) * 1e3)
                got = _host(h.streams)
                model.free(h)
            if ref is None:
                ref = got
            else:
                out["maxdiff"] = int((got != ref).sum())
            model.steady_state(POS)
            _rt_begin(mesh)
            h = _chunk(model, layers, tokens, POS, state)
            dev = _rt_end(mesh, model)
            model.free(h)
            out[side] = {
                "host_ms": statistics.median(hosts),
                "wall_ms": statistics.median(walls),
                "wall_spread": max(walls) - min(walls),
                **dev,
            }
            logger.info(f"P.61 p{POS} {side}: {out[side]}")
    finally:
        _free(state)
    return out


@mesh_parametrize(timeout_s=3 * 3600)
def test_moe_trace(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_prev_replicas import _ids, _side
    from models.demos.deepseek_v41_flash_d_p.tt import moe_trace as MT
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_TRACE", "moe")
    monkeypatch.setenv("DS41_MOE_REPLICAS", "prev")
    monkeypatch.setenv("DS41_MOE_FILL", "piggyback")
    monkeypatch.setenv("DS41_ENGRAM_TABLE", "resident")
    settings.thaw()
    layers = S.layers()[:4] if QUICK else S.layers()
    _time_segments(monkeypatch)
    t0 = time.time()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    logger.info(f"P.61 model loaded in {time.time() - t0:.0f} s")
    _traced(model, True)
    assert model._p61_segs, "MOE_TRACE=moe: no layer's MoE half can trace"
    data: dict = {"layers_traced": sorted(model._p61_segs)}
    fails = []
    for wl in WORKLOADS:
        tokens = _ids(wl)
        for side in SIDES:
            _side(mesh_device, side)
            ref: dict = {}
            d: dict = {}
            _traced(model, False)
            _pass(model, mesh_device, layers, tokens, "warm")
            d["untraced_clean"] = _pass(model, mesh_device, layers, tokens, "clean", ref)
            d["untraced_prof"] = _pass(model, mesh_device, layers, tokens, "prof")
            _traced(model, True)
            caps = MT.STATS["captures"]
            d["traced_warm"] = _pass(model, mesh_device, layers, tokens, "warm")
            d["captures"] = MT.STATS["captures"] - caps
            d["traced_clean"] = _pass(model, mesh_device, layers, tokens, "clean", ref)
            d["traced_prof"] = _pass(model, mesh_device, layers, tokens, "prof")
            d["traced_clean2"] = _pass(model, mesh_device, layers, tokens, "clean", ref)
            s = {
                k: round(_mean(d[p], m), 2)
                for k, p, m in (
                    ("untraced_host_ms", "untraced_clean", "host_ms"),
                    ("untraced_wall_ms", "untraced_clean", "wall_ms"),
                    ("untraced_moe_host_ms", "untraced_clean", "moe_host_ms"),
                    ("traced_moe_host_ms", "traced_clean", "moe_host_ms"),
                    ("untraced_device_ms", "untraced_prof", "device_ms"),
                    ("traced_host_ms", "traced_clean", "host_ms"),
                    ("traced_wall_ms", "traced_clean", "wall_ms"),
                    ("traced_device_ms", "traced_prof", "device_ms"),
                    ("traced_wall_ms_repeat", "traced_clean2", "wall_ms"),
                )
            }
            for k in ("span_ms", "idle_ms", "gap_entry_ms", "gap_exit_ms", "gap_eager_ms", "gap_trace_ms"):
                s[f"untraced_{k}"] = round(_mean(d["untraced_prof"], k), 2)
                s[f"traced_{k}"] = round(_mean(d["traced_prof"], k), 2)
            md = max(r.get("maxdiff", 0) for r in d["traced_clean"] + d["traced_clean2"])
            s["maxdiff_words"] = md
            s["captures"] = d["captures"]
            if md:
                fails.append(f"{wl} {side}: traced differs from untraced ({md} words)")
            d["mean_1_10"] = s
            for k, v in s.items():
                metrics.record(f"p61_{k}_{side}_{wl}", v)
            data[f"{wl}_{side}"] = d
            print(f"P.61 {wl} {side} c1-{CHUNKS - 1}: {s}")
    if not QUICK and not os.environ.get("P61_ONLY"):
        _side(mesh_device, "prev")
        data[f"p{POS}"] = pos = _position(model, mesh_device, layers)
        for side in ("untraced", "traced"):
            for k, v in pos[side].items():
                metrics.record(f"p61_{k}_{side}_p{POS}", round(v, 2))
        if pos.get("maxdiff"):
            fails.append(f"p{POS}: traced differs from untraced ({pos['maxdiff']} words)")
        print(f"P.61 p{POS}: {pos}")
    mem = {}
    for i, seg in model._p61_segs.items():
        for key, b in seg.memory().items():
            mem.setdefault(i, []).append(round(b / 2**20, 3))
    caps = [r.capture_s for seg in model._p61_segs.values() for r in seg.traces.values()]
    data["trace_mib_per_layer"] = mem
    data["capture_ms"] = {"median": 1e3 * statistics.median(caps), "max": 1e3 * max(caps), "n": len(caps)}
    data["stats"] = dict(MT.STATS)
    per = [v for vs in mem.values() for v in vs]
    metrics.record("p61_trace_mib_layer_median", round(statistics.median(per), 3))
    metrics.record("p61_trace_mib_total", round(sum(per), 1))
    metrics.record("p61_capture_ms_median", round(data["capture_ms"]["median"], 1))
    print(f"P.61 trace memory MiB a chip a layer (per key): {mem}; capture ms {data['capture_ms']}; {MT.STATS}")
    metrics.record_json("moe_trace", {"mesh": list(mesh_device.shape), "quick": QUICK, "data": data})
    for f in fails:
        logger.error(f"P.61: {f}")
    assert not fails, fails
