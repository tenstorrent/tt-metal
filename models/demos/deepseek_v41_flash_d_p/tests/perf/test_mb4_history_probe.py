# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.4 (OWNER 2026-11-21): where the 5120 ladder's long-history slowdown comes from. A measurement (no model change).

At each history p of MB4_POS (default 51200, 524288, 786432, 1024000) the ladder's STEADY chunk (test_skip_positions'
_point: the real previous chunk untimed first, then the timed chunk; same ids, same state) on both sides of the
call-time switch POS_DYNAMIC (on / off) in ONE load (loaded as the gate's ladder loads it: the serving runtime's
adapter.load_model, max_seq the deployment context in whole chunks). Per side:

    wall          warm_up, then MB4_RUNS timed runs with program-cache misses refused, the median (as the ladder)
    host items    MB4_HOST_RUNS more runs with the history-path functions wrapped (perf_counter on the main thread):
                  history_gather prefetch / gather / _full / _pack / _tail, index_keys prefetch / keys / _decode,
                  page_copy.page_copy, RoPE / compressed-RoPE rows, DeviceScalars._pick, CacheSlab.slice_rows,
                  WindowTables.srv_row: host ms a chunk (median), calls a chunk
    device        one real-time profiled run with op rows (profiler ops=True, rt: no syncs): the slowest chip's total,
                  the sections, each wrapped function's device ms (its device-operation id range: per chip summed,
                  the slowest chip), and the op rows (section, op) whose device ms differ on vs off by > 0.03 ms
    bit identity  the chunk's output on vs off (to_host), max |diff|

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=5120
        DS41_ENGRAM_TABLE=resident BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_mb4_history_probe.py"
"""

from __future__ import annotations

import importlib
import json
import os
import re
import time
from collections import defaultdict

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden, cache_entries, misses_allowed, warm_up
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("MB4_POS", "MB4_RUNS", "MB4_HOST_RUNS", "MB4_SIDES", "MB4_OUT", "WALLS_FRESH_SLOTS")
BATCH_POS_VARS = ("MB4_POS",)
BATCH_LAYERS = "executed"
POINTS = (51200, 524288, 786432, 1024000)


def _knob(name: str, default):
    v = os.environ.get(name, "").strip()
    return type(default)(v) if v else default


def _side(monkeypatch, v: str) -> None:
    monkeypatch.setenv(settings.env_name("POS_DYNAMIC"), v)
    if settings.snap is not None:
        settings.snap["POS_DYNAMIC"] = v
    assert settings.get("POS_DYNAMIC") == v


class Wraps:
    """The history path's functions wrapped: host ns and device-operation id ranges per call, per name."""

    TARGETS = [
        ("history_gather", None, "prefetch"),
        ("history_gather", None, "gather"),
        ("history_gather", None, "_full"),
        ("history_gather", None, "_pack"),
        ("history_gather", None, "_pack_rows"),
        ("history_gather", None, "_tail"),
        ("index_keys", None, "prefetch"),
        ("index_keys", None, "keys"),
        ("index_keys", None, "_decode"),
        ("page_copy", None, "page_copy"),
        ("page_copy", "IdsReorder", "__call__"),
        ("rope", "RopeTable", "_rows"),
        ("compressor", "CompressRope", "_rows"),
        ("compressor", "TtCompressor", "_write_from"),
        ("served", "DeviceScalars", "_pick"),
        ("served", "CacheSlab", "slice_rows"),
        ("attention", "WindowTables", "srv_row"),
    ]

    def __init__(self, monkeypatch):
        import ttnn

        self.ttnn = ttnn
        self.on = False
        self.host = defaultdict(float)
        self.calls = defaultdict(int)
        self.ids = defaultdict(list)
        for mod, cls, fn in self.TARGETS:
            m = importlib.import_module(f"models.demos.deepseek_v41_flash_d_p.tt.{mod}")
            owner = getattr(m, cls) if cls else m
            orig = getattr(owner, fn, None)
            if orig is None:
                print(f"MB4 probe: no {mod}.{cls or ''}.{fn}", flush=True)
                continue
            name = f"{mod}.{cls + '.' if cls else ''}{fn}"
            monkeypatch.setattr(owner, fn, self._wrap(name, orig))

    def _wrap(self, name, orig):
        w = self

        def f(*a, **k):
            if not w.on:
                return orig(*a, **k)
            i0 = w.ttnn._ttnn.get_device_operation_id()
            t0 = time.perf_counter_ns()
            try:
                return orig(*a, **k)
            finally:
                w.host[name] += time.perf_counter_ns() - t0
                w.calls[name] += 1
                w.ids[name].append((i0, w.ttnn._ttnn.get_device_operation_id()))

        return f

    def reset(self):
        self.host.clear()
        self.calls.clear()
        self.ids.clear()


def _dev_of(recs: dict, dev_to_chip: dict, ranges: list) -> float:
    """ms: the slowest chip's summed program time over the id ranges (nested ranges counted once)."""
    ids = set()
    for a, b in ranges:
        ids.update(range(a, b))
    per = defaultdict(float)
    for i in ids:
        for d, e in recs.get(i, {}).items():
            per[dev_to_chip.get(int(d), int(d))] += e[0]
    return max(per.values()) / 1e6 if per else 0.0


def _op_rows(res: dict) -> dict:
    """(section without the layer prefix, op) -> device ms summed over layers (slowest chip a call, as op rows)."""
    out = defaultdict(float)
    for key, rows in res.get("op_ns", {}).items():
        sec = re.sub(r"^L\d+\.", "", key or "")
        for r in rows:
            out[f"{sec} | {r['op']}"] += r["ns"] / 1e6
    return dict(out)


@mesh_parametrize(timeout_s=4 * 3600)
def test_mb4_history_probe(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode"
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    from models.demos.deepseek_v41_flash_d_p.tests import ladder_common as LC
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as TSP

    chunk = int(S.get("target.chunk"))
    pts = [int(x) for x in os.environ.get("MB4_POS", ",".join(map(str, POINTS))).split(",") if x.strip()]
    runs, hruns = _knob("MB4_RUNS", 10), _knob("MB4_HOST_RUNS", 5)
    sides = [s for s in os.environ.get("MB4_SIDES", "on,off").split(",") if s]
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    want = max(-(-deploy // chunk) * chunk, int(S.data["target"]["seq"]))  # (the gate ladder's load: serving column)
    monkeypatch.setitem(S.data["target"], "seq", want)
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(1 + LC.SPLIT_RUNS))
    rt = TSP._runtime(mesh_device, layers, chunk, want)
    model = rt[0].model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    W = Wraps(monkeypatch)
    out = {}
    try:
        for p in pts:
            g = PositionGolden(S, p, chunk)
            ids = g.tokens()
            prev_t, cur_t = ids[p - chunk : p], ids[p : p + chunk]
            state = model.new_state(p + chunk)
            res_p = {}
            try:

                def run(tokens, start, section=False, keep=False):
                    if section:
                        profiler.signpost("chunk_input")
                    model.begin_chunk(tokens, start, last=False)
                    h = model.embed(tokens)
                    kept = None
                    for i in layers:
                        h2 = model.layer(i, h, start, state)
                        if keep and i == max(j for j in layers if j not in model.kv_only):  # (layer 20 is KV-only)
                            kept = model.to_host(h2).float()
                        model.free(h)
                        h = h2
                    model.free(h)
                    return kept

                def one_run():
                    run(prev_t, p - chunk)
                    model.sync()
                    t0 = time.time()
                    run(cur_t, p)
                    model.sync()
                    return round((time.time() - t0) * 1e3, 2)

                outs = {}
                for side in sides:
                    _side(monkeypatch, side)
                    w = warm_up(mesh_device, one_run)
                    assert w["warm"], (p, side, w["entries"][-5:])
                    walls = []
                    misses_allowed(mesh_device, False)
                    try:
                        for _ in range(runs):
                            walls.append(one_run())
                    finally:
                        misses_allowed(mesh_device, True)
                    wall = sorted(walls)[len(walls) // 2]
                    # host items
                    per_run = []
                    hwalls = []
                    for _ in range(hruns):
                        run(prev_t, p - chunk)
                        model.sync()
                        W.reset()
                        W.on = True
                        t0 = time.time()
                        run(cur_t, p)
                        model.sync()
                        hwalls.append(round((time.time() - t0) * 1e3, 2))
                        W.on = False
                        per_run.append({k: (W.host[k] / 1e6, W.calls[k]) for k in W.host})
                    host = {}
                    for k in per_run[0]:
                        v = sorted(r.get(k, (0.0, 0))[0] for r in per_run)
                        host[k] = (round(v[len(v) // 2], 3), per_run[0][k][1])
                    # device, with op rows and the wrapped ranges
                    run(prev_t, p - chunk)
                    model.sync()
                    n0 = cache_entries(mesh_device)
                    W.reset()
                    profiler.enable(mesh_device, rt=True, ops=True)
                    W.on = True
                    try:
                        profiler.set_layer(None)
                        run(cur_t, p, section=True)
                        profiler.signpost("end")
                        profiler.finish_rt()
                        res = profiler.result()
                    finally:
                        W.on = False
                        profiler.disable()
                    n1 = cache_entries(mesh_device)
                    dev = TSP._device_split(res)
                    recs, d2c = res["rt_records"], res["dev_to_chip"]
                    wdev = {k: round(_dev_of(recs, d2c, v), 3) for k, v in W.ids.items()}
                    # bit identity: the chunk's output kept
                    run(prev_t, p - chunk)
                    outs[side] = run(cur_t, p, keep=True)
                    r = {
                        "wall": wall,
                        "walls": walls,
                        "host_walls": hwalls,
                        "device": round(dev["slowest"], 2),
                        "phases": {k: round(v, 2) for k, v in dev["phases"].items()},
                        "subs": {k: round(v, 2) for k, v in dev["subs"].items()},
                        "programs": dev["programs"],
                        "new_programs_profiled": None if n0 is None or n1 is None else n1 - n0,
                        "host": host,
                        "wrapped_device": wdev,
                        "ops": {k: round(v, 3) for k, v in _op_rows(res).items()},
                    }
                    res_p[side] = r
                    print(
                        f"MB4 p{p} POS_DYNAMIC={side}: wall {wall} ms (runs {walls}), device {r['device']} ms, "
                        f"exposed host {wall - r['device']:.2f} ms; host items {json.dumps(host)}; wrapped device "
                        f"{json.dumps(wdev)}; phases {json.dumps(r['phases'])}",
                        flush=True,
                    )
                if len(outs) == 2:
                    a, b = outs[sides[0]], outs[sides[1]]
                    res_p["on_off_max_abs_diff"] = float((a - b).abs().max())
                    print(f"MB4 p{p} on vs off output max |diff| {res_p['on_off_max_abs_diff']}", flush=True)
                if len(sides) == 2:
                    oa, ob = res_p[sides[0]]["ops"], res_p[sides[1]]["ops"]
                    diff = {
                        k: (oa.get(k, 0.0), ob.get(k, 0.0))
                        for k in set(oa) | set(ob)
                        if abs(oa.get(k, 0.0) - ob.get(k, 0.0)) > 0.03
                    }
                    res_p["op_diff"] = dict(sorted(diff.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1])))
                    print(f"MB4 p{p} op rows differing ({sides[0]}, {sides[1]}) ms: {json.dumps(res_p['op_diff'])}")
            finally:
                state.free()
            out[p] = res_p
            metrics.record(f"mb4_probe_p{p}", json.dumps(res_p)[:200000])
    finally:
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    dst = os.environ.get("MB4_OUT", "/tmp/mb4_probe.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"MB4 probe written: {dst}")
