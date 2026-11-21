# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.5 (OWNER 2026-11-21): an ASSESSMENT, tests only, never a serving switch. How much faster is the shipped 5120
single-request chunk at a 1M history without the per-chunk bf16 -> 4-bit history conversion (tt/history_gather._pack:
MB.4's ttnn.bringup.ckv_bfp4_rows, the served bf16 rows -> 320 B bfloat4_b records before every gather)?

Ported from the batching branch's PB.NC probe (48bb0ee8205), single-request part only (this branch has no packed path
and no PB.HG owned sends, so DS41-377's owned-send swap is not needed here: every gather reads ``_pack``'s return).

The steady chunk at history NOCONV_POS (default 1024000), PB.9C's ladder method (test_skip_positions._point: the real
previous chunk untimed first, testing/profile.warm_up, NOCONV_RUNS (10) timed runs with program-cache misses refused,
the median, one real-time profiled run), loaded as the gate's ladder loads it (the serving runtime's adapter.load_model,
max_seq the deployment context in whole chunks, layers 0-20 under DECODER_SKIP=decode). Two sides:
    A   today's path
    B   every history already held as 4-bit records: after one untimed collect run (prev + timed chunk: the slabs then
        hold every row any chunk of the probe reads), each cache's whole slab [0, R) is packed ONCE (the original
        _pack_rows) into a record buffer of its own, what a 4-bit served cache would hold; then history_gather._pack
        returns that buffer without launching anything (prefetch AND tail packs skipped). Gathers, attention and
        everything else unchanged. A 4-bit served cache would pay the tail's conversion at its compressor write
        instead (2 periods a chunk); B leaves that out too (the upper bound of the saving).
The pair A, B runs NOCONV_REPEAT (2) times in the one load.
Per side: harness wall, device ms (slowest chip), exposed host (wall - device), the pack's device ms (its programs) and
host ms (the wrapped _pack calls of a timed chunk, median), the history ops' device ms (prefetch + layer gathers,
packs included; "moves" = without the packs), the profile's history_gather section, and the last computing layer's
output A vs B (max |diff|: must be 0, the records are a function of each row).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=5120
        DS41_ENGRAM_TABLE=resident BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_no_conversion_probe.py"
"""

from __future__ import annotations

import importlib
import json
import os
import time
from collections import defaultdict

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden, misses_allowed, warm_up
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("NOCONV_POS", "NOCONV_RUNS", "NOCONV_REPEAT", "NOCONV_OUT")
BATCH_POS_VARS = ("NOCONV_POS",)
BATCH_LAYERS = "executed"  # (the ladder's load: layers 0-20 under decode)
POS = 1024000


def _knob(name: str, default):
    v = os.environ.get(name, "").strip()
    return type(default)(v) if v else default


def _hg():
    return importlib.import_module("models.demos.deepseek_v41_flash_d_p.tt.history_gather")  # (DS41-311)


class NoConv:
    """history_gather._pack wrapped. mode "a": today's (timed; its op ids kept when ``trace``); "collect": today's,
    every cache seen kept; "b": the cache's pre-packed records returned, nothing launched. prefetch / gather wrapped
    for their device-operation id ranges (``trace``)."""

    def __init__(self, monkeypatch, mesh):
        import ttnn

        self.ttnn, self.mesh, self.hg = ttnn, mesh, _hg()
        self.mode, self.count, self.trace = "a", False, False
        self.seen, self.pre = {}, {}
        self.host_ns, self.calls = 0, 0
        self.ids = defaultdict(list)  # kind -> [(id0, id1)]: pack / prefetch / gather
        self.orig_pack, self.orig_rows = self.hg._pack, self.hg._pack_rows
        monkeypatch.setattr(self.hg, "_pack", self._pack)
        for name in ("prefetch", "gather"):
            monkeypatch.setattr(self.hg, name, self._traced(name, getattr(self.hg, name)))

    def _traced(self, kind, orig):
        w = self

        def f(*a, **k):
            if not w.trace:
                return orig(*a, **k)
            i0 = w.ttnn._ttnn.get_device_operation_id()
            try:
                return orig(*a, **k)
            finally:
                w.ids[kind].append((i0, w.ttnn._ttnn.get_device_operation_id()))

        return f

    def _pack(self, ckv, lo, hi):
        t0 = time.perf_counter_ns()
        i0 = self.ttnn._ttnn.get_device_operation_id()
        try:
            if self.mode == "b":
                hit = self.pre.get(id(ckv))
                assert hit is not None, ("MB.5 side B: a cache the collect run never packed", id(ckv), lo, hi)
                return hit.tensor
            if self.mode == "collect":
                self.seen[id(ckv)] = ckv
            return self.orig_pack(ckv, lo, hi)
        finally:
            if self.count:
                self.host_ns += time.perf_counter_ns() - t0
                self.calls += 1
            if self.trace:
                self.ids["pack"].append((i0, self.ttnn._ttnn.get_device_operation_id()))

    def prepack(self) -> int:
        """Every collected cache's whole slab -> 320 B records in a buffer of its own (once, untimed)."""
        n = 0
        for k, ckv in self.seen.items():
            if k in self.pre:
                continue
            R = int(ckv.slab.rows)
            buf = self.hg.HistoryBuffer(self.mesh, R, None, True)
            self.orig_rows(ckv, buf, 0, R)  # (the records of every row: what a 4-bit served cache would hold)
            self.pre[k] = buf
            n += 1
        self.ttnn.synchronize_device(self.mesh)
        return n

    def free(self) -> None:
        for b in self.pre.values():
            b.free()
        self.pre.clear()
        self.seen.clear()
        self.mode = "a"


def _dev_ms(recs: dict, d2c: dict, ranges: list, minus: list = ()) -> float:
    """ms: the slowest chip's summed program time over the device-operation id ranges (nested counted once), less
    the ids of ``minus``."""
    ids = set()
    for a, b in ranges:
        ids.update(range(a, b))
    for a, b in minus:
        ids.difference_update(range(a, b))
    per = defaultdict(float)
    for i in ids:
        for d, e in (recs.get(i) or {}).items():
            per[d2c.get(int(d), int(d))] += e[0]
    return max(per.values()) / 1e6 if per else 0.0


@mesh_parametrize(timeout_s=4 * 3600)
def test_no_conversion_probe(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode"
    assert _hg().bfp4_on(), "MB.5 measures the 4-bit history path (HISTORY_FORMAT=bfp4)"
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    from models.demos.deepseek_v41_flash_d_p.tests import ladder_common as LC
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as TSP

    chunk = int(S.get("target.chunk"))
    p = _knob("NOCONV_POS", POS)
    runs, repeat = _knob("NOCONV_RUNS", 10), _knob("NOCONV_REPEAT", 2)
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    want = max(-(-deploy // chunk) * chunk, int(S.data["target"]["seq"]))  # (the gate ladder's load)
    monkeypatch.setitem(S.data["target"], "seq", want)
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(1 + LC.SPLIT_RUNS))
    assert p % 32 == 0 and p >= chunk and p + chunk <= want, (p, want)
    rt = TSP._runtime(mesh_device, layers, chunk, want)
    model = rt[0].model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    last = max(j for j in layers if j not in model.kv_only)  # (layer 20 is KV-only)
    ids = PositionGolden(S, p, chunk).tokens()
    prev_t, cur_t = ids[p - chunk : p], ids[p : p + chunk]
    state = model.new_state(p + chunk)
    W = NoConv(monkeypatch, mesh_device)
    label = f"single{chunk}"
    out = {"position": p, "chunk": chunk, "runs": []}

    def run(tokens, start, section=False, keep=False):
        if section:
            profiler.signpost("chunk_input")
        model.begin_chunk(tokens, start, last=False)
        h = model.embed(tokens)
        kept = None
        for i in layers:
            h2 = model.layer(i, h, start, state)
            if keep and i == last:
                kept = model.to_host(h2).float()
            model.free(h)
            h = h2
        model.free(h)
        return kept

    def one_run():
        run(prev_t, p - chunk)  # the real previous chunk, untimed
        model.sync()
        W.count, W.host_ns, W.calls = True, 0, 0
        t0 = time.time()
        try:
            run(cur_t, p)
            model.sync()
        finally:
            W.count = False
        hosts.append(W.host_ns / 1e6)
        pcalls.append(W.calls)
        return round((time.time() - t0) * 1e3, 2)

    def measure(side: str) -> tuple[dict, object]:
        w = warm_up(mesh_device, one_run)
        assert w["warm"], (side, w["entries"][-5:])
        hosts.clear()
        pcalls.clear()
        walls = []
        misses_allowed(mesh_device, False)  # F196: every timed run hit the program cache
        try:
            for _ in range(runs):
                walls.append(one_run())
        finally:
            misses_allowed(mesh_device, True)
        kept = sorted(walls)
        wall = kept[len(kept) // 2]
        host_pack = sorted(hosts)[len(hosts) // 2]
        calls = sorted(pcalls)[len(pcalls) // 2]
        # one real-time profiled run: device, sections, the pack's programs and the history ops
        run(prev_t, p - chunk)
        model.sync()
        W.ids.clear()
        W.trace = True
        profiler.enable(mesh_device, rt=True)
        try:
            profiler.set_layer(None)
            run(cur_t, p, section=True)
            profiler.signpost("end")
            profiler.finish_rt()
            prof = profiler.result()
        finally:
            W.trace = False
            profiler.disable()
        dev = TSP._device_split(prof)
        recs, d2c = prof.get("rt_records") or {}, prof.get("dev_to_chip") or {}
        hist_r = W.ids["prefetch"] + W.ids["gather"]
        r = {
            "wall": wall,
            "runs": walls,
            "spread": round(kept[-1] - kept[0], 2),
            "warm_runs": len(w["walls"]),
            "device": round(dev["slowest"], 2),
            "host_exposed": round(wall - dev["slowest"], 2),
            "pack_device": round(_dev_ms(recs, d2c, W.ids["pack"]), 3),
            "pack_host": round(host_pack, 3),
            "pack_calls": calls,
            "hist_ops_device": round(_dev_ms(recs, d2c, hist_r), 3),
            "hist_moves_device": round(_dev_ms(recs, d2c, hist_r, W.ids["pack"]), 3),
            "hist_prefetch_device": round(_dev_ms(recs, d2c, W.ids["prefetch"]), 3),
            "hist_gather_device": round(_dev_ms(recs, d2c, W.ids["gather"]), 3),
            "hist_section": round(dev["phases"].get("history_gather", 0.0), 3),
            "phases": {k: round(v, 3) for k, v in dev["phases"].items()},
        }
        r["tok_s"] = round(chunk / (wall / 1e3), 1)
        print(
            f"NOCONV {label} p{p} {side}: wall {wall} ms (runs {walls}, spread {r['spread']}), device {r['device']} ms, "
            f"exposed host {r['host_exposed']} ms; pack device {r['pack_device']} ms / host {r['pack_host']} ms "
            f"({calls} calls); history ops device {r['hist_ops_device']} ms (moves {r['hist_moves_device']}, prefetch "
            f"{r['hist_prefetch_device']}, layer gathers {r['hist_gather_device']}), history_gather section "
            f"{r['hist_section']} ms; phases {json.dumps(r['phases'])}",
            flush=True,
        )
        # the output (same tokens, same state rows on both sides)
        run(prev_t, p - chunk)
        o = run(cur_t, p, keep=True)
        return r, o

    hosts, pcalls = [], []
    try:
        for k in range(1, repeat + 1):
            res = {}
            W.mode = "a"
            res["a"], oa = measure(f"run {k} A (today: per-chunk pack)")
            # side B: collect every cache the chunks pack, pre-pack each slab once, then skip the pack
            W.mode = "collect"
            run(prev_t, p - chunk)
            run(cur_t, p)
            model.sync()
            n = W.prepack()
            W.mode = "b"
            print(f"NOCONV {label} run {k}: {n} caches pre-packed as 4-bit records (once, untimed)", flush=True)
            try:
                res["b"], ob = measure(f"run {k} B (history held as 4-bit records)")
            finally:
                W.free()
            diff = float((oa - ob).abs().max())
            res["max_abs_diff"] = diff
            res["prepacked_caches"] = n
            a, b = res["a"], res["b"]
            res["saved_ms"] = round(a["wall"] - b["wall"], 2)
            res["saved_device_ms"] = round(a["device"] - b["device"], 2)
            res["saved_host_exposed_ms"] = round(a["host_exposed"] - b["host_exposed"], 2)
            for side in ("a", "b"):
                r = res[side]
                pre = f"noconv_{label}_{side}"
                for key, v in (
                    ("wall_ms", r["wall"]),
                    ("runs", r["runs"]),
                    ("device_ms", r["device"]),
                    ("host_exposed_ms", r["host_exposed"]),
                    ("pack_device_ms", r["pack_device"]),
                    ("pack_host_ms", r["pack_host"]),
                    ("hist_ops_device_ms", r["hist_ops_device"]),
                    ("hist_section_ms", r["hist_section"]),
                    ("tok_s", r["tok_s"]),
                ):
                    LC.record(f"{pre}_{key}_p{p}_r{k}", v)
            LC.record(f"noconv_{label}_max_abs_diff_p{p}_r{k}", diff)
            LC.record(f"noconv_{label}_saved_ms_p{p}_r{k}", res["saved_ms"])
            print(
                f"NOCONV {label} p{p} run {k}: A {a['wall']} ms ({a['tok_s']} tok/s) vs B {b['wall']} ms "
                f"({b['tok_s']} tok/s): saved {res['saved_ms']} ms a chunk (device {res['saved_device_ms']}, exposed "
                f"host {res['saved_host_exposed_ms']}); A vs B output max |diff| {diff}",
                flush=True,
            )
            out["runs"].append(res)
            metrics.record(f"noconv_{label}_detail_r{k}", json.dumps(res)[:200000])
    finally:
        W.free()
        state.free()
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    dst = os.environ.get("NOCONV_OUT", "/tmp/mb5_noconv.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"NOCONV probe written: {dst}")
    for k, res in enumerate(out["runs"], 1):
        assert res["max_abs_diff"] == 0.0, (k, res["max_abs_diff"])
        assert res["a"]["pack_calls"] > 0 and res["a"]["pack_device"] > 0, ("side A packed nothing", k, res["a"])
        assert res["b"]["pack_device"] == 0.0, ("side B launched a pack", k, res["b"]["pack_device"])
