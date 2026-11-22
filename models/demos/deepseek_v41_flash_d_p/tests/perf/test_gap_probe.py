# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.GAP (OWNER 2026-11-22): the cost of a chip-aligned 4 x 2k chunk over ONE single 8192 chunk (same FLOPs), item
by item, at GAP_POS (139264,1024000), steady, the whole executed layer stack (layers 0-20 under decode).

Per (position, path): warm-up until no new program, 3 clean walls (with the host enqueue time: run_chunk's return
before the final sync), then ONE real-time profiled run (op rows: per (section, op) the slowest chip's device ms, the
calls and the host ms of the launches). Printed per section and per (section, op) single vs 4 x 2k side by side;
written to GAP_OUT (json) for the report (results/PB.GAP_ladder.md). The chunk method is PB.KR's test_kr_probe
(the real previous chunk untimed first; the 4 x 2k segments rotate their order every run).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "<the walls env> \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_gap_probe.py"
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
from models.demos.common.bringup.testing.profile import PositionGolden, warm_up
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request)

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = (
    "GAP_POS",
    "GAP_OUT",
    "GAP_PATHS",
    "GAP_WALLS",
    "GAP_CPROF",
    "GAP_PROF",
    "GAP_PP",
    "GAP_AB",
    "GAP_AB_KEY",
    "GAP_AB_POS",
)
BATCH_LAYERS = "executed"  # (the ladder's load: layers 0-20 under decode)
CHUNK = _pc.CHUNK
SHIFT = 4096
LENS = (2048, 2048, 2048, 2048)


def _positions() -> list[int]:
    return [int(x) for x in os.environ.get("GAP_POS", "139264,1024000").split(",") if x.strip()]


def _paths() -> list[str]:
    return [x for x in os.environ.get("GAP_PATHS", "single8192,4x2k").split(",") if x.strip()]


def _runner(runtime, kv, mesh, layers, text, path: str):
    """run(p, prev, rot, prof) -> (wall ms, host enqueue ms, profile or None) for ``path`` at history p."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as PD
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment

    model = runtime.model
    n = len(LENS)
    states = [runtime.state(kv, k) for k in range(n)]

    def run(p: int, prev: bool, rot: int, prof: bool = False):
        if path == "single8192":
            s = p - (CHUNK if prev else 0)
            desc = ChunkDescriptor.single(CHUNK, 0, s, s + CHUNK)
            ids, sts = [text[s : s + CHUNK].long()], [states[0]]
        else:
            order = [(k + rot) % n for k in range(n)]
            segs, ids, sts = [], [], []
            for k in order:
                s = p - (LENS[k] if prev else 0)
                segs.append(Segment(k, s, s + LENS[k], None))
                ids.append(text[k * SHIFT + s : k * SHIFT + s + LENS[k]].long())
                sts.append(states[k])
            desc = ChunkDescriptor.packed(CHUNK, segs)
        inp = PD.make_input(mesh, desc, ids)
        model.sync()
        if prof:
            profiler.enable(mesh, rt=True, ops=True)
            profiler.set_layer(None)
        res = None
        t0 = time.time()
        try:
            nat = PD.ids_from_input(mesh, inp, desc)
            PD.run_chunk(model, desc, nat, sts, layers, last=False, inp=inp)
            host = (time.time() - t0) * 1e3
            model.sync()
            wall = (time.time() - t0) * 1e3
            if prof:
                profiler.signpost("end")
                profiler.finish_rt()
                res = profiler.result()
        finally:
            if prof:
                profiler.disable()
            ttnn.deallocate(inp)
        return wall, host, res

    return run


class _PPBook:
    """GAP_PP=1: every ttnn.bringup.run_prepared launched outside a wrapped op is booked in a section of its own,
    "<section>|pp:<caller's caller>:<caller>" (the prepared programs are "(other)" rows otherwise)."""

    def __init__(self):
        import ttnn

        self.ttnn, self.orig = ttnn, ttnn.bringup.run_prepared

    def __enter__(self):
        import sys

        orig = self.orig

        def run_prepared(*a, **k):
            P = profiler._P
            if P.get("mesh") is None or P.get("depth") or not P.get("current_full"):
                return orig(*a, **k)
            f1 = sys._getframe(1)
            f2 = f1.f_back
            who = f"{os.path.basename(f2.f_code.co_filename)}:{f2.f_code.co_name}>{f1.f_code.co_name}"
            prev = P["current_full"]
            profiler.signpost(f"{P['current']}|pp:{who}")
            try:
                return orig(*a, **k)
            finally:
                profiler.signpost(prev)

        self.ttnn.bringup.run_prepared = run_prepared
        return self

    def __exit__(self, *exc):
        self.ttnn.bringup.run_prepared = self.orig


def _summary(prof: dict) -> dict:
    """{"device": slowest chip ms, "sections": {sec: slowest-chip ms}, "sections_min": {sec: fastest chip ms},
    "ops": {sec: [{op, shape, calls, programs, ms, ms_min_chip, host_ms, gap_ms}]}, "host_ms": all launches}."""
    per_chip = defaultdict(float)
    for d in prof["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    secs = {k: round(v / 1e6, 3) for k, v in sorted(prof["kernel_ns"].items(), key=lambda kv: -kv[1])}
    secs_min = {k: round(min(d.values()) / 1e6, 3) for k, d in prof["kernel_ns_dev"].items() if d}
    ops, host_total = {}, 0.0
    for sec, rows in prof["op_ns"].items():
        agg = {}
        for r in rows:
            k = (r["op"], r["shape"])
            a = agg.setdefault(
                k, {"op": r["op"], "shape": r["shape"], "calls": 0, "programs": 0, "dev": defaultdict(float)}
            )
            a["calls"] += r["calls"]
            a["programs"] += r["programs"]
            for c, v in r["ns_dev"].items():
                a["dev"][c] += v
            for f in ("gap_ns", "host_ns"):
                a[f] = a.get(f, 0.0) + r.get(f, 0.0)
        out = []
        for a in agg.values():
            dv = a.pop("dev")
            a["ms"] = round(max(dv.values()) / 1e6, 4) if dv else 0.0
            a["ms_min_chip"] = round(min(dv.values()) / 1e6, 4) if dv else 0.0
            a["gap_ms"] = round(a.pop("gap_ns", 0.0) / 1e6, 4)
            a["host_ms"] = round(a.pop("host_ns", 0.0) / 1e6, 4)
            host_total += a["host_ms"]
            out.append(a)
        ops[sec] = sorted(out, key=lambda r: -r["ms"])
    return {
        "device": round(max(per_chip.values()) / 1e6, 3) if per_chip else 0.0,
        "sections": secs,
        "sections_min": secs_min,
        "ops": ops,
        "host_ms": round(host_total, 3),
    }


def _free_states(runtime, kv):
    import ttnn

    for key in [k for k in runtime._states if k[0] == id(kv)]:
        try:
            runtime._states.pop(key).free()
        except Exception as e:  # noqa: BLE001
            print(f"state free: {e}")
    for t in kv.contract.by_config.values():
        if t.is_allocated():
            ttnn.deallocate(t)


def _write(key: str, val) -> None:
    dst = os.environ.get("GAP_OUT", "/tmp/pb_gap_probe.json")
    try:
        with open(dst) as f:
            cur = json.load(f)
    except Exception:  # noqa: BLE001
        cur = {}
    cur[key] = val
    with open(dst, "w") as f:
        json.dump(cur, f, indent=1)
    print(f"GAP probe: {key} written to {dst}")


def _side_by_side(p: int, a: dict, b: dict) -> None:
    print(f"GAP p{p}: device single {a['device']} 4x2k {b['device']}; walls {a['walls']} | {b['walls']}")
    print(f"GAP p{p}: host enqueue single {a['host_enqueue']} 4x2k {b['host_enqueue']}")
    secs = sorted(
        set(a["sections"]) | set(b["sections"]), key=lambda s: -(b["sections"].get(s, 0) - a["sections"].get(s, 0))
    )
    for s in secs:
        x, y = a["sections"].get(s, 0.0), b["sections"].get(s, 0.0)
        print(
            f"GAP section p{p} {s}: single {x} 4x2k {y} extra {round(y - x, 3)} (4x2k min chip "
            f"{b['sections_min'].get(s, 0.0)})"
        )
    for sec in sorted(set(a["ops"]) | set(b["ops"])):
        ra = {(r["op"], r["shape"]): r for r in a["ops"].get(sec, [])}
        rb = {(r["op"], r["shape"]): r for r in b["ops"].get(sec, [])}
        for k in sorted(set(ra) | set(rb), key=lambda k: -abs(rb.get(k, {}).get("ms", 0) - ra.get(k, {}).get("ms", 0))):
            x, y = ra.get(k, {}), rb.get(k, {})
            if abs(y.get("ms", 0.0) - x.get("ms", 0.0)) < 0.05 and abs(y.get("host_ms", 0) - x.get("host_ms", 0)) < 0.2:
                continue
            print(
                f"GAP op p{p} [{sec}] {k[0]} {k[1]}: single {x.get('ms', 0.0)} ms ({x.get('calls', 0)} calls, host "
                f"{x.get('host_ms', 0.0)}) | 4x2k {y.get('ms', 0.0)} ms ({y.get('calls', 0)} calls, host "
                f"{y.get('host_ms', 0.0)}, min chip {y.get('ms_min_chip', 0.0)})"
            )


@mesh_parametrize(timeout_s=4 * 3600)
def test_gap_ops(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    if os.environ.get("GAP_AB", "") == "1":
        pytest.skip("GAP_AB=1: the A/B check only")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192"
    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode"
    from contextlib import ExitStack

    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP

    positions = _positions()
    n_walls = int(os.environ.get("GAP_WALLS", "5"))
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = -(-deploy // CHUNK) * CHUNK
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(len(LENS)))
    runtime, adapter, hf, params = SP._runtime(mesh_device, layers, CHUNK, want)
    model = runtime.model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    text = PositionGolden(S, (len(LENS) - 1) * SHIFT + max(positions) + CHUNK, CHUNK).tokens()
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
    res = {}
    try:
        for path in _paths():
            ctx = ExitStack()
            if path == "single8192":
                ctx.enter_context(_pc.packed("off"))
            else:
                ctx.enter_context(_pc.packed("on"))
                ctx.enter_context(_pc.switch("PACKED_GLUED", "on"))
            with ctx:
                run = _runner(runtime, kv, mesh_device, layers, text, path)
                for p in positions:
                    rot = [0]
                    hosts = []

                    def one_run():
                        run(p, True, rot[0])
                        w, h, _ = run(p, False, rot[0])
                        hosts.append(round(h, 1))
                        rot[0] += 1
                        return round(w, 1)

                    w = warm_up(mesh_device, one_run)
                    hosts.clear()
                    walls = sorted(one_run() for _ in range(n_walls))
                    if os.environ.get("GAP_CPROF"):  # (host split: one clean chunk under cProfile, pstats to GAP_CPROF)
                        import cProfile

                        run(p, True, rot[0])
                        cp = cProfile.Profile()
                        cp.enable()
                        run(p, False, rot[0])
                        cp.disable()
                        cp.dump_stats(os.path.join(os.environ["GAP_CPROF"], f"{path}_p{p}.pstats"))
                    if os.environ.get("GAP_PROF", "1") == "1":
                        run(p, True, rot[0])
                        with _PPBook() if os.environ.get("GAP_PP") == "1" else ExitStack():
                            _, _, prof = run(p, False, rot[0], prof=True)
                        sm = _summary(prof)
                    else:
                        sm = {"device": 0.0, "sections": {}, "sections_min": {}, "ops": {}, "host_ms": 0.0}
                    sm["walls"] = walls
                    sm["wall_median"] = walls[len(walls) // 2]
                    sm["host_enqueue"] = sorted(hosts)
                    sm["warm"] = bool(w.get("warm"))
                    res.setdefault(f"p{p}", {})[path] = sm
                    print(
                        f"GAP {path} p{p}: walls {walls}, host enqueue {sorted(hosts)}, device {sm['device']} ms, "
                        f"launch host {sm['host_ms']} ms"
                    )
                    metrics.record(f"gap_{path}_device_ms_p{p}", sm["device"])
                    metrics.record(f"gap_{path}_wall_ms_p{p}", sm["wall_median"])
                    _write(f"p{p}", res[f"p{p}"])
    finally:
        _free_states(runtime, kv)
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    for p in positions:
        r = res.get(f"p{p}", {})
        if "single8192" in r and "4x2k" in r:
            _side_by_side(p, r["single8192"], r["4x2k"])


@mesh_parametrize(timeout_s=3600)
def test_gap_ab(mesh_device, monkeypatch):
    """PB.GAP's switch (GAP_AB_KEY, default PACKED_CWRITE_FAST) off vs on, bit for bit: the same 4 x 2k chunk (its real
    previous chunk first) at GAP_AB_POS on slots 0-3 with the switch off and on slots 4-7 with it on (slots nothing
    wrote before, DS41-348); every compressed served record of [0, p + 2048) of every segment compared by digest."""
    if mesh_device is None:
        pytest.skip("device test")
    if os.environ.get("GAP_AB", "") != "1":
        pytest.skip("GAP_AB")
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP

    key = os.environ.get("GAP_AB_KEY", "PACKED_CWRITE_FAST")
    p = int(os.environ.get("GAP_AB_POS", "139264"))
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = -(-deploy // CHUNK) * CHUNK
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(2 * len(LENS)))
    runtime, adapter, hf, params = SP._runtime(mesh_device, layers, CHUNK, want)
    model = runtime.model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    text = PositionGolden(S, (len(LENS) - 1) * SHIFT + p + CHUNK, CHUNK).tokens()
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as PD
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment

    n = len(LENS)
    digs = {}
    try:
        with _pc.packed("on"), _pc.switch("PACKED_GLUED", "on"):
            for side, val in (("off", "off"), ("on", "on")):
                base = 0 if side == "off" else n
                states = [runtime.state(kv, base + k) for k in range(n)]
                with _pc.switch(key, val):
                    for prev in (True, False):
                        segs, ids = [], []
                        for k in range(n):
                            s = p - (LENS[k] if prev else 0)
                            segs.append(Segment(base + k, s, s + LENS[k], None))
                            ids.append(text[k * SHIFT + s : k * SHIFT + s + LENS[k]].long())
                        desc = ChunkDescriptor.packed(CHUNK, segs)
                        inp = PD.make_input(mesh_device, desc, ids)
                        nat = PD.ids_from_input(mesh_device, inp, desc)
                        PD.run_chunk(model, desc, nat, states, layers, last=False, inp=inp)
                        model.sync()
                        ttnn.deallocate(inp)
                d = {}
                for k, st in enumerate(states):
                    for i in sorted(st.compress):
                        for name, t in st.compress[i].by_config().items():
                            d[(k, i, name)] = _pc.digest(t.to_host(p + LENS[k]))
                digs[side] = d
    finally:
        _free_states(runtime, kv)
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    a, b = digs["off"], digs["on"]
    bad = [k for k in a if a[k] != b.get(k)]
    print(f"GAP AB {key} p{p}: {len(a)} served records compared, {len(bad)} differ: {bad[:8]}")
    metrics.record(f"gap_ab_{key.lower()}_mismatches_p{p}", len(bad))
    assert a and not bad, bad[:8]
