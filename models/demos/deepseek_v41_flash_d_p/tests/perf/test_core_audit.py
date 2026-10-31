# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.15 (agent rule 18, owner 2026-10-30): how many Tensix cores each op of one warm chunk runs on, per chip.

Runs the profile rung's chunk (or one chunk at BRINGUP_POS, fresh state) through the all-device model, warm, with the
program real-time profiler: every program record carries its core count (``core_count``). Each record is charged to
the outermost ttnn call that launched it (testing/profiler.py op mode), so each row is (section, op, shape) with its
calls, the largest core count of any of its programs on any chip, and its kernel ms. Rows above
tt/common.max_worker_cores(mesh) are printed as OVER and listed in generated/ds41_cores/cores<tag>.json; the test
fails if any program exceeds the cap (CORES_AUDIT_STRICT=0: report only). Not a gate.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_core_audit.py
    DS41_MAX_WORKER_CORES=0 CORES_TAG=_before CORES_AUDIT_STRICT=0 ...   (no cap: the grid every op picks by itself)
    CORES_LOAD=1 ...   (also the model's load and its first, compiling chunk); CORES_SYNC=all (every ttnn op fenced)
"""

from __future__ import annotations

import bisect
import inspect
import json
import os
import pathlib
import time
from collections import defaultdict

import ttnn
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden, position, profile_rung
from models.demos.deepseek_v41_flash_d_p.tt.cores import max_worker_cores

OUT = pathlib.Path("generated/ds41_cores")


@mesh_parametrize
def test_core_audit(mesh_device):
    s = spec()
    mesh = mesh_device
    pos = position()
    if pos is not None:
        chunk = int(s.get("target.chunk"))
        rung = {"name": f"pos{pos}", "seq": pos + chunk, "chunk": chunk}
        g = PositionGolden(s, pos, chunk)
        s.data["target"]["seq"] = max(int(s.data["target"]["seq"]), pos + chunk)
    else:
        rung_name = profile_rung(s)
        rung = s.rung(rung_name)
        g = Golden.for_rung(s, rung_name)
    layers = [i for i in s.layers() if i in g.layers]
    chunk = rung["chunk"]
    start = rung["seq"] - chunk
    recs = []  # (runtime id, chip, cores), in arrival order

    def on_batch(batch):
        for r in batch.records:
            recs.append((int(r.runtime_id), int(r.chip_id), int(r.core_count)))

    # CORES_SYNC=1 (default): the programs Python dispatches itself (ttnn.bringup.run_prepared, ttnn.generic_op) share
    # runtime ids with the ops around them, so each such call is fenced (sync + settle before and after) and owns the
    # records that arrive inside its fence, named by its caller's file:line
    fenced, depth = [], [0]

    def settle():
        ttnn.synchronize_device(mesh)
        n, t0 = -1, time.monotonic()
        while n != len(recs) and time.monotonic() - t0 < 0.5:
            n = len(recs)
            time.sleep(0.01)

    def where() -> str:
        here = pathlib.Path(__file__).name
        skip = (here, "prepared_program", "testing/profiler.py", "ttnn/ttnn/decorators.py")
        fr = [f for f in inspect.stack()[2:] if not any(k in f.filename for k in skip)]
        rel = [f for f in fr if "models/demos" in f.filename or "ttnn/ttnn/bringup" in f.filename]
        f = (rel or fr)[0]
        return f"{f.filename.split('tt-metal/')[-1]}:{f.lineno}"

    def fence(name, fn):
        def wrapped(*a, **k):
            if depth[0]:
                return fn(*a, **k)
            depth[0] += 1
            try:
                settle()
                n0 = len(recs)
                out = fn(*a, **k)
                settle()
                fenced.append((name, where(), profiler._P.get("current_full"), n0, len(recs)))
                return out
            finally:
                depth[0] -= 1

        return wrapped

    patched = []
    mode = os.environ.get("CORES_SYNC", "1")

    def install():
        if patched or mode not in ("1", "all"):
            return
        # (and every public ttnn.bringup callable: a fork's C++ op the profiler's call bookkeeping does not wrap -
        # flat_routed_expert_with_program, the exchange's programs - is then named by its caller's file:line too)
        names = [(ttnn.bringup, "run_prepared"), (ttnn, "generic_op")]
        for n in dir(ttnn.bringup):
            f = getattr(ttnn.bringup, n, None)
            if n.startswith("_") or n in ("run_prepared", "prepare_program") or inspect.ismodule(f):
                continue
            if inspect.isclass(f) or not callable(f):
                continue
            names.append((ttnn.bringup, n))
        # (host uploads: a TILE upload may tilize on the device)
        names += [(ttnn, n) for n in ("from_torch", "as_tensor")]
        for mod, name in names:
            orig = getattr(mod, name)
            patched.append((mod, name, orig))
            setattr(mod, name, fence(f"{'bringup.' if mod is ttnn.bringup else ''}{name}", orig))  # noqa
        if mode == "all":  # every ttnn operation fenced: each record owned by the call it arrived in (slow, exact)
            FO = ttnn.decorators.FastOperation
            orig_call = FO.__call__

            def op_call(self, *a, **k):
                return fence(self.python_fully_qualified_name, lambda: orig_call(self, *a, **k))()

            patched.append((FO, "__call__", orig_call))
            FO.__call__ = op_call

    # CORES_LOAD=1: the model's load and its first (compile) chunk too - weight / table uploads, the MoE block's build
    # and warm-up - each record owned by the fenced call it arrived in, else "(load, no call)"
    load = os.environ.get("CORES_LOAD", "0") == "1"
    h = None
    if load:
        install()
        h = ttnn.device.RegisterProgramRealtimeProfilerCallback(on_batch)
    model = s.hooks().device_model(mesh, s, layers, lm_head=False)
    state = model.new_state(rung["seq"])
    prefix = {i: g.state(i, at=start) for i in layers}
    tokens = g.tokens()[start:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}

    def run():
        if pos is None:
            for i in layers:
                state.load_prefix(i, prefix[i], start)
        begin_chunk(model, tokens, start)
        h = None
        for i in layers:
            if i in starts:
                if h is not None:
                    model.free(h)
                h = model.embed(tokens) if i == 0 else model.from_host(g.layer(g.n_chunks - 1, i)["in"].float())
            profiler.set_layer(i)
            h2 = model.layer(i, h, start, state)
            model.free(h)
            h = h2
        model.sync()
        model.free(h)

    run()  # compile
    load_rows = []
    if load:
        settle()
        lr = defaultdict(lambda: {"calls": 0, "programs": 0, "cores": 0})
        owned_l = set()
        for name, at, _full, n0, n1 in fenced:
            r = lr[f"{name} @ {at}"]
            r["calls"] += 1
            r["programs"] += len({x[0] for x in recs[n0:n1]})
            r["cores"] = max([r["cores"]] + [x[2] for x in recs[n0:n1]])
            owned_l.update(range(n0, n1))
        free_l = defaultdict(int)
        for k, (rid, _chip, c) in enumerate(recs):
            if k not in owned_l:
                free_l[(rid, c)] += 0
        by_c = defaultdict(set)
        for rid, c in free_l:
            by_c[c].add(rid)
        for c, ids in by_c.items():
            lr[f"(load, no call) cores {c}"].update(programs=len(ids), cores=c)
        load_rows = sorted(({"op": k, **v} for k, v in lr.items()), key=lambda r: -r["cores"])
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(h)
        recs.clear()
        fenced.clear()
    install()
    profiler.enable(mesh, rt=True, calls=True)
    h = ttnn.device.RegisterProgramRealtimeProfilerCallback(on_batch)
    try:
        run()
        profiler.set_layer(None)
        profiler.signpost("end")
        calls = [dict(c) for c in profiler._P["calls"]]  # before finish_rt drops each call's id range
        marks = list(profiler._P.get("markers") or [])
        profiler.finish_rt()
        time.sleep(0.5)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(h)
        profiler.disable()
        for mod, name, orig in patched:
            setattr(mod, name, orig)
    rows = defaultdict(lambda: {"calls": 0, "programs": 0, "cores": 0, "ns": 0.0})
    in_fence = set()
    for name, at, full, n0, n1 in fenced:
        sec = ".".join(str(full or "?").split(".")[1:2]) or "?"
        mine = recs[n0:n1]
        in_fence.update(range(n0, n1))
        r = rows[(sec, f"{name} @ {at}", "")]
        r["calls"] += 1
        r["programs"] += len({x[0] for x in mine})
        r["cores"] = max([r["cores"]] + [x[2] for x in mine])
    cores = {}  # runtime id -> {chip: max cores}, the records no fence owns
    for k, (rid, chip, c) in enumerate(recs):
        if k not in in_fence:
            d = cores.setdefault(rid, {})
            d[chip] = max(d.get(chip, 0), c)
    owned = set()
    for c in calls:
        a, b = c["ids"]
        ids = [i for i in range(a, b) if i in cores]
        if not ids:
            continue
        owned.update(ids)
        sec = ".".join(str(c.get("key") or "?").split(".")[1:2]) or "?"
        r = rows[(sec, c["op"], c.get("shape", ""))]
        r["calls"] += 1
        r["programs"] += len(ids)
        r["cores"] = max(r["cores"], max(max(v.values()) for i in ids for v in [cores[i]]))
        r["layers"] = sorted(set(r.get("layers", [])) | {str(c.get("key") or "?").split(".")[0]})
    # programs no wrapped call launched (prepared / generic programs dispatched outside a ttnn op): by section
    starts = [m[0] for m in marks]
    for i in sorted(i for i in cores if i not in owned):
        k = bisect.bisect_right(starts, i) - 1
        full = str(marks[k][2]) if k >= 0 and marks[k][2] is not None else "?"
        sec = ".".join(full.split(".")[1:]) or "?"
        c = max(cores[i].values())
        r = rows[(sec, "(no call)", f"cores {c}")]
        r["programs"] += 1
        r["cores"] = max(r["cores"], c)
    # kernel ms per row from the profiler's own per-call bookkeeping
    for c in profiler._P.get("calls", []) or []:
        sec = ".".join(str(c.get("key") or "?").split(".")[1:2]) or "?"
        k = (sec, c["op"], c.get("shape", ""))
        if k in rows:
            rows[k]["ns"] += max((c.get("ns_dev") or {0: 0.0}).values())
    cap = max_worker_cores(mesh)
    grid = mesh.compute_with_storage_grid_size()
    out = sorted(
        ({"section": k[0], "op": k[1], "shape": k[2], **v, "ms": round(v.pop("ns") / 1e6, 4)} for k, v in rows.items()),
        key=lambda r: (-r["cores"], -r["ms"]),
    )
    over = [r for r in out if r["cores"] > cap]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("CORES_TAG", "") + ("" if pos is None else f"_p{pos}")
    path = OUT / f"cores{tag}.json"
    path.write_text(json.dumps({"grid": [grid.x, grid.y], "cap": cap, "pos": pos, "rows": out}, indent=1))
    print(f"[cores] grid {grid.x}x{grid.y}, cap {cap}: {len(out)} (section, op, shape) rows, {len(over)} over the cap")
    for r in out:
        flag = "OVER" if r["cores"] > cap else "    "
        print(
            f"[cores] {flag} {r['cores']:4d} {r['section']:16s} {r['op']:40s} calls {r['calls']:4d} "
            f"{r['ms']:9.4f} ms  {r['shape'][:110]}"
        )
    for r in load_rows:
        flag = "OVER" if r["cores"] > cap else "    "
        print(
            f"[cores] {flag} {r['cores']:4d} load/compile     {r['op'][:110]:110s} calls {r['calls']:5d} progs {r['programs']}"
        )
    if load_rows:
        over += [dict(r, section="load") for r in load_rows if r["cores"] > cap]
        path.write_text(
            json.dumps({"grid": [grid.x, grid.y], "cap": cap, "pos": pos, "rows": out, "load": load_rows}, indent=1)
        )
    print(f"[cores] wrote {path}")
    if os.environ.get("CORES_AUDIT_STRICT", "1") == "1":
        assert not over, f"{len(over)} op rows run on more than {cap} cores: " + "; ".join(
            f"{r.get('section', '?')} {r['op']} {r['cores']}" for r in over[:20]
        )
