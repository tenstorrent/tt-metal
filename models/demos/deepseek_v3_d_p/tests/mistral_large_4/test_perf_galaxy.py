# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mistral-Large-4-1T-A52B on a Blackhole Galaxy: where one warm prefill chunk spends its time.

The device model over layers 0..N (all 72 by default), built as it serves (56320-token capacity), runs one 5120-token
chunk at ``ML4_PERF_START`` (default 51200, the last chunk of a 56320-token prompt) four times:

1. compile: builds every layer (loads the checkpoint) and fills the program cache;
2. timed: warm, synced only at the end: ``chunk_wall_ms`` (the number a perf change must move);
3. steps: the device profiler read at every block step (the package's ``signpost`` per ``run_block`` step):
   device kernel time per layer and per step, and the host wall per step;
4. ops (``ML4_PERF_OP_LAYERS``, default one layer per variant: 0,1,3,17,39): the device profiler read after every
   outermost ttnn call on those layers: per op, its calls, programs and device time.

The profiler is read and its programs dropped between runs: without that, the unprofiled runs fill its DRAM buffers
("markers were dropped") before the profiled one starts, which is how the bring-up's profile of 9 layers failed.
Perf is measured with the KV and index caches empty before the chunk: top-k and attention do the same work on any
prefix, so the timing is that of a chunk at that position. Accuracy is the e2e test's (test_e2e_galaxy.py).

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 \
    ML4_PERF_GALAXY=1 python -m pytest -s \
    models/demos/deepseek_v3_d_p/tests/mistral_large_4/test_perf_galaxy.py

Results go to ML4_PERF_OUT (JSON). Without the device profiler only the compile and timed runs are meaningful
(``device_ms_total`` is then 0).
"""

from __future__ import annotations

import json
import os
import socket
import re
import time
from collections import defaultdict
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v3_d_p.tt.mistral_large_4.hooks import device_model  # noqa: E402
from models.demos.deepseek_v3_d_p.reference.mistral_large_4 import _bringup_interface as iface  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mistral_large_4 import CHECKPOINT_DIR  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import sub_device  # noqa: E402

CKPT = os.environ.get("ML4_CKPT") or os.environ.get("PREFILL_HF_MODEL") or CHECKPOINT_DIR
OUT = Path(os.environ.get("ML4_PERF_OUT", "generated/ml4_perf/result.json"))
CAPACITY = 56320
CHUNK = 5120


class _Spec:
    """What the device model's builder reads off a spec (bringup/hooks.py)."""

    def __init__(self, hf_dir: str, seq: int, chunk: int):
        self.hf_dir = Path(hf_dir)
        self._v = {"target.chunk": chunk, "target.seq": seq, "reference.weight_dtype": "bfloat16"}

    def get(self, key, default=None):
        return self._v.get(key, default)


def _layers() -> list[int]:
    a, _, b = os.environ.get("ML4_PERF_LAYERS", "0-71").partition("-")
    assert int(a) == 0, "the chunk enters through the embedding: the layers start at 0"
    return list(range(0, int(b or a) + 1))


def _tokens(start: int) -> list[int]:
    p = os.environ.get("ML4_PROMPT")
    if p:
        ids = json.loads(Path(p).read_text())["token_ids"]
        if len(ids) >= start + CHUNK:
            return ids[start : start + CHUNK]
    g = torch.Generator().manual_seed(0)
    return torch.randint(1000, 100000, (CHUNK,), generator=g).tolist()


class Profiler:
    """Device kernel ns of every program since the last read, charged to the current section (and, in op mode, to
    the outermost ttnn call). Per read: the slowest device's sum (the chunk waits for it)."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.section = None  # "L<i>.<step>"
        self.layer = None
        self.op_layers: set[int] = set()
        self.kernel_ns = defaultdict(float)
        self.programs = defaultdict(int)
        self.wall_s = defaultdict(float)
        self.ops = defaultdict(lambda: {"calls": 0, "programs": 0, "ns": 0.0})  # (layer, step, op) -> row
        self.t = time.time()
        self._draining = False
        self._depth = 0

    def drain(self) -> tuple[float, int]:
        self._draining = True
        try:
            ttnn.synchronize_device(self.mesh)
            ttnn.ReadDeviceProfiler(self.mesh)
            data = ttnn.get_latest_programs_perf_data() or {}
        finally:
            self._draining = False
        per_dev, n = defaultdict(float), 0
        for dev, progs in data.items():
            for prog in progs:
                e = (getattr(prog, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
                if e is not None:
                    per_dev[int(dev)] += float(e.duration)
                    n += 1
        return (max(per_dev.values()) if per_dev else 0.0), n

    def discard(self) -> None:
        self.drain()

    def _book(self, ns: float, n: int) -> None:
        if self.section is not None:
            self.kernel_ns[self.section] += ns
            self.programs[self.section] += n

    def signpost(self, name: str) -> None:
        ns, n = self.drain()
        self._book(ns, n)
        now = time.time()
        if self.section is not None:
            self.wall_s[self.section] += now - self.t
        self.section = None if name is None else f"L{self.layer}.{name}"
        self.t = time.time()

    def install(self) -> None:
        iface.SECTION_HOOK = self.signpost
        self.drain()  # start empty

    def uninstall(self) -> None:
        self.signpost(None)
        iface.SECTION_HOOK = None
        self.unpatch()

    # op mode: drain around every outermost ttnn call on the op layers
    def patch(self) -> None:
        import ttnn.decorators as D

        if getattr(D.FastOperation, "_ml4_orig_call", None) is not None:
            return
        orig = D.FastOperation.__call__
        prof = self

        def call(op, *args, **kwargs):
            if prof._draining or prof._depth or prof.layer not in prof.op_layers or prof.section is None:
                prof._depth += 1
                try:
                    return orig(op, *args, **kwargs)
                finally:
                    prof._depth -= 1
            ns, n = prof.drain()  # anything launched outside a wrapped call since the last read
            prof._book(ns, n)
            prof._row("(other)", ns, n)
            prof._depth = 1
            try:
                out = orig(op, *args, **kwargs)
            finally:
                prof._depth = 0
            ns, n = prof.drain()
            prof._book(ns, n)
            name = op.python_fully_qualified_name
            prof._row(name[5:] if name.startswith("ttnn.") else name, ns, n)
            return out

        D.FastOperation._ml4_orig_call = orig
        D.FastOperation.__call__ = call

    def unpatch(self) -> None:
        import ttnn.decorators as D

        orig = getattr(D.FastOperation, "_ml4_orig_call", None)
        if orig is not None:
            D.FastOperation.__call__ = orig
            D.FastOperation._ml4_orig_call = None

    def _row(self, op: str, ns: float, n: int) -> None:
        if not n or self.section is None:
            return
        layer, step = self.section.split(".", 1)
        r = self.ops[(layer, step, op)]
        r["calls"] += 1
        r["programs"] += n
        r["ns"] += ns


class WallTimer(Profiler):
    """The Profiler's sections and op rows timed by wall clock instead of the device profiler: every read is a device
    synchronize, and charges the wall time since the previous read (host dispatch and device time together, the
    sync's own cost included). Works without TT_METAL_DEVICE_PROFILER, so it never fills a buffer."""

    def __init__(self, mesh):
        super().__init__(mesh)
        self._t0 = time.perf_counter()

    def drain(self) -> tuple[float, int]:
        ttnn.synchronize_device(self.mesh)
        now = time.perf_counter()
        ns, self._t0 = (now - self._t0) * 1e9, now
        return ns, 1


def _dram_mb(mesh) -> float:
    """DRAM allocated on a chip (all banks), MB: before and after a capture, the trace's size (perf 7)."""
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return round(v.total_bytes_allocated_per_bank * v.num_banks / 2**20, 1)


def _dram_free_mb(mesh) -> float:
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return v.total_bytes_free_per_bank * v.num_banks / 2**20


class TraceSteps:
    """perf 7: one ttnn trace per block step, captured in order and replayed in order with a device synchronize
    after each, so each step's replay time is its device time without host dispatch (plus one synchronize). The
    final norm lands in the last step's trace. Replaying in capture order keeps every buffer as the capture left it.
    Quacks like a Profiler for _breakdown (kernel_ns per "L<i>.<step>").

    perf 8: on ``op_layers`` every outermost ttnn call is a trace of its own (op rows: device ms per op without host
    dispatch); anything those layers enqueue outside a ttnn call runs eagerly during the capture and is not timed."""

    def __init__(self, mesh, op_layers=()):
        self.mesh = mesh
        self.layer = None
        self.section = None
        self.cur = None
        self.op_layers = set(op_layers)
        self.traces = []  # [(section, trace id, op or None)] in capture order
        self.kernel_ns = defaultdict(float)
        self.ops = defaultdict(lambda: {"calls": 0, "ns": 0.0})
        self._depth = 0

    def _end(self) -> None:
        if self.cur is not None:
            sub_device.end_capture(self.mesh, self.cur[1])
            self.traces.append(self.cur)
            self.cur = None

    def signpost(self, name) -> None:
        self._end()
        self.section = None if name is None else f"L{self.layer}.{name}"
        if name is not None and self.layer not in self.op_layers:
            self.cur = (self.section, sub_device.begin_capture(self.mesh), None)

    def install(self) -> None:
        iface.SECTION_HOOK = self.signpost
        if self.op_layers:
            self._patch()

    def uninstall(self) -> None:
        self.signpost(None)
        iface.SECTION_HOOK = None
        self._unpatch()

    def _patch(self) -> None:
        import ttnn.decorators as D

        orig = D.FastOperation.__call__
        ts = self

        def call(op, *args, **kwargs):
            if ts._depth or ts.section is None or ts.layer not in ts.op_layers:
                ts._depth += 1
                try:
                    return orig(op, *args, **kwargs)
                finally:
                    ts._depth -= 1
            name = op.python_fully_qualified_name
            tid = sub_device.begin_capture(ts.mesh)
            ts._depth = 1
            try:
                return orig(op, *args, **kwargs)
            finally:
                ts._depth = 0
                sub_device.end_capture(ts.mesh, tid)
                ts.traces.append((ts.section, tid, name[5:] if name.startswith("ttnn.") else name))

        D.FastOperation._ml4_trace_orig_call = orig
        D.FastOperation.__call__ = call

    def _unpatch(self) -> None:
        import ttnn.decorators as D

        orig = getattr(D.FastOperation, "_ml4_trace_orig_call", None)
        if orig is not None:
            D.FastOperation.__call__ = orig
            D.FastOperation._ml4_trace_orig_call = None

    def replay(self) -> float:
        """Every trace once, in order; kernel_ns (and op rows) get this replay's times. Returns the total wall ms."""
        self.kernel_ns.clear()
        self.ops.clear()
        ttnn.synchronize_device(self.mesh)
        t0 = time.perf_counter()
        for sec, tid, op in self.traces:
            t1 = time.perf_counter()
            sub_device.execute(self.mesh, tid)
            ttnn.synchronize_device(self.mesh)
            ns = (time.perf_counter() - t1) * 1e9
            self.kernel_ns[sec] += ns
            if op is not None:
                layer, step = sec.split(".", 1)
                r = self.ops[(layer, step, op)]
                r["calls"] += 1
                r["ns"] += ns
        return (time.perf_counter() - t0) * 1e3

    def release(self) -> None:
        for _, tid, _ in self.traces:
            sub_device.release(self.mesh, tid)
        self.traces = []


_RID_MOD = 1 << 21  # the profiler keeps a program's runtime id as (id << 10 | chip) in 31 bits: 21 bits of id


class DeviceOps:
    """perf 11: exact device kernel time per op on a few layers, from the device profiler's C++ post-process
    (TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1).

    Every program carries its ttnn runtime id (the device-operation counter when it was enqueued; a trace's replay
    keeps the captured ids), so an outermost ttnn call on ``layers`` owns the ids it advanced the counter by, and a
    step's ids no call claimed are "(untracked)". The profiler is read only every DRAIN layers and after the profiled
    layers (a read of the 32 chips takes seconds), never per op; its programs are matched to the calls by id."""

    DRAIN = 8  # layers between reads: ~100 programs a layer stays under PROGRAM_SUPPORT_COUNT (1000 by default)

    def __init__(self, mesh, layers, devices=None):
        self.mesh = mesh
        self.layers = set(layers)
        self.devices = set(devices) if devices else None  # chip ids to aggregate over (all read anyway)
        self.layer = None
        self.section = None  # (layer, step) while on a profiled layer
        self._start = 0
        self._depth = 0
        self.tags = {}  # runtime id % _RID_MOD -> (layer, step, op)
        self.calls = defaultdict(int)  # (step, op) -> outermost calls
        self.dur = defaultdict(dict)  # runtime id -> {chip: kernel ns}
        self.reads, self.read_s = 0, 0.0

    @staticmethod
    def _rid() -> int:
        return int(ttnn._ttnn.get_device_operation_id())

    def drain(self) -> None:
        """Synchronize, read the profiler (empties its buffers) and keep the kernel time of the tagged programs."""
        t = time.time()
        ttnn.synchronize_device(self.mesh)
        ttnn.ReadDeviceProfiler(self.mesh)
        data = ttnn.get_latest_programs_perf_data() or {}
        for dev, progs in data.items():
            if self.devices is not None and int(dev) not in self.devices:
                continue
            for p in progs:
                rid = (int(p.program_execution_uid.runtime_id) & 0x7FFFFFFF) >> 10
                if rid not in self.tags:
                    continue
                e = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
                if e is not None and e.duration:
                    self.dur[rid][int(dev)] = float(e.duration)
        self.reads += 1
        self.read_s += time.time() - t

    def after_layer(self, i: int) -> None:
        """Between layers (eager only: a read synchronizes): read when the buffers could fill or the profiled layers
        are done."""
        self._close()
        if (i + 1) % self.DRAIN == 0 or (self.layers and i == max(self.layers)):
            self.drain()

    def _close(self) -> None:
        if self.section is not None:
            for r in range(self._start, self._rid()):
                self.tags.setdefault(r % _RID_MOD, (*self.section, "(untracked)"))
            self.section = None

    def signpost(self, name) -> None:
        self._close()
        if name is not None and self.layer in self.layers:
            self.section, self._start = (self.layer, name), self._rid()

    def install(self) -> None:
        import ttnn.decorators as D

        iface.SECTION_HOOK = self.signpost
        orig = D.FastOperation.__call__
        prof = self

        def call(op, *args, **kwargs):
            if prof._depth or prof.section is None:
                prof._depth += 1
                try:
                    return orig(op, *args, **kwargs)
                finally:
                    prof._depth -= 1
            sec, a = prof.section, prof._rid()
            prof._depth = 1
            try:
                return orig(op, *args, **kwargs)
            finally:
                prof._depth = 0
                name = op.python_fully_qualified_name
                name = name[5:] if name.startswith("ttnn.") else name
                for r in range(a, prof._rid()):
                    prof.tags[r % _RID_MOD] = (*sec, name)
                prof.calls[(sec[1], name)] += 1

        D.FastOperation._ml4_dev_orig_call = orig
        D.FastOperation.__call__ = call

    def uninstall(self) -> None:
        import ttnn.decorators as D

        self._close()
        iface.SECTION_HOOK = None
        orig = getattr(D.FastOperation, "_ml4_dev_orig_call", None)
        if orig is not None:
            D.FastOperation.__call__ = orig
            D.FastOperation._ml4_dev_orig_call = None

    def summary(self) -> dict:
        """Per (step, op) over the profiled layers: calls, programs, and ms summed over programs of the slowest
        chip's kernel time (``ms_max``) and of the median chip's (``ms_median``); per layer and in total."""
        rows = defaultdict(lambda: {"calls": 0, "programs": 0, "ms_max": 0.0, "ms_median": 0.0})
        per_layer, per_dev = defaultdict(float), defaultdict(float)
        missing, ndev = 0, []
        for rid, (lay, step, op) in self.tags.items():
            d = self.dur.get(rid)
            if not d:
                missing += 1  # dropped (buffer full) or not on the aggregated chips
                continue
            v = sorted(d.values())
            r = rows[(step, op)]
            r["programs"] += 1
            r["ms_max"] += v[-1] / 1e6
            r["ms_median"] += v[len(v) // 2] / 1e6
            per_layer[lay] += v[-1] / 1e6
            ndev.append(len(v))
            for dev, ns in d.items():
                per_dev[dev] += ns / 1e6
        for k, n in self.calls.items():
            rows[k]["calls"] = n
        ops = [
            {"step": st, "op": op, **{k: round(x, 4) if k.startswith("ms") else x for k, x in r.items()}}
            for (st, op), r in rows.items()
        ]
        return {
            "layers": sorted(self.layers),
            "ms_total_max": round(sum(r["ms_max"] for r in rows.values()), 3),
            "ms_total_median": round(sum(r["ms_median"] for r in rows.values()), 3),
            "ms_slowest_chip": round(max(per_dev.values()), 3) if per_dev else 0.0,  # one chip's sum
            "ms_per_layer": {i: round(x, 3) for i, x in sorted(per_layer.items())},
            "programs": len(ndev),
            "programs_without_data": missing,
            "chips_per_program": [min(ndev), max(ndev)] if ndev else None,
            "reads": self.reads,
            "read_s": round(self.read_s, 1),
            "ops": sorted(ops, key=lambda r: -r["ms_max"]),
        }


def _breakdown(timer: Profiler, cfg) -> dict:
    """Per step (summed over layers), per block variant and step, and per layer, in ms; op rows if timed per op."""
    per_layer, per_step, per_variant = defaultdict(float), defaultdict(float), defaultdict(float)
    for sec, ns in timer.kernel_ns.items():
        layer, step = sec.split(".", 1)
        if layer == "LNone":
            per_step[step] += ns / 1e6
            continue
        i = int(layer[1:])
        per_layer[i] += ns / 1e6
        per_step[step] += ns / 1e6
        per_variant[f"{_variant(cfg, i)}.{step}"] += ns / 1e6

    def ranked(d):
        return dict(sorted(((k, round(v, 2)) for k, v in d.items()), key=lambda x: -x[1]))

    out = {
        "ms_total": round(sum(per_step.values()), 1),
        "ms_per_step": ranked(per_step),
        "ms_per_variant_step": ranked(per_variant),
        "ms_per_layer": {i: round(v, 2) for i, v in sorted(per_layer.items())},
    }
    if timer.ops:
        rows = [
            {"layer": lay, "step": st, "op": op, "calls": r["calls"], "ms": round(r["ns"] / 1e6, 3)}
            for (lay, st, op), r in timer.ops.items()
        ]
        out["ops"] = sorted(rows, key=lambda r: (int(r["layer"][1:]), -r["ms"]))
        out["ccl_share"] = _ccl_share(rows)
    return out


# ops that move data between chips (fabric); everything else is local compute / data movement
_CCL_OPS = {  # last dotted segment of the op name; rms_norm_post_all_gather etc. are local
    "all_gather", "reduce_scatter", "all_reduce", "all_to_all_async_generic", "high_bw_all_gather",
    "dispatch", "combine", "offset_cumsum", "all_gather_async", "reduce_scatter_minimal_async",
}


def _ccl_share(rows):
    """Per op-mode layer: device ms of collectives vs local ops above the per-replay floor (an empty trace,
    i.e. the median ``deallocate`` row), so the share is not diluted by ops too small to resolve."""
    out = {}
    for lay in sorted({r["layer"] for r in rows}, key=lambda x: int(x[1:])):
        mine = [r for r in rows if r["layer"] == lay]
        dea = [r["ms"] / r["calls"] for r in mine if r["op"] == "deallocate" and r["calls"]]
        floor = sorted(dea)[len(dea) // 2] if dea else 0.0
        ccl = loc = 0.0
        for r in mine:
            if r["op"] == "deallocate":
                continue
            net = max(0.0, r["ms"] - floor * r["calls"])
            if r["op"].rsplit(".", 1)[-1] in _CCL_OPS:
                ccl += net
            else:
                loc += net
        out[lay] = {"ccl_ms": round(ccl, 2), "local_ms": round(loc, 2), "floor_ms": round(floor, 3),
                    "ccl_pct": round(100 * ccl / (ccl + loc), 1) if ccl + loc else None}
    return out


@pytest.fixture
def mesh_device():
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import fabric

    fabric.set_from_env()  # ML4_FABRIC / ML4_FABRIC_PAYLOAD; default the bring-up's 1D ring
    # perf 7: ML4_TRACE_REGION_MB reserves a trace region; 0 (default) = traces allocated from DRAM as captured
    region = int(os.environ.get("ML4_TRACE_REGION_MB", "0")) * 1024 * 1024
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), **({"trace_region_size": region} if region else {}))
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _variant(cfg, i: int) -> str:
    """The layer's variant, as the bring-up named the five block types."""
    if not cfg.is_moe(i):
        return "attn_dense"
    if not cfg.owns_kv(i):
        return "attn_nope_shared_kv_moe"
    if not cfg.is_full(i):
        return "attn_shared_index_moe"
    return "attn_moe" if cfg.use_rope(i) else "attn_nope_moe"


@pytest.mark.skipif(os.environ.get("ML4_PERF_GALAXY") != "1", reason="needs a Blackhole Galaxy (ML4_PERF_GALAXY=1)")
def test_perf_chunk(mesh_device):
    layers = _layers()
    start = int(os.environ.get("ML4_PERF_START", str(CAPACITY - CHUNK)))
    assert start % CHUNK == 0 and start + CHUNK <= CAPACITY, start
    op_layers = {int(x) for x in os.environ.get("ML4_PERF_OP_LAYERS", "0,1,3,17,39").split(",") if x} & set(layers)
    tokens = _tokens(start)
    result = {"layers": [layers[0], layers[-1]], "start": start, "chunk": CHUNK, "op_layers": sorted(op_layers)}
    for k in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"):
        result[k] = os.environ.get(k, "")
    result["settings"] = {k: v for k, v in os.environ.items() if k.startswith("MISTRAL_LARGE4_1T_")}

    t0 = time.time()
    model = device_model(mesh_device, _Spec(CKPT, CAPACITY, CHUNK), layers, lm_head=False)
    state = model.new_state(CAPACITY)
    prof = Profiler(mesh_device)
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"

    def run(per_layer=None):
        h = model.embed(tokens, start=start)
        for i in layers:
            prof.layer = i
            out = model.layer(i, h, start, state, end=start + CHUNK)
            model.free(h)
            h = out
            if per_layer is not None:
                per_layer(i)
        out = model.final_norm(h)
        model.free(h)
        model.sync()
        model.free(out)

    # 1. compile (builds and loads every layer); the profiler is read after every layer so it never fills
    run(per_layer=(lambda i: prof.discard()) if profiling else None)
    result["build_and_compile_seconds"] = round(time.time() - t0, 1)
    result["load_seconds"] = round(getattr(model, "load_seconds", 0.0), 1)
    if profiling:
        prof.discard()

    # 2. timed, warm: two runs, the faster counts (the first may still compile position-dependent programs)
    walls = []
    for _ in range(2):
        t1 = time.time()
        run()
        walls.append(round((time.time() - t1) * 1e3, 1))
        if profiling:
            prof.discard()
    result["chunk_wall_ms_runs"] = walls
    result["chunk_wall_ms"] = min(walls)

    if profiling:
        # 3. per step
        prof.install()
        run()
        prof.uninstall()
        per_layer, per_step, per_variant = defaultdict(float), defaultdict(float), defaultdict(float)
        for sec, ns in prof.kernel_ns.items():
            layer, step = sec.split(".", 1)
            i = int(layer[1:])
            per_layer[i] += ns / 1e6
            per_step[step] += ns / 1e6
            per_variant[f"{_variant(model.cfg, i)}.{step}"] += ns / 1e6
        total = sum(per_layer.values())
        result["device_ms_total"] = round(total, 2)
        result["host_overhead_ms"] = round(result["chunk_wall_ms"] - total, 1)
        result["device_ms_per_layer"] = {i: round(v, 2) for i, v in sorted(per_layer.items())}
        result["device_ms_per_step"] = dict(
            sorted(((k, round(v, 2)) for k, v in per_step.items()), key=lambda x: -x[1])
        )
        result["device_ms_per_variant_step"] = dict(
            sorted(((k, round(v, 2)) for k, v in per_variant.items()), key=lambda x: -x[1])
        )
        result["programs_per_layer"] = {
            i: sum(n for s, n in prof.programs.items() if s.startswith(f"L{i}.")) for i in layers
        }
        result["host_wall_ms_per_step"] = dict(
            sorted(
                (
                    (k, round(sum(v for s, v in prof.wall_s.items() if s.split(".", 1)[1] == k) * 1e3, 1))
                    for k in per_step
                ),
                key=lambda x: -x[1],
            )
        )

        # 4. ops on the representative layers
        if op_layers:
            prof = Profiler(mesh_device)  # run() sets the layer on the profiler it finds here
            prof.op_layers = op_layers
            prof.install()
            prof.patch()
            run()
            prof.uninstall()
            rows = [
                {
                    "layer": lay,
                    "step": st,
                    "op": op,
                    "calls": r["calls"],
                    "programs": r["programs"],
                    "ms": round(r["ns"] / 1e6, 3),
                }
                for (lay, st, op), r in prof.ops.items()
            ]
            result["ops"] = sorted(rows, key=lambda r: (int(r["layer"][1:]), -r["ms"]))
            by_op = defaultdict(float)
            for r in rows:
                by_op[re.sub(r"\(.*", "", r["op"])] += r["ms"]
            result["op_ms_on_op_layers"] = dict(
                sorted(((k, round(v, 2)) for k, v in by_op.items()), key=lambda x: -x[1])
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k not in ("ops", "device_ms_per_layer", "programs_per_layer")}
    print(json.dumps(summary, indent=1))
    assert result["chunk_wall_ms"] > 0
    if profiling:
        assert result["device_ms_total"] > 0, "the device profiler returned no durations"


# ------------------------------------------------------------------------------------------------- settings sweep
def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _top1_detail(lg: torch.Tensor, g: torch.Tensor) -> dict:
    """perf 8: how far a last-position top-1 flip is. Each side's rank of the other's top-1 (0 = its own top-1), the
    logit gap to it on each side, and the top-5 overlap, for the last row; top-5 containment over all rows."""
    a, b = lg[-1].double(), g[-1].double()
    ta, tb = int(a.argmax()), int(b.argmax())
    gold_in_top5 = (lg.topk(5, dim=-1).indices == g.argmax(-1, keepdim=True)).any(-1)
    return {
        "golden_rank_in_ours": int((a > a[tb]).sum()),
        "our_rank_in_golden": int((b > b[ta]).sum()),
        "our_gap": round(float(a[ta] - a[tb]), 4),  # our logit of our top-1 minus ours of the golden's
        "golden_gap": round(float(b[tb] - b[ta]), 4),  # the golden's margin over our top-1
        "last_top5_overlap": len(set(a.topk(5).indices.tolist()) & set(b.topk(5).indices.tolist())),
        "golden_top1_in_our_top5_rows": int(gold_in_top5.sum()),
    }


def _variants() -> list[dict]:
    """ML4_SWEEP_VARIANTS: a JSON file or inline JSON, [{"name": ..., "env": {"EXPERTS_IMPL": "loop", ...}}]; env keys
    are settings names (tt/settings.py) without the MISTRAL_LARGE4_1T_ prefix. Every variant runs on top of the
    defaults; the first is the reference the others are compared with."""
    raw = os.environ.get("ML4_SWEEP_VARIANTS", '[{"name": "default", "env": {}}]')
    return json.loads(Path(raw).read_text() if not raw.lstrip().startswith("[") else raw)


@pytest.mark.skipif(os.environ.get("ML4_SWEEP_GALAXY") != "1", reason="needs a Blackhole Galaxy (ML4_SWEEP_GALAXY=1)")
def test_sweep(mesh_device):
    """One model load, many settings variants: each runs the last chunk of the golden's prompt (its KV and index
    caches loaded from the golden's state up to that chunk) to compile, twice timed (warm, synced at the end), and once
    against the golden (layers 35 and 71, the final norm and the last 32 rows of logits). Only settings read when the
    model runs can differ between variants (all but the weight dtypes and the router / experts topology). A variant with ``"trace": true`` captures the chunk as one ttnn trace and times its replay (perf 7; without
    ML4_TRACE_REGION_MB the trace buffer comes from free DRAM); with ``"trace_steps": true`` also one trace per block step, for device ms per step. A variant with ``"breakdown": true`` also runs once with a device synchronize at every block step, and
    once more with one after every outermost ttnn call on its ``op_layers`` (default 0,1,3,17,39): wall ms per step
    and per op, profiler-free (WallTimer). ``dispatch_ms_runs`` is how long the host took to issue a timed run.
    A variant with ``"device_profile": true`` (perf 11; optional ``"profile_layers"``, default [16, 17, 18], and
    ``"profile_devices"``, chip ids to aggregate over) is not timed: it records the device kernel time of every op on
    those layers (``device_profile``; eager, or the layers as one trace with ``"trace": true``) and needs the device
    profiler for the whole job, so run it in a job of its own (the profiler slows every kernel a little, and the
    other variants would fill its buffers):

        TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
        TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 \
        TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1 \
        ML4_SWEEP_GALAXY=1 ML4_SWEEP_VARIANTS='[{"name": "prof", "device_profile": true, "trace": true}]' \
        python -m pytest -x -s test_perf_galaxy.py::test_sweep

    (plain pytest, not ``python -m tracy``: the per-program durations come from the profiler's in-memory analysis;
    without DISABLE_DUMP_TO_FILES the raw marker CSV grows by GBs.) Results go to ML4_PERF_OUT after every variant."""
    from safetensors.torch import load_file

    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

    golden = Path(os.environ["ML4_GOLDEN"])
    meta = json.loads((golden / "meta.json").read_text())
    seq, chunk, rows = meta["seq"], meta["chunk"], meta["logits_rows"]
    assert chunk == CHUNK and meta["layers"] == [0, 71], meta
    layers = list(range(72))
    c = seq // chunk - 1
    start = c * chunk
    tokens = json.loads(Path(meta["prompt"]).read_text())["token_ids"][start:seq]
    gold = load_file(str(golden / f"chunk_{c:02d}.safetensors"))
    dumps = [int(k[1:]) for k in gold if re.fullmatch(r"L\d+", k)]
    result = {"golden": str(golden), "start": start, "chunk": chunk, "variants": {}}

    t0 = time.time()
    model = device_model(mesh_device, _Spec(CKPT, max(seq, CAPACITY), chunk), layers, lm_head=True)
    state = model.new_state(max(seq, CAPACITY))
    result["build_seconds"] = round(time.time() - t0, 1)
    result["node"] = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    result["fabric"] = os.environ.get("ML4_FABRIC", "ring1d")
    result["fabric_payload"] = int(os.environ.get("ML4_FABRIC_PAYLOAD", "12288")) or None
    t0 = time.time()
    for i in layers:
        st = load_file(str(golden / "kv_cache" / f"layer_{i}.safetensors"))
        state.load_prefix(
            i,
            {n: st[f"{n}_cache_layer_{i}"] for n in ("kv_latent", "k_rope", "index_key")},
            start,
        )
    model.sync()
    result["prefix_seconds"] = round(time.time() - t0, 1)

    dispatched = [0.0]  # when run() returned from its last dispatch, before the final sync

    def run(watch=None, timer=None, after_layer=None):
        h = model.embed(tokens, start=start)
        for i in layers:
            if timer is not None:
                timer.layer = i
            out = model.layer(i, h, start, state, end=start + chunk)
            model.free(h)
            h = out
            if after_layer is not None:
                after_layer(i)
            if watch is not None and i in dumps:
                watch[f"L{i}"] = model.to_host(h)[:chunk]
        final = model.final_norm(h)
        model.free(h)
        if watch is not None:
            watch["final_norm"] = model.to_host(final)[:chunk]
            watch["logits"] = model.logits(final, list(range(chunk - rows, chunk)))
        dispatched[0] = time.time()
        model.sync()
        model.free(final)

    def traced(rec):
        """perf 7: the chunk's 72 layers and final norm captured as one ttnn trace (the embedding stays eager, its
        output the trace's persistent input), replayed three times timed, then read against the golden (final norm
        and logits; no per-layer dumps). Returns the persistent final-norm output; the caller releases the trace."""
        h_in = model.embed(tokens, start=start)
        model.sync()
        dram0 = _dram_mb(mesh_device)
        t1 = time.time()
        tid = sub_device.begin_capture(mesh_device)
        try:
            h = h_in
            for i in layers:
                out = model.layer(i, h, start, state, end=start + chunk)
                if h is not h_in:
                    model.free(h)
                h = out
            final = model.final_norm(h)
            model.free(h)
        finally:
            sub_device.end_capture(mesh_device, tid)
        model.sync()
        rec["capture_ms"] = round((time.time() - t1) * 1e3, 1)
        rec["trace_dram_mb"] = round(_dram_mb(mesh_device) - dram0, 1)  # the trace buffer and the final norm output
        rec["dram_free_mb"] = round(
            _dram_free_mb(mesh_device), 1
        )
        walls, issue = [], []
        for _ in range(3):
            t1 = time.time()
            sub_device.execute(mesh_device, tid)
            issue.append(round((time.time() - t1) * 1e3, 2))  # host time to enqueue the replay (all its segments)
            model.sync()
            walls.append(round((time.time() - t1) * 1e3, 1))
        rec["chunk_wall_ms_runs"], rec["chunk_wall_ms"] = walls, min(walls)
        rec["replay_issue_ms_runs"] = issue
        rec["trace_segments"] = getattr(tid, "segments", 1)
        return tid, h_in, final

    def traced_steps(rec, op_layers=()):
        """perf 7: the chunk again as one trace per block step (TraceSteps), replayed twice; the second replay's
        device time per step goes to rec["trace_steps"] (no accuracy read: the outputs are the full trace's)."""
        h_in = model.embed(tokens, start=start)
        model.sync()
        ts = TraceSteps(mesh_device, op_layers)
        ts.install()
        try:
            h = h_in
            for i in layers:
                ts.layer = i
                out = model.layer(i, h, start, state, end=start + chunk)
                if h is not h_in:
                    model.free(h)
                h = out
            final = model.final_norm(h)
            model.free(h)
        finally:
            ts.uninstall()
        try:
            ts.replay()
            rec["trace_steps_wall_ms"] = round(ts.replay(), 1)
            rec["trace_steps"] = _breakdown(ts, model.cfg)
        finally:
            ts.release()
            model.free(final)
            model.free(h_in)

    def eager(v, rec):
        t1 = time.time()
        run()  # compile this variant's programs
        rec["first_run_ms"] = round((time.time() - t1) * 1e3, 1)
        walls, disp = [], []
        for _ in range(2):
            t1 = time.time()
            run()
            walls.append(round((time.time() - t1) * 1e3, 1))
            disp.append(round((dispatched[0] - t1) * 1e3, 1))
        rec["chunk_wall_ms_runs"], rec["chunk_wall_ms"] = walls, min(walls)
        # host time to issue every op (the device runs behind it); close to the wall = host bound
        rec["dispatch_ms_runs"] = disp
        if v.get("breakdown"):
            timer = WallTimer(mesh_device)
            timer.install()
            run(timer=timer)
            timer.uninstall()
            rec["steps"] = _breakdown(timer, model.cfg)
            op_layers = set(v.get("op_layers", [0, 1, 3, 17, 39]))
            if op_layers:
                timer = WallTimer(mesh_device)
                timer.op_layers = op_layers
                timer.install()
                timer.patch()
                run(timer=timer)
                timer.uninstall()
                rec["op_steps"] = _breakdown(timer, model.cfg)
        got = {}
        run(watch=got)
        return got

    def device_profile(v, rec):
        """perf 11: per-op device kernel time on ``profile_layers`` (DeviceOps). Eager: one run, the profiler read
        every few layers. ``"trace": true``: layers 0..first eager, then the profiled layers (first..last) captured
        as one trace and replayed once (the whole chunk would overflow the profiler's buffers in one replay); the
        kernels are the same, only host dispatch differs. Accuracy from one more eager run (the profiler's builds)."""
        if not os.environ.get("TT_METAL_PROFILER_CPP_POST_PROCESS"):
            raise RuntimeError("device_profile needs TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 "
                               "TT_METAL_PROFILER_CPP_POST_PROCESS=1")
        pl = sorted(set(v.get("profile_layers", [16, 17, 18])) & set(layers))
        prof = DeviceOps(mesh_device, pl, v.get("profile_devices"))
        t1 = time.time()
        run(after_layer=prof.after_layer)  # compile this variant's programs
        prof.drain()
        rec["first_run_ms"] = round((time.time() - t1) * 1e3, 1)
        t1 = time.time()
        if v.get("trace"):
            a, b = pl[0], pl[-1]
            h = model.embed(tokens, start=start)
            for i in range(a):
                out = model.layer(i, h, start, state, end=start + chunk)
                model.free(h)
                h = out
                prof.after_layer(i)
            model.sync()
            prof.install()
            tid = sub_device.begin_capture(mesh_device)
            try:
                x = h
                for i in range(a, b + 1):
                    prof.layer = i
                    out = model.layer(i, x, start, state, end=start + chunk)
                    if x is not h:
                        model.free(x)
                    x = out
                    prof._close()  # no read while capturing
            finally:
                sub_device.end_capture(mesh_device, tid)
                prof.uninstall()
            try:
                prof.drain()  # empty before the replay
                sub_device.execute(mesh_device, tid)
                prof.drain()
            finally:
                sub_device.release(mesh_device, tid)
                model.free(x)
                model.free(h)
        else:
            prof.install()
            try:
                run(timer=prof, after_layer=prof.after_layer)
            finally:
                prof.uninstall()
            prof.drain()
        rec["profile_run_ms"] = round((time.time() - t1) * 1e3, 1)  # reads included: not a perf number
        rec["device_profile"] = prof.summary()
        got = {}
        run(watch=got, after_layer=prof.after_layer)
        prof.drain()
        return got

    prefix = settings.prefix
    for v in _variants():
        name, env = v["name"], {prefix + k: str(x) for k, x in v.get("env", {}).items()}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        rec = {"env": v.get("env", {})}
        try:
            if v.get("device_profile"):
                got = device_profile(v, rec)
            elif v.get("trace"):
                run()  # compile this variant's programs
                tid, h_in, final = traced(rec)
                try:
                    got = {"final_norm": model.to_host(final)[:chunk]}
                    got["logits"] = model.logits(final, list(range(chunk - rows, chunk)))
                finally:
                    sub_device.release(mesh_device, tid)
                    model.free(final)
                    model.free(h_in)
                if v.get("trace_steps"):
                    traced_steps(rec, v.get("trace_op_layers", ()))
            else:
                got = eager(v, rec)
            cmp = {}
            for k in [f"L{i}" for i in sorted(dumps) if f"L{i}" in got] + ["final_norm"]:
                cmp[k] = {
                    "pcc": round(_pcc(got[k].float(), gold[k].float()), 6),
                    "rel": round(_rel(got[k], gold[k]), 6),
                }
            lg, g = got["logits"].float(), gold["logits"].float()
            cmp["logits"] = {
                "pcc": round(_pcc(lg, g), 6),
                "last_row_pcc": round(_pcc(lg[-1], g[-1]), 6),
                "top1_match_rows": int((lg.argmax(-1) == g.argmax(-1)).sum()),
                "rows": rows,
                "last_top1": [int(lg[-1].argmax()), int(g[-1].argmax())],
                **_top1_detail(lg, g),
            }
            rec["accuracy"] = cmp
            rec["finite"] = bool(all(t.isfinite().all() for t in got.values()))
        except Exception as e:  # a variant that fails (an op rejects it) is recorded; the sweep goes on
            rec["error"] = f"{type(e).__name__}: {e}"[:2000]
            print(f"variant {name} failed: {rec['error']}", flush=True)
        finally:
            for k, x in old.items():
                if x is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = x
        result["variants"][name] = rec
        print(json.dumps({name: {k: rec.get(k) for k in ("chunk_wall_ms", "dispatch_ms_runs", "error")}}), flush=True)
        if "steps" in rec:
            print(f"  {name} steps: {json.dumps(rec['steps']['ms_per_step'])}", flush=True)
        if "device_profile" in rec:
            dp = rec["device_profile"]
            top = {f"{r['step']}:{r['op']}": r["ms_max"] for r in dp["ops"][:12]}
            print(f"  {name} device ms {dp['ms_total_max']} (layers {dp['layers']}, {dp['programs']} programs, "
                  f"{dp['programs_without_data']} without data): {json.dumps(top)}", flush=True)
        if "accuracy" in rec:
            a = rec["accuracy"]
            print(
                f"  {name}: final_norm {a['final_norm']['pcc']}, logits {a['logits']['pcc']}, "
                f"top-1 {a['logits']['top1_match_rows']}/{rows}, last {a['logits']['last_top1']}",
                flush=True,
            )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2) + "\n")
    assert any("chunk_wall_ms" in r or "device_profile" in r for r in result["variants"].values())


# ---- long context: device time per op at chunk positions up to 1M (PERF.md "Long context")
CTX_POSITIONS = (0, 51200, 512000, 1039360)
_CCL = ("all_gather", "reduce_scatter", "all_to_all", "dispatch", "combine", "all_reduce", "ring_", "point_to_point")


def _ctx_groups(ops: list, n_layers: int) -> dict:
    """ms per layer by kind: sparse_sdpa, the rest of attention, indexer, MoE (router, experts, shared expert),
    collectives (any step), other."""
    g = defaultdict(float)
    for r in ops:
        step, op, ms = r["step"], r["op"], r["ms_max"] / n_layers
        if any(k in op for k in _CCL):
            g["ccl"] += ms
        elif "sparse_sdpa" in op:
            g["sparse_sdpa"] += ms
        elif step == "attention":
            g["attention_other"] += ms
        elif step in ("indexer", "topk_shared"):
            g["indexer"] += ms
        elif step in ("router", "experts", "shared_expert", "moe_combine", "mlp"):
            g["moe"] += ms
        else:
            g["other"] += ms
    return {f"group:{k}": round(v, 4) for k, v in sorted(g.items())}


@pytest.mark.skipif(os.environ.get("ML4_CTX_PROFILE") != "1", reason="needs a Blackhole Galaxy (ML4_CTX_PROFILE=1)")
def test_ctx_profile(mesh_device):
    """Warm-chunk device kernel time per op (DeviceOps) on ``ML4_CTX_PROFILE_LAYERS`` (default 16, 17, 18) at chunk
    starts ``ML4_CTX_POSITIONS`` (default CTX_POSITIONS), on the 1-slot long build (max_seq ML4_CTX_MAX_SEQ, default
    1044480; SEQ_BUFFERS=shared, SSMAX_TABLE=shared_log, no LM head) and on the trace-safe (metadata) forward the traced
    serving runs, here eagerly so the profiler is read between layers. Per position: one unprofiled run (compiles),
    then the profiled one; the caches are empty (the indexer scores and the KV gather cover the whole prefix
    [0, start + chunk) whatever it holds; attention reads the top-2048). Needs the device profiler for the whole job:

        TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
        TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1 ML4_CTX_PROFILE=1 ...

    Results (ms per layer: ms_max of the profiled layers / their count) go to ML4_PERF_OUT."""
    os.environ.setdefault("MISTRAL_LARGE4_1T_SEQ_BUFFERS", "shared")
    os.environ.setdefault("MISTRAL_LARGE4_1T_SSMAX_TABLE", "shared_log")
    max_seq = int(os.environ.get("ML4_CTX_MAX_SEQ", "1044480"))
    positions = [int(x) for x in os.environ.get("ML4_CTX_POSITIONS", ",".join(map(str, CTX_POSITIONS))).split(",")]
    pl = [int(x) for x in os.environ.get("ML4_CTX_PROFILE_LAYERS", "16,17,18").split(",")]
    layers = _layers()
    profiling = os.environ.get("TT_METAL_PROFILER_CPP_POST_PROCESS") == "1"
    result = {"max_seq": max_seq, "profile_layers": pl, "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
              "settings": {k: v for k, v in os.environ.items() if k.startswith("MISTRAL_LARGE4_1T_")}, "positions": {}}

    def dump():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=1) + "\n")

    t0 = time.time()
    model = device_model(mesh_device, _Spec(CKPT, max_seq, CHUNK), layers, lm_head=False, num_users=1)
    meta = model.enable_meta()
    state = model.new_state(max_seq)

    def run(p, prof=None):
        meta.write(0, p, p + CHUNK)
        meta.unpack()
        state.slot = 0
        h = model.embed(_tokens(p), start=p)
        for i in layers:
            if prof is not None:
                prof.layer = i
            out = model.layer(i, h, p, state, end=p + CHUNK)
            model.free(h)
            h = out
            if prof is not None:
                prof.after_layer(i)
        model.free(h)
        model.sync()

    run(0)  # builds every layer (the weight load) and compiles
    result["build_seconds"] = round(time.time() - t0, 1)
    result["dram_free_mb"] = _dram_free_mb(mesh_device)
    dump()
    for p in positions:
        rec = {}
        try:
            t1 = time.time()
            run(p)
            rec["warm_wall_ms"] = round((time.time() - t1) * 1e3, 1)  # eager, host bound: not the traced time
            if profiling:
                prof = DeviceOps(mesh_device, pl)
                prof.drain()  # empty before the profiled run
                prof.tags.clear()
                prof.install()
                try:
                    run(p, prof)
                finally:
                    prof.uninstall()
                prof.drain()
                s = prof.summary()
                n = max(1, len(pl))
                ops = {f"{r['step']}:{r['op']}": round(r["ms_max"] / n, 4) for r in s["ops"][:20]}
                rec["ops_ms"] = {**_ctx_groups(s["ops"], n), **ops}
                rec["ms_per_layer"] = round(s["ms_total_max"] / n, 3)
                rec["programs_without_data"] = s["programs_without_data"]
                rec["profile"] = {k: v for k, v in s.items() if k != "ops"}
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:1500]
            print(f"position {p} failed: {rec['error']}", flush=True)
        result["positions"][str(p)] = rec
        print(f"position {p}: {json.dumps({k: rec.get(k) for k in ('warm_wall_ms', 'ms_per_layer', 'error')})} "
              f"{json.dumps({k: v for k, v in rec.get('ops_ms', {}).items() if k.startswith('group:')})}", flush=True)
        dump()
    assert any("ops_ms" in r for r in result["positions"].values()) or not profiling
