# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-section device time without Tracy (Tracy host capture crashed on the ERNIE box).

Model code calls ``signpost("L{i}.<phase>.<op>")`` before each section, e.g. ``L3.attn.sdpa``, ``L3.moe.experts``.
Every program that ran between a signpost and the next one is charged to that section, per chip. The layer prefix is
dropped, so a section sums over layers. Signposts are free when profiling is off.

F155 (owner 2026-10-22): the framework measures device time with the program real-time profiler (``backend()``
"rt", the default; ``enable(rt=True)``): the dispatcher streams every program's start and end to a host callback, a
signpost only reads the device-operation id counter, and ``finish_rt()`` charges each program to the section (and
op call) whose id range holds it. No syncs, no profiler build; one run gives sections, per-op rows and the pipelined
timeline. Its time is a program's start to end at dispatch, so a program that waits inside (a CCL waiting for its
neighbour) is charged the wait. The device profiler (``backend()`` "device": spec perf.profiler or
BRINGUP_PROFILER=device, with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
TT_METAL_PROFILER_CPP_POST_PROCESS=1) is kept for cross-checks only: kernel time alone, but each signpost syncs.

Op mode (``enable(mesh, ops=True)``, F43): every outermost ttnn operation call is also a charge point, so each section
is broken down into the ttnn ops that ran in it, per layer, in execution order (``op_ns``: "L3.attention.qkv" ->
[{op, shape, calls, ns, ns_dev}]). Programs no wrapped op launched are charged to "(other)". It syncs after every op,
so the host timing of that run is meaningless; device kernel durations are not affected.
"""

from __future__ import annotations

import bisect
import os
import re
import time

PROFILER_ENV = {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
}
_TRACY = bool(os.environ.get("BRINGUP_TRACY_SIGNPOSTS"))
_P = {"mesh": None}
BACKENDS = ("rt", "device")


def backend(s=None) -> str:
    """F155: where device time comes from: BRINGUP_PROFILER (one run), else the spec's perf.profiler, else "rt"."""
    b = os.environ.get("BRINGUP_PROFILER") or (s.get("perf.profiler") if s is not None else None) or "rt"
    if b not in BACKENDS:
        raise ValueError(f"profiler backend {b!r}: one of {BACKENDS} (perf.profiler / BRINGUP_PROFILER)")
    return b


def require_rt() -> None:
    """F155: no silent fallback to another source of device time."""
    import ttnn

    if not ttnn.device.IsProgramRealtimeProfilerActive():
        raise RuntimeError(
            "the program real-time profiler is not active on this mesh: the framework measures device time with it "
            "(F155) and does not fall back. For a device-profiler run set perf.profiler: device in the spec (or "
            "BRINGUP_PROFILER=device) with " + " ".join(f"{k}={v}" for k, v in PROFILER_ENV.items())
        )


def require_device_profiler() -> None:
    missing = [k for k, v in PROFILER_ENV.items() if os.environ.get(k) != v]
    if missing:
        raise RuntimeError(
            f"profiler backend 'device' needs {' '.join(f'{k}=1' for k in missing)} (set before the device opens); "
            "the default backend is the real-time profiler (F155)"
        )


def enable(mesh, ops: bool = False, timeline: bool = False, calls: bool = False, rt: bool = False) -> None:
    """ops: op mode (sync + drain per outermost ttnn call). timeline: no syncs at all; each outermost call's host
    dispatch time is timed and the device programs are read once, by ``collect_timeline()`` (F44).
    calls (with ops): also keep every call individually in ``calls`` with its per-chip device ns and full metadata
    (every tensor argument's and output's per-device shape / dtype / layout / buffer, the CCL and compute kwargs),
    for the per-op report (testing/op_report.py).
    rt (F155): the program real-time profiler instead of the device profiler: no syncs and no profiler build; each
    signpost records the device-operation id it starts at, each outermost call (with ops / calls) the id range it
    launched ([id before, id after)), and ``finish_rt()`` (after the run) charges every real-time record (per chip,
    per program: end - start at the chip's clock) to its section, its call and, with calls, the timeline."""
    import ttnn

    dev_to_chip = {int(d): c for c, d in enumerate(mesh.get_device_ids())}
    _P.update(
        mesh=mesh,
        current=None,
        current_full=None,
        layer=None,
        timeline=timeline,
        seq=[],
        last_n_dev={},
        op_ns={},
        depth=0,
        kernel_ns={},
        kernel_ns_dev={},
        programs={},
        wall_s={},
        t=time.time(),
        dev_to_chip=dev_to_chip,
        keep_calls=calls,
        calls=[],
        rt=rt,
        rt_records={},
        rt_handle=None,
    )
    ttnn.synchronize_device(mesh)
    if rt:
        require_rt()
        recs = _P["rt_records"]
        _P["markers"] = [(ttnn._ttnn.get_device_operation_id(), None, None)]  # (first id, section, full key)
        _P["keep_calls"] = calls or ops
        dropped = _P["rt_dropped"] = [0]

        def on_batch(batch):
            dropped[0] += int(getattr(batch, "dropped", 0) or 0)
            for r in batch.records:  # [runtime id] -> {chip: [ns, start ns, end ns, programs]}
                f = float(r.frequency)  # GHz: cycles / f = ns
                if f <= 0 or r.end_timestamp <= r.start_timestamp:
                    continue
                st, en = r.start_timestamp / f, r.end_timestamp / f
                e = recs.setdefault(int(r.runtime_id), {}).setdefault(int(r.chip_id), [0.0, st, en, 0])
                e[0] += en - st
                e[1], e[2], e[3] = min(e[1], st), max(e[2], en), e[3] + 1

        _P["rt_handle"] = ttnn.device.RegisterProgramRealtimeProfilerCallback(on_batch)
        if calls or ops:
            _patch_ops()
        return
    ttnn.ReadDeviceProfiler(mesh)
    ttnn.get_latest_programs_perf_data()
    if ops or timeline:
        _patch_ops()


def finish_rt(settle_s: float = 1.0) -> list:
    """rt mode: sync, let the receiver deliver the last records, unregister, then charge the records: per section
    (kernel_ns, kernel_ns_dev, programs, as the device profiler books them), per call (ns_dev, programs) and, with
    calls, the op rows (op_ns, with "(other)" for programs no wrapped call launched) and the pipelined timeline
    (``timeline_rt``). Returns the calls that launched programs."""
    import ttnn

    p = _P
    ttnn.synchronize_device(p["mesh"])
    time.sleep(settle_s)
    if p.get("rt_handle") is not None:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(p["rt_handle"])
        p["rt_handle"] = None
    if p.get("rt_dropped", [0])[0]:
        raise RuntimeError(
            f"the real-time profiler dropped {p['rt_dropped'][0]} record(s): the receiver could not keep up, the "
            "section times would under-report"
        )
    charge_rt(p, p["rt_records"])
    return p["calls"]


def charge_rt(p: dict, recs: dict) -> None:
    """The bookkeeping half of finish_rt (no device; the selftests feed it records)."""
    chip_of = p["dev_to_chip"]
    to_chip = lambda d: chip_of.get(int(d), int(d))  # noqa: E731
    ids = sorted(recs)

    def per_chip(lo: int, hi: int) -> tuple[dict, int]:
        ns, n = {}, 0
        for i in ids[bisect.bisect_left(ids, lo) : bisect.bisect_left(ids, hi)]:
            for d, e in recs[i].items():
                c = to_chip(d)
                ns[c] = ns.get(c, 0.0) + e[0]
                n += e[3]
        return ns, n

    # sections: every signposted range but the last (the one still open when the run ended), as _charge books them
    marks = p.get("markers") or []
    for (lo, sec, _), (hi, _, _) in zip(marks, marks[1:]):
        if sec is None:
            continue
        ns, n = per_chip(lo, hi)
        p["kernel_ns"][sec] = p["kernel_ns"].get(sec, 0.0) + (max(ns.values()) if ns else 0.0)
        dev = p["kernel_ns_dev"].setdefault(sec, {})
        for c, v in ns.items():
            dev[c] = dev.get(c, 0.0) + v
        p["programs"][sec] = p["programs"].get(sec, 0) + n
    calls = p["calls"]
    for c in calls:
        a, b = c.pop("ids")
        c["_ids"] = (a, b)
        c["ns_dev"], c["programs"] = per_chip(a, b)
    calls = [c for c in calls if c["programs"]]
    if not p.get("keep_calls"):
        p["calls"] = calls
        return
    # the timeline on the critical chip (the one whose programs span longest), call by call in launch order
    spans = {}
    for i in ids:
        for d, e in recs[i].items():
            c = to_chip(d)
            lo, hi = spans.get(c, (e[1], e[2]))
            spans[c] = (min(lo, e[1]), max(hi, e[2]))
    crit = max(spans, key=lambda c: spans[c][1] - spans[c][0]) if spans else None
    prev = spans[crit][0] if crit is not None else 0.0
    owner = {i: k for k, c in enumerate(calls) for i in range(*c["_ids"])}
    on_crit = lambda i: [e for d, e in recs[i].items() if to_chip(d) == crit]  # noqa: E731
    for c in calls:
        c["kernel_ns"] = c["gap_ns"] = c["slot_ns"] = 0.0
    seen = set()
    for i in ids:  # launch order; a program no call launched still moves the critical chip's clock
        k = owner.get(i)
        if k is None:
            prev = max([prev] + [e[2] for e in on_crit(i)])
            continue
        if k in seen:
            continue
        seen.add(k)
        c = calls[k]
        mine = [e for j in range(*c["_ids"]) if j in recs for e in on_crit(j)]
        c["kernel_ns"] = sum(e[0] for e in mine)
        if mine:
            st, en = min(e[1] for e in mine), max(e[2] for e in mine)
            c["gap_ns"], c["slot_ns"] = max(0.0, st - prev), en - prev
            prev = max(prev, en)
    # op rows in execution order, back-to-back repeats merged; "(other)": programs in a section no call launched
    done = set()
    for (lo, sec, full), (hi, _, _) in zip(marks, marks[1:]):
        if sec is None:
            continue
        other = []
        for i in ids[bisect.bisect_left(ids, lo) : bisect.bisect_left(ids, hi)]:
            k = owner.get(i)
            if k is None:
                other.append(i)
                continue
            if other:
                _rt_row(p, full, {"op": "(other)", "shape": "", "mem": ""}, *per_chip_ids(recs, other, to_chip))
                other = []
            if k not in done:
                done.add(k)
                c = calls[k]
                _rt_row(p, c["key"], c, c["ns_dev"], c["programs"])
        if other:
            _rt_row(p, full, {"op": "(other)", "shape": "", "mem": ""}, *per_chip_ids(recs, other, to_chip))
    host = sum(c.get("host_ns", 0.0) for c in calls)
    p["timeline_rt"] = {
        "critical_chip": crit,
        "device_timeline_ms": round((spans[crit][1] - spans[crit][0]) / 1e6, 3) if crit is not None else 0.0,
        "kernel_ms": round(sum(c["kernel_ns"] for c in calls) / 1e6, 3),
        "gap_ms": round(sum(c["gap_ns"] for c in calls) / 1e6, 3),
        "host_dispatch_ms": round(host / 1e6, 3),
        "calls": len(calls),
    }
    for c in calls:
        c.pop("_ids", None)
    p["calls"] = calls


def per_chip_ids(recs: dict, ids: list, to_chip) -> tuple[dict, int]:
    ns, n = {}, 0
    for i in ids:
        for d, e in recs[i].items():
            c = to_chip(d)
            ns[c] = ns.get(c, 0.0) + e[0]
            n += e[3]
    return ns, n


def _rt_row(p: dict, key: str, c: dict, ns_dev: dict, n: int) -> None:
    """One call (or "(other)" run) into its section's op rows, merged with an identical previous row."""
    rows = p["op_ns"].setdefault(key, [])
    op, shape, mem = c["op"], c.get("shape", ""), c.get("mem", "")
    row = rows[-1] if rows and (rows[-1]["op"], rows[-1]["shape"], rows[-1]["mem"]) == (op, shape, mem) else None
    if row is None:
        row = {"op": op, "shape": shape, "mem": mem, "calls": 0, "programs": 0, "ns": 0.0, "ns_dev": {}}
        row.update(gap_ns=0.0, slot_ns=0.0, host_ns=0.0, kernel_ns=0.0)
        rows.append(row)
    row["calls"] += 1
    row["programs"] += n
    row["ns"] += max(ns_dev.values()) if ns_dev else 0.0
    for ch, v in ns_dev.items():
        row["ns_dev"][ch] = row["ns_dev"].get(ch, 0.0) + v
    for k in ("gap_ns", "slot_ns", "host_ns", "kernel_ns"):
        row[k] += c.get(k, 0.0)


def disable() -> None:
    if _P.get("rt_handle") is not None:  # rt mode left before finish_rt (an exception): drop the callback
        import ttnn

        ttnn.device.UnregisterProgramRealtimeProfilerCallback(_P["rt_handle"])
        _P["rt_handle"] = None
    _P["mesh"] = None
    _unpatch_ops()


def set_layer(i: int | None) -> None:
    """The layer the next signposts belong to (the profile loop sets it; op mode keys its rows "L{i}.<section>")."""
    _P["layer"] = i


def result() -> dict:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _P.items() if k != "mesh"}


def _drain() -> tuple[dict, int]:
    """Device kernel ns per device, and the program count, of every program since the last drain."""
    import ttnn

    _P["draining"] = True
    try:
        ttnn.synchronize_device(_P["mesh"])
        ttnn.ReadDeviceProfiler(_P["mesh"])
        data = ttnn.get_latest_programs_perf_data() or {}
    finally:
        _P["draining"] = False
    per_dev, n, n_dev = {}, 0, {}
    for dev, programs in data.items():
        for prog in programs:
            e = (getattr(prog, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                per_dev[dev] = per_dev.get(dev, 0.0) + float(e.duration)
                n_dev[int(dev)] = n_dev.get(int(dev), 0) + 1
                n += 1
    _P["last_n_dev"] = n_dev
    return per_dev, n


def _book(per_dev: dict, n: int) -> None:
    """Add drained programs to the current section."""
    p, cur = _P, _P["current"]
    if cur is None:
        return
    p["kernel_ns"][cur] = p["kernel_ns"].get(cur, 0.0) + (max(per_dev.values()) if per_dev else 0.0)
    dev = p["kernel_ns_dev"].setdefault(cur, {})
    for d, ns in per_dev.items():
        chip = p["dev_to_chip"].get(int(d), int(d))
        dev[chip] = dev.get(chip, 0.0) + ns
    p["programs"][cur] = p["programs"].get(cur, 0) + n


def _book_op(op: str, shape: str, per_dev: dict, n: int, mem: str = "") -> None:
    p = _P
    # every real call is in the sequence (the timeline run sees them all), even with 0 programs
    if op != "(other)" or n:
        p["seq"].append(
            {
                "key": p["current_full"],
                "op": op,
                "shape": shape,
                "mem": mem,
                "n_dev": dict(p["last_n_dev"]) if n else {},
            }
        )
    # before the first section (e.g. the embedding): in the sequence, not in a row
    if not n or p["current_full"] is None:
        return
    rows = p["op_ns"].setdefault(p["current_full"], [])
    same = rows and rows[-1]["op"] == op and rows[-1]["shape"] == shape and rows[-1].get("mem", "") == mem
    row = rows[-1] if same else None  # execution order: only back-to-back repeats merge
    if row is None:
        row = {"op": op, "shape": shape, "mem": mem, "calls": 0, "programs": 0, "ns": 0.0, "ns_dev": {}}
        rows.append(row)
    row["calls"] += 1
    row["programs"] += n
    row["ns"] += max(per_dev.values()) if per_dev else 0.0
    for d, ns in per_dev.items():
        chip = p["dev_to_chip"].get(int(d), int(d))
        row["ns_dev"][chip] = row["ns_dev"].get(chip, 0.0) + ns


def _charge(name: str) -> None:
    p = _P
    per_dev, n = _drain()
    _book(per_dev, n)
    _book_op("(other)", "", per_dev, n)
    now = time.time()
    if p["current"] is not None:
        p["wall_s"][p["current"]] = p["wall_s"].get(p["current"], 0.0) + (now - p["t"])
    layer = p.get("layer")
    p["current_full"] = f"L{layer}.{name}" if layer is not None and not re.match(r"^L\d+\.", name) else name
    p["current"], p["t"] = re.sub(r"^L\d+\.", "", name), time.time()


def signpost(name: str) -> None:
    if _TRACY:
        import ttnn

        ttnn.tracy_message(f"`TT_SIGNPOST: {name}`")
    if _P["mesh"] is not None:
        if _P.get("timeline") or _P.get("rt"):  # no sync: only the section name changes
            layer = _P.get("layer")
            _P["current_full"] = f"L{layer}.{name}" if layer is not None and not re.match(r"^L\d+\.", name) else name
            _P["current"] = re.sub(r"^L\d+\.", "", name)
            if _P.get("rt"):  # F155: the section starts at the next device-operation id
                import ttnn

                _P["markers"].append((ttnn._ttnn.get_device_operation_id(), _P["current"], _P["current_full"]))
        else:
            _charge(name)


def collect_timeline() -> dict:
    """After a timeline-mode run: one sync and one profiler read. Programs per chip in launch order, as
    (start_ns, end_ns, kernel_ns) on the chip's own clock."""
    import ttnn

    _P["draining"] = True
    try:
        ttnn.synchronize_device(_P["mesh"])
        ttnn.ReadDeviceProfiler(_P["mesh"])
        data = ttnn.get_latest_programs_perf_data() or {}
    finally:
        _P["draining"] = False
    progs = {}
    for dev, programs in data.items():
        rows = []
        for prog in programs:
            e = (getattr(prog, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if e is None:
                continue
            rows.append(
                (prog.program_execution_uid.runtime_id, int(e.start_timestamp), int(e.end_timestamp), float(e.duration))
            )
        rows.sort()
        cyc = [(b - a) / d for _, a, b, d in rows if d > 0 and b > a]
        f = sorted(cyc)[len(cyc) // 2] if cyc else 1.0  # cycles per ns (device clock)
        progs[_P["dev_to_chip"].get(int(dev), int(dev))] = [(a / f, b / f, d) for _, a, b, d in rows]
    return progs


def align_timeline(op_seq: list[dict], tl_seq: list[dict], progs: dict, dev_to_chip: dict) -> dict:
    """Split the timeline run's programs into op calls using the op-mode run's per-call program counts (the model
    is deterministic), then per call: kernel ns, the device idle gap before it and its slot
    (end - previous end), all on the critical chip (the one whose timeline is longest), and its host dispatch ns.
    Returns {"calls": [...], "summary": {...}} or {"error": ...}."""
    real = [c for c in op_seq if c["op"] != "(other)"]
    if [(c["key"], c["op"], c["shape"]) for c in real] != [(c["key"], c["op"], c["shape"]) for c in tl_seq]:
        return {
            "error": f"op sequences differ between the op-mode and timeline runs ({len(real)} vs {len(tl_seq)} calls)"
        }
    host = iter(c["host_ns"] for c in tl_seq)
    need = {}
    for c in op_seq:
        for d, k in c["n_dev"].items():
            chip = dev_to_chip.get(int(d), int(d))
            need[chip] = need.get(chip, 0) + k
    got = {c: len(v) for c, v in progs.items()}
    if any(got.get(c, 0) != k for c, k in need.items()):
        return {"error": f"device programs per chip {got} != op-mode counts {need} (profiler buffer too small?)"}
    span = {c: (v[-1][1] - v[0][0]) if v else 0.0 for c, v in progs.items()}
    crit = max(span, key=span.get)
    pos = {c: 0 for c in progs}
    prev_end = {c: v[0][0] for c, v in progs.items() if v}
    calls = []
    for c in op_seq:
        k_dev, g_crit, s_crit = {}, 0.0, 0.0
        for d, k in c["n_dev"].items():
            chip = dev_to_chip.get(int(d), int(d))
            mine = progs[chip][pos[chip] : pos[chip] + k]
            pos[chip] += k
            k_dev[chip] = sum(x[2] for x in mine)
            if mine:
                st, en = min(x[0] for x in mine), max(x[1] for x in mine)
                if chip == crit:
                    g_crit, s_crit = max(0.0, st - prev_end[chip]), en - prev_end[chip]
                prev_end[chip] = max(prev_end[chip], en)
        # the critical chip's own kernel time: in a pipelined run a collective's kernel on an early chip includes
        # waiting for the others, so the slowest chip per op would count that wait (MiMo all_reduce: 10.6 ms synced,
        # 25.0 ms as max over chips pipelined)
        calls.append(
            {
                "key": c["key"],
                "op": c["op"],
                "shape": c["shape"],
                "mem": c.get("mem", ""),
                "programs": sum(c["n_dev"].values()),
                "kernel_ns": k_dev.get(crit, 0.0),
                "gap_ns": g_crit,
                "slot_ns": s_crit,
                "host_ns": 0.0 if c["op"] == "(other)" else next(host),
            }
        )
    summary = {
        "critical_chip": crit,
        "device_timeline_ms": round(span[crit] / 1e6, 3),
        "kernel_ms": round(sum(x["kernel_ns"] for x in calls) / 1e6, 3),
        "gap_ms": round(sum(x["gap_ns"] for x in calls) / 1e6, 3),
        "host_dispatch_ms": round(sum(x["host_ns"] for x in calls) / 1e6, 3),
        "calls": len(calls),
    }
    return {"calls": calls, "summary": summary}


_DT = {
    "BFLOAT16": "bf16",
    "BFLOAT8_B": "bfp8",
    "BFLOAT4_B": "bfp4",
    "FLOAT32": "fp32",
    "UINT32": "u32",
    "INT32": "i32",
    "UINT16": "u16",
    "UINT8": "u8",
}


_ML = {
    "INTERLEAVED": "interleaved",
    "HEIGHT_SHARDED": "height-sharded",
    "WIDTH_SHARDED": "width-sharded",
    "BLOCK_SHARDED": "block-sharded",
    "ND_SHARDED": "nd-sharded",
}


def _mem_of_tensor(t) -> str:
    """'DRAM interleaved', 'L1 height-sharded 64 cores', '+ RM' for a row-major tensor."""
    try:
        mc = t.memory_config()
        where = str(mc.buffer_type).split(".")[-1]
        lay = str(mc.memory_layout).split(".")[-1]
        txt = f"{where} {_ML.get(lay, lay.lower())}"
        spec = mc.shard_spec if mc.is_sharded() else None
        if spec is not None:
            txt += f" {spec.grid.num_cores()} cores"
        if "ROW_MAJOR" in str(t.layout):
            txt += " RM"
        return txt
    except Exception:
        return "?"


def _tensors(x, limit=2) -> list:
    import ttnn

    out = []
    for a in x if isinstance(x, (list, tuple)) else [x]:
        if isinstance(a, ttnn.Tensor):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.extend(t for t in a if isinstance(t, ttnn.Tensor))
        if len(out) >= limit:
            break
    return out[:limit]


def _desc(t) -> dict:
    """Per-device shape, dtype, layout and buffer of one tensor (the op report's cost models read these)."""
    try:
        mc = t.memory_config()
        return {
            "shape": [int(x) for x in t.shape],
            "dtype": str(t.dtype).split(".")[-1],
            "layout": "RM" if "ROW_MAJOR" in str(t.layout) else "TILE",
            "buffer": str(mc.buffer_type).split(".")[-1],
            "sharded": bool(mc.is_sharded()),
        }
    except Exception:
        return {"shape": [], "dtype": "?", "layout": "?", "buffer": "?", "sharded": False}


_KW = (
    "cluster_axis",
    "num_links",
    "dim",
    "in_dim",
    "out_dim",
    "topology",
    "kv_len",
    "chunk_start_idx",
    "key_compression_ratio",
    "k_chunk_size",
    "v_dim",
    "block_cyclic_chunk_local",
    "num_heads",
)


def _kw_of(kwargs) -> dict:
    """The scalar kwargs the op report uses: CCL geometry, attention extents, and from a compute config the math
    fidelity / fp32 accumulation / packer L1 accumulation, from a program config its type."""
    out = {}
    for k in _KW:
        v = kwargs.get(k)
        if v is not None:
            out[k] = v if isinstance(v, (int, float, bool)) else str(v).split(".")[-1]
    ckc = kwargs.get("compute_kernel_config")
    if ckc is not None:
        for f in ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc"):
            v = getattr(ckc, f, None)
            if v is not None:
                out[f] = v if isinstance(v, bool) else str(v).split(".")[-1]
    for k in ("program_config", "config"):
        pc = kwargs.get(k)
        if pc is not None:
            out["program_config"] = type(pc).__name__
    return out


def _mem_of(args, kwargs, out) -> str:
    """Where the first two tensor inputs and the outputs live: 'DRAM interleaved · L1 height-sharded 64 cores ->
    DRAM interleaved'."""
    ins = _tensors(list(args) + list(kwargs.values()))
    outs = _tensors(out)
    return " · ".join(_mem_of_tensor(t) for t in ins) + " -> " + " · ".join(_mem_of_tensor(t) for t in outs)


def _shape_of(args, kwargs) -> str:
    """'5120x4096 bf16 · 4096x3456 bfp8': the first two tensor arguments (per-device shape)."""
    import ttnn

    out = []
    flat = []
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, (list, tuple)) and a and all(isinstance(t, ttnn.Tensor) for t in a):
            flat.extend(a[:2])  # concat([a, b]), all_gather lists
        else:
            flat.append(a)
    for a in flat:
        if isinstance(a, ttnn.Tensor):
            try:
                dims = [int(x) for x in a.shape]
                while len(dims) > 2 and dims[0] == 1:
                    dims = dims[1:]
                dt = str(a.dtype).split(".")[-1]
                out.append("x".join(map(str, dims)) + " " + _DT.get(dt, dt.lower()))
            except Exception:
                out.append("?")
            if len(out) == 2:
                break
    return " · ".join(out)


def _patch_ops() -> None:
    import ttnn.decorators as D

    if getattr(D.FastOperation, "_bringup_orig_call", None) is not None:
        return
    orig = D.FastOperation.__call__

    def call(self, *args, **kwargs):
        p = _P
        if p["mesh"] is None or p.get("draining"):
            return orig(self, *args, **kwargs)
        if p["depth"]:  # an op called from inside another op: charged to the outer one
            p["depth"] += 1
            try:
                return orig(self, *args, **kwargs)
            finally:
                p["depth"] -= 1
        name = self.python_fully_qualified_name
        name = name[5:] if name.startswith("ttnn.") else name
        if p.get("rt"):  # no sync: the call's device-operation id range, joined with the real-time records later
            import ttnn

            p["depth"] = 1
            i0 = ttnn._ttnn.get_device_operation_id()
            t0 = time.perf_counter_ns()
            try:
                out = orig(self, *args, **kwargs)
            finally:
                p["depth"] = 0
            host_ns = time.perf_counter_ns() - t0
            i1 = ttnn._ttnn.get_device_operation_id()
            if p.get("keep_calls") and i1 > i0 and p["current_full"] is not None:
                p["calls"].append(
                    {
                        "key": p["current_full"],
                        "layer": p.get("layer"),
                        "op": name,
                        "shape": _shape_of(args, kwargs),
                        "mem": _mem_of(args, kwargs, out),
                        "host_ns": float(host_ns),
                        "ins": [_desc(t) for t in _tensors(list(args) + list(kwargs.values()), limit=8)],
                        "outs": [_desc(t) for t in _tensors(out, limit=4)],
                        "kw": _kw_of(kwargs),
                        "ids": (i0, i1),
                    }
                )
            return out
        if p.get("timeline"):  # host dispatch time of the call, no sync
            p["depth"] = 1
            t0 = time.perf_counter_ns()
            try:
                out = orig(self, *args, **kwargs)
            finally:
                p["depth"] = 0
            p["seq"].append(
                {
                    "key": p["current_full"],
                    "op": name,
                    "shape": _shape_of(args, kwargs),
                    "host_ns": time.perf_counter_ns() - t0,
                }
            )
            return out
        per_dev, n = _drain()  # anything launched outside a wrapped op since the last charge
        _book(per_dev, n)
        _book_op("(other)", "", per_dev, n)
        p["depth"] = 1
        try:
            out = orig(self, *args, **kwargs)
        finally:
            p["depth"] = 0
        per_dev, n = _drain()
        _book(per_dev, n)
        _book_op(name, _shape_of(args, kwargs), per_dev, n, _mem_of(args, kwargs, out))
        if p.get("keep_calls") and n and p["current_full"] is not None:
            p["calls"].append(
                {
                    "key": p["current_full"],
                    "layer": p.get("layer"),
                    "op": name,
                    "ins": [_desc(t) for t in _tensors(list(args) + list(kwargs.values()), limit=8)],
                    "outs": [_desc(t) for t in _tensors(out, limit=4)],
                    "kw": _kw_of(kwargs),
                    "ns_dev": {p["dev_to_chip"].get(int(d), int(d)): ns for d, ns in per_dev.items()},
                    "programs": n,
                }
            )
        return out

    D.FastOperation._bringup_orig_call = orig
    D.FastOperation.__call__ = call


def _unpatch_ops() -> None:
    try:
        import ttnn.decorators as D
    except ImportError:
        return
    orig = getattr(D.FastOperation, "_bringup_orig_call", None)
    if orig is not None:
        D.FastOperation.__call__ = orig
        D.FastOperation._bringup_orig_call = None


def phase_of(section: str) -> str:
    """'attn.sdpa' -> 'attn'; a section without a dot is its own phase."""
    return section.split(".", 1)[0]
