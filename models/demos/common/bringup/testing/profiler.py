# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-section device time without Tracy (Tracy host capture crashed on the ERNIE box).

Model code calls ``signpost("L{i}.<phase>.<op>")`` before each section, e.g. ``L3.attn.sdpa``, ``L3.moe.experts``.
While profiling is enabled, each signpost syncs the mesh, reads the device profiler and charges every program
that ran since the previous signpost to the previous section, per chip. The layer prefix is dropped, so a section
sums over layers. Signposts are free when profiling is off.

Needs TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1;
without the last two the perf data comes back empty.

Op mode (``enable(mesh, ops=True)``, F43): every outermost ttnn operation call is also a charge point, so each section
is broken down into the ttnn ops that ran in it, per layer, in execution order (``op_ns``: "L3.attention.qkv" ->
[{op, shape, calls, ns, ns_dev}]). Programs no wrapped op launched are charged to "(other)". It syncs after every op,
so the host timing of that run is meaningless; device kernel durations are not affected.
"""

from __future__ import annotations

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


def enable(mesh, ops: bool = False) -> None:
    import ttnn

    dev_to_chip = {int(d): c for c, d in enumerate(mesh.get_device_ids())}
    _P.update(
        mesh=mesh,
        current=None,
        current_full=None,
        layer=None,
        op_ns={},
        depth=0,
        kernel_ns={},
        kernel_ns_dev={},
        programs={},
        wall_s={},
        t=time.time(),
        dev_to_chip=dev_to_chip,
    )
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    ttnn.get_latest_programs_perf_data()
    if ops:
        _patch_ops()


def disable() -> None:
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
    per_dev, n = {}, 0
    for dev, programs in data.items():
        for prog in programs:
            e = (getattr(prog, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                per_dev[dev] = per_dev.get(dev, 0.0) + float(e.duration)
                n += 1
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


def _book_op(op: str, shape: str, per_dev: dict, n: int) -> None:
    p = _P
    if p["current_full"] is None or not n:
        return
    rows = p["op_ns"].setdefault(p["current_full"], [])
    row = rows[-1] if rows and rows[-1]["op"] == op and rows[-1]["shape"] == shape else None  # execution order
    if row is None:
        row = {"op": op, "shape": shape, "calls": 0, "programs": 0, "ns": 0.0, "ns_dev": {}}
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
        _charge(name)


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
        name = self.python_fully_qualified_name
        _book_op(name[5:] if name.startswith("ttnn.") else name, _shape_of(args, kwargs), per_dev, n)
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
