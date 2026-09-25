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


def enable(mesh) -> None:
    import ttnn

    dev_to_chip = {int(d): c for c, d in enumerate(mesh.get_device_ids())}
    _P.update(
        mesh=mesh,
        current=None,
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


def disable() -> None:
    _P["mesh"] = None


def result() -> dict:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _P.items() if k != "mesh"}


def _charge(name: str) -> None:
    import ttnn

    p = _P
    ttnn.synchronize_device(p["mesh"])
    ttnn.ReadDeviceProfiler(p["mesh"])
    per_dev, n = {}, 0
    for dev, programs in (ttnn.get_latest_programs_perf_data() or {}).items():
        for prog in programs:
            e = (getattr(prog, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                per_dev[dev] = per_dev.get(dev, 0.0) + float(e.duration)
                n += 1
    now = time.time()
    cur = p["current"]
    if cur is not None:
        p["kernel_ns"][cur] = p["kernel_ns"].get(cur, 0.0) + (max(per_dev.values()) if per_dev else 0.0)
        dev = p["kernel_ns_dev"].setdefault(cur, {})
        for d, ns in per_dev.items():
            chip = p["dev_to_chip"].get(int(d), int(d))
            dev[chip] = dev.get(chip, 0.0) + ns
        p["programs"][cur] = p["programs"].get(cur, 0) + n
        p["wall_s"][cur] = p["wall_s"].get(cur, 0.0) + (now - p["t"])
    p["current"], p["t"] = name, time.time()


def signpost(name: str) -> None:
    if _TRACY:
        import ttnn

        ttnn.tracy_message(f"`TT_SIGNPOST: {name}`")
    if _P["mesh"] is not None:
        _charge(re.sub(r"^L\d+\.", "", name))


def phase_of(section: str) -> str:
    """'attn.sdpa' -> 'attn'; a section without a dot is its own phase."""
    return section.split(".", 1)[0]
