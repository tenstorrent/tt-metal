# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profiler enablement + in-process device-kernel-time readback.

The three flags below are the *only* way to make tt-metal's device profiler
populate the in-process getters (`ttnn.get_latest_programs_perf_data`).  They are
read once at first device open; there is no pybind setter.  We set them with
`setdefault` so an outer caller can still override, but the default is that a
plain `python run_matmul_sweep.py` just works with no env vars typed anywhere.

`TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1` keeps everything in memory -> no CSV
is ever written (matches the "no terminal tracy / no CSV" requirement).

Requires a build with `ENABLE_TRACY=ON` (verified present in build_Release).
"""

from __future__ import annotations

import os

_PROFILER_ENV = {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
}

for _k, _v in _PROFILER_ENV.items():
    os.environ.setdefault(_k, _v)

_KERNEL_DUR_KEY = "DEVICE KERNEL DURATION [ns]"


def profiler_ready() -> bool:
    """True iff the profiler flags are set (data will populate)."""
    return all(
        os.environ.get(k) == "1"
        for k in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_PROFILER_MID_RUN_DUMP", "TT_METAL_PROFILER_CPP_POST_PROCESS")
    )


def read_latest_kernel_ns(mesh_device) -> dict[int, list[int]]:
    """Flush the device profiler and return {chip_id: [kernel_duration_ns, ...]}.

    Returns the DEVICE KERNEL DURATION for every program captured since the last
    ReadDeviceProfiler call, per chip.  Run the op(s) of interest in isolation
    right before calling this so the list only contains what you measured.
    """
    import ttnn

    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    latest = ttnn.get_latest_programs_perf_data() or {}

    out: dict[int, list[int]] = {}
    for chip_id, programs in latest.items():
        durs: list[int] = []
        for prog in programs:
            res = prog.program_analyses_results.get(_KERNEL_DUR_KEY)
            if res is not None and res.duration:
                durs.append(int(res.duration))
        out[int(chip_id)] = durs
    return out


def dominant_kernel_ns(mesh_device) -> dict[int, int]:
    """Per-chip *largest* kernel duration since last read.

    Isolation heuristic: when a single measured op also drags in a tiny
    incidental reshard/typecast program, the matmul is the dominant (largest)
    kernel, so max-per-chip picks it out.
    """
    per_chip = read_latest_kernel_ns(mesh_device)
    return {chip: (max(durs) if durs else 0) for chip, durs in per_chip.items()}


def sum_kernel_ns(mesh_device) -> dict[int, int]:
    """Per-chip *sum* of kernel durations since last read (for CCL fan-out)."""
    per_chip = read_latest_kernel_ns(mesh_device)
    return {chip: (sum(durs) if durs else 0) for chip, durs in per_chip.items()}
