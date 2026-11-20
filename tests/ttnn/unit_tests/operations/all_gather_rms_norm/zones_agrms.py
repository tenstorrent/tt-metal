# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage zone breakdown of a perf case (AGRMS_ZONES=1 device zones), same launch as eval/ccl_perf: one traced
pattern = spin + ITERS calls, so every chip starts together. Not collected by default (no test_ prefix):

    CCL_PERF_CASES=0 scripts/run_safe_pytest.sh --no-precompile \\
        tests/ttnn/unit_tests/operations/all_gather_rms_norm/zones_agrms.py -s

Prints, per chip, the median over the traced calls of every zone's start / end relative to the earliest kernel
start of that call on that chip (us), per (core, RISC) for the port cores and min/median/max over compute cores.
"""

from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

os.environ["AGRMS_ZONES"] = "1"
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")

import pytest  # noqa: E402

from eval import ccl_perf  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.conftest import _fabric_mesh_cache  # noqa: E402,F401
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases  # noqa: E402
from eval.golden_tests.all_gather_rms_norm.topology import resolve_pinned_mesh, system_mesh_shape  # noqa: E402
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm  # noqa: E402
import ttnn.operations.all_gather_rms_norm.all_gather_rms_norm_program_descriptor as pdm  # noqa: E402

ITERS = int(os.environ.get("AGRMS_ZONE_ITERS", "5"))
PATTERNS = int(os.environ.get("AGRMS_ZONE_PATTERNS", "4"))
FREQ_MHZ = 1350.0


def _selected():
    cases = _loose_cases()
    idx = [int(i) for i in os.environ.get("CCL_PERF_CASES", "0").split(",")]
    return [pytest.param(*cases[i].values, i, id=f"case{i}") for i in idx]


@pytest.fixture
def perf_mesh(request, _fabric_mesh_cache):  # noqa: F811
    extras, axes = request.node.callspec.params["extras"], request.node.callspec.params["axes"]
    shape = system_mesh_shape()
    if extras.get("mesh_shape"):
        resolved, _ = resolve_pinned_mesh(extras["mesh_shape"], axes["topology"], axes["cluster_axis"])
        shape = resolved or shape
    return _fabric_mesh_cache.get(shape, trace_region_size=ccl_perf.TRACE_REGION_SIZE)


def _parse(path: Path):
    """{chip: [(x, y, risc, zone, start_cycle, end_cycle)]}"""
    rows = defaultdict(list)
    opened = {}
    with open(path) as f:
        next(f)  # arch / freq line
        rd = csv.DictReader(f, skipinitialspace=True)
        for r in rd:
            chip = int(r["PCIe slot"])
            key = (chip, int(r["core_x"]), int(r["core_y"]), r["RISC processor type"], r["zone name"])
            t = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                opened.setdefault(key, []).append(t)
            elif r["type"] == "ZONE_END" and opened.get(key):
                s = opened[key].pop()
                rows[chip].append((key[1], key[2], key[3], key[4], s, t))
    return rows


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("inputs,axes,extras,index", _selected())
def test_zones(inputs, axes, extras, index, perf_mesh):
    import ttnn

    call, check = prepare_all_gather_rms_norm(
        inputs, mesh_device=perf_mesh, extras=extras, op=all_gather_rms_norm, **axes
    )
    out = call()
    if not os.environ.get("AGRMS_ABL"):
        check(out)
    info = pdm.LAST_INFO
    print("PLAN", {k: v for k, v in (info or {}).items() if not isinstance(v, (list, dict))})
    print("GROUPS", (info or {}).get("group_cores"))
    spin = ccl_perf.spin_op(perf_mesh)
    spin()
    call()
    ttnn.synchronize_device(perf_mesh)
    ttnn.ReadDeviceProfiler(perf_mesh)
    tid = ttnn.begin_trace_capture(perf_mesh, cq_id=0)
    spin()
    for _ in range(ITERS):
        call()
    ttnn.end_trace_capture(perf_mesh, tid, cq_id=0)
    for _ in range(PATTERNS):
        ttnn.execute_trace(perf_mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(perf_mesh)
    ttnn.release_trace(perf_mesh, tid)
    ttnn.ReadDeviceProfiler(perf_mesh)
    # the device log is written when the mesh closes: parse it afterwards with `python <this file> [csv]`


def report(log: Path):
    rows = _parse(log)
    for chip in sorted(rows):
        zs = rows[chip]
        # firmware zones mark the calls: per call, the earliest BRISC-FW start on the chip
        fw = sorted({s for (x, y, rv, z, s, e) in zs if z.endswith("-FW")})
        # group zone instances by call: the call = latest kernel start <= zone start; spin calls have few cores
        starts = defaultdict(list)
        for x, y, rv, z, s, e in zs:
            if z == "BRISC-FW":
                starts[s].append((x, y))
        call_starts = sorted(s for s, c in starts.items())
        # cluster FW starts into calls (cores of one call start within ~2 us)
        groups = []
        for s in call_starts:
            if groups and s - groups[-1][-1] < 2 * FREQ_MHZ * 1000 / 1000 * 1:
                groups[-1].append(s)
            else:
                groups.append([s])
        big = [g[0] for g in groups if sum(len(starts[s]) for s in g) > 8]
        big = big[-(ITERS * PATTERNS) :]
        per_zone = defaultdict(lambda: defaultdict(list))  # (role, zone) -> call -> [(s, e)]
        for x, y, rv, z, s, e in zs:
            if z.endswith("-FW") or z.endswith("-KERNEL"):
                if z != "BRISC-KERNEL" and z != "NCRISC-KERNEL" and z != "TRISC-KERNEL":
                    continue
            ci = None
            for i, b in enumerate(big):
                if b <= s:
                    ci = i
            if ci is None:
                continue
            per_zone[(x, y, rv, z)][ci].append((s - big[ci], e - big[ci]))
        print(f"== chip {chip}: {len(big)} calls")
        agg = defaultdict(list)
        for key, calls in per_zone.items():
            x, y, rv, z = key
            ss = [min(v[0] for v in c) for c in calls.values()]
            ee = [max(v[1] for v in c) for c in calls.values()]
            agg[(rv, z)].append((x, y, st.median(ss) / FREQ_MHZ, st.median(ee) / FREQ_MHZ))
        for (rv, z), lst in sorted(agg.items(), key=lambda kv: min(v[2] for v in kv[1])):
            if z.startswith("p") or len(lst) <= 8:
                for x, y, s, e in sorted(lst):
                    print(f"  {rv:7s} {z:14s} ({x},{y})  {s:7.2f} -> {e:7.2f}  ({e - s:6.2f})")
            else:
                s_ = sorted(v[2] for v in lst)
                e_ = sorted(v[3] for v in lst)
                print(
                    f"  {rv:7s} {z:14s} n={len(lst):3d} start min/med/max {s_[0]:6.2f}/{st.median(s_):6.2f}/{s_[-1]:6.2f}"
                    f"  end min/med/max {e_[0]:6.2f}/{st.median(e_):6.2f}/{e_[-1]:6.2f}"
                )


if __name__ == "__main__":
    import sys

    report(Path(sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"))
