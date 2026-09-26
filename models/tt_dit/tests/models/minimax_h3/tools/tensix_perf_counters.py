# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-op Tensix hardware perf counters from a Tracy device log, grouped by run host ID.

    TT_METAL_DEVICE_ARCH=wormhole_b0 python -m tracy -r -p --profiler-capture-perf-counters=fpu,pack,unpack,instrn \
        --perf-counter-multipass <bench.py> ...
    python models/tt_dit/tests/models/minimax_h3/tools/tensix_perf_counters.py generated/profiler/.logs/profile_log_device.csv \
        '{"case A": ["1024","2048"], "case B": ["3072","4096"]}'

Reads the timer-id 9090 marker rows the BRISC firmware writes after each kernel (tt_metal/tools/profiler/perf_counters.hpp)
and prints, per group of run host IDs, the mean [min..max] over cores of the derived metrics (FPU util, src write requests
and how often they were blocked by overwrite protection or the port, thread stalls, src valid/clear waits, instruction
availability). Use it when tracy's own post-processing cannot merge the multipass logs (run-host-id assertion in
process_ops_logs.py); TT_METAL_DEVICE_ARCH is needed on a Galaxy or the parent tracy process holds the chip lock. Not a test.
"""
import collections
import csv
import json
import statistics as st
import sys

F = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
groups = {"2x2 fp32 (production)": {"1024", "2048", "3072", "4096"}, "4x2 fp32-off": {"5120", "6144", "7168", "8192"}}
if len(sys.argv) > 2:
    groups = json.loads(sys.argv[2])
rows = list(csv.DictReader(open(F).read().split("\n", 1)[1].splitlines()))
# per (run, core) -> counter -> (value, ref)
data = collections.defaultdict(dict)
for r in rows:
    if r[" timer_id"].strip() != "9090":
        continue
    md = json.loads(r[" meta data"].replace(";", ","))
    data[(r[" run host ID"].strip(), r[" core_x"], r[" core_y"])][md["counter type"]] = (md["value"], md["ref cnt"])


def agg(runs, f):
    vals = []
    for (run, x, y), c in data.items():
        if run not in runs:
            continue
        try:
            vals.append(f(c))
        except (KeyError, ZeroDivisionError):
            pass
    return (st.mean(vals), min(vals), max(vals), len(vals)) if vals else None


ref = lambda c: next(iter(c.values()))[1]
V = lambda c, k: c[k][0]
metrics = [
    ("elapsed cycles (ref) per core", lambda c: ref(c)),
    ("FPU util % (FPU_COUNTER/ref)", lambda c: 100 * V(c, "FPU_COUNTER") / ref(c)),
    ("FPU active cycles", lambda c: V(c, "FPU_COUNTER")),
    ("math instr started / available %", lambda c: 100 * V(c, "MATH_INSTRN_STARTED") / V(c, "MATH_INSTRN_AVAILABLE")),
    ("math src data ready / ref %", lambda c: 100 * V(c, "MATH_SRC_DATA_READY") / ref(c)),
    ("fidelity stall cycles", lambda c: V(c, "MATH_FIDELITY_STALL")),
    ("unpack0 busy % (srcA side)", lambda c: 100 * V(c, "UNPACK0_BUSY_THREAD0") / ref(c)),
    ("unpack1 busy % (srcB side)", lambda c: 100 * V(c, "UNPACK1_BUSY_THREAD0") / ref(c)),
    ("srcA write req % of ref", lambda c: 100 * V(c, "SRCA_WRITE_REQ") / ref(c)),
    ("srcB write req % of ref", lambda c: 100 * V(c, "SRCB_WRITE_REQ") / ref(c)),
    (
        "srcA write blocked by overwrite % of req",
        lambda c: 100 * (1 - V(c, "SRCA_WRITE_NOT_BLOCKED_OVR") / V(c, "SRCA_WRITE_REQ")),
    ),
    (
        "srcB write blocked by overwrite % of req",
        lambda c: 100 * (1 - V(c, "SRCB_WRITE_NOT_BLOCKED_OVR") / V(c, "SRCB_WRITE_REQ")),
    ),
    (
        "srcA write blocked by port % of req",
        lambda c: 100 * (1 - V(c, "SRCA_WRITE_NOT_BLOCKED_PORT") / V(c, "SRCA_WRITE_REQ")),
    ),
    (
        "srcB write blocked by port % of req",
        lambda c: 100 * (1 - V(c, "SRCB_WRITE_NOT_BLOCKED_PORT") / V(c, "SRCB_WRITE_REQ")),
    ),
    ("packer busy % (PACKER_BUSY/ref)", lambda c: 100 * V(c, "PACKER_BUSY") / ref(c)),
    ("T0 unpack thread stall %", lambda c: 100 * V(c, "THREAD_STALLS_0") / ref(c)),
    ("T1 math thread stall %", lambda c: 100 * V(c, "THREAD_STALLS_1") / ref(c)),
    ("T2 pack thread stall %", lambda c: 100 * V(c, "THREAD_STALLS_2") / ref(c)),
    ("T0 instr issued / ref", lambda c: V(c, "THREAD_INSTRUCTIONS_0") / ref(c)),
    ("T1 instr issued / ref", lambda c: V(c, "THREAD_INSTRUCTIONS_1") / ref(c)),
    ("T2 instr issued / ref", lambda c: V(c, "THREAD_INSTRUCTIONS_2") / ref(c)),
    ("wait srcA valid % (T1)", lambda c: 100 * V(c, "WAITING_FOR_SRCA_VALID") / ref(c)),
    ("wait srcB valid % (T1)", lambda c: 100 * V(c, "WAITING_FOR_SRCB_VALID") / ref(c)),
    ("wait srcA clear % (T0)", lambda c: 100 * V(c, "WAITING_FOR_SRCA_CLEAR") / ref(c)),
    ("wait srcB clear % (T0)", lambda c: 100 * V(c, "WAITING_FOR_SRCB_CLEAR") / ref(c)),
    ("wait math idle T1 %", lambda c: 100 * V(c, "WAITING_FOR_MATH_IDLE_1") / ref(c)),
    ("wait unpack idle T0 %", lambda c: 100 * V(c, "WAITING_FOR_UNPACK_IDLE_0") / ref(c)),
    ("wait pack idle T2 %", lambda c: 100 * V(c, "WAITING_FOR_PACK_IDLE_2") / ref(c)),
    ("wait cfg idle T0 %", lambda c: 100 * V(c, "WAITING_FOR_CFG_IDLE_0") / ref(c)),
    ("THCON instr avail T0 %", lambda c: 100 * V(c, "THCON_INSTRN_AVAILABLE_0") / ref(c)),
    ("MATH instr avail T1 %", lambda c: 100 * V(c, "MATH_INSTRN_AVAILABLE_1") / ref(c)),
    ("UNPACK instr avail T0 %", lambda c: 100 * V(c, "UNPACK_INSTRN_AVAILABLE_0") / ref(c)),
    ("PACK instr avail T2 %", lambda c: 100 * V(c, "PACK_INSTRN_AVAILABLE_2") / ref(c)),
]
names = sorted({k for c in data.values() for k in c})
print("counters present:", len(names))
print(f"{'metric':46s}" + "".join(f"{g:>28s}" for g in groups))
for name, f in metrics:
    line = f"{name:46s}"
    for g, runs in groups.items():
        a = agg(runs, f)
        line += f"{a[0]:>14.2f} [{a[1]:.1f}..{a[2]:.1f}]".rjust(28) if a else f"{'n/a':>28s}"
    print(line)
missing = [
    n
    for n in [
        "WAITING_FOR_SRCA_VALID",
        "WAITING_FOR_SRCB_VALID",
        "WAITING_FOR_SRCA_CLEAR",
        "WAITING_FOR_SRCB_CLEAR",
        "MATH_FIDELITY_STALL",
        "THREAD_STALLS_1",
    ]
    if n not in names
]
print("missing counters:", missing)
print("all names:", names)
