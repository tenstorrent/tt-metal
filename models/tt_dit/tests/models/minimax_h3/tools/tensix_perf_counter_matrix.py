# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Before / after hardware-counter table for the K-loop engine-isolation variants, from Tracy device logs.

    python models/tt_dit/tests/models/minimax_h3/tools/tensix_perf_counter_matrix.py "full, old=<csv>" "full, alt=<csv>" ...

Each argument is `column name=path` to a `profile_log_device.csv` written by a `--profiler-capture-perf-counters=fpu,pack,unpack,instrn
--perf-counter-multipass` run of transformer_op_single_device_bench.py with a mm_kloop_variants.py variant applied (copy
generated/profiler/.logs/profile_log_device.csv after each run; tracy's own merge asserts on the multipass logs). Prints a
Markdown table: K-loop pace from the KLOOP zone, FPU utilisation, source-ready fraction, thread stalls and waits, unpacker busy,
write-request / unblocked / refused cycles per K-tile step (per core / 63,504), L1-port refusals, packer busy. Not a test.
"""
import collections
import csv
import json
import statistics as st
import sys

STEPS = 63504


def load(path):
    rows = list(csv.DictReader(open(path).read().split("\n", 1)[1].splitlines()))
    cnt = collections.defaultdict(dict)
    starts = {}
    kl = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in rows:
        rid, core = r[" run host ID"].strip(), (r[" core_x"], r[" core_y"])
        if r[" timer_id"].strip() == "9090":
            md = json.loads(r[" meta data"].replace(";", ","))
            cnt[(rid, core)][md["counter type"]] = (md["value"], md["ref cnt"])
            continue
        if r[" RISC processor type"].strip() != "TRISC_1" or r[" zone name"].strip() != "KLOOP":
            continue
        key = (rid, core)
        tm = int(r[" time[cycles since reset]"])
        if r[" type"].strip() == "ZONE_START":
            starts[key] = tm
        elif r[" type"].strip() == "ZONE_END" and key in starts:
            kl[rid][core] += tm - starts.pop(key)
    return cnt, kl


def summarize(path):
    cnt, kl = load(path)
    runs = sorted({rid for rid, _ in cnt}, key=int)
    if not runs:
        return None

    # use all runs with counter data (2 calls x 2 passes merged); per-core mean
    def m(f):
        v = []
        for (rid, core), c in cnt.items():
            try:
                v.append(f(c, kl[rid][core] if core in kl[rid] else None))
            except (KeyError, ZeroDivisionError, TypeError):
                pass
        return st.mean(v) if v else float("nan")

    ref = lambda c: next(iter(c.values()))[1]
    V = lambda c, k: c[k][0]
    kloop_step = st.mean([sum(kl[r].values()) / len(kl[r]) for r in kl if kl[r]]) / STEPS if kl else float("nan")
    return {
        "K-loop pace, cycles per step (KLOOP zone)": kloop_step,
        "elapsed cycles per core (kernel, M)": m(lambda c, k: ref(c) / 1e6),
        "FPU util % of kernel": m(lambda c, k: 100 * V(c, "FPU_COUNTER") / ref(c)),
        "FPU util % of K loop (FPU active / KLOOP)": m(lambda c, k: 100 * V(c, "FPU_COUNTER") / k),
        "src data ready % of kernel": m(lambda c, k: 100 * V(c, "MATH_SRC_DATA_READY") / ref(c)),
        "math thread stalled %": m(lambda c, k: 100 * V(c, "THREAD_STALLS_1") / ref(c)),
        "T1 wait srcA valid / srcB valid %": (
            m(lambda c, k: 100 * V(c, "WAITING_FOR_SRCA_VALID") / ref(c)),
            m(lambda c, k: 100 * V(c, "WAITING_FOR_SRCB_VALID") / ref(c)),
        ),
        "unpacker0 / unpacker1 busy %": (
            m(lambda c, k: 100 * V(c, "UNPACK0_BUSY_THREAD0") / ref(c)),
            m(lambda c, k: 100 * V(c, "UNPACK1_BUSY_THREAD0") / ref(c)),
        ),
        "srcA / srcB write-request cycles per step": (
            m(lambda c, k: V(c, "SRCA_WRITE_REQ") / STEPS),
            m(lambda c, k: V(c, "SRCB_WRITE_REQ") / STEPS),
        ),
        "srcA / srcB unblocked write cycles per step": (
            m(lambda c, k: V(c, "SRCA_WRITE_NOT_BLOCKED_OVR") / STEPS),
            m(lambda c, k: V(c, "SRCB_WRITE_NOT_BLOCKED_OVR") / STEPS),
        ),
        "srcA / srcB refused by overwrite, % of requests": (
            m(lambda c, k: 100 * (1 - V(c, "SRCA_WRITE_NOT_BLOCKED_OVR") / V(c, "SRCA_WRITE_REQ"))),
            m(lambda c, k: 100 * (1 - V(c, "SRCB_WRITE_NOT_BLOCKED_OVR") / V(c, "SRCB_WRITE_REQ"))),
        ),
        "srcA / srcB refused by overwrite, cycles per step": (
            m(lambda c, k: (V(c, "SRCA_WRITE_REQ") - V(c, "SRCA_WRITE_NOT_BLOCKED_OVR")) / STEPS),
            m(lambda c, k: (V(c, "SRCB_WRITE_REQ") - V(c, "SRCB_WRITE_NOT_BLOCKED_OVR")) / STEPS),
        ),
        "srcA / srcB refused by L1 port, % of requests": (
            m(lambda c, k: 100 * (1 - V(c, "SRCA_WRITE_NOT_BLOCKED_PORT") / V(c, "SRCA_WRITE_REQ"))),
            m(lambda c, k: 100 * (1 - V(c, "SRCB_WRITE_NOT_BLOCKED_PORT") / V(c, "SRCB_WRITE_REQ"))),
        ),
        "T0 wait srcA clear / srcB clear %": (
            m(lambda c, k: 100 * V(c, "WAITING_FOR_SRCA_CLEAR") / ref(c)),
            m(lambda c, k: 100 * V(c, "WAITING_FOR_SRCB_CLEAR") / ref(c)),
        ),
        "unpack thread stalled %": m(lambda c, k: 100 * V(c, "THREAD_STALLS_0") / ref(c)),
        "packer busy %": m(lambda c, k: 100 * V(c, "PACKER_BUSY") / ref(c)),
        "pack thread stalled %": m(lambda c, k: 100 * V(c, "THREAD_STALLS_2") / ref(c)),
    }


def fmt(x):
    if isinstance(x, tuple):
        return " / ".join(fmt(y) for y in x)
    return "n/a" if x != x else (f"{x:.1f}" if abs(x) < 1000 else f"{x:,.0f}")


configs = [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]
data = {name: summarize(path) for name, path in configs}
keys = list(next(d for d in data.values() if d).keys())
print("| metric | " + " | ".join(data) + " |")
print("|---|" + "---|" * len(data))
for k in keys:
    print(f"| {k} | " + " | ".join(fmt(d[k]) if d else "n/a" for d in data.values()) + " |")
