# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-phase inner-loop budget of the ring SDPA compute kernel from a Tracy profile_log_device.csv
recorded with TT_EXP_SDPA_PROFILE_INNER=1 (exp ring op). Picks one core, the last run, and one
inner-loop step (4 Phase-1 "Softmax(Q@KT)" zones + 4 Phase-2 "Softmax(Q@KT)@V" zones at q256/k512),
then prints per compute thread (TRISC_0 unpack, TRISC_1 math, TRISC_2 pack) the time in each leaf
zone and the time outside all leaf zones (handshakes, CB waits).

    python sdpa_phase_zones.py <profile_log_device.csv> [device id, default 0] [core "x,y", default 1,1]
"""
import csv
import sys
from collections import defaultdict

LEAF = {
    "Q@KT MM+Pack",
    "Reduce max",
    "SUB_EXP_BLOCK_INIT",
    "SUB",
    "EXP",
    "PACK SUB_EXP",
    "QKT@V MM+Pack",
    "S_CORR_FUSED",
    "ROW_NORM",
}


def main() -> None:
    path = sys.argv[1]
    dev = sys.argv[2] if len(sys.argv) > 2 else "0"
    core = tuple((sys.argv[3] if len(sys.argv) > 3 else "1,1").split(","))
    ev = defaultdict(list)
    with open(path) as f:
        next(f)  # ARCH header line
        for r in csv.DictReader(f, skipinitialspace=True):
            if r["PCIe slot"] != dev or (r["core_x"], r["core_y"]) != core:
                continue
            ev[(r["RISC processor type"], r["run host ID"])].append(
                (int(r["time[cycles since reset]"]), r["zone name"], r["type"])
            )
    for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
        runs = sorted(k[1] for k in ev if k[0] == risc)
        if not runs:
            print(f"{risc}: no zones (was the kernel built with SDPA_PROFILE_INNER?)")
            continue
        seq = sorted(ev[(risc, runs[-1])])
        p1 = [t for t, z, ty in seq if z == "Softmax(Q@KT)" and ty == "ZONE_START"]
        p2e = [t for t, z, ty in seq if z == "Softmax(Q@KT)@V" and ty == "ZONE_END"]
        if len(p1) >= 9 and len(p2e) >= 8:
            t0, t1, label = p1[4], p2e[7], "step 1"
        elif len(p1) >= 5 and len(p2e) >= 4:
            t0, t1, label = p1[0], p2e[3], "step 0 (includes the first K-chunk wait)"
        else:
            print(f"{risc}: only {len(p1)} phase-1 / {len(p2e)} phase-2 zones captured")
            continue
        dur, cnt, st = defaultdict(int), defaultdict(int), {}
        for t, z, ty in seq:
            if t < t0 or t > t1 or z not in LEAF:
                continue
            if ty == "ZONE_START":
                st[z] = t
            elif z in st:
                dur[z] += t - st.pop(z)
                cnt[z] += 1
        tot = sum(dur.values())
        print(
            f"--- {risc} {label}: period {(t1 - t0) / 1000:.1f} us; in leaf zones {tot / 1000:.1f} us; "
            f"outside leaves {(t1 - t0 - tot) / 1000:.1f} us"
        )
        for z, v in sorted(dur.items(), key=lambda kv: -kv[1]):
            print(f"   {z:20s} n={cnt[z]:3d}  {v / 1000:6.1f} us  ({100 * v / (t1 - t0):4.1f}%)")


if __name__ == "__main__":
    main()
