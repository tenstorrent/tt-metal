#!/usr/bin/env python3
"""stream_init_scan.py [BUILD]: the stream-register writes of the router's init, today's rule (main's order:
init_link_streams + init_2d_streams) against A2's (init_streams: every receiver channel the router has, every sender
channel the router has at its fabric position), on every config of a cov_m build's named-arg headers (default
v4txo8_O3). Prints, per difference kind, how many configs have it, and checks whether a register today's rule writes
and A2's does not shares its stream id with a register A2 writes (a collision would make the write order matter)."""
import collections, glob, re, sys
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
UNUSED = 32
build = sys.argv[1] if len(sys.argv) > 1 else "v4txo8_O3"


def fabric_position(a, ch):
    v0, v1 = a["ACTUAL_VC0_SENDER_CHANNELS"], a["ACTUAL_VC1_SENDER_CHANNELS"]
    if ch < v0:
        return ch
    if ch < v0 + v1:
        return a["VC1_FABRIC_POSITION_START"] + ch - v0
    return a["VC2_FABRIC_POSITION_START"] + ch - v0 - v1


def compact(a, p):
    v0, v1 = a["ACTUAL_VC0_SENDER_CHANNELS"], a["ACTUAL_VC1_SENDER_CHANNELS"]
    s1, s2 = a["VC1_FABRIC_POSITION_START"], a["VC2_FABRIC_POSITION_START"]
    return p if p < s1 else (v0 + p - s1 if p < s2 else v0 + v1 + p - s2)


def services(a, p):
    s1, s2 = a["VC1_FABRIC_POSITION_START"], a["VC2_FABRIC_POSITION_START"]
    if p < s1:
        return p < a["ACTUAL_VC0_SENDER_CHANNELS"]
    if p < s2:
        return p < s1 + a["ACTUAL_VC1_SENDER_CHANNELS"]
    return p < s2 + a["ACTUAL_VC2_SENDER_CHANNELS"]


def streams(a):
    rx = [a[f"TO_RECEIVER_{i}_PKTS_SENT_ID"] for i in range(3)]
    acked = [a.get(f"TO_SENDER_{i}_PKTS_ACKED_ID", UNUSED) for i in range(5)] + [UNUSED] * 5
    completed = [a[f"TO_SENDER_{i}_PKTS_COMPLETED_ID"] for i in range(9)] + [UNUSED]
    free = [a[f"SENDER_CHANNEL_{i}_FREE_SLOTS_STREAM_ID"] for i in range(10)]
    return rx, acked, completed, free


def old_writes(a):
    rx, acked, completed, free = streams(a)
    w = [("rx", 0, rx[0], 0)]  # init_ptr_val: written even if it were the sentinel
    w += [("acked", p, acked[p], 0) for p in (0, 1) if acked[p] != UNUSED]
    w += [("completed", p, completed[p], 0) for p in (0, 1) if completed[p] != UNUSED]
    w += [("free", p, free[p], "slots[%d]" % p) for p in (0, 1)]
    if a["IS_2D_FABRIC"]:
        w += [("rx", r, rx[r], 0) for r in (1, 2) if rx[r] != UNUSED]
        w += [("acked", p, acked[p], 0) for p in (2, 3, 4) if acked[p] != UNUSED]
        for p in range(2, 10):
            if completed[p] != UNUSED:
                w.append(("completed", p, completed[p], 0))
            if services(a, p):
                w.append(("free", p, free[p], "slots[%d]" % compact(a, p)))
    return w


def new_writes(a):
    rx, acked, completed, free = streams(a)
    w = [("rx", r, rx[r], 0) for r in range(a["NUM_RECEIVER_CHANNELS"]) if rx[r] != UNUSED]
    for ch in range(a["NUM_SENDER_CHANNELS"]):
        p = fabric_position(a, ch)
        w += [("acked", p, acked[p], 0)] if acked[p] != UNUSED else []
        w += [("completed", p, completed[p], 0)] if completed[p] != UNUSED else []
        w.append(("free", p, free[p], "slots[%d]" % ch))
    return w


kinds = collections.Counter()
examples = {}
n = 0
for f in sorted(glob.glob(f"{CB}/{build}/*/named_ct_arg_map_generated.h")):
    a = {k: int(v) for k, v in re.findall(r'\{"(\w+)",(\d+)\}', open(f).read())}
    n += 1
    o, w = set(old_writes(a)), set(new_writes(a))
    gone, added = o - w, w - o
    case = f.split("/")[-2].split("__")[0]
    for g in sorted(gone):
        collide = any(x[2] == g[2] for x in w)
        k = ("only today", g[0], g[1], "id shared with an A2 write" if collide else "id written by nothing else")
        kinds[k] += 1
        examples.setdefault(k, case)
    for g in sorted(added):
        k = ("only A2", g[0], g[1], "")
        kinds[k] += 1
        examples.setdefault(k, case)
    if not gone and not added:
        kinds[("same set",)] += 1
print(f"{build}: {n} configs")
for k, c in sorted(kinds.items()):
    print(f"  {c:5d}  {' '.join(map(str, k))}  (e.g. {examples.get(k, '-')})")
