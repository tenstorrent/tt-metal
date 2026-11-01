#!/usr/bin/env python3
"""cfgs.py BUILD [CASE ...]: per receiver image of a coverage build, the named CT args that shape the receiver loop
(first-level acks, receiver channels, VC1/VC2, direction, edge/intermesh, speedy, ERISCs) and its loop function's size
and stack accesses (rxcmp.rx_stats)."""
import collections, glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rxcmp
COV = rxcmp.COV
KEYS = ["ENABLE_FIRST_LEVEL_ACK_VC0", "NUM_RECEIVER_CHANNELS", "ACTUAL_VC1_SENDER_CHANNELS", "ACTUAL_VC2_SENDER_CHANNELS",
        "MY_DIRECTION", "IS_INTRAMESH_ROUTER_ON_EDGE", "IS_INTERMESH_ROUTER", "ENABLE_SPEEDY_VC0", "NUM_ACTIVE_ERISCS",
        "IS_RECEIVER_CHANNEL_0_SERVICED", "IS_RECEIVER_CHANNEL_1_SERVICED", "IS_RECEIVER_CHANNEL_2_SERVICED",
        "ENABLE_DEADLOCK_AVOIDANCE", "IS_2D_FABRIC", "UDM_MODE"]
SHORT = ["fla", "nrx", "vc1", "vc2", "dir", "edge", "imesh", "spd", "nE", "rx0", "rx1", "rx2", "dla", "2d", "udm"]
b = sys.argv[1]
cases = set(sys.argv[2:])
rows = collections.Counter()
for e in sorted(glob.glob(f"{COV}/{b}/*/*/*.elf")):
    rel = os.path.relpath(e, f"{COV}/{b}")
    case = rel.split("__")[0]
    if cases and case not in cases:
        continue
    st = rxcmp.rx_stats(rxcmp.functions(e))
    if not st:
        continue
    h = open(os.path.join(os.path.dirname(os.path.dirname(e)), "named_ct_arg_map_generated.h")).read()
    v = dict(re.findall(r'\{"(\w+)",(-?\d+)\}', h))
    key = " ".join(f"{s}={v.get(k, '?')}" for k, s in zip(KEYS, SHORT))
    rows[(case, key, st)] += 1
for (case, key, st), n in sorted(rows.items()):
    print(f"{case:22s} {key}  rx {st[0]} insns sp {st[1]}+{st[2]}; loop {st[3]} insns sp {st[4]}+{st[5]}  x{n}")
