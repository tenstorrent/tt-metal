#!/usr/bin/env python3
"""cfgprops.py BUILD [CASE_PREFIX]: per config: direction, ERISCs, edge/intermesh, VC0/VC1 downstream counts, UDM, DA."""
import os, re, sys
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
b = sys.argv[1]; pre = sys.argv[2] if len(sys.argv) > 2 else ""
KEYS = ["MY_DIRECTION", "NUM_ACTIVE_ERISCS", "IS_INTRAMESH_ROUTER_ON_EDGE", "IS_INTERMESH_ROUTER", "IS_INTERMESH_ROUTER_ON_EDGE",
        "VC0_DOWNSTREAM_EDM_SIZE", "VC1_DOWNSTREAM_EDM_SIZE", "UDM_MODE", "ENABLE_DEADLOCK_AVOIDANCE", "IS_2D_FABRIC", "MY_ERISC_ID"]
for d in sorted(os.listdir(f"{CB}/{b}")):
    if not d.startswith(pre) or not os.path.isdir(f"{CB}/{b}/{d}"):
        continue
    p = f"{CB}/{b}/{d}/named_ct_arg_map_generated.h"
    if not os.path.exists(p):
        continue
    s = open(p).read()
    v = {k: (re.search(r'\{"' + k + r'",(\d+)\}', s) or [None, "-"])[1] for k in KEYS}
    print(d, " ".join(f"{k.replace('IS_','').replace('_ROUTER','')[:14]}={v[k]}" for k in KEYS))
