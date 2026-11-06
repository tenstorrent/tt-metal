#!/usr/bin/env python3
"""vc1_active_check.py COVDIR...: for every router compile in each capture case (mock.log compile commands), whether
-DFABRIC_2D_VC1_ACTIVE is defined, against ACTUAL_VC1_SENDER_CHANNELS > 0 from that kernel's named compile-time args
(named_ct_arg_map_generated.h in the case's cache). Prints per case the counts and any disagreement."""
import glob, os, re, sys
tot = bad = 0
for cov in sys.argv[1:]:
    for case in sorted(os.listdir(os.path.join(cov, "cases"))):
        d = os.path.join(cov, "cases", case)
        logs = glob.glob(os.path.join(d, "**", "mock.log"), recursive=True) + glob.glob(os.path.join(d, "*.log"))
        maps = {}
        for m in glob.glob(os.path.join(d, "**", "fabric_erisc_router", "*", "named_ct_arg_map_generated.h"), recursive=True):
            t = open(m).read()
            v = re.search(r'\{"ACTUAL_VC1_SENDER_CHANNELS",(\d+)\}', t)
            maps[m.split("/")[-2]] = int(v.group(1)) if v else None
        seen = {}
        for lg in logs:
            for l in open(lg, errors="replace"):
                if "g++ compile cmd" not in l or "fabric_erisc_router/" not in l:
                    continue
                h = re.search(r"fabric_erisc_router/(\d+)/", l)
                if not h or " -c " not in l or "(build.cpp:97" in l:  # compile lines only (not link lines)
                    continue
                a = " -DFABRIC_2D_VC1_ACTIVE" in l
                if h.group(1) in seen and seen[h.group(1)] != a:
                    print(f"   MIXED {case} {h.group(1)}: compile lines of one kernel disagree on the define")
                seen[h.group(1)] = seen.get(h.group(1), a) or a
        n = disagree = missing = 0
        for h, active in seen.items():
            v = maps.get(h)
            if v is None:
                missing += 1
                continue
            n += 1
            if active != (v > 0):
                disagree += 1
                print(f"   DISAGREE {cov.split('/')[-1]} {case} {h}: define {active}, ACTUAL_VC1_SENDER_CHANNELS {v}")
        tot += n; bad += disagree
        print(f"{cov.split('/')[-1]:32s} {case:28s} router kernels {n:4d}  VC1 active {sum(1 for h, a in seen.items() if a and maps.get(h) is not None):4d}  disagree {disagree}  no map {missing}")
print(f"TOTAL router kernels {tot}, disagree {bad}")
