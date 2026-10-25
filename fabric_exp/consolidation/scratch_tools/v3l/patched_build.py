#!/usr/bin/env python3
"""patched_build.py VARIANT OUT OPT HASHES KEY=VAL,...: offline build with named compile-time args overridden/added
(compile-coverage only: telemetry-on and similar builds that no logged config has)."""
import sys, os, re, json
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import offline_build as ob
variant, out, opt, hashes, kv = sys.argv[1:6]
over = dict(x.split("=") for x in kv.split(",") if x)
db = json.load(open(ob.DB))[opt]
out = os.path.abspath(out); os.makedirs(out, exist_ok=True)
ov = ob.make_overlay(os.path.abspath(variant), out)
bad = 0
for h in hashes.split(","):
    ob.prep_headers(h, next(iter(db[h].values()))["compile"], ov, out)
    p = os.path.join(out, h, "named_ct_arg_map_generated.h")
    s = open(p).read()
    for k, v in over.items():
        if f'{{"{k}",' in s:
            s = re.sub(r'\{"' + k + r'",-?\d+\}', f'{{"{k}",{v}}}', s)
        else:
            s = s.replace('#define KERNEL_COMPILE_TIME_ARG_MAP ', f'#define KERNEL_COMPILE_TIME_ARG_MAP {{"{k}",{v}}},')
    open(p, "w").write(s)
    for proc, c in db[h].items():
        if "compile" in c and "link" in c:
            _, _, err = ob.build_one(h, proc, c, ov, out, [], None)
            print(h, proc, "ok" if not err else err[:3000])
            bad += bool(err)
sys.exit(1 if bad else 0)
