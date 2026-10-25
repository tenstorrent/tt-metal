#!/usr/bin/env python3
# categorize.py BASE NEW: per config category (ERISC, 1D/2D, ring, speedy, E/W vs N/S), how many configs' .text changed
# and the instruction-count deltas.
import collections, glob, os, re, subprocess, sys, concurrent.futures
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
def text(elf):
    out = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    return [re.sub(r" <[^>]*>", "", re.sub(r"\s*#.*$", "", l.split("\t", 2)[-1])) for l in out.splitlines() if re.match(r"\s+[0-9a-f]+:\t", l)]
def args(d):
    s = open(os.path.join(d, "named_ct_arg_map_generated.h")).read()
    return {k: int(v) for k, v in re.findall(r'\{"(\w+)",(-?\d+)\}', s)}
base, new = sys.argv[1], sys.argv[2]
elfs = sorted(glob.glob(f"{base}/*/*/*.elf"))
def one(e):
    rel = os.path.relpath(e, base)
    h, proc = rel.split("/")[:2]
    a = args(os.path.join(base, h))
    tb, tn = text(e), text(os.path.join(new, rel))
    dirn = "EW" if a["MY_DIRECTION"] in (0, 1) else "NS"
    tx = any(a.get(f"IS_SENDER_CHANNEL_{i}_SERVICED", 0) for i in range(10))
    rx = any(a.get(f"IS_RECEIVER_CHANNEL_{i}_SERVICED", 0) for i in range(3))
    role = ("TX+RX" if tx and rx else "TX" if tx else "RX") + ("/1E" if a["NUM_ACTIVE_ERISCS"] == 1 else "")
    cat = (("E0 " if proc == "active_erisc" else "E1 ") + role, "2D" if a["IS_2D_FABRIC"] else "1D",
           "ring" if a["ENABLE_DEADLOCK_AVOIDANCE"] else ("speedy" if a["ENABLE_SPEEDY_VC0"] else "-"),
           dirn if a["IS_2D_FABRIC"] else "-")
    return cat, tb != tn, len(tn) - len(tb)
with concurrent.futures.ThreadPoolExecutor(48) as ex:
    res = list(ex.map(one, elfs))
agg = collections.defaultdict(lambda: [0, 0, []])
for cat, ch, d in res:
    agg[cat][0] += 1; agg[cat][1] += ch; agg[cat][2].append(d)
print(f"{'ERISC role':12} {'fabric':6} {'mode':6} {'dir':4} {'changed':>9}  d insns min/max")
for cat in sorted(agg):
    n, c, ds = agg[cat]
    print(f"{cat[0]:12} {cat[1]:6} {cat[2]:6} {cat[3]:4} {c:4d}/{n:<4d}  {min(ds):+d}/{max(ds):+d}")
