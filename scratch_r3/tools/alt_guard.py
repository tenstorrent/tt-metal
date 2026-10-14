#!/usr/bin/env python3
"""#58817 legacy port, void A/B guard for the in-process alternation: per legacy bmm build in a JIT cache, group the
builds by their generated files without the LEG_VARIANT define, and print per thread whether each variant's loaded
image equals variant 0's. Expected: 4 equals 0 everywhere; 1, 2 and 3 differ where their opt-in is selected.
usage: alt_guard.py <cache>"""
import hashlib, os, re, subprocess, sys
from collections import defaultdict

READELF = os.environ.get("READELF", "riscv-tt-elf-readelf")
root = sys.argv[1]
K = "bmm_large_block_zm_fused_bias_activation"
builds = defaultdict(dict)  # key -> variant -> {thread: hash}


def image(p):
    secs = []
    for line in subprocess.run([READELF, "-S", "-W", p], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", line)
        if m and m.group(2) == "PROGBITS" and "A" in m.group(3):
            secs.append(m.group(1))
    h = hashlib.sha256()
    for s in secs:
        h.update(subprocess.run([READELF, "-x", s, p], capture_output=True, text=True).stdout.encode())
    return h.hexdigest()[:12]


for kdir in sorted(set(os.path.dirname(os.path.dirname(p)) for p in __import__("glob").glob(f"{root}/*/kernels/{K}/*/trisc*/trisc*.elf"))):
    if os.path.basename(os.path.dirname(kdir)) != K:
        continue
    defs = open(os.path.join(kdir, "defines_generated.h")).read()
    m = re.search(r"#define\s+LEG_VARIANT\s+(\d+)", defs)
    if not m:
        continue
    h = hashlib.sha256()
    for g in sorted(os.listdir(kdir)):
        gp = os.path.join(kdir, g)
        if os.path.isfile(gp) and g.endswith(".h"):
            h.update(re.sub(rb"#define\s+LEG_VARIANT\s+\d+\n?", b"", open(gp, "rb").read()))
    th = {t: image(os.path.join(kdir, t, f"{t}.elf")) for t in ("trisc0", "trisc1", "trisc2") if os.path.isfile(os.path.join(kdir, t, f"{t}.elf"))}
    builds[h.hexdigest()[:12]][int(m.group(1))] = th
print(f"== alt guard: {len(builds)} configurations of {K}")
for key, vv in builds.items():
    if 0 not in vv:
        print(f"   {key}: no variant 0 build")
        continue
    row = []
    for v in sorted(vv):
        if v == 0:
            continue
        diff = [t for t in vv[0] if vv[v].get(t) != vv[0][t]]
        row.append(f"v{v} differs in {','.join(diff) if diff else 'nothing'}")
    print(f"   {key}: " + "; ".join(row))
