#!/usr/bin/env python3
"""§66 diagnosis on bh-lb-120-a08u24: §63 rebuilt (a8s63, a8s63b), §66 (dq66, dq66r2, dq66r3) and §66 + [[unlikely]] on
the admission drain (dq66u, diagnostic) and §66 with the drain back at the inner-loop end (dq66d) on 4352 / mix / flushed-signal; then E1 code size per config from the saved ELFs
(variants/<v>/elf/<config hash>/; the hash names the compile-time config, so the same hash pairs across variants).
Cycles/packet; MixCmd excluded."""
import csv, glob, os, statistics, subprocess

X = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(X, "results")
OD = os.path.join(os.path.dirname(X), "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump")
V = {"§63r": ["a8s63", "a8s63b"], "§66": ["dq66", "dq66r2", "dq66r3"], "§66u": ["dq66u"], "§66d": ["dq66d"]}


def bw(label):
    f = os.path.join(R, label + ".csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r.get("packet_size", ""))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def avg(variant, suite):
    runs = [d for d in (bw(f"{p}_13129088_{suite}") for p in V[variant]) if d]
    return {k: statistics.mean(r[k] for r in runs) for k in runs[0]}, len(runs)


for suite in ("4352", "mix", "flushmix"):
    d = {}
    for v in V:
        d[v], n = avg(v, suite)
        print(f"{suite}: {v} averaged over {n} run(s)")
    keys = [k for k in d["§63r"] if "MixCmd" not in k[0]]
    print(
        f"{suite}: mean vs §63r: "
        + ", ".join(f"{v} {statistics.mean(d[v][k] - d['§63r'][k] for k in keys):+.2f}" for v in V if v != "§63r")
    )
    print(f"   {'row':30s}" + "".join(f"{v:>8s}" for v in V))
    for k in keys:
        name = k[0] + ("" if suite != "4352" else f" {k[1][:7]}")
        print(f"   {name[:30]:30s}" + "".join(f"{d[v][k]:8.1f}" for v in V))
    print()


def insns(elf):
    out = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    return sum(1 for l in out.splitlines() if l[:1] == " " and ":\t" in l)


print("E1 (subordinate_active_erisc) .text instructions per config")
sizes = {}
for v in V:
    for e in glob.glob(os.path.join(X, "variants", v.replace("§", "s"), "elf", "*", "subordinate_active_erisc.elf")):
        sizes.setdefault(os.path.basename(os.path.dirname(e)), {})[v] = insns(e)
rows = sorted({tuple(s.get(v) for v in V) for s in sizes.values() if len(s) == len(V)})
for r in rows:
    n = sum(1 for s in sizes.values() if tuple(s.get(v) for v in V) == r)
    print("   " + "  ".join(f"{v} {x}" for v, x in zip(V, r)) + f"   x{n} configs")
