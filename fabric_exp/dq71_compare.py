#!/usr/bin/env python3
"""§71 on bh-lb-120-a08u24: the user's exact-C1 FLUSH_RELAXED (s71: dq71, dq71r2 relaxed; dq71s strict packets)
against §63+guard (s63g, RELAXED treated as STRICT: a8s63* runs, byte-identical ERISC code) and §70's s69a (no completion
guard, 4-trid window: dq69a, dq69ar2), and s71b (two-read exact check at the first bypass: dq71b, dq71br2). flushmix4 = chunk of 4 (flush_mix4.yaml). 4352 / mix / flushed-signal (cycles/packet, MixCmd excluded), flush cost per signal,
latency (this host only), and E1 code size per config from the saved ELFs (variants/<v>/elf/<config hash>/)."""
import csv, glob, os, re, statistics, subprocess

X = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(X, "results")
OD = os.path.join(os.path.dirname(X), "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump")
V = {
    "§62x": ["a8s62x"],
    "§63g": ["a8s63", "a8s63b", "a8s63g", "a8s63g2", "a8s63g3", "a8s63g4"],
    "§69a": ["dq69a", "dq69ar2"],
    "§71 strict": ["dq71s"],
    "§71": ["dq71", "dq71r2"],
    "§71b": ["dq71b", "dq71br2"],
}
BASE = "§63g"


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
    if not runs:
        return None
    return {k: statistics.mean(r[k] for r in runs if k in r) for k in runs[0]}


for suite in ("4352", "mix", "flushmix", "flushmix4"):
    d = {v: a for v in V if (a := avg(v, suite))}
    keys = [k for k in d[BASE] if "MixCmd" not in k[0] and all(k in d[v] for v in d)]
    print(
        f"{suite}: mean vs {BASE}: "
        + ", ".join(f"{v} {statistics.mean(d[v][k] - d[BASE][k] for k in keys):+.2f}" for v in d if v != BASE)
    )
    print(f"   {'row':30s}" + "".join(f"{v:>12s}" for v in d))
    for k in keys:
        name = k[0] + ("" if suite != "4352" else f" {k[1][:7]}")
        print(f"   {name[:30]:30s}" + "".join(f"{d[v][k]:12.1f}" for v in d))
    if suite.startswith("flushmix"):
        fm = {v: {k[0]: x for k, x in d[v].items()} for v in d}
        print("\n   cost per flushed signal (flushed - unflushed) x chunk")
        shapes = (
            {"Chunk4Fused": 4}
            if suite == "flushmix4"
            else {"Fused": 1, "Chunk8Fused": 8, "Chunk16Fused": 16, "Chunk8Atomic": 8}
        )
        for dr in ("EW", "NS"):
            for sh, c in shapes.items():
                print(
                    f"   {dr}_{sh:27s}"
                    + "".join(f"{(fm[v][f'{dr}_{sh}_Flush'] - fm[v][f'{dr}_{sh}_NoFlush']) * c:12.1f}" for v in fm)
                )
    print()


def lat(label):
    f = os.path.join(R, label + "_latency.csv")
    if not os.path.exists(f):
        return None
    g = {}
    for r in csv.DictReader(open(f)):
        g.setdefault(re.sub(r"_sample\d+$", "", r["test_name"]), []).append(float(r["net_avg_ns"]))
    return {k: statistics.median(v) for k, v in g.items()}


L = {
    n: lat(p)
    for n, p in [("§62x", "a8s62x_13129088_lat"), ("§63r", "a8s63_13129088_lat"), ("§71", "dq71_13129088_lat")]
}
L = {n: v for n, v in L.items() if v}
print("latency ns (a08u24)" + " " * 11 + "".join(f"{n:>8s}" for n in L))
for k in L["§62x"]:
    print(
        f"   {k.replace('_ntype_unicast_write_size', '').replace('_ntype_fused_atomic_inc_size', '_fused'):28s}"
        + "".join(f"{L[n][k]:8.0f}" for n in L)
    )


def insns(elf):
    out = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    return sum(1 for l in out.splitlines() if l[:1] == " " and ":\t" in l)


print("\nE1 (subordinate_active_erisc) .text instructions per config")
EV = ["s62x", "s63g", "s69a", "s71", "s71b"]
sizes = {}
for v in EV:
    for e in glob.glob(os.path.join(X, "variants", v, "elf", "*", "subordinate_active_erisc.elf")):
        sizes.setdefault(os.path.basename(os.path.dirname(e)), {})[v] = insns(e)
rows = {}
for s in sizes.values():
    if len(s) == len(EV):
        t = tuple(s[v] for v in EV)
        rows[t] = rows.get(t, 0) + 1
for t, n in sorted(rows.items()):
    print("   " + "  ".join(f"{v} {x}" for v, x in zip(EV, t)) + f"   x{n} configs")
