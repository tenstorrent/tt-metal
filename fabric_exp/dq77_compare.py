#!/usr/bin/env python3
"""§77: the user's s77 (no sender slot table, compile-time free-slots stream ids, no [[unlikely]]) against the committed
tree (s74c), s77 with the [[unlikely]] restored (s77u), and s77u split into its slot-table removal (s77t) and
compile-time stream ids (s77s; s77sn = s77s without the [[unlikely]]), dq77_runs.sh. Cycles/packet per suite (2D 4352, mix,
flushed-signal mixes, 1D 4352; MixCmd excluded), mean change vs s74c, flush cost per signal, and per config from the
saved ELFs: .text instructions and the largest stack frame on E0 (active_erisc) and E1 (subordinate_active_erisc)."""
import csv, glob, os, re, statistics, subprocess

X = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(X, "results")
OD = os.path.join(os.path.dirname(X), "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump")
V = {
    "s74c": ["dq74c", "dq74cr2", "dq74cr3"],
    "s77t": ["dq77t"],
    "s77s": ["dq77s", "dq77sv"],
    "s77sn": ["dq77sn"],
    "s77u": ["dq77u"],
    "s77": ["dq77", "dq77r2"],
}
BASE = "s74c"


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


def spread(variant, suite, k):
    runs = [d for d in (bw(f"{p}_13129088_{suite}") for p in V[variant]) if d and k in d]
    return max(r[k] for r in runs) - min(r[k] for r in runs) if len(runs) > 1 else float("nan")


for suite in ("4352", "mix", "flushmix", "flushmix4", "1d"):
    d = {v: a for v in V if (a := avg(v, suite))}
    if BASE not in d:
        continue
    keys = [k for k in d[BASE] if "MixCmd" not in k[0] and all(k in d[v] for v in d)]
    print(
        f"{suite}: mean vs {BASE}: "
        + ", ".join(f"{v} {statistics.mean(d[v][k] - d[BASE][k] for k in keys):+.2f}" for v in d if v != BASE)
    )
    print(f"   {'row':34s}" + "".join(f"{v:>9s}" for v in d) + "   run spread " + " ".join(V_ for V_ in d if len(V[V_]) > 1))
    for k in keys:
        name = k[0] + ("" if suite not in ("4352", "1d") else f" {k[1][:7]}")
        print(
            f"   {name[:34]:34s}"
            + "".join(f"{d[v][k]:9.1f}" for v in d)
            + "   "
            + " ".join(f"{spread(v, suite, k):4.1f}" for v in d if len(V[v]) > 1)
        )
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
                    f"   {dr}_{sh:31s}"
                    + "".join(f"{(fm[v][f'{dr}_{sh}_Flush'] - fm[v][f'{dr}_{sh}_NoFlush']) * c:9.1f}" for v in fm)
                )
    print()


def elf_stats(elf):
    out = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    n = sum(1 for l in out.splitlines() if l[:1] == " " and ":\t" in l)
    frames = [int(m) for m in re.findall(r"addi\s+sp,sp,-(\d+)", out)]
    return n, max(frames, default=0)


EV = ["s74c", "s77t", "s77s", "s77sn", "s77u", "s77"]
for proc, name in (("active_erisc", "E0"), ("subordinate_active_erisc", "E1")):
    stats = {}
    for v in EV:
        for e in glob.glob(os.path.join(X, "variants", v, "elf", "*", f"{proc}.elf")):
            stats.setdefault(os.path.basename(os.path.dirname(e)), {})[v] = elf_stats(e)
    rows = {}
    for s in stats.values():
        if len(s) == len(EV):
            t = tuple(s[v][0] - s[BASE][0] for v in EV[1:]) + tuple(s[v][1] for v in EV)
            rows.setdefault(t, []).append(s[BASE][0])
    print(f"{name} ({proc}): .text instructions vs {BASE} | largest stack frame (bytes) {' / '.join(EV)}")
    for t, base_sizes in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        dv = "  ".join(f"{v} {x:+d}" for v, x in zip(EV[1:], t[: len(EV) - 1]))
        fr = " / ".join(str(x) for x in t[len(EV) - 1 :])
        print(f"   {dv}   frame {fr}   x{len(base_sizes)} configs ({min(base_sizes)}-{max(base_sizes)} insns)")
