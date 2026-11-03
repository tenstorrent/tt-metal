#!/usr/bin/env python3
"""bnd_report.py [--arms A,B,...] [--tag TAG] [SUITE ...]: which side bounds each row, from bnd_delay_runs.sh
(V4_BOUNDS.md); --tag reads the runs made with BND_TAG=TAG (labels bnd<TAG>_..., default: no tag).

prof_report.py's reading on the lead's hook variant (v4bdd at 0 = v4rxv2i3's images). Per row (cycles/packet,
1.35 GHz): the base rate; per side the cost of a 32-cycle spin (c32 = tx32 - base; a spin of N costs about N+12 on its
path, so a side with no slack pays ~44) and the slack it implies (44 - c32, floored at 0). Side: "TX" (E0, the send
side) or "RX" (E1, the receive side) when only that side has under 8 cycles of slack, "both" when both do, "-" when
neither (the row is bound elsewhere: the wire, the worker, a window). With tx96/rx96 arms the 32 -> 96 step is printed
too. On one ERISC both spins run on the same core: TX vs RX then says which half of the pass the row is sensitive to.
VC2 traffic rows (v2t_*) are read per sender (VC0 / VC2), from the profiler's per-sender log lines.
Suites: 4352 1d mix ring torusy vc2_<row> v2t_<row> 4352_1e 1d_1e rse mux vc1 vc1i."""
import csv, glob, os, re, statistics, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
T = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--tag=")), "")
if "--tag" in sys.argv:
    i = sys.argv.index("--tag")
    T = sys.argv[i + 1]
    del sys.argv[i:i + 2]
sys.argv = [x for x in sys.argv if not x.startswith("--tag=")]
SENDER = re.compile(r"(?:Sender core \S+ \(VC(\d)\) on device \d+ direction \S+ link \d+: (\d+) packets in (\d+) cycles"
                    r"|another sender core \S+ \(VC(\d)\) with (\d+) packets in (\d+) cycles)")


def rows(f):
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        if "avg_packets_per_s" in r:
            k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r["packet_size"])
            v = 1.35e9 / float(r["avg_packets_per_s"])
        else:
            k = (r["test_name"], "", "")
            v = 1.35e9 / float(r["packets_per_second"])
            if k + (0,) in out:
                out[k + (0,)] = max(out[k + (0,)], v)
                continue
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = v
    return out


def sender_rows(log):
    if not os.path.exists(log):
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read())
    out = {}
    for g in SENDER.findall(text):
        vc, p, c = (g[0], g[1], g[2]) if g[0] else (g[3], g[4], g[5])
        out[(f"VC{vc} sender", "", "", 0)] = int(c) / int(p)
    return out


def load(s, a):
    if s in ("vc1", "vc1i"):
        return rows(f"{R}/bnd{T}{s}_{a}_default_p1_o3_13129088_vc1_tests.csv")
    if s.startswith("v2t_"):
        return sender_rows(f"{R}/bnd{T}_{s}_{a}.log")
    return rows(f"{R}/bnd{T}_{s}_{a}.csv")


def side(slack_t, slack_r):
    t, r = slack_t < 8, slack_r < 8
    return "both" if t and r else "TX" if t else "RX" if r else "-"


def suites():
    named = [a for a in sys.argv[1:] if not a.startswith("--") and "," not in a]
    if named:
        return named
    found = set()
    for f in glob.glob(f"{R}/bnd{T}_*_base.csv") + glob.glob(f"{R}/bnd{T}_v2t_*_base.log"):
        found.add(re.sub(rf"^bnd{re.escape(T)}_|_base\.(csv|log)$", "", os.path.basename(f)))
    for s in ("vc1", "vc1i"):
        if glob.glob(f"{R}/bnd{T}{s}_base_*_vc1_tests.csv"):
            found.add(s)
    order = ["4352", "1d", "mix", "ring", "torusy", "4352_1e", "1d_1e", "rse", "mux", "vc1", "vc1i"]
    return [s for s in order if s in found] + sorted(s for s in found if s not in order)


def main():
    counts = {}
    for s in suites():
        d = {a: load(s, a) for a in ("base", "tx32", "rx32", "tx96", "rx96")}
        if not (d["base"] and d["tx32"] and d["rx32"]):
            continue
        deep = bool(d["tx96"] and d["rx96"])
        ks = [k for k in d["base"] if k in d["tx32"] and k in d["rx32"]]
        print(f"\n## {s} ({len(ks)} rows)")
        print(f"{'row':52}{'base':>8}{'c32 tx':>8}{'c32 rx':>8}{'slack tx':>9}{'slack rx':>9}"
              + (f"{'step tx':>8}{'step rx':>8}" if deep else "") + "  side")
        c = {}
        for k in ks:
            b = d["base"][k]
            t1, r1 = d["tx32"][k] - b, d["rx32"][k] - b
            st, sr = max(0.0, 44 - t1), max(0.0, 44 - r1)
            sd = side(st, sr)
            c[sd] = c.get(sd, 0) + 1
            lab = f"{k[0]} {k[1]} {k[2]}" + (f" #{k[3]}" if k[3] else "")
            line = f"{lab[:52]:52}{b:8.1f}{t1:+8.1f}{r1:+8.1f}{st:9.1f}{sr:9.1f}"
            if deep and k in d["tx96"] and k in d["rx96"]:
                line += f"{d['tx96'][k] - d['tx32'][k]:+8.1f}{d['rx96'][k] - d['rx32'][k]:+8.1f}"
            print(f"{line}  {sd}")
        counts[s] = c
    print("\n## summary: rows per side (TX = send side, E0; RX = receive side, E1)")
    for s, c in counts.items():
        print(f"  {s:22} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
