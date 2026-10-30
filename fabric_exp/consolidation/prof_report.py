#!/usr/bin/env python3
"""prof_report.py [SUITE ...]: which side each row is sensitive to, from prof_delay_runs.sh (V4_PROFILE.md).

Per row (cycles/packet, 1.35 GHz): the base rate (v4trmd without spins = v4trm's images); per side the cost of a
32-cycle spin (c32 = tx32 - base; a spin of N costs about N+12, so a side with no slack pays ~44) and the slack it
implies (44 - c32, floored at 0: how many cycles of work that side could add per packet before the row slows); and the
32 -> 96 step (64 when the side has no slack; more than 64 means the spin costs more than its cycles there). At 96 both
sides are past their slack on every row so far, so the side is read from the 32-cycle point: "TX" (send side) or "RX"
(receive side) when only that side has under 8 cycles of slack, "both" when both do, "-" when neither (the row is bound
elsewhere: the link, the worker, the far router). The send spin runs after every send of every sender channel; the
receive spin after every packet taken off the link."""
import csv, os, statistics, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SUITES = ["4352", "1d", "mix", "ring", "torusy", "4352_1e", "1d_1e", "rse", "mux", "vc1"]
ARMS = ["base", "tx32", "tx96", "rx32", "rx96"]


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


PASS = "--pass" in sys.argv  # per-pass spins (prof_delay_runs.sh GROUP pass): arms base, sp16, sp48, rp16, rp48
P = "profp" if PASS else "prof"


def path(s, a):
    if s == "vc1":
        return f"{R}/{P}vc1_{a}_default_p1_o3_13129088_vc1_tests.csv"
    return f"{R}/{P}_{s}_{a}.csv"


def main_pass():
    """Per row: base; per pass kind (sender, receiver) the cost of a 16-cycle spin per pass and the 16 -> 48 step over 32:
    passes of that kind on the critical path per packet (what one cycle off that pass is worth on the row)."""
    arms = ["base", "sp16", "sp48", "rp16", "rp48"]
    summary = {}
    for s in [a for a in sys.argv[1:] if not a.startswith("--")] or SUITES:
        d = {a: rows(path(s, a)) for a in arms}
        if not all(d.values()):
            continue
        ks = [k for k in d["base"] if all(k in d[a] for a in arms)]
        print(f"\n## {s} ({len(ks)} rows)")
        print(f"{'row':52}{'base':>8}{'s16-b':>8}{'s/pkt':>7}{'r16-b':>8}{'r/pkt':>7}")
        ss, rr = [], []
        for k in ks:
            b = d["base"][k]
            s1, s2 = d["sp16"][k] - b, (d["sp48"][k] - d["sp16"][k]) / 32
            r1, r2 = d["rp16"][k] - b, (d["rp48"][k] - d["rp16"][k]) / 32
            ss.append(s2); rr.append(r2)
            lab = f"{k[0]} {k[1]} {k[2]}" + (f" #{k[3]}" if k[3] else "")
            print(f"{lab[:52]:52}{b:8.1f}{s1:+8.1f}{s2:7.2f}{r1:+8.1f}{r2:7.2f}")
        summary[s] = (len(ks), statistics.mean(ss), max(ss), statistics.mean(rr), max(rr))
    print("\n## summary: passes on the critical path per packet (mean, max over rows)")
    for s, (n, sm, sx, rm, rx) in summary.items():
        print(f"  {s:9} {n:3} rows   sender pass {sm:5.2f} (max {sx:5.2f})   receiver pass {rm:5.2f} (max {rx:5.2f})")


def side(slack_t, slack_r):
    t, r = slack_t < 8, slack_r < 8
    return "both" if t and r else "TX" if t else "RX" if r else "-"


def main():
    counts = {}
    for s in [a for a in sys.argv[1:] if not a.startswith("--")] or SUITES:
        d = {a: rows(path(s, a)) for a in ARMS}
        if not all(d.values()):
            continue
        ks = [k for k in d["base"] if all(k in d[a] for a in ARMS)]
        print(f"\n## {s} ({len(ks)} rows)")
        print(f"{'row':52}{'base':>8}{'c32 tx':>8}{'c32 rx':>8}{'slack tx':>9}{'slack rx':>9}{'step tx':>8}{'step rx':>8}  side")
        c = {}
        for k in ks:
            b = d["base"][k]
            t1, t2 = d["tx32"][k] - b, d["tx96"][k] - d["tx32"][k]
            r1, r2 = d["rx32"][k] - b, d["rx96"][k] - d["rx32"][k]
            st, sr = max(0.0, 44 - t1), max(0.0, 44 - r1)
            sd = side(st, sr)
            c[sd] = c.get(sd, 0) + 1
            lab = f"{k[0]} {k[1]} {k[2]}" + (f" #{k[3]}" if k[3] else "")
            print(f"{lab[:52]:52}{b:8.1f}{t1:+8.1f}{r1:+8.1f}{st:9.1f}{sr:9.1f}{t2:+8.1f}{r2:+8.1f}  {sd}")
        counts[s] = c
    print("\n## summary: rows per side (TX = send side, RX = receive side)")
    for s, c in counts.items():
        print(f"  {s:9} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main_pass() if PASS else main()
