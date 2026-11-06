#!/usr/bin/env python3
"""multivc_json.py: the link-L rows of glx_full_split.yaml (VC2_LINE_BRINGUP.md, "Full VCs and total throughput") as
report records (fabric_exp/report/data/SCHEMA.md), and a table for the bring-up log.

    multivc_json.py --pass1 RESULTS/full.log --pass2 RESULTS/full_rev.log --job J --node N --date D --out OUT.json

Link L is mesh 1 chip 2 -> chip 3 on the Galaxy 2x2 split: VC0 and VC2 from chip 2's workers, VC1 = mesh 0 chip 3's VC0
flow over the intermesh link, crossing over at chip 2. Per pass and size, cycles per packet of each flow (the profiler's
per-sender lines, benchmark mode without sync: each sender's own send loop):
  vc0_alone, vc2_alone, vc1_alone          LVC0, LVC2, LVC1
  vc0_with_vc2, vc2_with_vc0               LVC0VC2
  vc0_all3, vc2_all3, vc1_all3             LAll
  total_two, total_all3                    the link's total while the flows overlap (overlap accounting): the packets
                                           every flow moved before the first one ended, counting each other flow's
                                           packets after that point at its alone rate of the same pass; cycles per
                                           packet = the first flow's cycles over those packets. Exact for two flows.
                                           For three, the two flows left after the first ends share the link, so they
                                           run slower than alone: the packets moved before that point are undercounted,
                                           and the total is a lower bound on throughput (an upper bound in cycles per
                                           packet).
A record's value is the mean of the two passes; `passes` lists them, `value_pass1` is pass 1.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_vcs_report as pvr  # noqa: E402

SIZES = (256, 1024, 4352)
# (test, VC, mesh, chip, direction) of each flow on link L
FLOWS = {
    "vc0": (0, 1, 2, "E"),
    "vc2": (2, 1, 2, "E"),
    "vc1": (0, 0, 3, "E"),  # mesh 0 chip 3's VC0 flow, VC1 on link L after the crossover
}


def flow(recs, test, size, name):
    vc, mesh, chip, d = FLOWS[name]
    got = [r for r in recs if r["test"] == test and r["size"] == size and r["vc"] == vc and r["mesh"] == mesh
           and r["chip"] == chip and r["dir"] == d]
    if len(got) != 1:
        raise SystemExit(f"{test} {size} {name}: {len(got)} sender lines")
    return got[0]


def overlap_total(flows, alone):
    """flows: name -> sender record; alone: name -> cycles per packet alone. Cycles per packet of the link while every
    flow runs: the first flow to end bounds the window; the others' packets after it are counted at their alone rate."""
    first = min(flows, key=lambda n: flows[n]["cycles"])
    t1 = flows[first]["cycles"]
    packets = flows[first]["packets"]
    for n, r in flows.items():
        if n == first:
            continue
        after = (r["cycles"] - t1) / alone[n]
        packets += r["packets"] - after
    return t1 / packets, first


def one_pass(log):
    recs = pvr.senders(log)
    out = {}
    for z in SIZES:
        a = {n: pvr.cpp(flow(recs, t, z, n)) for n, t in (("vc0", "LVC0"), ("vc2", "LVC2"), ("vc1", "LVC1"))}
        two = {n: flow(recs, "LVC0VC2", z, n) for n in ("vc0", "vc2")}
        three = {n: flow(recs, "LAll", z, n) for n in ("vc0", "vc2", "vc1")}
        t2, first2 = overlap_total(two, a)
        t3, first3 = overlap_total(three, a)
        out[z] = {
            "vc0_alone": a["vc0"], "vc2_alone": a["vc2"], "vc1_alone": a["vc1"],
            "vc0_with_vc2": pvr.cpp(two["vc0"]), "vc2_with_vc0": pvr.cpp(two["vc2"]),
            "vc0_all3": pvr.cpp(three["vc0"]), "vc2_all3": pvr.cpp(three["vc2"]), "vc1_all3": pvr.cpp(three["vc1"]),
            "total_two": t2, "total_all3": t3, "_first2": first2, "_first3": first3,
        }
    return out


TEST = {"vc0_alone": "LVC0", "vc2_alone": "LVC2", "vc1_alone": "LVC1", "vc0_with_vc2": "LVC0VC2",
        "vc2_with_vc0": "LVC0VC2", "vc0_all3": "LAll", "vc2_all3": "LAll", "vc1_all3": "LAll",
        "total_two": "LVC0VC2", "total_all3": "LAll"}
WHAT = {
    "vc0_alone": "VC0 flow alone (mesh 1 chip 2 -> chip 3)",
    "vc2_alone": "VC2 flow alone (mesh 1 chip 2 -> chip 3, line mode, 1 hop)",
    "vc1_alone": "VC1 flow alone on link L (mesh 0 chip 3's VC0 flow, crossover to VC1 at mesh 1 chip 2)",
    "vc0_with_vc2": "VC0 flow while VC2's runs on the same link",
    "vc2_with_vc0": "VC2 flow while VC0's runs on the same link",
    "vc0_all3": "VC0 flow while VC1's and VC2's run on the same link",
    "vc2_all3": "VC2 flow while VC0's and VC1's run on the same link",
    "vc1_all3": "VC1 flow while VC0's and VC2's run on the same link",
    "total_two": "link total while VC0 and VC2 both run",
    "total_all3": "link total while VC0, VC1 and VC2 all run",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass1", required=True)
    ap.add_argument("--pass2", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--slots", required=True, help="the slot rows the runs logged")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p1, p2 = one_pass(a.pass1), one_pass(a.pass2)
    recs = []
    print(f"{'size':>5} {'metric':<14} {'pass 1':>8} {'pass 2':>8} {'mean':>8}")
    for z in SIZES:
        for m in TEST:
            v1, v2 = round(p1[z][m], 1), round(p2[z][m], 1)
            mean = round((p1[z][m] + p2[z][m]) / 2, 1)
            print(f"{z:>5} {m:<14} {v1:>8.1f} {v2:>8.1f} {mean:>8.1f}")
            notes = [WHAT[m] + "."]
            if m.startswith("total"):
                first = (p1[z]["_first2"] if m == "total_two" else p1[z]["_first3"]).upper()
                notes.append("Overlap accounting: the packets every flow moved before the first flow ended "
                             f"({first} in pass 1), the other flows' later packets counted at their alone rates of the "
                             "same pass; cycles per packet = the first flow's cycles over those packets.")
                notes.append("Exact for two flows." if m == "total_two" else
                             "Three flows: a lower bound on throughput (an upper bound in cycles/packet): after the "
                             "first flow ends the other two share the link and run slower than alone.")
            notes.append("Benchmark mode, no sync, 40,000 packets per flow, unicast writes; per-flow cycles from the "
                         "profiler's per-sender lines (each sender's own send loop). Pass 1 glx_full_split.yaml "
                         "(tests in file order, sizes ascending); pass 2 glx_full_split_rev.yaml (reverse order, sizes "
                         f"descending). Pass values: {v1}, {v2}.")
            recs.append({
                "id": f"glx-allvc-L-{m}-{z}-vc2l_g3a",
                "system": "BH Galaxy",
                "topology": "VC1 split (two 2x2 meshes)",
                "config": f"two ERISCs, VC2 line mode, VC1 on (multi-mesh), -O3, default slots ({a.slots})",
                "test": TEST[m],
                "send_type": "write",
                "packet_size_B": z,
                "router": "vc2l_g3a",
                "metric": m,
                "value": mean,
                "unit": "cycles/packet",
                "bound_side": None,
                "bound_status": None,
                "source_file": "fabric_exp/consolidation/VC2_LINE_BRINGUP.md",
                "source_section": "Rebase onto v4rxg3: all three VCs on link L (job %s)" % a.job,
                "job": f"{a.job} ({a.node})",
                "date": a.date,
                "same_job_with_main": False,
                "pair_key": None,
                "notes": " ".join(notes),
                "passes": [v1, v2],
                "value_pass1": v1,
                "slots": a.slots,
                "opt_level": "-O3",
            })
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(recs, open(a.out, "w"), indent=1)
    print(f"{len(recs)} records -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
