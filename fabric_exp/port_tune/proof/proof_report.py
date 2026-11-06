#!/usr/bin/env python3
"""proof_report.py RESULTS_DIR [--min-slots N]: the proof-count diagnostic (proof_runs.sh). Per row, pass and receiver
channel (chip, Ethernet channel, the chip it receives from, VC) with at least N slots credited (default 1000; the
sync kernels' few packets are left out): group proofs (G > 1) with average status reads per proof, the histogram of
reads at success (1 / 2-4 / 5-8 / 9-16 / 17-32, % of proofs) and cap hits per 1,000 proofs; single-slot checks (G 1)
with their deferral rate. Records marked stale (left by a router that run did not launch) are skipped."""
import argparse, glob, json, os, re

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("results")
ap.add_argument("--min-slots", type=int, default=1000)
a = ap.parse_args()
runs = {}
for f in sorted(glob.glob(os.path.join(a.results, "pr*_proof.json"))):
    m = re.match(r"pr([ab])_(.*)_proof\.json", os.path.basename(f))
    runs.setdefault(m.group(2), {})[m.group(1)] = json.load(open(f))
print("| row | pass | receiver (chip.eth <- chip) | VC | G | slots credited | proofs | reads / proof | 1 | 2-4 | 5-8 "
      "| 9-16 | 17-32 | cap hits / 1k | G1 checks | G1 deferred % |")
print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for row in sorted(runs):
    for p in sorted(runs[row]):
        shown = 0
        for r in runs[row][p]:
            if r.get("stale"):
                continue
            for vc, c in sorted(r["vcs"].items()):
                if c["slots_credited"] < a.min_slots:
                    continue
                shown += 1
                n = c["attempts"]
                hist = [c[k] for k in ("h1", "h2_4", "h5_8", "h9_16", "h17_32")]
                pct = [f"{100 * h / n:.1f}" if n else "-" for h in hist]
                rpp = f"{c['reads'] / n:.2f}" if n else "-"
                cap = f"{1000 * c['cap_hits'] / n:.2f}" if n else "-"
                g1 = c["g1_checks"]
                g1d = f"{100 * c['g1_deferred'] / g1:.1f}" if g1 else "-"
                src = r["remote"][0] if r.get("remote") else "?"
                print(f"| {row} | {p} | {r['chip']}.{r['eth']} <- {src} | {vc} | {c['group']} | {c['slots_credited']} "
                      f"| {n} | {rpp} | " + " | ".join(pct) + f" | {cap} | {g1} | {g1d} |")
        if not shown:
            print(f"| {row} | {p} | (no receiver with >= {a.min_slots} slots credited) | | | | | | | | | | | | | |")
