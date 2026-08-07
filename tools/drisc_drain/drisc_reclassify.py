#!/usr/bin/env python3
"""Reclassify DRISC harness runs from the LOGS, not from exit code alone.

The live harness keys TEARDOWN off a log signature ("waiting for physical cores to finish" /
"Continuing with cleanup"). That message is only emitted when TT_METAL_OPERATION_TIMEOUT_SECONDS
is armed and the teardown wait times out. UNARMED runs hang at the same place silently, so they
fell into OTHER. The robust discriminator is how far the log got:

    clean run     -> reaches "Cluster destructor completed"  (cluster.cpp:811)
    teardown hang -> stops at the profiler teardown block, card HEALTHY
    wedge         -> stops earlier, card reads Unknown|63

Usage:  drisc_reclassify.py <run_dir>          # dir holding runs.csv and *.log
"""
import sys, os, csv, glob, collections

END_MARK = "Cluster destructor completed"
MASK_MARK = ("Continuing with cleanup", "waiting for physical cores to finish")


def find_log(d, k, armed):
    for pat in (f"{k}_a{armed}.log", f"{k}.log"):
        p = os.path.join(d, pat)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(d, f"{k}_*.log"))
    return hits[0] if hits else None


def classify(rc, card, text):
    wedged = card.startswith("Unknown")
    reached_end = END_MARK in text
    masked_sig = any(m in text for m in MASK_MARK)
    if wedged:
        return "WEDGE"
    if rc != 0:
        return "TEARDOWN" if not reached_end else "OTHER"
    if masked_sig:
        return "MASKED"
    return "CLEAN" if reached_end else "OTHER"


def main(d):
    rows = list(csv.DictReader(open(os.path.join(d, "runs.csv"))))
    out, changed = [], 0
    for r in rows:
        log = find_log(d, r["k"], r["armed"])
        text = open(log, errors="ignore").read() if log else ""
        new = classify(int(r["rc"]), r.get("card", "-") or "-", text)
        if new != r["class"]:
            changed += 1
            print(f"  k={r['k']:>4} armed={r['armed']}  {r['class']:8s} -> {new}")
        r["class_orig"], r["class"] = r["class"], new
        out.append(r)

    with open(os.path.join(d, "runs_reclassified.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"\nreclassified {changed} of {len(out)} rows\n")
    print(f"{'arm':8s} {'n':>4} {'WEDGE':>6} {'TEARDOWN':>9} {'MASKED':>7} {'OTHER':>6} {'CLEAN':>6}  median  max")
    for armed in ("0", "1"):
        r = [x for x in out if x["armed"] == armed]
        if not r:
            continue
        c = collections.Counter(x["class"] for x in r)
        dur = sorted(float(x["dur_s"]) for x in r)
        lbl = "UNARMED" if armed == "0" else "ARMED"
        print(f"{lbl:8s} {len(r):>4} {c['WEDGE']:>6} {c['TEARDOWN']:>9} {c['MASKED']:>7} "
              f"{c['OTHER']:>6} {c['CLEAN']:>6}  {dur[len(dur)//2]:.1f}s  {dur[-1]:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
