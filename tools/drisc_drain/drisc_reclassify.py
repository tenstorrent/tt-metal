#!/usr/bin/env python3
"""Reclassify DRISC runs from the LOGS, not from exit code alone. Handles both csv schemas (the Mac
harness's `card` column and the on-box watcher's ep_link/devsta/sweep).

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
# A ~220 ms 4-byte MMIO load is the root-port completion timeout firing: the endpoint never completed the
# read, so the UMD's 2 ms per-op budget throws and the process aborts. The wedge MECHANISM, transient --
# the card answers again by the time card state is sampled, so no link-state check can catch it.
MMIO_MARK = "MMIO per-op timeout"
# A real core-wait hang sits there: 45 s with the timeout armed, RUN_TIMEOUT unarmed. Anything that failed
# in less than this did NOT hang in wait_until_cores_done, whatever else it did.
TEARDOWN_MIN_S = 40.0


def find_log(d, k, armed, sweep=None):
    pats = [f"{sweep}_{k}.log"] if sweep else []
    for pat in pats + [f"{k}_a{armed}.log", f"{k}.log"]:
        p = os.path.join(d, pat)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(d, f"{k}_*.log"))
    return hits[0] if hits else None


def classify(rc, card, text, dur):
    """Card state first, then the log. Never infer a hang from 'did not reach the end marker' alone:
    an early abort fails to reach it too, and filing a 3 s abort as a 45 s hang pools two distinct
    failures -- the error this investigation has paid for most."""
    if card.startswith("Unknown"):
        return "WEDGE"  # hard wedge: config space all-ones
    if MMIO_MARK in text:
        return "MMIO_STALL"  # transient wedge: completion timeout, card recovered
    reached_end = END_MARK in text
    masked_sig = any(m in text for m in MASK_MARK)
    if rc != 0:
        if masked_sig or (not reached_end and dur >= TEARDOWN_MIN_S):
            return "TEARDOWN"
        return "ABORT"
    if masked_sig:
        return "MASKED"  # rc=0 but the armed timeout caught a core-wait: not clean
    return "CLEAN" if reached_end else "ABORT"


def main(d):
    rows = list(csv.DictReader(open(os.path.join(d, "runs.csv"))))
    if not rows:
        print("no rows")
        return

    # Two csv schemas exist: the Mac harness wrote a single `card` column, the on-box watcher writes
    # ep_link/rp_link/devsta (+ sweep). Accept either -- one reclassifier, or the two drift.
    def card_of(r):
        return (r.get("card") or r.get("ep_link") or "-") or "-"

    # If a csv holds several sweeps that each numbered k from 1 and the logs are NOT namespaced by
    # sweep, then a later sweep overwrote the earlier one's logs and a bare k.log cannot be attributed.
    # Re-judging an earlier row from a later sweep's log invents a verdict, which is worse than leaving
    # it alone -- so only the LAST occurrence of an ambiguous k is reclassified.
    seen = collections.Counter(r["k"] for r in rows)
    last_idx = {}
    for i, r in enumerate(rows):
        last_idx[r["k"]] = i

    out, changed, nolog, ambig = [], 0, 0, 0
    for i, r in enumerate(rows):
        if seen[r["k"]] > 1 and not r.get("sweep") and last_idx[r["k"]] != i:
            ambig += 1
            r["class_orig"] = r["class"]
            out.append(r)
            continue
        log = find_log(d, r["k"], r.get("armed", "0"), r.get("sweep"))
        text = open(log, errors="ignore").read() if log else ""
        if not log:
            nolog += 1
            r["class_orig"] = r["class"]  # no evidence -> do NOT invent a verdict
            out.append(r)
            continue
        new = classify(int(r["rc"]), card_of(r), text, float(r.get("dur_s") or 0))
        if new != r["class"]:
            changed += 1
            print(f"  k={r['k']:>4} sweep={r.get('sweep','-')}  {r['class']:10s} -> {new}")
        r["class_orig"], r["class"] = r["class"], new
        out.append(r)

    with open(os.path.join(d, "runs_reclassified.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"\nreclassified {changed} of {len(out)} rows")
    if nolog:
        print(f"  {nolog} left as-is: no log found")
    if ambig:
        print(f"  {ambig} left as-is: DUPLICATE k with un-namespaced logs -- an earlier sweep's log was")
        print(f"      overwritten by a later one, so those rows have no attributable evidence. Their")
        print(f"      original class stands (rc alone still rules out MMIO_STALL, which aborts rc=134).")

    # Group by sweep when the column is populated, else by arm. Pooling sweeps that ran in different
    # places (host vs container: devsta readable or not) yields a rate that describes neither.
    keyname = "sweep" if any(r.get("sweep") for r in out) else "armed"
    print(
        f"\n{keyname:16s} {'n':>5} {'WEDGE':>6} {'MMIO':>5} {'TEARDOWN':>9} {'MASKED':>7} {'ABORT':>6} {'CLEAN':>6}  median   max"
    )
    for key in sorted({(r.get(keyname) or "-") for r in out}):
        g = [x for x in out if (x.get(keyname) or "-") == key]
        c = collections.Counter(x["class"] for x in g)
        dur = sorted(float(x["dur_s"] or 0) for x in g)
        n = len(g)
        print(
            f"{str(key):16s} {n:>5} {c['WEDGE']:>6} {c['MMIO_STALL']:>5} {c['TEARDOWN']:>9} "
            f"{c['MASKED']:>7} {c['ABORT']:>6} {c['CLEAN']:>6}  {dur[len(dur)//2]:>5.0f}s {dur[-1]:>5.0f}s"
        )
        hard, trans = c["WEDGE"], c["MMIO_STALL"]
        print(
            f"{'':16s}   hard wedge {hard}/{n} = {100.0*hard/n:.2f}%   transient {trans}/{n} = "
            f"{100.0*trans/n:.2f}%   combined {100.0*(hard+trans)/n:.2f}%  (recorded rate ~2-3%)"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
