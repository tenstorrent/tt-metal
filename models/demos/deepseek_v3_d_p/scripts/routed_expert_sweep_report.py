#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn a routed expert sweep run directory into its results table."""

import argparse
import bisect
import csv
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RINGS = ("ring8", "ring4")
CHUNKS = ("2.5k", "5k", "10k")
TRAFFIC = ("balanced", "hot_expert")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--score", type=Path, help="write 'score: <pass fraction>' here, which robo_test reads")
    return ap.parse_args()


def run_dir():
    directory = os.environ.get("RE_SWEEP_DIR")
    if not directory:
        sys.exit("RE_SWEEP_DIR is not set; it names the run directory to report")
    return Path(directory)


def device_spread(run):
    per_phase = defaultdict(lambda: defaultdict(list))
    for path in sorted(run.glob("reports/*/ops_perf_results*.csv")):
        rows = list(csv.DictReader(path.open()))
        posts = sorted(
            (int(r["HOST START TS"]), r["OP CODE"])
            for r in rows
            if r["OP TYPE"] == "signpost" and re.fullmatch(r"\S+-ring\d+-\S+-\S+", r["OP CODE"])
        )
        if not posts:
            continue
        starts = [ts for ts, _ in posts]
        for r in rows:
            if r["OP TYPE"] == "tt_dnn_device" and r["OP CODE"].startswith("HybridRoutedExpert"):
                i = bisect.bisect_right(starts, int(r["HOST START TS"])) - 1
                if i >= 0:
                    per_phase[posts[i][1]][r["DEVICE ID"]].append(r)

    spreads = {}
    for name, chips in per_phase.items():
        medians = []
        for chip_rows in chips.values():
            last = max(int(r["GLOBAL CALL COUNT"]) for r in chip_rows)
            us = [
                int(r["DEVICE KERNEL DURATION [ns]"]) / 1000
                for r in chip_rows
                if int(r["GLOBAL CALL COUNT"]) == last and r["DEVICE KERNEL DURATION [ns]"]
            ]
            if us:
                medians.append(statistics.median(us))
        if medians:
            spreads[name] = (min(medians), max(medians))
    return spreads


def load(run):
    rows = [line.split("\t") for line in (run / "state.tsv").read_text().splitlines() if line.strip()]
    outcome = {r[1]: r[0] for r in rows if len(r) > 1}
    measured = {r[1]: dict(f.split("=", 1) for f in r[2].split()) for r in rows if len(r) > 2 and r[0] == "pass"}
    return [r[1] for r in rows if len(r) > 1], outcome, measured, device_spread(run)


def main():
    args = parse_args()
    phases, outcome, measured, device = load(run_dir())

    def metric(model, ring, chunk, traffic, which):
        name = f"{model}-{ring}-{chunk}-{traffic}"
        if name not in phases:
            return ""
        if outcome.get(name) != "pass":
            return outcome.get(name, "?") if which == "dev" else ""
        span = device.get(name)
        if which == "dev":
            return f"{span[0]:.0f} - {span[1]:.0f}" if span else "-"
        value = measured.get(name, {}).get("traced_us" if which == "ht" else "untraced_us")
        if not value:
            return "-"
        if not span:
            return f"{float(value):.0f}"
        return f"{float(value):.0f} ({float(value) / span[1] - 1:+.0%})"

    def alloc_mb(model, ring):
        """One figure when the row's cells agree, a range when they do not.

        A phase that reuses a cached program allocates nothing of its own and records a zero, so
        those are dropped rather than widening every range down to nothing.
        """
        mb = sorted(
            {
                int(f["op_alloc_bytes"]) / 1000**2
                for p, f in measured.items()
                if p.startswith(f"{model}-{ring}-") and f.get("op_alloc_bytes", "0") != "0"
            }
        )
        return "" if not mb else f"{mb[0]:.0f}" if round(mb[0]) == round(mb[-1]) else f"{mb[0]:.0f} - {mb[-1]:.0f}"

    def field(which, ring):
        """The distinct values of a recorded field over this ring's phases."""
        return sorted(
            {f[which] for p, f in measured.items() if which in f and ring in p.split("-")},
            key=lambda v: int(v) if v.isdigit() else v,
        )

    parts = [p.split("-") for p in phases]
    models = list(dict.fromkeys(p[0] for p in parts))
    rings = [r for r in RINGS if any(len(p) > 1 and p[1] == r for p in parts)]
    counts = Counter(outcome.values())
    failed = counts["FAIL"] + counts["HANG"] + counts["unrun"] + counts["?"]
    thresholds = {("none" if f["threshold"] == "0" else f["threshold"]) for f in measured.values() if "threshold" in f}
    shown = ", ".join(sorted(thresholds)) if thresholds else "unknown"

    report = [
        f"{counts['pass']} pass, {counts['unsup']} unsup, {failed} failed",
        "",
        f"hybrid token threshold {shown}, capacity factor 2, bfloat8_b tiled input",
        "",
        "dev is the device op time per chip, min - max over the chips. ht and hu are the host wall",
        "clock per launch, traced and untraced, in us, each with what it adds over the slowest chip.",
        "",
        "Chunk is the tokens per dispatch group, so a chip carries chunk/ring of them. This op takes",
        "no cluster axis: the ring reaches it as twice the experts and twice the tokens per chip.",
    ]
    # Chunk and traffic make the columns; the ring crosses with the model to make the rows, which
    # keeps the table narrow enough to read however many axes the run swept.
    cases = [(c, t) for c in CHUNKS for t in TRAFFIC if any(metric(m, r, c, t, "dev") for m in models for r in rings)]
    groups = [(m, r) for m in models for r in rings if any(metric(m, r, c, t, "dev") for c, t in cases)]
    for ring in rings:
        chips = field("chips", ring)
        experts = field("experts_per_chip", ring)
        spread = " - ".join(dict.fromkeys([experts[0], experts[-1]])) if experts else "?"
        report += ["", f"{ring}: {' / '.join(chips) or '?'} chips, {spread} experts per chip across these models"]
    report.append("")
    head = ["case", ""] + [f"{c} {t}" for c, t in cases] + ["op alloc MB"]
    body = [
        row
        for m, r in groups
        for row in (
            [f"{m} {r}", "dev"] + [metric(m, r, c, t, "dev") for c, t in cases] + [alloc_mb(m, r)],
            ["", "ht"] + [metric(m, r, c, t, "ht") for c, t in cases] + [""],
            ["", "hu"] + [metric(m, r, c, t, "hu") for c, t in cases] + [""],
        )
    ]
    width = [max(3, *(len(row[i]) for row in [head] + body)) for i in range(len(head))]
    rule = [":" + "-" * (width[0] - 1), ":" + "-" * (width[1] - 1)] + ["-" * (w - 1) + ":" for w in width[2:]]
    report += [
        "| " + " | ".join(x.ljust(width[i]) if i < 2 else x.rjust(width[i]) for i, x in enumerate(row)) + " |"
        for row in [head, rule] + body
    ]

    blind = [p for p in phases if outcome.get(p) == "pass" and p not in device]
    if blind:
        print(f"WARNING: {len(blind)} of {counts['pass']} passing phases have no device samples:", file=sys.stderr)
        print(f"  {', '.join(blind)}", file=sys.stderr)

    text = "\n".join(report) + "\n"
    (args.out.write_text(text) if args.out else sys.stdout.write(text))
    if args.score:
        graded = counts["pass"] + failed
        args.score.write_text(f"score: {counts['pass'] / graded if graded else 0:.3f}\n")


if __name__ == "__main__":
    main()
