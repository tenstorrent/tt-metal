#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn a combine sweep run directory into its results table."""

import argparse
import bisect
import csv
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RINGS = ("ring8", "ring4")
CHUNKS = ("2.5k", "5k", "10k")
TRAFFIC = ("balanced", "hot_expert")
LAYOUT = (("row_major", "row major"), ("tile", "tiled"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--score", type=Path, help="write 'score: <pass fraction>' here, which robo_test reads")
    ap.add_argument("--device-from", type=Path, metavar="RUN")
    return ap.parse_args()


def run_dir(given=None):
    directory = given or os.environ.get("CMB_SWEEP_DIR")
    if not directory:
        sys.exit("CMB_SWEEP_DIR is not set; it names the run directory to report")
    return Path(directory)


def device_spread(run):
    """Per phase, the min and max over chips of that chip's median replay, in us.

    The sweep emits one signpost carrying the phase name, so a phase's samples are the ops between
    its signpost and the next one. GLOBAL CALL COUNT is kept per device, so the timed replays are
    picked per device.
    """
    per_phase = defaultdict(lambda: defaultdict(list))
    for path in sorted(run.glob("reports/*/ops_perf_results*.csv")):
        # One CSV per profiled process: timestamps only order within one, so each is matched alone.
        rows = list(csv.DictReader(path.open()))
        posts = sorted(
            (int(r["HOST START TS"]), r["OP CODE"])
            for r in rows
            if r["OP TYPE"] == "signpost" and "-cmb_" in r["OP CODE"]
        )
        if not posts:
            continue
        starts = [ts for ts, _ in posts]
        for r in rows:
            if r["OP TYPE"] == "tt_dnn_device" and r["OP CODE"].startswith("Combine"):
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


def table(head, body):
    """One markdown table, every column as wide as its widest cell. The first two are row labels."""
    width = [max(3, *(len(row[i]) for row in [head] + body)) for i in range(len(head))]
    rule = [":" + "-" * (width[0] - 1), ":" + "-" * (width[1] - 1)] + ["-" * (w - 1) + ":" for w in width[2:]]
    return [
        "| " + " | ".join(x.ljust(width[i]) if i < 2 else x.rjust(width[i]) for i, x in enumerate(row)) + " |"
        for row in [head, rule] + body
    ]


def main():
    args = parse_args()
    phases, outcome, measured, device = load(run_dir())
    if args.device_from:
        *_, device = load(args.device_from)

    def metric(model, version, ring, chunk, traffic, layout, which):
        name = f"{model}-{ring}-{chunk}-{traffic}-{layout}-cmb_{version}"
        if name not in outcome:
            return ""
        if outcome[name] != "pass":
            return outcome[name] if which == "dev" else ""
        span = device.get(name)
        if which == "dev":
            return f"{span[0]:.0f} - {span[1]:.0f}" if span else "-"
        value = measured.get(name, {}).get("traced_us" if which == "ht" else "untraced_us")
        if not value:
            return "-"
        if not span:
            return f"{float(value):.0f}"
        # Not a bare "|": that is the table's own column separator, and a cell holding one splits
        # into two, leaving the row wider than its header.
        return f"{float(value):.0f} ({float(value) / span[1] - 1:+.0%})"

    def field(which, **where):
        """The distinct values of a recorded field over the phases matching the given name parts."""
        return sorted(
            {
                f[which]
                for p, f in measured.items()
                if which in f and all(part in p.split("-") for part in where.values())
            },
            key=lambda v: int(v) if v.isdigit() else v,
        )

    def alloc_mb(model, version, ring, layout):
        """One figure when the row's cells agree, a range when they do not.

        A phase that reuses a cached program allocates nothing of its own and records a zero, so
        those are dropped rather than widening every range down to nothing.
        """
        mb = sorted(
            {
                int(f["op_alloc_bytes"]) / 1000**2
                for p, f in measured.items()
                if p.startswith(f"{model}-{ring}-")
                and p.endswith(f"-{layout}-cmb_{version}")
                and f.get("op_alloc_bytes") != "0"
            }
        )
        return "" if not mb else f"{mb[0]:.0f}" if round(mb[0]) == round(mb[-1]) else f"{mb[0]:.0f} - {mb[-1]:.0f}"

    parts = [p.split("-") for p in phases]
    models = list(dict.fromkeys(p[0] for p in parts))
    versions = list(dict.fromkeys(p[5].removeprefix("cmb_") for p in parts if len(p) > 5))
    rings = [r for r in RINGS if any(p[1] == r for p in parts)]
    counts = Counter(outcome.values())
    failed = counts["FAIL"] + counts["HANG"] + counts["unrun"] + counts["?"]

    report = [
        f"{counts['pass']} pass, {counts['unsup']} unsup, {failed} failed",
        "",
        "dev is the device op time per chip, min - max over the chips. ht and hu are the host wall",
        "clock per launch, traced and untraced, in us, each with what it adds over the slowest chip.",
        "",
        "Chunk is the tokens per dispatch group, so a chip carries chunk/ring of them. The forwarding",
        "buffer is held fixed across chunks by scaling the reduction factor with that per-chip share.",
    ]
    # Chunk and traffic make the columns; everything else crosses with the model to make the rows,
    # which keeps the table narrow enough to read however many axes the run swept.
    cases = [
        (c, t)
        for c in CHUNKS
        for t in TRAFFIC
        if any(
            metric(m, v, r, c, t, layout, "dev")
            for m in models
            for v in versions
            for r in rings
            for layout, _ in LAYOUT
        )
    ]
    groups = [
        (m, v, r, layout, label)
        for m in models
        for r in rings
        for layout, label in LAYOUT
        for v in versions
        if any(metric(m, v, r, c, t, layout, "dev") for c, t in cases)
    ]
    for ring in rings:
        chips = field("chips", ring=ring)
        experts = field("experts_per_chip", ring=ring)
        reductions = field("fwd_reduction", ring=ring)
        spread = " - ".join(dict.fromkeys([experts[0], experts[-1]])) if experts else "?"
        report += [
            "",
            f"{ring}: {' / '.join(chips) or '?'} chips, {spread} experts per chip across these models, "
            f"reduction factor {', '.join(r + 'x' for r in reductions) or 'unknown'}",
        ]
    report.append("")
    head = ["case", ""] + [f"{c} {t}" for c, t in cases] + ["op alloc MB"]
    body = [
        row
        for m, v, r, layout, label in groups
        for row in (
            [" ".join([m, r, label] + ([v] if len(versions) > 1 else [])), "dev"]
            + [metric(m, v, r, c, t, layout, "dev") for c, t in cases]
            + [alloc_mb(m, v, r, layout)],
            ["", "ht"] + [metric(m, v, r, c, t, layout, "ht") for c, t in cases] + [""],
            ["", "hu"] + [metric(m, v, r, c, t, layout, "hu") for c, t in cases] + [""],
        )
    ]
    report += table(head, body)

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
