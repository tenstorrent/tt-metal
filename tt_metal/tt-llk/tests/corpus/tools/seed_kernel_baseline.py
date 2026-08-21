#!/usr/bin/env python3
"""Seed the KERNEL-scoped (v2) VERDICT baseline from one sweep run's
evidence (lane ET, owner ratification 2026-08-21).

Reads <evidence-root>/scoreboard.tsv (the KERNEL_*_E2E scope rows the
dual-metric sweep emits) plus each op's ROW-VERDICT.json (for the
KERNEL-decided class), and writes a candidate
sfpu_device_baseline_<class>_v2.tsv with the conf-anchored header that
conf_lint R5b/R6b enforces.

BASELINES ARE NEVER MODIFIED BY SWEEPS: this tool writes a CANDIDATE file;
committing it is a reviewed manual step (corpus/README.md baseline update
procedure) — review the KERNEL-DELTA.md of the seeding run alongside it.

Usage:
  seed_kernel_baseline.py --evidence-root <run> [--conf <sweep_2x2.conf>]
      [--chip-class p150] [--out <path>] [--force]
"""

import argparse
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent

CLASS_MAP = {"WIN": "win", "PARITY": "parity", "LOSS": "loss", "REFUSAL": "refusal"}


def conf_pin(conf, name):
    for line in conf.read_text().splitlines():
        if line.startswith(f"_REVIEWED_{name}_SHA256="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit(f"seed_kernel_baseline: conf lacks _REVIEWED_{name}_SHA256: {conf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-root", type=pathlib.Path, required=True)
    ap.add_argument("--conf", type=pathlib.Path, default=HERE / "sweep_2x2.conf")
    ap.add_argument("--chip-class", default="p150")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing output file"
    )
    a = ap.parse_args()
    out = a.out or HERE / f"sfpu_device_baseline_{a.chip_class}_v2.tsv"
    if out.exists() and not a.force:
        sys.exit(
            f"seed_kernel_baseline: {out} exists — v2 updates are a reviewed "
            "manual step; pass --force only for a deliberate re-seed"
        )
    sb = a.evidence_root / "scoreboard.tsv"
    if not sb.is_file():
        sys.exit(f"seed_kernel_baseline: no scoreboard.tsv in {a.evidence_root}")
    cc1 = conf_pin(a.conf, "CC1PLUS")
    sim_bh = conf_pin(a.conf, "SIM_BH")

    # KERNEL-decided class per op, from the streamed ROW-VERDICTs.
    op_class, op_ratio = {}, {}
    for rv in sorted(a.evidence_root.glob("*/ROW-VERDICT.json")):
        try:
            p = json.loads(rv.read_text())
        except ValueError:
            continue
        op_class[p["op"]] = CLASS_MAP.get(p.get("class"), "")
        op_ratio[p["op"]] = p.get("kernel_vs_hand_pct")

    rows_out, seen = [], set()
    with sb.open() as f:
        for rec in csv.DictReader(
            (x for x in f if not x.startswith("#")), delimiter="\t"
        ):
            scope = rec["scope"]
            if not scope.endswith("_E2E") or not scope.startswith("KERNEL_"):
                continue
            key = (rec["id"], scope, rec["selector"])
            if key in seen:
                continue
            seen.add(key)
            # selector is '<op>:<cell>' (pinpair rows keep native selectors)
            op = rec["selector"].split(":", 1)[0] if ":" in rec["selector"] else ""
            cls = op_class.get(op, "")
            status = rec["status"]
            if status == "REFUSAL_BYTE_IDENTICAL":
                status, cls = "refusal_byte_identical", "refusal"
            ratio = op_ratio.get(op)
            prov = (
                f"seeded from {a.evidence_root.name} (e2e-metric re-measure; "
                f"KERNEL-decided class"
                + (
                    f", kernel vs_hand {ratio:+.2f}%"
                    if isinstance(ratio, (int, float))
                    else ""
                )
                + ")"
            )
            rows_out.append(
                [
                    rec["id"],
                    rec.get("arch", "bh"),
                    a.chip_class,
                    "device_cycles",
                    scope,
                    rec["selector"],
                    rec["cycles"],
                    status,
                    cls,
                    rec.get("compiler_sha", cc1),
                    prov,
                ]
            )
    if not rows_out:
        sys.exit(
            "seed_kernel_baseline: scoreboard.tsv carries no KERNEL_*_E2E rows "
            "— was the run made with the dual-metric sweep?"
        )
    hdr = [
        f"# schema=2; chip_class={a.chip_class}; KERNEL-scoped (v2) VERDICT baseline — "
        "cycles are ABSOLUTE end-to-end device kernel time: mean(<metric>) at the "
        "drain-inclusive KERNEL profiler marker (harness trisc.cpp wraps run_kernel() + "
        'tensix_sync() in ZONE_SCOPED("KERNEL")), never per-tile divided.',
        "# VERDICT METRIC (owner ratification 2026-08-21, lane ET): WIN/PARITY/LOSS verdicts",
        "# are decided by these anchors for every row; sfpu_device_baseline_*_v1.tsv keeps the",
        "# DIAGNOSTIC (body-zone) anchors.  The issue-slot lower-bound gate applies only to the",
        "# diagnostic zone (KERNEL is structurally drain-inclusive).",
        f"# Seeded from {a.evidence_root} by tools/seed_kernel_baseline.py; committing an",
        "# update is a REVIEWED manual step (review the run's KERNEL-DELTA.md alongside).",
        "# Pin anchors (conf_lint R5b/R6b enforce these against sweep_2x2.conf):",
        f"# {cc1} (CURRENT sweep_2x2.conf PINNED_CC1PLUS_SHA256)",
        f"# {sim_bh} (CURRENT sweep_2x2.conf PINNED_SIM_BH_SHA256)",
    ]
    cols = (
        "id\tarch\tchip_class\tmetric\tscope\tselector\tcycles\tstatus\t"
        "expected_class\tcompiler_sha\tprovenance"
    )
    out.write_text(
        "\n".join(hdr)
        + "\n"
        + cols
        + "\n"
        + "\n".join("\t".join(str(x) for x in r) for r in rows_out)
        + "\n"
    )
    print(
        f"seed_kernel_baseline: wrote {len(rows_out)} KERNEL-scope rows -> {out}\n"
        "REMINDER: committing this file is a reviewed baseline update "
        "(cite the seeding run + KERNEL-DELTA.md in the commit)."
    )


if __name__ == "__main__":
    main()
