#!/usr/bin/env python3
"""Regenerate the TTNN comparison Table 2 from frontier manifests.

This report intentionally reads the live frontier selection manifests instead of
the older static best.csv predictors. Rows that are known stale or semantically
invalid can be quarantined through the audit override table below; those rows
remain visible in the CSV/Markdown and are excluded from win counts.
"""

import argparse
import csv
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "results" / "frontier"


# Audit corrections from the June 25 basis-factor runs. These are intentionally
# small and explicit: the normal path is still manifest-driven.
AUDIT_OVERRIDES = {
    ("bf16", "multigammaln"): {
        "status": "excluded_invalid_old_target",
        "note": (
            "frontier/native rows were generated before multigammaln.json was "
            "fixed to the true p=4 target; rerun p=4 sweep before claiming"
        ),
    },
    ("fp32", "multigammaln"): {
        "status": "excluded_invalid_old_target",
        "note": (
            "frontier/native rows were generated before multigammaln.json was "
            "fixed to the true p=4 target; rerun p=4 sweep before claiming"
        ),
    },
    ("bf16", "gelu"): {
        "role": "fastest_waived_ulp",
        "method": "basis",
        "degree": "6",
        "segments": "1",
        "max_ulp": "0.25",
        "runtime_us": "3.15",
        "csv": "gelu_p6_s1_uniform_basis_ulp.csv",
        "coeff_csv": (
            "/home/ttuser/tt-polynomial-fitter/data/coefficients/"
            "gelu_p6_s1_uniform_basis_ulp.csv"
        ),
        "status": "waived_gelu_0p25ulp_fast",
        "note": (
            "explicit GELU waiver: accept the 0.25 ULP basis-factor candidate "
            "because it is substantially faster than TTNN"
        ),
    },
}


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, digits=2):
    x = _f(value)
    if x is None:
        return "--"
    return f"{x:.{digits}f}"


def _load_manifest(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            method = (row.get("method") or "").strip()
            role = (row.get("role") or "").strip()
            status = (row.get("status") or "").strip()
            if method == "ttnn" or role == "ttnn" or status == "ttnn_ref":
                continue
            rows.append(row)
    return rows


def _row_result(row):
    ours_ulp = _f(row.get("max_ulp"))
    ours_us = _f(row.get("runtime_us"))
    ttnn_ulp = _f(row.get("ttnn_maxulp"))
    ttnn_us = _f(row.get("ttnn_us"))
    if None in (ours_ulp, ours_us, ttnn_ulp, ttnn_us):
        return "incomplete"
    if ours_ulp <= ttnn_ulp and ours_us <= ttnn_us:
        return "win_both"
    if ours_ulp <= ttnn_ulp:
        return "accuracy_match_slow"
    if ours_us < ttnn_us:
        return "faster_less_accurate"
    return "loss"


def _apply_audit(dtype, row):
    out = dict(row)
    override = AUDIT_OVERRIDES.get((dtype, row.get("activation")))
    if override:
        out.update(override)
    status = str(out.get("status", ""))
    if status.startswith("excluded_"):
        out["audited_result"] = "excluded"
    elif status.startswith("waived_"):
        out["audited_result"] = "waived_fast"
    else:
        out["audited_result"] = _row_result(out)
    ours_us = _f(out.get("runtime_us"))
    ttnn_us = _f(out.get("ttnn_us"))
    out["speedup_vs_ttnn"] = f"{ttnn_us / ours_us:.3f}" if ours_us and ttnn_us else ""
    return out


def regenerate(dtype, manifest):
    return [_apply_audit(dtype, row) for row in _load_manifest(manifest)]


def write_csv(path, rows):
    fields = [
        "activation",
        "dtype",
        "role",
        "method",
        "degree",
        "segments",
        "max_ulp",
        "runtime_us",
        "ttnn_maxulp",
        "ttnn_us",
        "speedup_vs_ttnn",
        "status",
        "audited_result",
        "note",
        "csv",
        "coeff_csv",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows_by_dtype):
    lines = [
        "# Table 2 - Frontier Pareto TTNN Comparison",
        "",
        "Source: `results/frontier/<dtype>/data/csv/pareto_winners.csv`.",
        "Rows marked `excluded` or `incomplete` are visible but are not counted as wins.",
        "",
    ]
    for dtype, rows in rows_by_dtype.items():
        comparable = [r for r in rows if r["audited_result"] not in ("excluded", "incomplete")]
        incomplete = [r for r in rows if r["audited_result"] == "incomplete"]
        wins = [r for r in comparable if r["audited_result"] == "win_both"]
        waived = [r for r in comparable if r["audited_result"] == "waived_fast"]
        acc = [r for r in comparable if r["audited_result"] == "accuracy_match_slow"]
        faster = [r for r in comparable if r["audited_result"] == "faster_less_accurate"]
        excluded = [r for r in rows if r["audited_result"] == "excluded"]
        lines += [
            f"## {dtype}",
            "",
            (
                f"Comparable rows: {len(comparable)}. Win on ULP and runtime: {len(wins)}. "
                f"Waived faster exceptions: {len(waived)}. Accuracy match but slower: "
                f"{len(acc)}. Faster but less accurate: {len(faster)}. "
                f"Incomplete TTNN refs: {len(incomplete)}. Excluded: {len(excluded)}."
            ),
            "",
            "| activation | ours cfg | ours ULP | ours us | TTNN ULP | TTNN us | speedup | result |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in sorted(rows, key=lambda x: x.get("activation", "")):
            cfg = f"{r.get('method', '')}:s{r.get('segments', '')}/d{r.get('degree', '')}"
            lines.append(
                "| {activation} | {cfg} | {ulp} | {us} | {tulp} | {tus} | {speed} | {result} |".format(
                    activation=r.get("activation", ""),
                    cfg=cfg,
                    ulp=_fmt(r.get("max_ulp")),
                    us=_fmt(r.get("runtime_us")),
                    tulp=_fmt(r.get("ttnn_maxulp")),
                    tus=_fmt(r.get("ttnn_us")),
                    speed=(r.get("speedup_vs_ttnn") or "--"),
                    result=r.get("audited_result", ""),
                )
            )
        if excluded:
            lines += ["", "Excluded rows:"]
            for r in excluded:
                lines.append(f"- {r.get('activation')}: {r.get('note', r.get('status'))}")
        if waived:
            lines += ["", "Waived rows:"]
            for r in waived:
                lines.append(f"- {r.get('activation')}: {r.get('note', r.get('status'))}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-manifest", default=str(DEFAULT_ROOT / "bf16" / "data" / "csv" / "pareto_winners.csv"))
    parser.add_argument("--fp32-manifest", default=str(DEFAULT_ROOT / "fp32" / "data" / "csv" / "pareto_winners.csv"))
    parser.add_argument("--outdir", default=str(DEFAULT_ROOT / "table2"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows_by_dtype = {
        "bf16": regenerate("bf16", Path(args.bf16_manifest)),
        "fp32": regenerate("fp32", Path(args.fp32_manifest)),
    }
    all_rows = [row for rows in rows_by_dtype.values() for row in rows]
    write_csv(outdir / "table2_frontier_ttnn.csv", all_rows)
    write_markdown(outdir / "table2_frontier_ttnn.md", rows_by_dtype)
    print(outdir / "table2_frontier_ttnn.md")
    print(outdir / "table2_frontier_ttnn.csv")


if __name__ == "__main__":
    main()
