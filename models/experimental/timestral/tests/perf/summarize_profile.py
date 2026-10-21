# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Summarize one optimization round: tt-perf-report tables plus the full-model device-time estimate.

    python models/experimental/timestral/tests/perf/summarize_profile.py <iter_dir>

Reads ``<iter_dir>/ops_perf_results.csv`` (Tracy, from ``test_profile_forward.py``),
``layer_counts.json`` and, when present, ``bench.json``. Writes:

    perf_report.txt                 advice-enabled table for the whole reduced forward
    perf_report.csv, *_stacked.*    the same window as CSV, stacked by op type, and the PNG
    sections/<section>.csv          one window per signposted section
    summary.json, summary.md        section device times, the 24-layer estimate vs the measured
                                    device time, op categories and the slowest ops
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

WINDOW = ("TIMESTRAL_START", "TIMESTRAL_END")
TOP_OPS = 15


def _perf_report(ops: Path, start: str, end: str, *args: str, stdout=None) -> None:
    cmd = ["tt-perf-report", str(ops), "--start-signpost", start, "--end-signpost", end, "--no-color", *args]
    subprocess.run(cmd, check=True, stdout=stdout or subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _section_stats(rows: list[dict]) -> dict:
    return {
        "device_us": sum(_num(r["Device Time"]) for r in rows),
        "op_to_op_gap_us": sum(_num(r["Op-to-Op Gap"]) for r in rows),
        "ops": len(rows),
    }


def _measured(bench: Path, shape: str, precision: str) -> dict | None:
    if not bench.is_file():
        return None
    for record in json.loads(bench.read_text()):
        if record.get("test") == "predict_block" and record["name"] == shape and record["precision"] == precision:
            return record
    return None


def main(iter_dir: Path) -> None:
    ops = iter_dir / "ops_perf_results.csv"
    meta = json.loads((iter_dir / "layer_counts.json").read_text())

    with (iter_dir / "perf_report.txt").open("w") as f:
        _perf_report(ops, *WINDOW, stdout=f)
    _perf_report(ops, *WINDOW, "--csv", str(iter_dir / "perf_report.csv"))

    sections_dir = iter_dir / "sections"
    sections_dir.mkdir(exist_ok=True)
    sections = {}
    for name in meta["sections"]:
        out = sections_dir / f"{name}.csv"
        _perf_report(ops, f"{name}_start", f"{name}_end", "--no-advice", "--csv", str(out))
        (sections_dir / f"{name}_stacked.png").unlink(missing_ok=True)
        sections[name] = _section_stats(_rows(out))

    counts = meta["layer_counts"]
    estimate_us = sections["embed"]["device_us"] + sections["head"]["device_us"]
    estimate_us += sum(counts[k] * sections[f"layer_{k}"]["device_us"] for k in counts)
    estimate_ops = sections["embed"]["ops"] + sections["head"]["ops"]
    estimate_ops += sum(counts[k] * sections[f"layer_{k}"]["ops"] for k in counts)

    window_rows = _rows(iter_dir / "perf_report.csv")
    stacked = _rows(iter_dir / "perf_report_stacked.csv")
    categories: dict[str, float] = {}
    for row in stacked:
        categories[row["Op Category"]] = categories.get(row["Op Category"], 0.0) + _num(row["Total % [%]"])
    matmul_rows = [r for r in stacked if r["Op Code"].startswith("Matmul")]
    matmul_us = sum(_num(r["Device Time Sum [μs]"]) for r in matmul_rows)
    matmul_flops = (
        sum(_num(r["Weighted Mean FLOPs [%]"]) * _num(r["Device Time Sum [μs]"]) for r in matmul_rows) / matmul_us
        if matmul_us
        else None
    )
    slowest = sorted(window_rows, key=lambda r: -_num(r["Device Time"]))[:TOP_OPS]

    measured = _measured(iter_dir / "bench.json", meta["shape"], meta["precision"])
    summary = {
        "shape": meta["shape"],
        "precision": meta["precision"],
        "reduced_window": _section_stats(window_rows),
        "sections": sections,
        "layer_counts": counts,
        "full_model_estimate": {"device_ms": estimate_us / 1e3, "ops": estimate_ops},
        "measured": None
        if measured is None
        else {k: measured[k] for k in ("steady_s", "device_s", "cold_s", "pcc_normalized", "pcc_raw")},
        "category_percent": categories,
        "matmul_weighted_flops_percent": matmul_flops,
        "stacked": [
            {
                "op": r["Op Code"],
                "percent": _num(r["Total % [%]"]),
                "device_us": _num(r["Device Time Sum [μs]"]),
                "count": int(_num(r["Op Count"])),
                "category": r["Op Category"],
            }
            for r in stacked
        ],
        "slowest_ops": [
            {
                "id": r["ID"],
                "op": r["OP Code"],
                "device_us": _num(r["Device Time"]),
                "bound": r["Bound"],
                "cores": r["Cores"],
                "dram_percent": _num(r["DRAM %"]),
                "flops_percent": _num(r["FLOPs %"]),
                "fidelity": r["Math Fidelity"],
                "in0_block_w": r["Inner Dim Block Size"],
                "subblock": f"{r['Output Subblock H']}x{r['Output Subblock W']}",
            }
            for r in slowest
        ],
    }
    (iter_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    est_ms = estimate_us / 1e3
    lines = [
        f"# Round summary: {meta['shape']} / {meta['precision']}",
        "",
        "| section | device ms | ops | x count | contribution ms |",
        "|---|---|---|---|---|",
    ]
    for name, stats in sections.items():
        n = counts.get(name.removeprefix("layer_"), 1) if name.startswith("layer_") else 1
        lines.append(
            f"| {name} | {stats['device_us'] / 1e3:.2f} | {stats['ops']} | {n} | {n * stats['device_us'] / 1e3:.1f} |"
        )
    lines += ["", f"Estimated full-model device time: **{est_ms:.1f} ms** over {estimate_ops} ops."]
    if measured is not None:
        dev_ms = measured["device_s"] * 1e3
        lines.append(
            f"Measured: device {dev_ms:.1f} ms (estimate off by {100 * (est_ms - dev_ms) / dev_ms:+.1f}%), "
            f"steady wall {measured['steady_s'] * 1e3:.1f} ms, PCC {measured['pcc_normalized']:.5f} normalized."
        )
    flops = "n/a" if matmul_flops is None else f"{matmul_flops:.1f}%"
    lines += [
        "",
        "Category share of the reduced window: "
        + ", ".join(f"{k} {v:.1f}%" for k, v in sorted(categories.items(), key=lambda kv: -kv[1]))
        + f"; matmul weighted FLOPs utilization {flops}.",
        "",
        "| op type | % | device ms | count |",
        "|---|---|---|---|",
        *(
            f"| {r['op']} | {r['percent']:.1f} | {r['device_us'] / 1e3:.2f} | {r['count']} |"
            for r in summary["stacked"]
        ),
    ]
    (iter_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
