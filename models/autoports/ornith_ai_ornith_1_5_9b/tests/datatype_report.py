# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Export measured full-model policies and accuracy/throughput Pareto plots."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
    rows = []
    p1 = {}
    for path in sorted(root.glob("p1_*.json")):
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict):
            continue
        data = doc.get("data", {})
        for name, value in data.get("candidates", {}).items():
            if "within_noise" in value:
                p1[name] = {"pass": value["within_noise"], "artifact": str(path), **value}
        if data.get("reference"):
            for check in doc.get("checks", []):
                if check.get("id") == "reference_vs_hf" and check.get("status") == "pass":
                    p1[data["reference"]] = {
                        "pass": True,
                        "artifact": str(path),
                        "basis": "Reference directly passes the runner's HF comparison bound",
                        "hf": data.get("hf"),
                    }
        if data.get("selected") == data.get("reference"):
            for check in doc.get("checks", []):
                if check.get("id") == "selected_within_noise":
                    p1[data["reference"]] = {"pass": check.get("status") == "pass", "artifact": str(path)}
    for name in json.loads((root / "candidate_matrix.json").read_text()):
        path = root / f"{name}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        refs = result.get("references", {})
        policy = json.loads((root / "configs" / f"{name}.json").read_text())
        row = {
            "config_id": name,
            "precision_config": str(root / "configs" / f"{name}.json"),
            "dtype_policy": {k: v for k, v in policy.items() if k != "compute_fidelities"},
            "compute_fidelity_policy": policy["compute_fidelities"],
            "accuracy_status": result["status"],
            "p1": p1.get(name),
            "selected": name == args.selected,
            "command": result.get("command"),
            "hardware": result.get("hardware"),
            "mesh": result.get("mesh"),
            "measurement_regime": result.get("measurement_regime"),
            "runtime_evidence": str(path),
            "references": {
                ref: {
                    phase: {k: v for k, v in values[phase].items() if k != "generator_metrics"}
                    for phase in ("prefill", "teacher_forcing")
                }
                for ref, values in refs.items()
            },
        }
        if len(refs) == 2:
            row.update(
                top1=min(v["teacher_forcing"]["top1"] for v in refs.values()),
                top5=min(v["teacher_forcing"]["top5"] for v in refs.values()),
                top100=min(v["teacher_forcing"]["top100"] for v in refs.values()),
                ttft_ms=refs["aime24"]["teacher_forcing"]["ttft_ms"],
                traced_decode_tps=refs["aime24"]["teacher_forcing"]["decode_t/s/u"],
                trace_verified=all(
                    s["generator_metrics"]["trace_verified"]
                    for v in refs.values()
                    for s in v["teacher_forcing_samples"]
                ),
            )
        if name == "bfp8_mlp_lofi":
            repeats = [root / f"{value}.json" for value in (name + "_run1", name + "_run2", name)]
            values = [json.loads(p.read_text())["references"]["aime24"]["teacher_forcing"] for p in repeats]
            speeds = [v["decode_t/s/u"] for v in values]
            row["baseline_repeats"] = {
                "artifacts": [str(p) for p in repeats],
                "traced_decode_tps": speeds,
                "spread_tps": max(speeds) - min(speeds),
                "aggregation": "median of three serial runs, each using two warmed traced requests",
            }
            row["traced_decode_tps"] = statistics.median(speeds)
            row["ttft_ms"] = statistics.median(v["ttft_ms"] for v in values)
        row["status"] = (
            "fail_accuracy"
            if row["accuracy_status"] != "pass"
            else "not_evaluated_p1"
            if not row["p1"]
            else "pass"
            if row["p1"]["pass"]
            else "fail_p1"
        )
        row["decision"] = (
            "selected"
            if row["selected"]
            else "rejected_full_model_accuracy"
            if row["accuracy_status"] != "pass"
            else "rejected_p1"
            if row["status"] == "fail_p1"
            else "slower_than_selected; P1 not needed for selection"
            if not row["p1"]
            else "passed; slower than selected"
        )
        rows.append(row)
    if args.selected not in {r["config_id"] for r in rows}:
        raise ValueError("Selected config has no evaluated artifact")
    (root / "sweep_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "sweep_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, threshold in (("top1", 0.90), ("top5", 0.98)):
        points = [row for row in rows if row.get("trace_verified")]
        frontier = [
            r
            for r in points
            if not any(
                s[metric] >= r[metric]
                and s["traced_decode_tps"] >= r["traced_decode_tps"]
                and (s[metric] > r[metric] or s["traced_decode_tps"] > r["traced_decode_tps"])
                for s in points
            )
        ]
        frontier.sort(key=lambda r: r[metric])
        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
            ax.plot(
                [r[metric] * 100 for r in frontier],
                [r["traced_decode_tps"] for r in frontier],
                color="#425f83",
                linewidth=1.5,
                label="Evaluated Pareto frontier",
            )
            for i, row in enumerate(points):
                color = "#d62728" if row["selected"] else "#278a79" if row["status"] == "pass" else "#8994a4"
                marker = "*" if row["selected"] else "o" if row["accuracy_status"] == "pass" else "x"
                ax.scatter(
                    row[metric] * 100,
                    row["traced_decode_tps"],
                    c=color,
                    marker=marker,
                    s=220 if row["selected"] else 55,
                    zorder=10 if row["selected"] else 5,
                )
                if (
                    row["selected"]
                    or row in frontier
                    or row["config_id"] in ("bfp8_mlp_lofi", "bf16_hifi4", "bfp4_head_lofi")
                ):
                    ax.annotate(
                        row["config_id"],
                        (row[metric] * 100, row["traced_decode_tps"]),
                        xytext={
                            "bfp8_all_lofi": (10, 12),
                            "bfp8_mlp_lofi": (10, -28),
                            "bfp8_attention_lofi": (10, 27),
                            "bfp8_all_hifi2": (10, -7),
                        }.get(row["config_id"], (7, 10 if i % 2 else -16)),
                        textcoords="offset points",
                        fontsize=8,
                        color="#b31b1b" if row["selected"] else "#354252",
                    )
            ax.axvline(
                threshold * 100,
                color="#555555",
                linestyle=":",
                linewidth=1.8,
                label=f"Minimum {metric}: {threshold:.0%}",
            )
            ax.set(
                xlabel=f"Teacher-forced {metric} agreement with HF — minimum across both references (%)",
                ylabel="Warmed traced teacher-forcing decode (tokens/s/user)",
                title=f"Ornith-1.5-9B · TP4 Blackhole · {metric} / throughput",
            )
            ax.legend(loc="best", frameon=True)
            ax.margins(x=0.18, y=0.12)
            fig.savefig(root / f"{metric}_perf_pareto.png", dpi=160)
            plt.close(fig)


if __name__ == "__main__":
    main()
