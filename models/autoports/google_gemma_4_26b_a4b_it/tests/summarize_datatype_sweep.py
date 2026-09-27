# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate full-model precision runs and plot their measured Pareto fronts."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep")


def main():
    rows = []
    for p in sorted((ROOT / "results").glob("*.json")):
        row = json.loads(p.read_text())
        row["evidence"] = str(p)
        policy = row.get("dtype_policy", {})

        def fidelities(value):
            return {
                k: fidelities(v) if isinstance(v, dict) else v
                for k, v in value.items()
                if isinstance(v, dict) or "fidelity" in k
            }

        row["compute_fidelity_policy"] = fidelities(policy)
        row.pop("runtime_policy", None)
        row.pop("decode_runs", None)
        for metric in ("top1", "top5", "top100"):
            row["prefill_" + metric] = min((r[metric] for r in row.get("prefill", [])), default=None)
        row["plot_id"] = f"C{len(rows)+1:02d}"
        rows.append(row)
    eligible = [
        r for r in rows if r.get("trace_verified") and r.get("performance_eligible", True) and r.get("status") == "pass"
    ]
    if not eligible:
        raise RuntimeError("No accuracy-passing traced full-model candidate")
    winner = max(eligible, key=lambda r: r["decode_tps"])
    for row in rows:
        row["selection_decision"] = (
            "selected"
            if row["config_id"] == winner["config_id"]
            else "slower_passing_policy"
            if row.get("status") == "pass"
            else row.get("status", "unmeasured")
        )
    (ROOT / "sweep_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = [
        "plot_id",
        "config_id",
        "status",
        "selection_decision",
        "top1",
        "top5",
        "top100",
        "prefill_top1",
        "prefill_top5",
        "prefill_top100",
        "ttft_ms",
        "decode_tps",
        "trace_verified",
        "measurement_regime",
        "dtype_policy",
        "compute_fidelity_policy",
        "evidence",
        "precision_config_path",
        "command",
        "hardware",
        "mesh",
        "commit",
        "error",
    ]
    with (ROOT / "sweep_results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(row[k], separators=(",", ":")) if isinstance(row.get(k), (dict, list)) else row.get(k)
                    for k in fields
                }
            )
    measured = [
        r
        for r in rows
        if r.get("trace_verified") and r.get("performance_eligible", True) and r.get("decode_tps") is not None
    ]
    for metric, threshold in [("top1", 0.9), ("top5", 0.98)]:
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        fig.patch.set_facecolor("#f7f9fc")
        ax.set_facecolor("#f7f9fc")
        frontier = sorted(
            [
                r
                for r in measured
                if not any(
                    s[metric] >= r[metric]
                    and s["decode_tps"] >= r["decode_tps"]
                    and (s[metric] > r[metric] or s["decode_tps"] > r["decode_tps"])
                    for s in measured
                )
            ],
            key=lambda r: r[metric],
        )
        ax.plot(
            [100 * r[metric] for r in frontier],
            [r["decode_tps"] for r in frontier],
            color="#247c8c",
            linewidth=1.5,
            marker="o",
            markersize=12,
            fillstyle="none",
            label="Measured Pareto frontier",
        )
        for i, row in enumerate(measured):
            selected = row["config_id"] == winner["config_id"]
            ax.scatter(
                100 * row[metric],
                row["decode_tps"],
                s=100 if selected else 42,
                color="#d62728" if selected else "#247c8c" if row["status"] == "pass" else "#9aa3af",
                marker="*" if selected else "o",
                zorder=4,
            )
            if (
                selected
                or row in frontier
                or row["config_id"] in ("baseline_mixed_lofi", "decode_bf16_hifi4", "activation_bfp8")
            ):
                ax.annotate(
                    row["plot_id"],
                    (100 * row[metric], row["decode_tps"]),
                    xytext=(-38 if selected else 8, 10 if selected else -4),
                    textcoords="offset points",
                    fontsize=8,
                    color="#b51f26" if selected else "#354253",
                )
        ax.axvline(100 * threshold, linestyle=":", color="#545a66", label=f"Minimum {metric}: {100*threshold:g}%")
        ax.set_xlabel(f'Teacher-forcing {metric.replace("top", "Top-")} accuracy (%)')
        ax.set_ylabel("Traced teacher-forcing decode (tokens/s/user)")
        ax.set_xlim(
            100 * min(threshold, min(r[metric] for r in measured)) - 0.4,
            min(100.4, 100 * max(r[metric] for r in measured) + 0.4),
        )
        ax.set_title(
            "Gemma 4 26B A4B · full model precision sweep\n100 AIME24 continuation positions · batch 1 · TP4",
            loc="left",
        )
        ax.text(
            0.01,
            0.02,
            f"Selected {winner['plot_id']}: {winner['config_id']}",
            transform=ax.transAxes,
            fontsize=9,
            color="#b51f26",
        )
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", fontsize=9)
        fig.savefig(ROOT / f"{metric}_perf_pareto.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"fastest_passing": winner["config_id"], "decode_tps": winner["decode_tps"]}, indent=2))


if __name__ == "__main__":
    main()
