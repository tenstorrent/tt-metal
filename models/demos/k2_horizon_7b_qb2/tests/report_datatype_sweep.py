"""Rebuild tables and pyplot Pareto figures from inspectable full-model runs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DOC = Path("models/demos/k2_horizon_7b_qb2/doc/datatype_sweep")


def main():
    selected = json.loads((DOC / "selected_precision_config.json").read_text())["config_id"]
    by_id = {}
    for path in sorted((DOC / "runs").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("layers") != 36:
            continue
        row["artifact"] = str(path)
        by_id[row["config_id"]] = row
    for path in sorted((DOC / "qualifications").glob("*_qualification.json")):
        row = json.loads(path.read_text())
        if row.get("completed_at"):
            row["coarse_artifact"] = by_id.get(row["config_id"], {}).get("artifact")
            row["artifact"] = str(path)
            by_id[row["config_id"]] = row
    verdict_path = DOC / "quality_verdicts.json"
    final_path = DOC / "final_selected.json"
    if final_path.exists():
        row = json.loads(final_path.read_text())
        if row.get("completed_at"):
            previous = by_id.get(row["config_id"], {})
            row["qualification_artifact"] = previous.get("artifact")
            row["coarse_artifact"] = previous.get("coarse_artifact")
            row["artifact"] = str(final_path)
            by_id[row["config_id"]] = row
    verdicts = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
    rows = []
    retained = {
        "config_id",
        "precision_config",
        "dtype_policy",
        "compute_fidelity_policy",
        "command",
        "started_at",
        "completed_at",
        "commit",
        "hardware",
        "mesh",
        "reference",
        "reference_sha256",
        "prompt_len",
        "generation_len",
        "batch_size",
        "layers",
        "measurement_regime",
        "prefill",
        "teacher_runs",
        "teacher_perf",
        "top1",
        "top5",
        "top100",
        "token_count",
        "ttft_ms",
        "decode_tokens_per_second_per_user",
        "trace_verified",
        "accuracy_pass",
        "status",
        "artifact",
        "coarse_artifact",
        "qualification_artifact",
        "continuation",
        "capability_pass",
        "quality_artifact",
        "quality_skipped",
        "error",
        "cache_capacity_tokens",
        "environment",
    }
    for full in by_id.values():
        row = {key: value for key, value in full.items() if key in retained}
        row["runtime_propagation_evidence"] = {
            "artifact": full["artifact"],
            "key": "runtime_summary",
            "actual_weights_and_kernel_configs_checked": "runtime_summary" in full,
        }
        row["quality_verdict"] = verdicts.get(row["config_id"])
        if row["quality_verdict"] and not row["quality_verdict"]["pass"]:
            row["status"] = "quality_fail"
        row["selection_eligible"] = bool(
            row.get("accuracy_pass")
            and row.get("capability_pass")
            and row["quality_verdict"]
            and row["quality_verdict"]["pass"]
        )
        geometry_path = DOC / "head_cores_full36.json"
        if row["config_id"] == selected and geometry_path.exists():
            geometry = json.loads(geometry_path.read_text())
            row["geometry_control"] = {
                "artifact": str(geometry_path),
                "scope": "ABBA head working-layout control for this same precision policy",
                "pass": geometry.get("pass", False),
                "medians_tps": geometry.get("medians"),
            }
        for key in ["top1", "top5", "top100"]:
            row["prefill_" + key] = min((score[key] for score in row.get("prefill", [])), default=None)
        rows.append(row)
    rows.sort(key=lambda r: r.get("decode_tokens_per_second_per_user", -1), reverse=True)
    for index, row in enumerate(rows, 1):
        row["plot_id"] = index
    report = {
        "thresholds": {"top1": 0.90, "top5": 0.98, "top100": 1.0},
        "selected_config_id": selected,
        "selection_regime": "warmed traced teacher forcing,156 prompt/100 generated/B1",
        "results": rows,
    }
    (DOC / "sweep_results.json").write_text(json.dumps(report, indent=1) + "\n")
    fields = [
        "plot_id",
        "config_id",
        "precision_config",
        "dtype_policy",
        "compute_fidelity_policy",
        "top1",
        "top5",
        "top100",
        "ttft_ms",
        "decode_tokens_per_second_per_user",
        "trace_verified",
        "measurement_regime",
        "command",
        "hardware",
        "mesh",
        "status",
        "accuracy_pass",
        "prefill_top1",
        "prefill_top5",
        "prefill_top100",
        "capability_pass",
        "quality_verdict",
        "selection_eligible",
        "geometry_control",
        "commit",
        "reference",
        "token_count",
        "artifact",
    ]
    with (DOC / "sweep_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key)) if isinstance(row.get(key), (dict, list)) else row.get(key)
                    for key in fields
                }
            )
    drawable = [row for row in rows if row.get("trace_verified") and row.get("completed_at")]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    for metric, minimum in [("top1", 90), ("top5", 98)]:
        fig, ax = plt.subplots(figsize=(11.5, 7.2))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.87, bottom=0.14)
        fig.set_facecolor("#fbfcfe")
        ax.set_facecolor("#fbfcfe")
        frontier = [
            row
            for row in drawable
            if not any(
                other[metric] >= row[metric]
                and other["decode_tokens_per_second_per_user"] >= row["decode_tokens_per_second_per_user"]
                and (
                    other[metric] > row[metric]
                    or other["decode_tokens_per_second_per_user"] > row["decode_tokens_per_second_per_user"]
                )
                for other in drawable
            )
        ]
        frontier.sort(key=lambda row: row[metric])
        ax.plot(
            [r[metric] * 100 for r in frontier],
            [r["decode_tokens_per_second_per_user"] for r in frontier],
            color="#4b7897",
            linewidth=1.7,
            alpha=0.8,
            marker="D",
            markersize=9,
            markerfacecolor="none",
            label="Numerical Pareto frontier",
        )
        for pass_gate, color, label in [
            (False, "#aeb5c1", "Fails accuracy gate"),
            (True, "#167d8d", "Passes accuracy gate"),
        ]:
            group = [
                r
                for r in drawable
                if r["accuracy_pass"] == pass_gate
                and r["config_id"] != selected
                and r["status"] not in {"continuation_fail", "quality_fail"}
            ]
            if not group:
                continue
            ax.scatter(
                [r[metric] * 100 for r in group],
                [r["decode_tokens_per_second_per_user"] for r in group],
                s=62,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=label,
                zorder=3,
            )
        rejected = [r for r in drawable if r["status"] in {"continuation_fail", "quality_fail"}]
        if rejected:
            ax.scatter(
                [r[metric] * 100 for r in rejected],
                [r["decode_tokens_per_second_per_user"] for r in rejected],
                s=75,
                color="#bc814c",
                marker="X",
                label="Rejected: continuation / quality",
                zorder=3,
            )
        labeled = {r["config_id"] for r in frontier}
        labeled.add("baseline_mixed_hifi2_lofi")
        failures = [r for r in drawable if not r["accuracy_pass"]]
        if failures:
            labeled.add(max(failures, key=lambda r: r["decode_tokens_per_second_per_user"])["config_id"])
        for r in drawable:
            x, y = r[metric] * 100, r["decode_tokens_per_second_per_user"]
            if r["config_id"] in labeled and r["config_id"] != selected:
                ax.annotate(
                    str(r["plot_id"]), (x, y), xytext=(6, 8), textcoords="offset points", fontsize=9, color="#465365"
                )
            if r["config_id"] == selected:
                ax.scatter([x], [y], s=150, color="#d22f43", marker="*", label="Selected", zorder=5)
                ax.annotate(
                    r["config_id"],
                    (x, y),
                    xytext=(-15, -26),
                    textcoords="offset points",
                    ha="right",
                    fontsize=10,
                    color="#bd2035",
                    weight="bold",
                )
        ax.axvline(minimum, linestyle=":", color="#49566b", linewidth=1.8, label=f"Minimum {metric}: {minimum}%")
        ax.set_xlabel(f"Full-model {metric.replace('top','top-')} agreement (%)")
        ax.set_ylabel("Traced teacher-forcing decode (tokens/s/user)")
        ax.set_title(
            f"K2-Horizon-7B · {metric.replace('top','Top-')} precision frontier",
            loc="left",
            fontsize=18,
            pad=22,
            weight="bold",
        )
        ax.grid(axis="y", alpha=0.16)
        ax.legend(loc="best", frameon=False, fontsize=9)
        ax.margins(x=0.08, y=0.15)
        fig.text(
            0.10,
            0.025,
            "TP4 Blackhole · 36 layers · AIME24 chat · 156 + 100 tokens · labels map to sweep_results.csv",
            fontsize=9,
            color="#5e6c80",
        )
        fig.savefig(DOC / f"{metric}_perf_pareto.png", dpi=180)
        plt.close(fig)
    print("Recorded", len(rows), "full-model configs; selected artifact:", selected)


if __name__ == "__main__":
    main()
