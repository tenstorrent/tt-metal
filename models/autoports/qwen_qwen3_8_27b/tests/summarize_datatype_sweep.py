# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build machine-readable full-model results and measured Pareto plots."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def main():
    rows = []
    commands = (DOC / "commands.log").read_text().splitlines()
    for path in sorted(DOC.glob("*.json")):
        data = json.loads(path.read_text())
        if path.stem == "selected_confirmation":
            continue
        if not isinstance(data, dict) or not data.get("warmed_teacher_forcing_perf"):
            continue
        perf = data["warmed_teacher_forcing_perf"]
        if not isinstance(perf, list) or "runtime" not in data:
            continue
        exit_path = path.with_suffix(".exit_status")
        if not exit_path.exists() or exit_path.read_text().strip() != "0":
            continue
        policy = data["runtime"]["policy"]
        if len(data["runtime"]["layers"]) != 64:
            continue
        measured = perf[-2:]
        for p in measured:
            c = p["steady_state_counters"]
            assert p["teacher_forcing"] and not p["host_sampling"]
            assert c["model_replays"] == c["sampling_replays"] == 99
            assert not c.get("trace_captures", 0)
        accuracy = data["teacher_forcing_runs"][-1]["accuracy"][0]
        row = dict(
            config_id=policy["config_id"],
            precision_config_path=str((DOC / "configs" / (policy["config_id"] + ".json")).relative_to(ROOT)),
            dtype_policy={k: v for k, v in policy.items() if k != "compute_fidelities"},
            compute_fidelity_policy=policy["compute_fidelities"],
            fp32_dest_acc_en=policy["fp32_dest_acc_en"],
            top1=accuracy["top1"],
            top5=accuracy["top5"],
            top100=accuracy["top100"],
            prefill_accuracy=data["prefill"],
            token_count=accuracy["total"],
            ttft_ms=1000 * statistics.median(p["ttft_s"] for p in measured),
            traced_teacher_forcing_decode_t_s_u=statistics.median(p["tokens_per_second"] for p in measured),
            traced_teacher_forcing_repeats_t_s_u=[p["tokens_per_second"] for p in measured],
            measurement_regime=data["measurement_regime"],
            workload=data["workload"],
            command=next(c for c in reversed(commands) if " " + path.stem + " " in c),
            commit=path.with_suffix(".commit").read_text().strip(),
            hardware=data["hardware"],
            mesh=data["mesh"],
            status=data["status"],
            trace_verified=True,
            reference=data["reference"],
            runtime_propagation_evidence=str(path.relative_to(ROOT)) + ":runtime",
            artifact=str(path.relative_to(ROOT)),
        )
        rows.append(row)
    assert rows, "No completed full-model measurements"
    # Confirmation is reported separately; rank the fixed candidate experiments.
    latest = {}
    for row in rows:
        latest[row["config_id"]] = row
    candidates = list(latest.values())
    passing = [r for r in candidates if r["status"] == "pass"]
    winner = max(passing, key=lambda r: r["traced_teacher_forcing_decode_t_s_u"]) if passing else None
    selected_path = DOC / "selected_precision_config.json"
    selected = json.loads(selected_path.read_text())["config_id"] if selected_path.exists() else None
    for row in rows:
        row["decision"] = "selected" if row["config_id"] == selected else "rejected"
        row["decision_reason"] = (
            "Fastest evaluated passing traced teacher-forcing candidate"
            if row["decision"] == "selected"
            else "Slower passing full-model candidate"
            if row["status"] == "pass"
            else "Accuracy gate failed"
        )
    report = dict(
        thresholds=dict(top1=0.90, top5=0.98, top100=1.0),
        ranking_metric="traced_teacher_forcing_decode_t_s_u",
        provisional_fastest=winner["config_id"] if winner else None,
        results=rows,
    )
    (DOC / "sweep_results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (DOC / "sweep_results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
    with (DOC / "matmul_group_results.csv").open("w") as f:
        fields = [
            "group",
            "config_id",
            "weight_dtype",
            "compute_fidelity",
            "fp32_dest_acc_en",
            "layer_exceptions",
            "top1",
            "top5",
            "top100",
            "traced_teacher_forcing_decode_t_s_u",
            "status",
            "decision",
            "decision_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for role, dtype in row["dtype_policy"]["weight_groups"].items():
                writer.writerow(
                    dict(
                        group=role,
                        config_id=row["config_id"],
                        weight_dtype=dtype,
                        compute_fidelity=row["compute_fidelity_policy"][role],
                        fp32_dest_acc_en=row["fp32_dest_acc_en"],
                        layer_exceptions=json.dumps(row["dtype_policy"]["layer_exceptions"]),
                        **{
                            k: row[k]
                            for k in (
                                "top1",
                                "top5",
                                "top100",
                                "traced_teacher_forcing_decode_t_s_u",
                                "status",
                                "decision",
                                "decision_reason",
                            )
                        },
                    )
                )
    selected_path = DOC / "selected_precision_config.json"
    selected = json.loads(selected_path.read_text())["config_id"] if selected_path.exists() else None
    for metric, threshold in (("top1", 0.90), ("top5", 0.98)):
        fig, (ax, key) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.5, 1.1]}, constrained_layout=True
        )
        key.axis("off")
        key.set_title("Evaluated full-model policies", loc="left", fontsize=12)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        for index, r in enumerate(candidates):
            x, y = r[metric] * 100, r["traced_teacher_forcing_decode_t_s_u"]
            color = "#dc2626" if r["config_id"] == selected else ("#2563eb" if r["status"] == "pass" else "#94a3b8")
            ax.scatter(x, y, s=110 if r["config_id"] == selected else 55, color=color, zorder=5)
            ax.annotate(
                str(index + 1),
                (x, y),
                xytext=(8, (index % 3 - 1) * 13 + 6),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
            key.text(
                0,
                0.97 - index * min(0.052, 0.9 / max(len(candidates), 1)),
                f"{index + 1:02d}  {r['config_id']}\n       {r[metric]:.0%}  ·  {y:.3f} t/s/u  ·  {r['status']}",
                transform=key.transAxes,
                fontsize=8,
                va="top",
                color=color,
            )
        frontier = [
            r
            for r in candidates
            if not any(
                q[metric] >= r[metric]
                and q["traced_teacher_forcing_decode_t_s_u"] >= r["traced_teacher_forcing_decode_t_s_u"]
                and (
                    q[metric] > r[metric]
                    or q["traced_teacher_forcing_decode_t_s_u"] > r["traced_teacher_forcing_decode_t_s_u"]
                )
                for q in candidates
            )
        ]
        frontier.sort(key=lambda r: r[metric])
        ax.plot(
            [r[metric] * 100 for r in frontier],
            [r["traced_teacher_forcing_decode_t_s_u"] for r in frontier],
            color="#0f766e",
            linewidth=1.8,
            marker="o",
            markersize=12,
            markerfacecolor="none",
            label="Measured Pareto frontier",
        )
        ax.axvline(threshold * 100, color="#64748b", linestyle=":", label=f"Minimum {metric}: {threshold:.0%}")
        ax.set(
            xlabel=f"Full-model {metric} accuracy (%)",
            ylabel="Traced teacher-forcing decode (tokens/s/user)",
            title=f"Qwen3.8-27B · {metric} / throughput",
        )
        fig.suptitle("TP4 Blackhole · batch 1 · AIME24 chat S203 / G100 · selected policy in red", fontsize=12)
        ax.grid(alpha=0.16)
        ax.margins(x=0.18, y=0.2)
        ax.legend(loc="best", frameon=False)
        fig.savefig(DOC / f"{metric}_perf_pareto.png", dpi=180)
        plt.close(fig)
    print(
        json.dumps(
            {
                r["config_id"]: [r["status"], r["top1"], r["top5"], r["traced_teacher_forcing_decode_t_s_u"]]
                for r in rows
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
