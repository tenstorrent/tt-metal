# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize measured full-model rows; never rank smoke or eager evidence."""

import argparse
import csv
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    rows = []
    rejected = json.loads((DOC / "capacity_rejections.json").read_text())
    capacity_ids = {r["config_id"] for r in rejected}
    for p in sorted((DOC / "runs").glob("*.json")):
        d = json.loads(p.read_text())
        capacity_failure = d["status"] == "runtime_fail" and p.stem in capacity_ids
        if d["status"] not in ("pass", "accuracy_fail") and not capacity_failure:
            continue
        assert d["layers"] == 48 and d["context"] == 524288
        assert capacity_failure or d["trace_verified"]
        policy = d["policy"]
        rows.append(
            dict(
                config_id=policy["config_id"],
                precision_config_path=d["config_path"],
                dtype_policy={
                    k: v
                    for k, v in policy.items()
                    if k
                    not in ("config_id", "compute_fidelities", "sdpa_fidelities", "lm_head_fidelity", "norm_fidelity")
                },
                compute_fidelity_policy={
                    k: policy[k] for k in ("compute_fidelities", "sdpa_fidelities", "lm_head_fidelity", "norm_fidelity")
                },
                top1=d.get("top1"),
                top5=d.get("top5"),
                top100=d.get("top100"),
                tokens=d.get("total", 0),
                prefill=d.get("prefill", []),
                ttft_ms=d.get("ttft_ms"),
                decode_tps=d.get("decode_t/s/u"),
                measurement_regime=(
                    "Full-model preparation rejected: physical DRAM exhausted; no timing or accuracy"
                    if capacity_failure
                    else d["measurement_regime"]
                ),
                trace_verified=d.get("trace_verified", False),
                command=d["command"],
                hardware=d["hardware"],
                mesh=d["mesh"],
                commit=d["commit"],
                branch=d.get("branch", "main"),
                environment={
                    "TT_METAL_TRACE_ALLOC_TRACKING": d["tracking"],
                    "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0",
                    "OMP_NUM_THREADS": "8",
                    "MKL_NUM_THREADS": "8",
                },
                status="capacity_fail" if capacity_failure else d["status"],
                artifact=str(p.relative_to(DOC)),
                runtime_evidence=str(p.relative_to(DOC)) + "#runtime",
                reference=d["reference"],
                reference_sha256=d["reference_sha256"],
                repeats=[x["rows"][0]["decode_t/s/u"] for x in d.get("teacher_forcing", [])],
            )
        )
    (DOC / "sweep_results.json").write_text(
        json.dumps(
            dict(
                thresholds=dict(top1=0.90, top5=0.98, top100=1.0), full_model_results=rows, capacity_rejections=rejected
            ),
            indent=2,
        )
        + "\n"
    )
    if not rows:
        return
    with (DOC / "sweep_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
    for row in sorted(rows, key=lambda x: -(x["decode_tps"] or -1)):
        print(row["config_id"], row["top1"], row["top5"], row["decode_tps"], row["status"])
    if not args.select:
        return
    selected = max((r for r in rows if r["status"] == "pass"), key=lambda r: r["decode_tps"])
    # Use the exact policy stored in the measured artifact, not a subsequently edited config file.
    policy = json.loads((DOC / selected["artifact"]).read_text())["policy"]
    (DOC / "selected_precision_config.json").write_text(json.dumps(policy, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    # Capacity failures have no measured accuracy/throughput and cannot be plotted or ranked.
    rows = [r for r in rows if r["trace_verified"] and r["decode_tps"] is not None]
    for metric, threshold in [("top1", 90), ("top5", 98)]:
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        for r in rows:
            chosen = r["config_id"] == selected["config_id"]
            color = "#dc2626" if chosen else ("#167d9a" if r["status"] == "pass" else "#929aaa")
            ax.scatter(
                100 * r[metric],
                r["decode_tps"],
                s=135 if chosen else 55,
                c=color,
                marker="*" if chosen else "o",
                zorder=4,
            )
        frontier = [
            r
            for r in rows
            if not any(
                o[metric] >= r[metric]
                and o["decode_tps"] >= r["decode_tps"]
                and (o[metric] > r[metric] or o["decode_tps"] > r["decode_tps"])
                for o in rows
            )
        ]
        frontier.sort(key=lambda r: r[metric])
        ax.plot(
            [100 * r[metric] for r in frontier],
            [r["decode_tps"] for r in frontier],
            color="#167d9a",
            lw=1.8,
            marker="o",
            markersize=11,
            markerfacecolor="none",
            label="Non-dominated frontier",
        )
        ax.scatter([], [], marker="*", s=135, c="#dc2626", label="Selected fastest passing policy")
        ax.scatter([], [], s=55, c="#167d9a", label="Passed full-model gates")
        if any(r["status"] != "pass" for r in rows):
            ax.scatter([], [], s=55, c="#929aaa", label="Failed full-model gates")
        ax.axvline(threshold, color="#555", linestyle=":", label=f"Minimum {metric}: {threshold}%")
        labels = {r["config_id"] for r in frontier} | {selected["config_id"], "baseline_bfp4_lofi"}
        for i, r in enumerate(rows):
            if r["config_id"] in labels:
                ax.annotate(
                    r["config_id"],
                    (100 * r[metric], r["decode_tps"]),
                    xytext=(
                        (-8, -24)
                        if r["config_id"] == "baseline_bfp4_lofi"
                        else ((8, 12) if r["config_id"] == selected["config_id"] and metric == "top1" else (-8, 12))
                    ),
                    ha="left" if r["config_id"] == selected["config_id"] and metric == "top1" else "right",
                    textcoords="offset points",
                    fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.6},
                )
        ax.set(
            xlabel=f'{metric.replace("top","Top-")} accuracy (%)',
            ylabel="Traced teacher-forcing decode (tokens/s/user)",
            title="K2 Horizon MoVA · full 48 layers · TP4 Blackhole\nAIME24 chat · 156 input / 100 output · 524288 cache capacity",
        )
        ax.margins(x=0.08, y=0.13)
        ax.grid(alpha=0.18)
        ax.legend(loc="lower left")
        fig.savefig(DOC / f"{metric}_perf_pareto.png", dpi=180)
        plt.close(fig)
    print("SELECTED", selected["config_id"])


if __name__ == "__main__":
    main()
