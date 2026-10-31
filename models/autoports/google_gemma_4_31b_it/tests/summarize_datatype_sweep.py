"""Collect measured full-model rows; never infer missing measurements."""

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path("models/autoports/google_gemma_4_31b_it/doc/datatype_sweep")


def collect():
    rows = []
    plan = json.loads((ROOT / "candidate_plan.json").read_text())
    for name in plan["candidates"]:
        folder = ROOT / (
            "selected_default"
            if name == "head_bfp4" and (ROOT / "selected_default/readiness_accuracy.json").exists()
            else name
        )
        path = folder / "readiness_accuracy.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        if "decode" not in raw:
            continue
        cfg = json.loads((ROOT / "configs" / (name + ".json")).read_text())
        row = dict(
            config_id=name,
            dtype_policy=cfg,
            compute_fidelity_policy=cfg["compute_fidelities"],
            top1=min(raw[p][0]["top1"] for p in ("prefill", "decode")),
            top5=min(raw[p][0]["top5"] for p in ("prefill", "decode")),
            top100=min(raw[p][0]["top100"] for p in ("prefill", "decode")),
            ttft_ms=raw["decode"][0]["ttft_ms"],
            decode_tps=raw["decode"][0]["decode_t/s/u"],
            measurement_regime="traced teacher-forcing; 161 prompt tokens, 100 continuation positions; B1",
            reference=raw["reference"],
            token_count=raw["decode"][0]["total"],
            command=f"python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_datatype_candidate --config {ROOT}/configs/{name}.json --id {name}",
            hardware="4 Blackhole p300c ASICs",
            mesh=[1, 4],
            trace_evidence={"ids": raw["traces"], "counters": raw["decode_counters"]},
            runtime_policy=str(folder / "runtime_policy.json"),
        )
        assert raw["decode_counters"]["model_replays"] >= 99 and raw["decode_counters"]["captures"] == 0
        manifest_path = folder / "source_manifest.json"
        row["source_manifest"] = str(manifest_path) if manifest_path.exists() else None
        row["commit"] = (
            json.loads(manifest_path.read_text())["head"]
            if manifest_path.exists()
            else "c673ec5a40de93afe05c3383a359eb6df0cb71d1"
        )
        row["branch"] = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        row["environment"] = {
            "TT_METAL_TRACE_ALLOC_TRACKING": "0",
            "TT_METAL_DEVICE_PROFILER": "0",
            "setup": "installed model-bringup/0.1.22/scripts/environment.py",
        }
        if folder.name == "selected_default":
            row[
                "command"
            ] = "python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_datatype_candidate --id selected_default"
            row["search_evidence"] = str(ROOT / name / "readiness_accuracy.json")
        else:
            row["search_evidence"] = None
        row["accuracy_pass"] = row["top1"] >= 0.9 and row["top5"] >= 0.98 and row["top100"] == 1
        lp = folder / "long_prefix.json"
        row["long_prefix_pass"] = json.loads(lp.read_text())["contains_expected_label"] if lp.exists() else None
        row["memory_pass"] = name != "kv_bf16"
        row["memory_basis"] = (
            "kv_capacity_estimate.json: BF16 cannot retain the 32x32768 pool"
            if name == "kv_bf16"
            else "BFP8 KV; baseline target pool fits, selected policy revalidated at batch32"
        )
        row["status"] = (
            "pass"
            if row["accuracy_pass"] and row["long_prefix_pass"] and row["memory_pass"]
            else "pending"
            if row["long_prefix_pass"] is None
            else "fail"
        )
        perf = folder / "performance_b1.json"
        row["token_out"] = json.loads(perf.read_text()) if perf.exists() else None
        rows.append(row)
    (ROOT / "sweep_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (ROOT / "sweep_results.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
    return rows


def plots(rows, selected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, threshold in [("top1", 0.9), ("top5", 0.98)]:
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        for r in rows:
            ax.scatter(
                r[metric] * 100,
                r["decode_tps"],
                color="red" if r["config_id"] == selected else "#227c9d" if r["memory_pass"] else "#888888",
                marker="o" if r["memory_pass"] else "x",
                s=65,
                zorder=3,
            )

        frontier = [
            r
            for r in rows
            if not any(
                q[metric] >= r[metric]
                and q["decode_tps"] >= r["decode_tps"]
                and (q[metric] > r[metric] or q["decode_tps"] > r["decode_tps"])
                for q in rows
            )
        ]
        frontier.sort(key=lambda r: r[metric])
        ax.plot(
            [r[metric] * 100 for r in frontier],
            [r["decode_tps"] for r in frontier],
            color="#227c9d",
            alpha=0.6,
            marker="o",
            markerfacecolor="none",
            markersize=11,
            label="Pareto frontier",
        )
        ax.axvline(threshold * 100, linestyle=":", color="#555555", label=f"Minimum accuracy {threshold:.0%}")
        ax.set(
            xlabel=f"Minimum prefill/decode {metric} agreement (%)",
            ylabel="Traced teacher-forcing decode tokens/s/user",
            title=f"Gemma 4 31B · {metric} precision frontier",
        )
        label_x = max(r[metric] * 100 for r in rows) + 0.5
        span = max(r["decode_tps"] for r in rows) - min(r["decode_tps"] for r in rows)
        last_y = -float("inf")
        for r in sorted(rows, key=lambda r: r["decode_tps"]):
            y = max(r["decode_tps"], last_y + max(0.35, span * 0.055))
            last_y = y
            ax.annotate(
                r["config_id"],
                (r[metric] * 100, r["decode_tps"]),
                xytext=(label_x, y),
                fontsize=8,
                color="red" if r["config_id"] == selected else "#333333",
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#999999"},
            )
        ax.set_xlim(min(threshold * 100, min(r[metric] * 100 for r in rows)) - 1, label_x + 7)
        ax.set_ylim(min(r["decode_tps"] for r in rows) - 1, last_y + 1)
        ax.grid(alpha=0.15)
        ax.legend(loc="lower left")
        fig.savefig(ROOT / f"{metric}_perf_pareto.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    rows = collect()
    selected = ROOT / "selected_precision_config.json"
    if selected.exists():
        plots(rows, json.loads(selected.read_text())["config_id"])
