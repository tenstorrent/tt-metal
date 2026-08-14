"""Build sweep tables and Pareto plots from immutable candidate logs."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REFERENCE = "models/autoports/qwen_qwen3_6_27b/doc/full_model/readiness_aime24_chat.refpt"
COMMAND = (
    "QWEN36_PRECISION_CONFIG={config} python -m models.common.readiness_check.run_teacher_forcing "
    "--model-dir models/autoports/qwen_qwen3_6_27b --reference "
    + REFERENCE
    + " --mesh-device P300X2 --fabric-config FABRIC_1D_RING"
)
AGGREGATE = re.compile(
    r"AGGREGATE\s+top1=(?P<top1>[0-9.]+).*top5=(?P<top5>[0-9.]+).*top100=(?P<top100>[0-9.]+).*"
    r"TTFT=(?P<ttft>[0-9.]+)ms\s+decode=(?P<decode>[0-9.]+) t/s/u\s+e2e=(?P<e2e>[0-9.]+) t/s/u"
)
ORDER = [
    "baseline_optimized_default",
    "baseline_bf16_kv",
    "full_attention_bfp8_hifi2",
    "full_attention_bfp8_lofi",
    "full_attention_bfp4_hifi2",
    "full_attention_bfp4_lofi",
    "all_projection_bfp8_hifi2",
    "all_projection_bfp8_lofi",
    "baseline_bfp8_ccl",
    "selected_bfp4_mlp_hifi2",
    "selected_bfp4_linear_hifi2",
    "selected_bfp8_activation_ccl",
]
SELECTED = "full_attention_bfp4_lofi"


def build_row(config_id):
    config_path = ROOT / "candidates" / f"{config_id}.json"
    policy = json.loads(config_path.read_text())
    log_path = ROOT / "logs" / f"{config_id}_teacher_forcing.log"
    text = log_path.read_text(errors="replace")
    matches = list(AGGREGATE.finditer(text))
    metrics = matches[-1].groupdict() if matches else None
    row = {
        "config_id": config_id,
        "precision_config": str(config_path.relative_to(Path.cwd())),
        "dtype_policy": {
            "weight_groups": policy["weight_groups"],
            "activation_residual_dtype": policy["activation_residual_dtype"],
            "ccl_dtype": policy["ccl_dtype"],
            "kv_cache_dtype": policy["kv_cache_dtype"],
            "linear_recurrent_state_dtype": policy["linear_recurrent_state_dtype"],
            "logits_sampling": policy["logits_sampling"],
            "layer_exceptions": policy["layer_exceptions"],
        },
        "compute_fidelity_policy": policy["compute_fidelities"],
        "top1": float(metrics["top1"]) if metrics else None,
        "top5": float(metrics["top5"]) if metrics else None,
        "top100": float(metrics["top100"]) if metrics else None,
        "token_count": 100,
        "ttft_ms": float(metrics["ttft"]) if metrics else None,
        "teacher_forcing_decode_t_s_u": float(metrics["decode"]) if metrics else None,
        "e2e_t_s_u": float(metrics["e2e"]) if metrics else None,
        "trace_verified": bool(metrics),
        "trace_evidence": "generator.generate(enable_trace=True); Qwen36Generator rejects enable_trace=False",
        "measurement_regime": "full-model AIME24 chat-template, S161 + 100 teacher-forced tokens, batch 1, traced decode",
        "reference": REFERENCE,
        "command": COMMAND.format(config=str(config_path.relative_to(Path.cwd()))),
        "log": str(log_path.relative_to(Path.cwd())),
        "hardware": "4x Blackhole p300c, firmware 19.8.0",
        "mesh": [1, 4],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "pass": bool(
            metrics
            and float(metrics["top1"]) >= 0.90
            and float(metrics["top5"]) >= 0.98
            and float(metrics["top100"]) == 1.0
        ),
        "status": "pass"
        if metrics
        and float(metrics["top1"]) >= 0.90
        and float(metrics["top5"]) >= 0.98
        and float(metrics["top100"]) == 1.0
        else "runtime-fail"
        if not metrics
        else "accuracy-fail",
        "failure": None
        if metrics
        else next(
            (line.strip() for line in text.splitlines() if "Statically allocated circular buffers" in line), "see log"
        ),
    }
    return row


def plot(rows, metric, threshold, output):
    valid = [row for row in rows if row[metric] is not None and row["teacher_forcing_decode_t_s_u"] is not None]
    frontier = [
        row
        for row in valid
        if not any(
            other is not row
            and other[metric] >= row[metric]
            and other["teacher_forcing_decode_t_s_u"] >= row["teacher_forcing_decode_t_s_u"]
            and (
                other[metric] > row[metric]
                or other["teacher_forcing_decode_t_s_u"] > row["teacher_forcing_decode_t_s_u"]
            )
            for other in valid
        )
    ]
    frontier.sort(key=lambda item: item[metric])
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for row in valid:
        selected = row["config_id"] == SELECTED
        ax.scatter(
            row[metric] * 100,
            row["teacher_forcing_decode_t_s_u"],
            s=130 if selected else 70,
            color="red" if selected else "#2878B5",
            zorder=3,
        )
        ax.annotate(
            row["config_id"],
            (row[metric] * 100, row["teacher_forcing_decode_t_s_u"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.plot(
        [x[metric] * 100 for x in frontier],
        [x["teacher_forcing_decode_t_s_u"] for x in frontier],
        color="#F28E2B",
        linewidth=2.2,
        marker="o",
        label="Pareto frontier",
    )
    ax.axvline(threshold * 100, color="#555", linestyle=":", linewidth=2, label=f"minimum {threshold:.0%}")
    ax.set(
        xlabel=f"{metric.replace('top', 'Top-')} accuracy (%)",
        ylabel="Traced teacher-forcing decode (tokens/s/user)",
        title=f"Qwen3.6-27B {metric.replace('top', 'Top-')} Accuracy / Decode Pareto",
    )
    ax.grid(alpha=0.22)
    ax.legend()
    fig.savefig(ROOT / output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    rows = [build_row(config_id) for config_id in ORDER]
    (ROOT / "sweep_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    flat = []
    for row in rows:
        flat.append(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    with (ROOT / "sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plot(rows, "top1", 0.90, "top1_perf_pareto.png")
    plot(rows, "top5", 0.98, "top5_perf_pareto.png")
    print(f"wrote {len(rows)} rows; selected={SELECTED}")
