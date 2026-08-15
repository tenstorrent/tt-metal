#!/usr/bin/env python3
"""Build deterministic datatype-sweep tables and Pareto plots from retained logs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
MODEL = ROOT.parents[1]
COMMAND = (
    "GEMMA4_PRECISION_CONFIG={config} TTNN_CONFIG_OVERRIDES='{{\"throw_exception_on_fallback\":true}}' "
    "python -m models.common.readiness_check.run_teacher_forcing --model-dir "
    "models/autoports/google_gemma_4_26b_a4b_it --reference "
    "models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/readiness_aime24_chat.refpt "
    "--mesh-device P300X2 --fabric-config FABRIC_1D_RING"
)
RUNS = [
    ("baseline_bfp8_hifi2_bf16_cache_ccl", "baseline.json", "baseline/policy_path_teacher_forcing.log"),
    ("kv_bfp8", "kv_bfp8.json", "kv_bfp8/teacher_forcing.log"),
    ("ccl_bfp8", "ccl_bfp8.json", "ccl_bfp8/teacher_forcing.log"),
    ("attention_bfp8_lofi", "attention_bfp8_lofi.json", "attention_bfp8_lofi/teacher_forcing.log"),
    ("expert_bfp4_lofi", "expert_bfp4_lofi.json", "expert_bfp4_lofi/teacher_forcing.log"),
    ("expert_bfp4_hifi2", "expert_bfp4_hifi2.json", "expert_bfp4_hifi2/teacher_forcing.log"),
    ("dense_down_bfp4_lofi", "dense_down_bfp4_lofi.json", "dense_down_bfp4_lofi/teacher_forcing.log"),
    ("activation_residual_bfp8", "activation_bfp8.json", "activation_bfp8/teacher_forcing.log"),
    ("decode_dense_gate_up_bfp4_hifi2", "dense_bfp4_hifi2.json", "dense_bfp4_hifi2/teacher_forcing.log"),
]
PATTERN = re.compile(
    r"AGGREGATE\s+top1=(?P<top1>[\d.]+).*top5=(?P<top5>[\d.]+).*top100=(?P<top100>[\d.]+).*"
    r"TTFT=(?P<ttft>[\d.]+)ms\s+decode=(?P<decode>[\d.]+)"
)


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
        )
    return result


def policy(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "extends" in data:
        base = policy(path.parent / data["extends"])
        data = merge(base, data.get("overrides", {}))
    return data


rows = []
for config_id, config_name, log_name in RUNS:
    config_path = ROOT / "configs" / config_name
    log_path = ROOT / "artifacts" / log_name
    match = PATTERN.search(log_path.read_text(errors="replace"))
    if match is None:
        raise RuntimeError(f"missing aggregate row: {log_path}")
    metrics = {key: float(value) for key, value in match.groupdict().items()}
    row = {
        "config_id": config_id,
        "precision_config": str(config_path.relative_to(MODEL)),
        "dtype_policy": policy(config_path),
        "compute_fidelity_policy": policy(config_path)["compute_fidelities"],
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "top100": metrics["top100"],
        "token_count": 100,
        "ttft_ms": metrics["ttft"],
        "teacher_forcing_decode_t_s_u": metrics["decode"],
        "trace_verified": True,
        "measurement_regime": "AIME24 chat-template, 100-token shifted-left teacher forcing, traced decode, batch 1, warmed per-run program cache state as logged",
        "command": COMMAND.format(config=config_path),
        "hardware": "4x P300C Blackhole",
        "mesh": [1, 4],
        "branch": "mvasiljevic/fmf/google-gemma-4-26b-a4b-it",
        "measured_base_commit": "bcb21b8d026464190a3ffceae72e420e2e026c56",
        "source_state": "live datatype-sweep worktree applied over measured_base_commit; exact stage diff is captured by the post-review checkpoint",
        "environment_notes": "Python 3.12.13; repo-built TTNN; firmware 19.8.0; KMD 2.8.0; TTNN fallback exceptions enabled; FABRIC_1D_RING; serialized device runs; tensor/program caches may be warm and TTFT is not ranked",
        "reference": "doc/full_model/readiness_aime24_chat.refpt",
        "pass": metrics["top1"] >= 0.90 and metrics["top5"] >= 0.98,
        "status": "pass" if metrics["top1"] >= 0.90 and metrics["top5"] >= 0.98 else "fail_accuracy",
        "evidence_log": str(log_path.relative_to(MODEL)),
    }
    rows.append(row)

(ROOT / "sweep_results.json").write_text(json.dumps(rows, indent=2) + "\n")
fields = [
    "config_id",
    "dtype_policy",
    "compute_fidelity_policy",
    "top1",
    "top5",
    "top100",
    "token_count",
    "ttft_ms",
    "teacher_forcing_decode_t_s_u",
    "trace_verified",
    "measurement_regime",
    "command",
    "hardware",
    "mesh",
    "branch",
    "measured_base_commit",
    "source_state",
    "environment_notes",
    "pass",
    "status",
    "evidence_log",
]
with (ROOT / "sweep_results.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {field: json.dumps(row[field]) if isinstance(row[field], (dict, list)) else row[field] for field in fields}
        )

selected = rows[0]
for metric, threshold, filename, label in (
    ("top1", 0.90, "top1_perf_pareto.png", "Top-1 accuracy"),
    ("top5", 0.98, "top5_perf_pareto.png", "Top-5 accuracy"),
):
    ordered = sorted(rows, key=lambda row: row[metric], reverse=True)
    frontier = []
    best_perf = -1.0
    for row in ordered:
        if row["teacher_forcing_decode_t_s_u"] > best_perf:
            frontier.append(row)
            best_perf = row["teacher_forcing_decode_t_s_u"]
    frontier.sort(key=lambda row: row[metric])
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.scatter(
        [row[metric] for row in rows], [row["teacher_forcing_decode_t_s_u"] for row in rows], s=65, color="#2878b5"
    )
    ax.plot(
        [row[metric] for row in frontier],
        [row["teacher_forcing_decode_t_s_u"] for row in frontier],
        color="#54a24b",
        linewidth=2.5,
        label="Pareto frontier",
    )
    ax.scatter(
        [selected[metric]],
        [selected["teacher_forcing_decode_t_s_u"]],
        s=150,
        color="red",
        edgecolor="black",
        zorder=5,
        label="Selected",
    )
    ax.axvline(threshold, linestyle=":", color="#555555", linewidth=2, label=f"Minimum {threshold:.0%}")
    for row in rows:
        ax.annotate(
            row["config_id"],
            (row[metric], row["teacher_forcing_decode_t_s_u"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel(label)
    ax.set_ylabel("Traced teacher-forcing decode (tokens/s/user)")
    ax.set_title(f"Gemma-4 26B A4B datatype sweep: {label} / performance")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.savefig(ROOT / filename, dpi=180)
    plt.close(fig)
