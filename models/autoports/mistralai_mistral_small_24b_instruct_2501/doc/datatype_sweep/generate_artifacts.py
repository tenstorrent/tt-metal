#!/usr/bin/env python3
"""Generate normalized datatype-sweep result tables and Pareto charts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parents[1]
REFERENCE = MODEL_DIR / "doc/full_model/artifacts/aime24_chat_100.refpt"
COMMAND = (
    "MISTRAL_SMALL_24B_PRECISION_CONFIG={config_path} HF_HUB_OFFLINE=1 TT_LOGGER_LEVEL=info "
    "python -m models.common.readiness_check.run_teacher_forcing "
    f"--model-dir {MODEL_DIR.relative_to(Path.cwd())} --reference {REFERENCE.relative_to(Path.cwd())} "
    "--mesh-device P300_QUAD --fabric-config FABRIC_1D --trace-region-size 200000000"
)
THRESHOLDS = {"top1": 0.90, "top5": 0.98, "top100": 1.0}
HARDWARE = "4x Blackhole p300c, firmware 19.9.0"
MESH = "1x4 TP4, FABRIC_1D"
REGIME = "full-model AIME24 chat-template, 100 forced tokens, split traced decode; rank by internal traced interval"
MEASUREMENT_BASE_COMMIT = "d182c2fe795610b7622205b4a06e98457b2a6e93"
MEASUREMENT_WORKTREE_STATE = "base commit plus live datatype-policy plumbing and candidate JSON in this stage"


RAW = [
    {
        "config_id": "baseline_bfp4_lofi_bfp8kv_bfp8ccl",
        "file": "baseline_bfp4_lofi_bfp8kv_bfp8ccl.json",
        "top1": 0.97,
        "top5": 1.0,
        "top100": 1.0,
        "ttft_ms": 229.367625,
        "traced_decode_t_s_u": 52.829349,
        "samples": [52.745610, 52.913089],
        "log": ["logs/baseline_policy_teacher_forcing.log", "logs/baseline_policy_teacher_forcing_repeat.log"],
        "capability_pass": True,
        "decision": "rejected: slower than BF16 decode CCL in repeated teacher-forcing and matched token-out",
    },
    {
        "config_id": "bfp4_hifi2_bfp8kv_bfp8ccl",
        "file": "bfp4_hifi2_bfp8kv_bfp8ccl.json",
        "top1": 0.97,
        "top5": 1.0,
        "top100": 1.0,
        "ttft_ms": 1469.744461,
        "traced_decode_t_s_u": 36.609832,
        "samples": [36.609832],
        "log": ["logs/bfp4_hifi2_teacher_forcing.log"],
        "capability_pass": True,
        "decision": "rejected: fidelity-only HiFi2 is much slower than LoFi",
    },
    {
        "config_id": "bfp8_lofi_bfp8kv_bfp8ccl",
        "file": "bfp8_lofi_bfp8kv_bfp8ccl.json",
        "top1": 0.98,
        "top5": 1.0,
        "top100": 1.0,
        "ttft_ms": 1910.366622,
        "traced_decode_t_s_u": 38.339460,
        "samples": [38.339460],
        "log": ["logs/bfp8_lofi_teacher_forcing.log"],
        "capability_pass": True,
        "decision": "rejected: adapted legal block-8 BFP8 path is slower than BFP4 LoFi",
    },
    {
        "config_id": "bfp8_hifi2_bfp8kv_bfp8ccl",
        "file": "bfp8_hifi2_bfp8kv_bfp8ccl.json",
        "top1": 0.99,
        "top5": 1.0,
        "top100": 1.0,
        "ttft_ms": 1476.891191,
        "traced_decode_t_s_u": 35.950609,
        "samples": [35.950609],
        "log": ["logs/bfp8_hifi2_teacher_forcing.log"],
        "capability_pass": True,
        "decision": "rejected: slower than BFP8 LoFi and BFP4 LoFi",
    },
    {
        "config_id": "bfp4_lofi_bfp8kv_bf16ccl",
        "file": "bfp4_lofi_bfp8kv_bf16ccl.json",
        "top1": 0.97,
        "top5": 1.0,
        "top100": 1.0,
        "ttft_ms": 226.993467,
        "traced_decode_t_s_u": 54.212248,
        "samples": [54.155957, 54.268538],
        "log": ["logs/bf16_ccl_teacher_forcing.log", "logs/bf16_ccl_teacher_forcing_repeat.log"],
        "capability_pass": True,
        "decision": "selected: fastest repeated passing trace-verified teacher-forcing policy",
    },
    {
        "config_id": "bfp4_lofi_bfp8act_bfp8kv_bfp8ccl",
        "file": "bfp4_lofi_bfp8act_bfp8kv_bfp8ccl.json",
        "top1": 0.98,
        "top5": 1.0,
        "top100": 1.0,
        "ttft_ms": 220.932642,
        "traced_decode_t_s_u": 51.600778,
        "samples": [51.600778],
        "log": ["logs/bfp8_activations_teacher_forcing.log"],
        "capability_pass": True,
        "decision": "rejected: reduced matmul inputs are slower",
    },
    {
        "config_id": "bfp4_lofi_bf16kv_bfp8ccl",
        "file": "bfp4_lofi_bf16kv_bfp8ccl.json",
        "top1": 0.97,
        "top5": 1.0,
        "top100": 1.0,
        "ttft_ms": 627.998432,
        "traced_decode_t_s_u": 53.326028,
        "samples": [53.326028],
        "log": ["logs/bf16_kv_teacher_forcing.log"],
        "capability_pass": False,
        "decision": "rejected: 18,304-token physical ceiling cannot preserve advertised 32,768 context",
    },
]


def result_rows() -> list[dict]:
    rows = []
    for raw in RAW:
        config_path = ROOT / "candidates" / raw["file"]
        policy = json.loads(config_path.read_text())
        accuracy_pass = raw["top1"] >= THRESHOLDS["top1"] and raw["top5"] >= THRESHOLDS["top5"]
        row = {
            "config_id": raw["config_id"],
            "precision_config_path": str(config_path.relative_to(Path.cwd())),
            "dtype_policy": {
                "weight_groups": policy["weight_groups"],
                "activation_residual": policy["activation_residual"],
                "ccl": policy["ccl"],
                "kv_cache": policy["kv_cache"],
                "logits_sampling": policy["logits_sampling"],
                "layer_exceptions": policy["layer_exceptions"],
            },
            "compute_fidelity_policy": policy["compute_fidelities"],
            "runtime_geometry": policy["runtime"],
            "top1": raw["top1"],
            "top5": raw["top5"],
            "top100": raw["top100"],
            "token_count": 100,
            "ttft_ms": raw["ttft_ms"],
            "trace_verified_teacher_forcing_decode_t_s_u": raw["traced_decode_t_s_u"],
            "trace_verified_samples_t_s_u": raw["samples"],
            "trace_verified": True,
            "measurement_regime": REGIME,
            "reference_path": str(REFERENCE.relative_to(Path.cwd())),
            "reference_sha256": "e88a9c2fe1d59448231e5edc4260f306328e4e4fdeef878d05166d2e4d9bbbc9",
            "command": COMMAND.format(config_path=config_path.relative_to(Path.cwd())),
            "hardware": HARDWARE,
            "mesh": MESH,
            "git_branch": "mvasiljevic/fast-models/mistralai-mistral-small-24b-instruct-2501",
            "measurement_base_commit": MEASUREMENT_BASE_COMMIT,
            "measurement_worktree_state": MEASUREMENT_WORKTREE_STATE,
            "evidence_logs": raw["log"],
            "accuracy_pass": accuracy_pass,
            "capability_pass": raw["capability_pass"],
            "pass": accuracy_pass and raw["capability_pass"],
            "selected": raw["config_id"] == "bfp4_lofi_bfp8kv_bf16ccl",
            "decision": raw["decision"],
        }
        rows.append(row)
    return rows


def write_tables(rows: list[dict]) -> None:
    payload = {
        "schema_version": 1,
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "thresholds": THRESHOLDS,
        "ranking_metric": "trace_verified_teacher_forcing_decode_t_s_u",
        "selected_config_id": "bfp4_lofi_bfp8kv_bf16ccl",
        "baseline_prefill": {"top1": 0.99, "top5": 1.0, "top100": 1.0, "log": "logs/baseline_prefill.log"},
        "post_selection_token_out": {
            "precision_config_path": "models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/datatype_sweep/selected_precision_config.json",
            "command": "MISTRAL_SMALL_24B_PRECISION_CONFIG=models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/datatype_sweep/selected_precision_config.json MISTRAL_SMALL_24B_OPTIMIZED_FULL_MODEL_BENCHMARK=/home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724 MISTRAL_SMALL_24B_OPT_FULL_LAYERS=40 MISTRAL_SMALL_24B_OPT_FULL_STEPS=128 pytest -q -s models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_full_model.py::test_optimized_full_model_token_out_benchmark",
            "prompt_tokens": 128,
            "steps": 128,
            "ttft_ms_samples": [57.294177, 56.826756],
            "no_readback_t_s_u": 55.930356,
            "no_readback_ms_per_token": 17.879379,
            "trace_verified": True,
            "host_boundaries_inside_window": 0,
            "evidence_log": "logs/bf16_ccl_token_out.log",
        },
        "results": rows,
    }
    (ROOT / "sweep_results.json").write_text(json.dumps(payload, indent=2) + "\n")

    columns = [
        "config_id",
        "weight_dtypes",
        "compute_fidelities",
        "activation_residual_dtype",
        "ccl_dtype",
        "kv_cache_dtype",
        "logits_sampling_dtype",
        "layer_exceptions",
        "top1",
        "top5",
        "top100",
        "token_count",
        "ttft_ms",
        "trace_verified_teacher_forcing_decode_t_s_u",
        "trace_verified_samples_t_s_u",
        "measurement_regime",
        "command",
        "hardware",
        "mesh",
        "accuracy_pass",
        "capability_pass",
        "pass",
        "selected",
        "decision",
        "precision_config_path",
        "reference_path",
        "git_branch",
        "measurement_base_commit",
        "measurement_worktree_state",
        "evidence_logs",
    ]
    with (ROOT / "sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "config_id": row["config_id"],
                    "weight_dtypes": json.dumps(row["dtype_policy"]["weight_groups"], sort_keys=True),
                    "compute_fidelities": json.dumps(row["compute_fidelity_policy"], sort_keys=True),
                    "activation_residual_dtype": json.dumps(row["dtype_policy"]["activation_residual"], sort_keys=True),
                    "ccl_dtype": json.dumps(row["dtype_policy"]["ccl"], sort_keys=True),
                    "kv_cache_dtype": row["dtype_policy"]["kv_cache"]["dtype"],
                    "logits_sampling_dtype": json.dumps(row["dtype_policy"]["logits_sampling"], sort_keys=True),
                    "layer_exceptions": json.dumps(row["dtype_policy"]["layer_exceptions"], sort_keys=True),
                    "top1": row["top1"],
                    "top5": row["top5"],
                    "top100": row["top100"],
                    "token_count": row["token_count"],
                    "ttft_ms": row["ttft_ms"],
                    "trace_verified_teacher_forcing_decode_t_s_u": row["trace_verified_teacher_forcing_decode_t_s_u"],
                    "trace_verified_samples_t_s_u": json.dumps(row["trace_verified_samples_t_s_u"]),
                    "measurement_regime": row["measurement_regime"],
                    "command": row["command"],
                    "hardware": row["hardware"],
                    "mesh": row["mesh"],
                    "accuracy_pass": row["accuracy_pass"],
                    "capability_pass": row["capability_pass"],
                    "pass": row["pass"],
                    "selected": row["selected"],
                    "decision": row["decision"],
                    "evidence_logs": json.dumps(row["evidence_logs"]),
                    "precision_config_path": row["precision_config_path"],
                    "reference_path": row["reference_path"],
                    "git_branch": row["git_branch"],
                    "measurement_base_commit": row["measurement_base_commit"],
                    "measurement_worktree_state": row["measurement_worktree_state"],
                }
            )


def pareto(rows: list[dict], accuracy_key: str, threshold: float, filename: str) -> None:
    valid = [row for row in rows if row["trace_verified"]]
    points = sorted(valid, key=lambda row: (row[accuracy_key], row["trace_verified_teacher_forcing_decode_t_s_u"]))
    frontier = []
    best = float("-inf")
    for row in reversed(points):
        speed = row["trace_verified_teacher_forcing_decode_t_s_u"]
        if speed > best:
            frontier.append(row)
            best = speed
    frontier.reverse()

    fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
    fig.patch.set_facecolor("#f7f3ea")
    ax.set_facecolor("#fffdf8")
    for row in valid:
        selected = row["selected"]
        feasible = row["pass"]
        ax.scatter(
            row[accuracy_key] * 100,
            row["trace_verified_teacher_forcing_decode_t_s_u"],
            s=180 if selected else 90,
            color="#d62728" if selected else ("#176b87" if feasible else "#8b8b8b"),
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
        )
        label = row["config_id"].replace("bfp4_lofi_", "").replace("_bfp8kv", "")
        ax.annotate(
            label,
            (row[accuracy_key] * 100, row["trace_verified_teacher_forcing_decode_t_s_u"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.plot(
        [row[accuracy_key] * 100 for row in frontier],
        [row["trace_verified_teacher_forcing_decode_t_s_u"] for row in frontier],
        color="#d18f00",
        linewidth=2.5,
        marker="o",
        label="Pareto frontier",
        zorder=3,
    )
    ax.axvline(
        threshold * 100,
        color="#333333",
        linestyle=":",
        linewidth=2,
        label=f"minimum {accuracy_key}: {threshold * 100:.0f}%",
    )
    ax.set_title(f"Mistral Small 24B datatype sweep: {accuracy_key} vs traced decode", fontsize=15, weight="bold")
    ax.set_xlabel(f"Full-model {accuracy_key} accuracy (%)")
    ax.set_ylabel("Trace-verified teacher-forcing decode (tokens/s/user)")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / filename)
    plt.close(fig)


def main() -> None:
    rows = result_rows()
    write_tables(rows)
    pareto(rows, "top1", THRESHOLDS["top1"], "top1_perf_pareto.png")
    pareto(rows, "top5", THRESHOLDS["top5"], "top5_perf_pareto.png")


if __name__ == "__main__":
    main()
