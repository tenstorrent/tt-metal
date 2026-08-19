# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build datatype-sweep aggregate files, Pareto plots, and selected config."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = Path("models/autoports/qwen_qwen3_6_35b_a3b")
SWEEP_DIR = MODEL_DIR / "doc/datatype_sweep"
CONFIG_DIR = SWEEP_DIR / "configs"
ARTIFACT_DIR = SWEEP_DIR / "artifacts"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _normalise_config(config_dir: Path, config_id: str) -> dict[str, Any]:
    baseline = _load_json(config_dir / "baseline_default.json")
    candidate_path = config_dir / f"{config_id}.json"
    if not candidate_path.exists():
        return baseline
    return _deep_update(baseline, _load_json(candidate_path))


def _aggregate_results(
    *,
    sweep_dir: Path,
    config_dir: Path,
    artifact_dir: Path,
    top1_min: float,
    top5_min: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quality_overrides_path = sweep_dir / "quality_gate_overrides.json"
    quality_overrides = _load_json(quality_overrides_path) if quality_overrides_path.exists() else {}
    rows: list[dict[str, Any]] = []
    for result_path in sorted(artifact_dir.glob("*_result.json")):
        result = _load_json(result_path)
        config_id = result["config_id"]
        config = _normalise_config(config_dir, config_id)
        prefill = result["prefill"]["aggregate"]
        teacher = result["teacher_forcing"]["aggregate"]
        trace_verified = bool(result["teacher_forcing"].get("trace_verified"))
        accuracy_gate_pass = (
            result.get("status") == "pass"
            and trace_verified
            and teacher["top1"] >= top1_min
            and teacher["top5"] >= top5_min
        )
        quality_gate = quality_overrides.get(config_id, {"status": "pass", "pass": True})
        quality_gate_pass = bool(quality_gate.get("pass", quality_gate.get("status") == "pass"))
        gate_pass = accuracy_gate_pass and quality_gate_pass
        row = {
            "config_id": config_id,
            "precision_config_path": result.get("precision_config_path"),
            "result_artifact": str(result_path),
            "dtype_policy": {
                "weight_groups": config["weight_groups"],
                "layer_exceptions": config.get("layer_exceptions", {}),
                "activation_dtype": config["activation_dtype"],
                "residual_dtype": config["residual_dtype"],
                "ccl_dtype": config["ccl_dtype"],
                "kv_cache_dtype": config["kv_cache_dtype"],
                "linear_state_dtype": config["linear_state_dtype"],
                "logits_dtype": config["logits_dtype"],
                "sampling_dtype": config["sampling_dtype"],
            },
            "compute_fidelity_policy": config.get("compute_fidelities", {}),
            "prefill_top1": prefill["top1"],
            "prefill_top5": prefill["top5"],
            "prefill_top100": prefill["top100"],
            "teacher_forcing_top1": teacher["top1"],
            "teacher_forcing_top5": teacher["top5"],
            "teacher_forcing_top100": teacher["top100"],
            "ttft_ms": teacher["ttft_ms"],
            "trace_verified_teacher_forcing_decode_t_s_u": teacher["decode_t_s_u"],
            "teacher_forcing_e2e_t_s_u": teacher["e2e_t_s_u"],
            "measurement_regime": result["measurement_regime"],
            "command": result.get("command"),
            "env": result.get("env", {}),
            "hardware": result.get("hardware", {}),
            "mesh": {
                "mesh_device": result.get("hardware", {}).get("mesh_device"),
                "mesh_shape": result.get("hardware", {}).get("mesh_shape"),
                "fabric_config": result.get("hardware", {}).get("fabric_config"),
            },
            "trace_counters": result["teacher_forcing"].get("trace_counters", {}),
            "runtime_policy_summary": result.get("runtime_policy_summary", {}),
            "accuracy_gate_pass": accuracy_gate_pass,
            "quality_gate": quality_gate,
            "quality_gate_pass": quality_gate_pass,
            "pass": gate_pass,
            "status": "pass" if gate_pass else str(quality_gate.get("status", "fail")),
        }
        rows.append(row)

    passing = [row for row in rows if row["pass"]]
    if not passing:
        raise RuntimeError("no datatype-sweep candidate satisfied the traced teacher-forcing gate")
    selected = max(passing, key=lambda row: row["trace_verified_teacher_forcing_decode_t_s_u"])
    for row in rows:
        row["selected"] = row["config_id"] == selected["config_id"]
    return rows, selected


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "config_id",
        "precision_config_path",
        "result_artifact",
        "dtype_policy",
        "compute_fidelity_policy",
        "prefill_top1",
        "prefill_top5",
        "prefill_top100",
        "teacher_forcing_top1",
        "teacher_forcing_top5",
        "teacher_forcing_top100",
        "ttft_ms",
        "trace_verified_teacher_forcing_decode_t_s_u",
        "teacher_forcing_e2e_t_s_u",
        "measurement_regime",
        "command",
        "env",
        "hardware",
        "mesh",
        "trace_counters",
        "runtime_policy_summary",
        "accuracy_gate_pass",
        "quality_gate",
        "quality_gate_pass",
        "pass",
        "status",
        "selected",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _pareto_frontier(rows: list[dict[str, Any]], accuracy_key: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row[accuracy_key], row["trace_verified_teacher_forcing_decode_t_s_u"]))
    frontier: list[dict[str, Any]] = []
    best_perf = float("-inf")
    for row in ordered:
        perf = row["trace_verified_teacher_forcing_decode_t_s_u"]
        if perf > best_perf:
            frontier.append(row)
            best_perf = perf
    return frontier


def _plot_pareto(
    *,
    rows: list[dict[str, Any]],
    selected: dict[str, Any],
    accuracy_key: str,
    min_accuracy: float,
    output: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10.5, 6.5))
    xs = [row[accuracy_key] * 100.0 for row in rows]
    ys = [row["trace_verified_teacher_forcing_decode_t_s_u"] for row in rows]
    plt.scatter(xs, ys, color="#4c78a8", s=55, label="Evaluated configs")
    for row in rows:
        plt.annotate(
            row["config_id"],
            (row[accuracy_key] * 100.0, row["trace_verified_teacher_forcing_decode_t_s_u"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    frontier = _pareto_frontier(rows, accuracy_key)
    plt.plot(
        [row[accuracy_key] * 100.0 for row in frontier],
        [row["trace_verified_teacher_forcing_decode_t_s_u"] for row in frontier],
        color="#54a24b",
        linewidth=2,
        marker="o",
        label="Pareto frontier",
    )
    plt.scatter(
        [selected[accuracy_key] * 100.0],
        [selected["trace_verified_teacher_forcing_decode_t_s_u"]],
        color="#e45756",
        s=120,
        zorder=5,
        label="Selected",
    )
    plt.axvline(min_accuracy * 100.0, color="#555555", linestyle=":", linewidth=1.5, label="Minimum accuracy")
    plt.xlabel(f"{title} (%)")
    plt.ylabel("Trace-verified teacher-forcing decode (t/s/u)")
    plt.title(f"{title} vs decode throughput")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _write_selected_config(
    *,
    config_dir: Path,
    output: Path,
    selected: dict[str, Any],
    rows: list[dict[str, Any]],
    top1_min: float,
    top5_min: float,
) -> dict[str, Any]:
    config = _normalise_config(config_dir, selected["config_id"])
    config["selected_from_config_id"] = selected["config_id"]
    config["selection_metric"] = "trace_verified_teacher_forcing_decode_t_s_u"
    config["selection_rule"] = (
        "fastest evaluated config satisfying teacher-forcing top-1/top-5 gates, trace_verified=true, "
        "and the qualitative quality gate"
    )
    config["accuracy_gate"] = {
        "teacher_forcing_top1_min": top1_min,
        "teacher_forcing_top5_min": top5_min,
    }
    config["selected_measurement"] = {
        "teacher_forcing_top1": selected["teacher_forcing_top1"],
        "teacher_forcing_top5": selected["teacher_forcing_top5"],
        "teacher_forcing_top100": selected["teacher_forcing_top100"],
        "ttft_ms": selected["ttft_ms"],
        "decode_t_s_u": selected["trace_verified_teacher_forcing_decode_t_s_u"],
        "measurement_regime": selected["measurement_regime"],
        "result_artifact": selected["result_artifact"],
    }
    rejected_quality = [
        {
            "config_id": row["config_id"],
            "reason": row.get("quality_gate", {}).get("reason", "failed qualitative quality gate"),
            "artifacts": row.get("quality_gate", {}).get("artifacts", []),
        }
        for row in rows
        if not row.get("quality_gate_pass", True)
    ]
    config["quality_gate"] = {
        "applied": True,
        "status": "pass",
        "selected_quality_artifact": "doc/datatype_sweep/artifacts/qualitative_chat_suite_64/degenerate_output_report.json",
        "rejected_faster_accuracy_pass_candidates": [
            item
            for item in rejected_quality
            if next(row for row in rows if row["config_id"] == item["config_id"])["accuracy_gate_pass"]
            and next(row for row in rows if row["config_id"] == item["config_id"])[
                "trace_verified_teacher_forcing_decode_t_s_u"
            ]
            > selected["trace_verified_teacher_forcing_decode_t_s_u"]
        ],
    }
    config[
        "logits_sampling_assumptions"
    ] = "LM-head emits BF16 logits for greedy top-k1 device sampling; token feedback buffers use uint32."
    config["default_consumption_path"] = {
        "generator": "models/autoports/qwen_qwen3_6_35b_a3b/tt/generator.py::build_generator",
        "loader": "models/autoports/qwen_qwen3_6_35b_a3b/tt/precision_config.py::load_precision_config",
        "default_file": str(output),
        "override_env": "QWEN36_PRECISION_CONFIG",
        "evidence": [
            selected["result_artifact"],
            "doc/datatype_sweep/artifacts/selected_precision_default_load_check.json",
            "doc/datatype_sweep/artifacts/token_out_no_readback_selected_prompt128_gen128_warmed.json",
        ],
    }
    config["bfp4_material_group_coverage"] = {
        "routed_moe": ["routed_all_bfp4_lofi", "routed_all_bfp4_hifi2"],
        "shared_moe": ["shared_moe_bfp4_lofi", "shared_moe_bfp4_hifi2"],
    }
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--top1-min", type=float, default=0.90)
    parser.add_argument("--top5-min", type=float, default=0.98)
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    config_dir = sweep_dir / "configs"
    artifact_dir = sweep_dir / "artifacts"
    rows, selected = _aggregate_results(
        sweep_dir=sweep_dir,
        config_dir=config_dir,
        artifact_dir=artifact_dir,
        top1_min=args.top1_min,
        top5_min=args.top5_min,
    )
    aggregate = {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "stage": "datatype_sweep",
        "thresholds": {
            "teacher_forcing_top1_min": args.top1_min,
            "teacher_forcing_top5_min": args.top5_min,
        },
        "selection_metric": "trace_verified_teacher_forcing_decode_t_s_u",
        "selected_config_id": selected["config_id"],
        "rows": rows,
    }
    (sweep_dir / "sweep_results.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    _write_csv(sweep_dir / "sweep_results.csv", rows)
    selected_config = _write_selected_config(
        config_dir=config_dir,
        output=sweep_dir / "selected_precision_config.json",
        selected=selected,
        rows=rows,
        top1_min=args.top1_min,
        top5_min=args.top5_min,
    )
    _plot_pareto(
        rows=rows,
        selected=selected,
        accuracy_key="teacher_forcing_top1",
        min_accuracy=args.top1_min,
        output=sweep_dir / "top1_perf_pareto.png",
        title="Teacher-forcing top-1",
    )
    _plot_pareto(
        rows=rows,
        selected=selected,
        accuracy_key="teacher_forcing_top5",
        min_accuracy=args.top5_min,
        output=sweep_dir / "top5_perf_pareto.png",
        title="Teacher-forcing top-5",
    )
    print(
        json.dumps(
            {
                "selected_config_id": selected["config_id"],
                "selected_decode_t_s_u": selected["trace_verified_teacher_forcing_decode_t_s_u"],
                "selected_top1": selected["teacher_forcing_top1"],
                "selected_top5": selected["teacher_forcing_top5"],
                "selected_precision_config": str(sweep_dir / "selected_precision_config.json"),
                "selected_weight_groups": selected_config["weight_groups"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
