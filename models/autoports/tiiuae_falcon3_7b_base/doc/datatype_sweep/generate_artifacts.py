#!/usr/bin/env python3
"""Build machine-readable Falcon3 datatype-sweep tables and Pareto plots."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TOP1_GATE = 0.90
TOP5_GATE = 0.98
REFERENCE = "models/autoports/tiiuae_falcon3_7b_base/doc/datatype_sweep/results/reference/aime24_100.refpt"
ACCURACY_COMMAND = (
    "FALCON3_PRECISION_CONFIG=<config> TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
    "python -m models.common.readiness_check.run_teacher_forcing --model-dir "
    "models/autoports/tiiuae_falcon3_7b_base --reference "
    + REFERENCE
    + " --mesh-device P300X2 --fabric-config FABRIC_1D_RING"
)
PERF_COMMAND = (
    "FALCON3_PRECISION_CONFIG=<config> TTNN_CONFIG_OVERRIDES='{\"throw_exception_on_fallback\": true}' "
    "python models/autoports/tiiuae_falcon3_7b_base/tests/full_model_evidence.py --model-dir "
    "models/autoports/tiiuae_falcon3_7b_base --reference "
    + REFERENCE
    + " --output <evidence> --weight-cache-path /tmp/falcon3-datatype-sweep-cache"
)


def accuracy(log: Path):
    text = log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"AGGREGATE\s+top1=([0-9.]+).*top5=([0-9.]+).*top100=([0-9.]+)", text)
    return tuple(map(float, match.groups())) if match else (None, None, None)


def main():
    rows = []
    for directory in sorted(path for path in RESULTS.iterdir() if path.is_dir() and path.name != "reference"):
        config_path = (
            ROOT / "configs" / "all_bfp4_lofi_bfp8_act_ccl_kv.json"
            if directory.name == "baseline"
            else ROOT / "configs" / f"{directory.name}.json"
        )
        if not config_path.is_file():
            continue
        policy = json.loads(config_path.read_text(encoding="utf-8"))
        top1, top5, top100 = accuracy(directory / "run_teacher_forcing.log")
        evidence_path = directory / "full_model_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else None
        blocker = None
        if evidence is None:
            log = (directory / "run_teacher_forcing.log").read_text(encoding="utf-8", errors="replace")
            match = re.search(r"Statically allocated circular buffers[^\n]+", log)
            blocker = match.group(0) if match else "candidate did not construct"
            if policy["config_id"] == "bf16_hifi4":
                blocker = (
                    "AutoFix final control: BF16 LM head requires 2,003,712 B/core at the minimum legal "
                    "in0_block_w=1 for 32768, 16384, and 8192-column splits; hardware L1 is 1,572,864 B/core"
                )
        perf = evidence["performance"] if evidence else {}
        passed = bool(
            evidence and evidence.get("passed") and top1 is not None and top1 >= TOP1_GATE and top5 >= TOP5_GATE
        )
        rows.append(
            {
                "config_id": policy["config_id"],
                "config_path": str(config_path.relative_to(Path.cwd())),
                "dtype_policy": {
                    "weight_groups": policy["weight_groups"],
                    "layer_exceptions": policy["layer_exceptions"],
                    "activation_dtype": policy["activation_dtype"],
                    "residual_dtype": policy["residual_dtype"],
                    "ccl_dtype": policy["ccl_dtype"],
                    "kv_cache_dtype": policy["kv_cache_dtype"],
                    "logits_dtype": policy["logits_dtype"],
                    "sampling_dtype_assumptions": policy["sampling_dtype_assumptions"],
                },
                "compute_fidelity_policy": policy["compute_fidelities"],
                "top1": top1,
                "top5": top5,
                "top100": top100,
                "tokens": 100 if top1 is not None else None,
                "ttft_ms": perf.get("warm_ttft_ms"),
                "teacher_forcing_decode_t_s_u": perf.get("teacher_forcing_trace_t_s_u"),
                "trace_verified": bool(evidence and evidence.get("passed")),
                "measurement_regime": "batch=1, prompt=128, 128 warmed trace replays, real 28-layer weights",
                "reference": REFERENCE,
                "command": (
                    ACCURACY_COMMAND.replace("<config>", str(config_path))
                    + "; "
                    + PERF_COMMAND.replace("<config>", str(config_path)).replace("<evidence>", str(evidence_path))
                ),
                "hardware": "4x Blackhole p300c",
                "mesh": "1x4 FABRIC_1D_RING, TP4, two links",
                "status": "pass" if passed else ("fail_accuracy" if evidence else "runtime_blocker"),
                "passed": passed,
                "blocker": blocker,
                "evidence": str(evidence_path.relative_to(Path.cwd()))
                if evidence
                else str(directory.relative_to(Path.cwd())),
            }
        )
    (ROOT / "sweep_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    flat_fields = [
        "config_id",
        "dtype_policy",
        "compute_fidelity_policy",
        "top1",
        "top5",
        "top100",
        "tokens",
        "ttft_ms",
        "teacher_forcing_decode_t_s_u",
        "trace_verified",
        "measurement_regime",
        "command",
        "hardware",
        "mesh",
        "status",
        "passed",
        "blocker",
        "config_path",
        "reference",
        "evidence",
    ]
    with (ROOT / "sweep_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(row[key]) if isinstance(row[key], dict) else row[key] for key in flat_fields}
            )

    valid = [row for row in rows if row["trace_verified"] and row["top1"] is not None]
    selected = max((row for row in valid if row["passed"]), key=lambda row: row["teacher_forcing_decode_t_s_u"])
    for accuracy_key, gate, filename, label in (
        ("top1", TOP1_GATE, "top1_perf_pareto.png", "Top-1 accuracy"),
        ("top5", TOP5_GATE, "top5_perf_pareto.png", "Top-5 accuracy"),
    ):
        frontier = []
        for row in valid:
            dominated = any(
                other[accuracy_key] >= row[accuracy_key]
                and other["teacher_forcing_decode_t_s_u"] >= row["teacher_forcing_decode_t_s_u"]
                and (
                    other[accuracy_key] > row[accuracy_key]
                    or other["teacher_forcing_decode_t_s_u"] > row["teacher_forcing_decode_t_s_u"]
                )
                for other in valid
                if other is not row
            )
            if not dominated:
                frontier.append(row)
        frontier.sort(key=lambda item: item[accuracy_key])
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.scatter([r[accuracy_key] for r in valid], [r["teacher_forcing_decode_t_s_u"] for r in valid], s=65)
        ax.plot(
            [r[accuracy_key] for r in frontier],
            [r["teacher_forcing_decode_t_s_u"] for r in frontier],
            "-",
            label="Pareto frontier",
        )
        ax.scatter(
            [selected[accuracy_key]],
            [selected["teacher_forcing_decode_t_s_u"]],
            color="red",
            s=120,
            zorder=5,
            label="Selected",
        )
        ax.axvline(gate, color="black", linestyle=":", label=f"Minimum {gate:.0%}")
        for row in valid:
            ax.annotate(
                row["config_id"],
                (row[accuracy_key], row["teacher_forcing_decode_t_s_u"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Trace-verified teacher-forcing decode (t/s/u)")
        ax.set_title(f"Falcon3-7B datatype sweep: {label} / performance")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(ROOT / filename, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
