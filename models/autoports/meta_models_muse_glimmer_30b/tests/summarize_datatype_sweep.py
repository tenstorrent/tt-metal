# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Produce inspectable results and traced teacher-forcing Pareto charts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"
parser = argparse.ArgumentParser()
parser.add_argument("--select", action="store_true")
args = parser.parse_args()
commands = [json.loads(line)["command"] for line in (DOC / "commands.jsonl").read_text().splitlines()]


def exact_command(attempt):
    # Early raw runner records used sys.argv; the serialized driver retained
    # their interpreter as well. Preserve those raw records and use its command.
    return next((cmd for cmd in commands if cmd[1:] == attempt["command"]), attempt["command"])


capability_evidence = {}
manifest = DOC / "capability_checks.json"
if manifest.exists():
    for record in json.loads(manifest.read_text()):
        check = json.loads((DOC / record["evidence_path"]).read_text())
        config_id = check.get("config_id") or check["precision_config"]["config_id"]
        assert config_id == record["config_id"]
        assert check["context"] == 131072
        assert check["status"] in ("pass", "capability_gate_fail")
        policy = check.get("precision_config")
        if policy is None:
            policy = json.loads((DOC / record["precision_config_path"]).read_text())
        assert policy == json.loads((DOC / "configs" / f"{config_id}.json").read_text())
        if check["status"] == "pass":
            assert check["layers"] == 52 and check["nonaligned_prefill"] == 131071
            assert check["aligned_prefill"] == 131072 and check["decode_position"] == 131071
            assert check["finite_logits"] and check["decode_token_in_prefill_top5"]
            assert check["prefill_decode_cosine"] > 0.99 and check["counters"]["captures"] == 25
        capability_evidence.setdefault(config_id, []).append(
            dict(
                check_type="context",
                status=check["status"],
                evidence_path=record["evidence_path"],
                prefill_decode_cosine=check["prefill_decode_cosine"],
                required_cosine_gt=0.99,
            )
        )

batch_manifest = DOC / "batch_capability_checks.json"
if batch_manifest.exists():
    for record in json.loads(batch_manifest.read_text()):
        check = json.loads((DOC / record["evidence_path"]).read_text())
        config_id = check["precision_config"]["config_id"]
        assert config_id == record["config_id"]
        assert check["precision_config"] == json.loads((DOC / "configs" / f"{config_id}.json").read_text())
        assert check["layers"] == 52 and check["batch"] == 32 and check["captures"] == 25
        assert check["status"] in ("pass", "capability_gate_fail")
        if check["status"] == "pass":
            assert check["append_cosine_min"] > 0.99 and check["append_argmax_equal"]
            assert check["inactive_cache_preserved"] and check["reordered_slots_preserved"]
        else:
            assert check["cosine_min"] <= 0.99 or not check["argmax_equal"]
        capability_evidence.setdefault(config_id, []).append(
            dict(
                check_type="batch32",
                status=check["status"],
                evidence_path=record["evidence_path"],
                append_cosine_min=check.get("append_cosine_min", check.get("cosine_min")),
                append_argmax_equal=check.get("append_argmax_equal", check.get("argmax_equal")),
            )
        )


def check_status(checks, check_type):
    matching = [c for c in checks if c["check_type"] == check_type]
    return (
        "fail"
        if any(c["status"] == "capability_gate_fail" for c in matching)
        else "pass"
        if matching
        else "not_evaluated"
    )


rows = []
for path in sorted((DOC / "runs").glob("*.json")):
    d = json.loads(path.read_text())
    if not d.get("trace_verified") or "smoke" in d:
        continue
    c = d["precision_config"]
    checks = capability_evidence.get(d["config_id"], [])
    capability_status = (
        "fail"
        if any(check["status"] == "capability_gate_fail" for check in checks)
        else "pass"
        if checks
        else "not_evaluated"
    )
    rows.append(
        dict(
            config_id=d["config_id"],
            precision_config_path=d["config_path"],
            dtype_policy=c,
            compute_fidelity_policy={
                "decoder_kinds": {
                    kind: {
                        "matmul_groups": {
                            name: {"fidelity": g["fidelity"], "fp32": g["fp32"]}
                            for name, g in spec["weight_groups"].items()
                        },
                        "norm_and_attention": spec["compute"],
                    }
                    for kind, spec in c["decoder_kinds"].items()
                },
                "terminal": {key: c["terminal"][key] for key in ("fidelity", "fp32")},
                "final_norm": {key: c["final_norm_" + key] for key in ("fidelity", "fp32")},
                "short_prefill_sliding_input_norm": c["short_prefill_sliding_input_norm"],
            },
            terminal_compute_fidelity=c["terminal"]["fidelity"],
            top1=d["top1"],
            top5=d["top5"],
            top100=d["top100"],
            prefill=d["prefill"],
            ttft_ms=d["ttft_ms"],
            teacher_forcing_decode_tps=d["decode_tps"],
            trace_verified=True,
            teacher_forcing_decode_tps_trials=[s["decode_t/s/u"] for t in d["teacher_trials"][1:] for s in t["stats"]],
            measurement_regime=d["measurement_regime"],
            token_count=100,
            command=exact_command(d),
            hardware=d["hardware"],
            mesh=d["mesh"],
            git_commit=d["git_commit"],
            reference_path=d["reference_path"],
            reference_sha256=d["reference_sha256"],
            environment=d["environment"],
            status="capability_gate_fail" if capability_status == "fail" else d["status"],
            accuracy_pass=d["accuracy_pass"],
            capability_status=capability_status,
            context_status=check_status(checks, "context"),
            batch_status=check_status(checks, "batch32"),
            capability_evidence=checks,
            selection_eligible=d["accuracy_pass"] and capability_status != "fail",
            evidence_path=str(path.relative_to(ROOT)),
            runtime_policy_evidence=str(path.relative_to(ROOT)) + "#runtime_summary",
        )
    )
excluded = []
for path in sorted((DOC / "failures").glob("*.json")):
    attempt = json.loads(path.read_text())
    if isinstance(attempt, dict) and attempt.get("status") == "runtime_error":
        excluded.append(
            dict(
                config_id=attempt["config_id"],
                status="runtime_error_before_measurement",
                dtype_policy=attempt["precision_config"],
                command=exact_command(attempt),
                hardware=attempt["hardware"],
                mesh=attempt["mesh"],
                error=attempt["error"],
                trace_verified=False,
                evidence_path=str(path.relative_to(ROOT)),
                top1=None,
                top5=None,
                top100=None,
                ttft_ms=None,
                teacher_forcing_decode_tps=None,
                measurement_regime="construction failure; excluded from Pareto and ranking",
            )
        )
passing = [r for r in rows if r["selection_eligible"]]
if not passing:
    raise SystemExit("No evaluated passing config")
winner = max(passing, key=lambda r: r["teacher_forcing_decode_tps"])
if args.select:
    (DOC / "selected_precision_config.json").write_text(json.dumps(winner["dtype_policy"], indent=2) + "\n")
selected = json.loads((DOC / "selected_precision_config.json").read_text())["config_id"]
result = dict(
    selection_metric="median warmed trace-verified teacher-forcing decode tokens/s/user",
    thresholds={"top1": 0.90, "top5": 0.98, "top100": 1.0},
    selected_config_id=selected,
    fastest_passing_evaluated_config_id=winner["config_id"],
    capability_selection_rule="Reject any measured capability failure; the fastest remaining accuracy-passing candidate must pass full selected-policy capability validation before completion. Untested slower policies are not claimed capability-validated.",
    configs=rows,
    excluded_attempts=excluded,
)
(DOC / "sweep_results.json").write_text(json.dumps(result, indent=2) + "\n")
with (DOC / "sweep_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows
    )
    for attempt in excluded:
        writer.writerow(
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in attempt.items()
                if k in rows[0]
            }
        )
for metric, threshold in [("top1", 90), ("top5", 98)]:
    fig, (ax, key_ax) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.4, 1]}, layout="constrained"
    )
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    key_ax.axis("off")
    frontier = [
        r
        for r in rows
        if not any(
            s[metric] >= r[metric]
            and s["teacher_forcing_decode_tps"] >= r["teacher_forcing_decode_tps"]
            and (s[metric] > r[metric] or s["teacher_forcing_decode_tps"] > r["teacher_forcing_decode_tps"])
            for s in rows
        )
    ]
    frontier.sort(key=lambda r: r[metric])
    ax.plot(
        [100 * r[metric] for r in frontier],
        [r["teacher_forcing_decode_tps"] for r in frontier],
        color="#0d9488",
        lw=2,
        marker="D",
        markersize=11,
        markerfacecolor="none",
        label="Accuracy/speed frontier (before capability gates)",
    )
    key_ax.text(0, 1, "EVALUATED FULL-MODEL POLICIES", fontsize=10, weight="bold", color="#0f172a")
    key_ax.text(
        0, 0.966, "ID  Policy                                      accuracy / t/s/u", fontsize=8, color="#475569"
    )
    spacing = min(0.038, 0.88 / max(len(rows), 1))
    for i, r in enumerate(rows):
        is_selected = r["config_id"] == selected
        color = "#dc2626" if is_selected else "#2563eb" if r["selection_eligible"] else "#94a3b8"
        number = f"{i+1:02}"
        ax.scatter(
            100 * r[metric],
            r["teacher_forcing_decode_tps"],
            s=190 if is_selected else 55,
            color=color,
            marker="*" if is_selected else "o" if r["selection_eligible"] else "x",
            zorder=10 if is_selected else 5,
        )
        if (
            is_selected
            or r in frontier
            or r["config_id"]
            in (
                "baseline_bfp4_lofi_head_bfp8_hifi2",
                "decoder_bfloat4_b_HiFi2",
                "decoder_bfloat8_b_HiFi2",
            )
            or (metric == "top1" and r["config_id"] == "kv_bfloat4_b")
        ):
            ax.annotate(
                f"Selected · {number}" if is_selected else number,
                (100 * r[metric], r["teacher_forcing_decode_tps"]),
                xytext=((-65 if is_selected else 7), 12 if is_selected or r in frontier else -12),
                textcoords="offset points",
                fontsize=9 if is_selected else 8,
                color=color,
                weight="bold" if is_selected else "normal",
            )
        label = r["config_id"].replace("bfloat4_b", "B4").replace("bfloat8_b", "B8").replace("bfloat16", "BF16")
        label = label.replace("baseline_bfp4_lofi_head_bfp8_hifi2", "baseline B4/LoFi + B8/HiFi2 head")
        label = label.replace("activations", "act").replace("sliding_norms", "slide norms").replace("_", " ")
        key_ax.text(
            0,
            0.92 - i * spacing,
            f"{number}  {label}   {100*r[metric]:.0f}% / {r['teacher_forcing_decode_tps']:.2f}",
            fontsize=7.5,
            color=color,
            weight="bold" if is_selected else "normal",
        )
    ax.axvline(threshold, color="#475569", ls=":", lw=1.8, label=f"Minimum {metric}: {threshold}%")
    ax.set(
        xlabel=f"{metric.replace('top','Top-')} accuracy (%)",
        ylabel="Traced teacher-forcing decode (tokens/s/user)",
        title=f"Muse-Glimmer-30B · {metric.replace('top','Top-')} / decode frontier",
    )
    ax.grid(alpha=0.18)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlim(min(threshold, min(100 * r[metric] for r in rows)) - 1, 101.3)
    ax.margins(y=0.15)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle("QB2 1×4 · AIME24 chat · 204 input / 100 generated tokens · median of 3 warmed trials", fontsize=11)
    key_ax.text(
        0,
        -0.01,
        "Red star: selected · Blue: accuracy pass, no known capability failure · Grey: fails a gate\nOnly traced measurements enter this frontier.",
        fontsize=8,
        color="#475569",
    )
    fig.savefig(DOC / f"{metric}_perf_pareto.png", dpi=180)
    plt.close(fig)
print("FASTEST_PASSING", winner["config_id"], winner["teacher_forcing_decode_tps"], "SELECTED", selected)
