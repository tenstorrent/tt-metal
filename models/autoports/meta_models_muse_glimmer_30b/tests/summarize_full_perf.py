# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize measured full-stack host timing and reduced-only device reports."""

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/full_model"
full = json.loads((ROOT / "full_model_ready.json").read_text())
profile = json.loads((ROOT / "tracy_reduced/perf_run.json").read_text())
iterations = profile["iterations"]


def number(row, key):
    value = row[key]
    return float(value) if value and value != "nan" else 0.0


def totals(rows, repeats=iterations):
    return {
        "classification": "measured reduced device kernel sum",
        "kernel_ms_per_iteration": sum(number(row, "Device Time") for row in rows) / repeats / 1000,
        "op_gap_ms_per_iteration": sum(number(row, "Op-to-Op Gap") for row in rows) / repeats / 1000,
    }


phases = {}
for phase in ("prefill", "decode", "sample", "argmax", "token_out"):
    with (ROOT / f"tracy_reduced/{phase}_perf_report.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    phases[phase] = totals(rows)
    phases[phase]["measured_host_ms_per_iteration"] = profile[f"{phase}_wall_ms"]
    if phase == "decode":
        decode_rows = rows

assert len(decode_rows) == iterations * 96
assert all(decode_rows[i * 96]["OP Code"] == "EmbeddingsDeviceOperation" for i in range(iterations))
segments = {}
for name, lo, hi in (
    ("embedding_rope", 0, 8),
    ("sliding_layer", 8, 49),
    ("full_layer", 49, 87),
    ("terminal_state", 87, 96),
    ("lm_head", 88, 89),
):
    rows = [row for i in range(iterations) for row in decode_rows[i * 96 + lo : i * 96 + hi]]
    segments[name] = totals(rows)

selected = {}
for mode in ("sample", "argmax"):
    rows = [row for row in full["perf"] if row["mode"] == mode]
    selected[mode] = {
        "classification": "measured full-stack host",
        "batch": 1,
        "prompt_tokens": 128,
        "generated_tokens": 128,
        "repeats": len(rows),
        **{key: statistics.median(row[key] for row in rows) for key in ("ttft_ms", "decode_ms", "decode_tps")},
    }

# Stored BFP4 tile bytes and physical widths from stage-5 PERF_REVIEW.md.
# This is a one-read DRAM lower bound, excluding activation/CCL/compute costs.
layer_weight_bytes = 69636096
head_weight_bytes = 6656 * 50688 * 2
kv_bytes = statistics.mean(math.ceil(tokens / 32) * 2 * 4 * 1088 * 52 for tokens in range(129, 256))
norm_bytes = 52 * 4 * 6656 * 2
modeled_bytes = 52 * layer_weight_bytes + head_weight_bytes + kv_bytes + norm_bytes + 6656 * 2 / 4
stack_best = 39 * 0.320355 + 13 * 0.309949
stack_1024 = 39 * 0.323808 + 13 * 0.310441
scaled_device = (
    sum(
        segments[name]["kernel_ms_per_iteration"] * count
        for name, count in (("sliding_layer", 39), ("full_layer", 13), ("terminal_state", 1), ("embedding_rope", 1))
    )
    + phases["sample"]["kernel_ms_per_iteration"]
)
result = {
    "source": "full_model_ready.json",
    "primary": selected,
    "readiness_teacher_forcing_token_out": full["teacher"][0],
    "traced_logits_only": full["logits_only"],
    "greedy_tie_control": full["greedy_tie_control"],
    "construction_and_preparation_seconds": full["construction_and_preparation_seconds"],
    "preparation_seconds": full["preparation_seconds"],
    "reduced_layers": [0, 3],
    "reduced_phases": phases,
    "reduced_decode_segments": segments,
    "full_stack_device_ms": None,
    "full_stack_device_reason": "Not measured: full-model skill prohibits all-layer profiler collection.",
    "modeled_full_stack_kernel_sum_ms": scaled_device,
    "modeled_kernel_sum_caveat": "Extrapolates reduced layer kernel sums by39/13. Not full-stack measured device latency or a critical path.",
    "modeled_one_read_roofline": {
        "classification": "modeled",
        "bytes_per_chip": modeled_bytes,
        "bandwidth_bytes_per_second_per_chip": 512e9,
        "ms_per_token": modeled_bytes / 512e9 * 1000,
        "excludes": ["activation traffic", "CCL", "programs and command buffers", "compute", "host dispatch"],
    },
    "layer_stack_accounting": {
        "classification": "modeled sum of measured prior layer latencies",
        "counts": {"sliding": 39, "full": 13},
        "best_short_context_ms": stack_best,
        "context1024_ms": stack_1024,
        "full_minus_best_ms": selected["sample"]["decode_ms"] - stack_best,
        "caveat": "Stage5 standalone per-layer traces at context33/1024; primary is one full-stack trace with positions128..254. This is a comparison, not isolated wrapper cost.",
    },
    "sample_fraction_of_full_decode": profile["sample_wall_ms"] / selected["sample"]["decode_ms"],
    "head_modeled_read_bandwidth_gbps": head_weight_bytes / (segments["lm_head"]["kernel_ms_per_iteration"] * 1e6),
}
(ROOT / "perf_summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {"primary": selected, "roofline": result["modeled_one_read_roofline"], "modeled_device_ms": scaled_device},
        indent=2,
    )
)
