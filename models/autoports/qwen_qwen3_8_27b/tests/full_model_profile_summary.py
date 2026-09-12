# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize per-device reports and explicitly bounded representative layers."""

import csv
import hashlib
import json
from pathlib import Path

root = Path("models/autoports/qwen_qwen3_8_27b/doc/full_model")
summary = {}
for name in ("profile_split_drained", "profile_selected", "profile_final"):
    devices = []
    for device in range(4):
        phases = {}
        for phase in ("prefill", "model", "sample", "token_out"):
            path = root / "tracy" / name / f"device{device}_{phase}_perf_report.csv"
            rows = list(csv.DictReader(path.open()))
            duration = sum(float(r["Device Time"] or 0) for r in rows)
            gap = sum(float(r["Op-to-Op Gap"] or 0) for r in rows)
            phases[phase] = dict(
                device_us=duration,
                gap_us=gap,
                window_us=duration + gap,
                ops=len(rows),
                report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        devices.append(dict(device=device, phases=phases))
    summary[name] = devices
(root / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

path = root / "tracy/profile_final/device0_model_perf_report.csv"
rows = list(csv.DictReader(path.open()))
by_id = {int(r["ID"]): r for r in rows}
# These boundaries follow model entry, layer0 (linear), layer3 (full), terminal.
for start in (1106, 1154, 1196):
    assert by_id[start]["OP Code"] == "LayerNormDeviceOperation"
assert "ChunkGdnScan" in by_id[1133]["OP Code"]
assert "SdpaDecode" in by_id[1177]["OP Code"]
assert "8192" in by_id[1198]["OP Code"]
blocks = {}
for name, start, end in (("entry", 1097, 1105), ("linear", 1106, 1153), ("full", 1154, 1195), ("terminal", 1196, 1217)):
    selected = [by_id[i] for i in range(start, end + 1)]
    kernel = sum(float(r["Device Time"] or 0) for r in selected)
    gap = sum(float(r["Op-to-Op Gap"] or 0) for r in selected)
    blocks[name] = dict(first_id=start, last_id=end, kernel_us=kernel, gap_us=gap, window_us=kernel + gap)
stack = 48 * blocks["linear"]["window_us"] + 16 * blocks["full"]["window_us"]
floor = 48 * blocks["linear"]["kernel_us"] + 16 * blocks["full"]["kernel_us"]
sample = summary["profile_final"][0]["phases"]["sample"]["window_us"]
report = dict(
    source=str(path),
    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    blocks=blocks,
    kernel_only_stack_floor_us=floor,
    stack_only_estimate_us=stack,
    token_out_estimate_us=stack + blocks["entry"]["window_us"] + blocks["terminal"]["window_us"] + sample,
    sampler_window_us=sample,
    previous_stage_primary_comparison=dict(
        source="optimized_multichip_decoder/README.md",
        linear_ms=0.421990,
        full_ms=0.306304,
        weighted_stack_ms=48 * 0.421990 + 16 * 0.306304,
        scope="Isolated layer trace replay and synchronization included; not a pure kernel floor.",
    ),
    note="Representative reduced device0 layer extrapolation, not an all-layer measured profile. Gaps include instrumentation; kernel-only sum is an accounting floor.",
)
(root / "stack_accounting.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
