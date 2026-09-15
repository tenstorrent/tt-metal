# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate saved research evidence using only Python's standard library."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    rows = [json.loads(line) for line in (HERE / (name + ".jsonl")).read_text().splitlines()]
    assert rows[0]["kind"] == "provenance", name
    assert rows[-1]["kind"] == "completed", f"Incomplete run: {name}"
    return rows


def check_sources(rows):
    hashes = rows[0]["source_sha256"]
    if isinstance(hashes, str):
        hashes = {"probe.py": hashes}
    for name, expected in hashes.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name


probe = read("probe-final")
check_sources(probe)
packs = [r for r in probe if r["kind"] == "pack"]
assert len(packs) == 8
for record in packs:
    assert record["candidates"]["device"]["unequal"] == 0
    assert record["host_vs_model"]["unequal"] == 0
differences = [r for r in probe if r["kind"] == "fidelity_difference"]
assert len(differences) == 15
for record in differences:
    if record["right"] == "4":
        assert record["metrics"]["unequal"] == 0, record
    else:
        assert record["metrics"]["l2_pct"] > 1, record
print("PASS: 8 exact host/device quantizer checks and 15 fidelity-pair probes")

device_files = {
    "device-attention-host-qkv-v1": 36,
    "device-attention-device-qkv-v1": 36,
    "device-attention-host-qkv-32k-v2": 24,
    "device-attention-host-qkv-32k-seed1241": 16,
}
device_records = []
for name, expected_count in device_files.items():
    rows = read(name)
    if "32k" in name:
        check_sources(rows)
    records = [r for r in rows if r["kind"] == "device_attention"]
    assert len(records) == expected_count, name
    for record in records:
        assert record["finite"] and math.isfinite(record["l2_pct"])
        assert record["quantizer_exact_checks"] == 1 + 3 * (record["length"] // 512)
        assert abs(record["l2_pct"] - record["model_l2_pct"]) < 0.01, record
    device_records.extend(records)
print(f"PASS: {len(device_records)} device-backed output/denominator records agree with the model")

for name, expected_count in (("numerics-host-final", 300), ("numerics-host-256k-final", 14)):
    rows = read(name)
    check_sources(rows)
    records = [r for r in rows if r["kind"] == "attention_model"]
    assert len(records) == expected_count, (name, len(records))
    for record in records:
        assert record["finite"] and math.isfinite(record["l2_pct"])
        if record["variant"] == "unquantized":
            assert record["before_output_rounding_l2_pct"] < 1e-7, record
print("PASS: 314 primary quantization-model records and unquantized controls")

# Pin the rounded normal-input entries displayed in REPORT.md.
for name, length, entries in (
    (
        "device-attention-host-qkv-v1",
        4096,
        (
            ("all4", "original", 31.201, 0.951983),
            ("all4", "matched", 45.466, 0.951742),
            ("q8k4p8v4", "matched", 16.434, 0.986643),
        ),
    ),
    (
        "device-attention-host-qkv-32k-v2",
        32768,
        (
            ("all4", "original", 31.029, 0.953462),
            ("all4", "matched", 43.646, 0.953279),
            ("q8k4p8v4", "matched", 16.023, 0.987224),
        ),
    ),
    (
        "numerics-host-256k-final",
        262144,
        (
            ("all4", "original", 31.339, 0.952289),
            ("all4", "matched", 44.566, 0.952169),
            ("q8k4p8v4", "matched", 16.944, 0.985824),
        ),
    ),
):
    records = read(name)
    for variant, denominator, l2, pcc in entries:
        matched = [
            r
            for r in records
            if r.get("length") == length
            and r.get("distribution") == "normal"
            and r.get("preprocessing") == "none"
            and r.get("variant") == variant
            and r.get("denominator") == denominator
        ]
        assert len(matched) == 1
        assert round(matched[0]["l2_pct"], 3) == l2
        assert round(matched[0]["pcc"], 6) == pcc
print("PASS: report normal-input L2/PCC table matches saved evidence")
