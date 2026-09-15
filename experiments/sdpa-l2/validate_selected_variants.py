# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate the four-option source/evidence checkpoint without device access."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def checked_path(relative):
    path = (ROOT / relative).resolve()
    assert ROOT in path.parents and path.is_file(), relative
    return path


def read_records(relative):
    records = {}
    for line in checked_path(relative).read_text().splitlines():
        row = json.loads(line)
        key = (row["kv_length"], row["seed"], row["distribution"], row["mode"])
        assert key not in records, key
        assert math.isfinite(row["l2_pct"]), key
        assert row["pcc"] is None or math.isfinite(row["pcc"]), key
        assert row["distinct_kv"] and row["q_chunk"] == 128 and row["k_chunk"] == 512, key
        records[key] = row
    return records


def main():
    manifest = json.loads((HERE / "selected_variants.json").read_text())
    expected = {
        "main_bf16": "main",
        "fast_bf16": "fast",
        "balanced_fp32": "qk4_pv2",
        "accurate_fp32": "accurate",
    }
    variants = manifest["variants"]
    assert len(variants) == 4
    assert {v["id"]: v["harness_mode"] for v in variants} == expected
    hashes = dict(manifest["harness_source_sha256"])
    for variant in variants:
        assert len(variant["source_sha256"]) == 3, variant["id"]
        for path, digest in variant["source_sha256"].items():
            assert path not in hashes or hashes[path] == digest, path
            hashes[path] = digest
        perf = json.loads(checked_path(variant["performance_record"]).read_text())
        assert perf["mode"] == variant["harness_mode"]
        assert perf["q_chunk"] == 256 and perf["k_chunk"] == 512
        assert perf["cores"] == 1 and perf["q_repeats"] == 16 and perf["k_chunks"] == 512
        expected_tflops = perf["useful_flops"] / (perf["median_ms"] * 1e9)
        assert math.isclose(perf["tflops_per_core"], expected_tflops, rel_tol=1e-12)
    for path, expected_digest in hashes.items():
        digest = hashlib.sha256(checked_path(path).read_bytes()).hexdigest()
        assert digest == expected_digest, f"Selected source changed: {path}"

    initial, repeat = [read_records(p) for p in manifest["accuracy_results"]]
    assert len(initial) == 136 and initial.keys() == repeat.keys()
    cases = {}
    for key, record in initial.items():
        other = repeat[key]
        for field in ("full_output_sha256", "input_sha256", "l2_pct", "pcc"):
            assert record[field] == other[field], (key, field)
        cases.setdefault(key[:3], []).append(record)
    assert len(cases) == 34
    for key, records in cases.items():
        assert {r["mode"] for r in records} == set(expected.values()), key
        assert all(r["input_sha256"] == records[0]["input_sha256"] for r in records), key

    print(f"PASS: 4 selected variants; {len(hashes)} pinned sources; 34 shared-input cases.")
    print("PASS: all 136 repeated outputs and numerical metrics are identical.")


if __name__ == "__main__":
    main()
