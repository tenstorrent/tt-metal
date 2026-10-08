# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A throughput comparison cannot hide changed outputs, precision or workload."""

import copy
import json

import pytest

from models.demos.qwen38_27b_qb2.tests.compare_gdn_sweeps import compare
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, summarize


def pair(tmp_path, defect=None):
    paths = []
    for i, (variant, policy) in enumerate((("single-step", "single_step"), ("shared-qk", "single_step_shared_qk"))):
        report = make_plan(batches=(1,), input_lengths=(8192,))
        report.update(
            state="completed",
            cleanup_completed=True,
            recurrence_variant=variant,
            source_sha256={"model.py": "same", "effective_precision_override": str(i)},
            precision={"decode_recurrence": policy, "kv_cache_dtype": "bfloat8_b"},
            configuration={"environment": {"QWEN_PRECISION_CONFIG": policy, "QWEN_PREFILL_MAX_BATCH_TOKENS": "32768"}},
        )
        sample = dict(
            prefill_s=2,
            decode_s=3 - i,
            elapsed_s=6 - i,
            ttft_s=[2.5],
            trace_captures=0,
            output_sha256_per_replica=["tokens"],
        )
        if i and defect == "output":
            sample["output_sha256_per_replica"] = ["changed"]
        cell = report["cells"][0]
        cell.update(
            status="completed",
            prompt_sha256="same",
            warmup=copy.deepcopy(sample),
            samples=[copy.deepcopy(sample) for _ in range(3)],
        )
        cell["summary"] = summarize(cell["samples"], concurrency=1, input_tokens=8192, output_tokens=128)
        if i and defect == "precision":
            report["precision"]["kv_cache_dtype"] = "bfloat4_b"
        if i and defect == "runtime":
            report["configuration"]["environment"]["QWEN_PREFILL_MAX_BATCH_TOKENS"] = "65536"
        if i and defect == "source":
            report["source_sha256"]["model.py"] = "changed"
        path = tmp_path / f"{variant}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


def checked(paths):
    return compare(
        *paths,
        variants=("single-step", "shared-qk"),
        recurrence_policies=("single_step", "single_step_shared_qk"),
        require_same_output=True,
    )


def test_shared_qk_comparison_preserves_raw_receipts_and_requires_equal_outputs(tmp_path):
    paths = pair(tmp_path)
    original = [path.read_bytes() for path in paths]
    result = checked(paths)
    assert result["cells"][0]["decode_uplift_percent"] == 50
    assert result["cells"][0]["same_output_hash_as_native"]
    assert result["output_equivalence_required"]
    assert [path.read_bytes() for path in paths] == original


@pytest.mark.parametrize(
    "defect,message",
    [
        ("output", "output differs"),
        ("precision", "Precision differs"),
        ("runtime", "runtime settings"),
        ("source", "model sources"),
    ],
)
def test_comparison_rejects_unmatched_or_changed_model(tmp_path, defect, message, expect_error):
    with expect_error(ValueError, message):
        checked(pair(tmp_path, defect))
