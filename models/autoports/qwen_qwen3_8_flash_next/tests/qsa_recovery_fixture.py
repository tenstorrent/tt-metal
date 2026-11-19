# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed loader for reduced QSA boundary evidence, never all-output acceptance.

Version 1 passing-prefill exports retain their separate, unchanged loader.
This version 2 path requires a closed successful recovery run and retains the
original full-prefill failure in every consuming bridge report.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_boundary_recovery import (
    CAPACITY,
    CHECKPOINT,
    MUTABLE,
    ORIGIN,
    PREFIX,
    bind_inputs,
    file_identity,
)
from models.autoports.qwen_qwen3_8_flash_next.tests.qsa_state_fixture import STATE_TENSORS, runtime_source_hashes


def validate_boundary_evidence(payload, report, *, source_key, prefix, capacity, runtime_hashes):
    """Pure metadata checks, also exercised by CPU-only negative controls."""
    assert payload["version"] == 2 and payload["origin"] == ORIGIN
    assert payload["prefix"] == prefix == PREFIX and payload["capacity"] == capacity == CAPACITY
    assert payload["page_size"] == 32 and payload["source_fixture_key"] == source_key
    assert payload["runtime_source_sha256"] == report["runtime_source_sha256"] == runtime_hashes
    assert payload["bindings"] == report["bindings"]
    assert payload["strict_full_prefill_passed"] is False and report["strict_full_prefill_passed"] is False
    assert payload["original_full_prefill"] == report["original_full_prefill"]
    assert payload["original_full_prefill"]["passed"] is False
    assert payload["original_full_prefill"]["rows_compared"] == CAPACITY
    assert report["boundary_state_passed"] is True and report["device_executed"] is True
    assert "error" not in report and report["stage_acceptance"] == "not_evaluated"
    assert report["capture_count"] == 1 and report["replay_count"] == 4
    assert report["runtime_fallback_audit"] == "clean" and report["allocation_tracking"] is True
    assert report["prefix_state_build"]["rows"] == CHECKPOINT
    assert report["prefix_state_build"]["origin"] == ORIGIN
    evidence = payload["boundary_evidence"]
    for name, value in evidence.items():
        assert report[name] == value, name
    assert set(evidence) == {
        "state_stop_control",
        "final32_saved_TT_bit_exact",
        "final32_HF",
        "nonaligned_tail_HF",
        "decode",
        "repeated_output_and_state_exact",
    }
    assert evidence["final32_saved_TT_bit_exact"] is True
    assert evidence["repeated_output_and_state_exact"] is True
    controls = evidence["state_stop_control"]
    assert [(item["start"], item["count"]) for item in controls] == [(0, 32), (32, 31), (63, 1)]
    for item in controls:
        assert set(item["all_mutable_state_equal"]) == set(MUTABLE)
        assert all(value is True for value in item["all_mutable_state_equal"].values())
    decode = evidence["decode"]
    assert [item["branch"] for item in decode] == [0, 1, 0, 1]
    assert all(item["position"] == CAPACITY for item in decode)
    for metric, rows in [(evidence["final32_HF"], 32), (evidence["nonaligned_tail_HF"], 31)] + [
        (item, 1) for item in decode
    ]:
        assert metric["passed"] is True and metric["finite"] is True
        assert metric["rows_compared"] == rows and metric["rows_below_threshold"] == 0
        assert all(math.isfinite(metric[key]) and metric[key] >= 0.995 for key in ("whole_pcc", "minimum_row_pcc"))


def load_recovery_prefix(path, *, recovery_report, recovery_provenance, source_path, source_key, prefix, capacity):
    """Validate immutable inputs and completed recovery before device upload."""
    path, recovery_report, recovery_provenance = map(Path, (path, recovery_report, recovery_provenance))
    report = json.loads(recovery_report.read_text())
    record = json.loads(recovery_provenance.read_text())
    assert record["status"] == "finished" and record["exit_code"] == 0 and record["finished_utc"]
    assert record["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert record["environment"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    assert "--cpu-only" not in record["argv"]
    assert "models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_boundary_recovery" in record["argv"]
    for flag, expected in (
        ("--output", recovery_report),
        ("--prefix-state-cache", path),
        ("--source-cache", source_path),
    ):
        actual = Path(record["cwd"]) / record["argv"][record["argv"].index(flag) + 1]
        assert actual.resolve() == Path(expected).resolve(), flag
    log = Path(record["cwd"]) / record["log"]
    assert file_identity(log)["sha256"] == record["log_sha256"]
    assert file_identity(path) == report["recovery_state_export"], "Recovery export identity changed"
    payload = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    runtime = runtime_source_hashes()
    validate_boundary_evidence(
        payload, report, source_key=source_key, prefix=prefix, capacity=capacity, runtime_hashes=runtime
    )
    assert {
        key.removeprefix("tt/"): value for key, value in record["source_sha256"].items() if key.startswith("tt/")
    } == runtime
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "tests/probe_qsa_boundary_recovery.py",
        "tests/probe_qsa.py",
        "tests/probe_qsa_prefill.py",
        "tests/probe_linear_contract.py",
        "tests/qsa_state_fixture.py",
    ):
        assert file_identity(root / relative)["sha256"] == record["source_sha256"][relative], relative
    bindings = payload["bindings"]
    for identity in bindings.values():
        assert file_identity(identity["path"]) == identity, identity["path"]
    for flag, name in (
        ("--full-report", "full_report"),
        ("--full-provenance", "full_provenance"),
        ("--full-output", "full_output"),
    ):
        actual = Path(record["cwd"]) / record["argv"][record["argv"].index(flag) + 1]
        assert actual.resolve() == Path(bindings[name]["path"]), flag
    reference_path = Path(record["cwd"]) / record["argv"][record["argv"].index("--reference-cache") + 1]
    assert file_identity(reference_path) == report["reference_cache"]
    assert file_identity(source_path) == bindings["source_fixture"]
    assert Path(payload["source_fixture_path"]).resolve() == Path(source_path).resolve()
    assert bindings["recovery_harness"]["sha256"] == record["source_sha256"]["tests/probe_qsa_boundary_recovery.py"]
    # Re-run the original failed-run identity/source/page-permutation checks.
    source, saved, full, old_runtime, checked_bindings = bind_inputs(
        SimpleNamespace(
            full_provenance=Path(bindings["full_provenance"]["path"]),
            full_report=Path(bindings["full_report"]["path"]),
            full_output=Path(bindings["full_output"]["path"]),
            source_cache=Path(source_path),
            snapshot=Path(source_key["snapshot"]),
        )
    )
    assert checked_bindings == bindings and old_runtime == runtime
    assert source["key"] == source_key and full["prefill"] == payload["original_full_prefill"]
    tensors = payload["tensors"]
    assert set(tensors) == set(STATE_TENSORS)
    specifications = {
        "position": ((1,), torch.int32),
        "page_table": ((1, 8192), torch.int32),
        "k_cache": ((8192, 2, 32, 256), torch.bfloat16),
        "v_cache": ((8192, 2, 32, 256), torch.bfloat16),
        "raw_history": ((1, 1, 3, 128), torch.bfloat16),
        "compressed_keys": ((1, 1, 65536, 128), torch.bfloat16),
        "block_numbers": ((1, 1, 1, 65536), torch.float32),
        "token_numbers": ((1, 1, 1, CAPACITY), torch.float32),
    }
    for name, (shape, dtype) in specifications.items():
        assert tuple(tensors[name].shape) == shape and tensors[name].dtype == dtype, name
        assert torch.isfinite(tensors[name]).all(), name
    assert int(tensors["position"].item()) == PREFIX
    assert torch.equal(tensors["page_table"], saved["page_table"])
    for name, count in (("block_numbers", 65536), ("token_numbers", CAPACITY)):
        assert torch.equal(tensors[name].flatten(), torch.arange(count, dtype=torch.float32)), name
    payload["verified_recovery"] = {
        "report": file_identity(recovery_report),
        "provenance": file_identity(recovery_provenance),
        "log": file_identity(log),
        "export": report["recovery_state_export"],
        "strict_full_prefill_passed": False,
        "all_output_gate": False,
        "boundary_state_passed": True,
    }
    return payload
