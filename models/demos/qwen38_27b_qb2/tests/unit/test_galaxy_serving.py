# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete qualification and accidental worker chip overlap."""

import json

import pytest

from models.demos.qwen38_27b_qb2.demo.galaxy_serving import (
    additional_config,
    model_source_hashes,
    qualified_groups,
    qualified_runtime_environment,
    server_command,
    verify_qualified_source,
    verify_worker_bindings,
)


def qualification():
    return dict(
        passed=True,
        state="completed",
        replicas_requested=8,
        replicas_executed=8,
        replica_mesh=[1, 4],
        topology="linear",
        comparisons=[dict(replica=i, ratio=1.01) for i in range(8)],
        loaded=[dict(replica=i, device_ids=list(range(i * 4, i * 4 + 4))) for i in range(8)],
    )


@pytest.mark.parametrize("failure", ["partial", "failed", "slow", "overlap", "missing", "nan"])
def test_invalid_qualification_cannot_authorize_serving(failure, expect_error):
    receipt = qualification()
    if failure == "partial":
        receipt["replicas_executed"] = 7
    elif failure == "failed":
        receipt["passed"] = False
    elif failure == "slow":
        receipt["comparisons"][0]["ratio"] = 1.04
    elif failure == "overlap":
        receipt["loaded"][1]["device_ids"] = receipt["loaded"][0]["device_ids"]
    elif failure == "missing":
        receipt["loaded"].pop()
    else:
        receipt["comparisons"][0]["ratio"] = float("nan")
    with expect_error(ValueError, "[Qq]ualification"):
        qualified_groups(receipt)


def test_launch_preserves_qualified_groups_and_independent_engine_capacity():
    groups = qualified_groups(qualification())
    command = server_command("/task", "/weights", groups)
    assert command[command.index("--data-parallel-size") + 1] == "8"
    assert command[command.index("--max-num-seqs") + 1] == "16"
    assert "--no-enable-prefix-caching" in command and "--no-enable-chunked-prefill" in command
    config = json.loads(command[command.index("--additional-config") + 1])
    assert config == additional_config(groups)
    assert config["_tt_standard_dp_visible_groups"] == groups
    assert config["_tt_standard_dp_mesh_grids"] == {group: [1, 4] for group in groups}
    assert config["tt"]["fabric_config"] == "FABRIC_1D"


def test_every_worker_must_match_its_qualified_group(expect_error):
    groups = qualified_groups(qualification())
    rows = [
        f"TT worker standard-DP binding: data_parallel_index={rank} "
        f"data_parallel_rank_local={rank} TT_VISIBLE_DEVICES={group} MESH_DEVICE=(8, 4)"
        for rank, group in enumerate(groups)
    ]
    assert verify_worker_bindings("\n".join(rows), groups) == dict(enumerate(groups))
    with expect_error(ValueError, "all eight"):
        verify_worker_bindings("\n".join(rows[:-1]), groups)
    with expect_error(ValueError, "differs"):
        verify_worker_bindings(
            "\n".join(rows).replace("TT_VISIBLE_DEVICES=0,1,2,3", "TT_VISIBLE_DEVICES=4,5,6,7"), groups
        )


def test_changed_precision_or_model_source_requires_new_qualification(tmp_path, expect_error):
    (tmp_path / "tt").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "tt/model.py").write_text("# synthetic model source\n")
    config = tmp_path / "config/precision.json"
    config.write_text('{"state": "float32"}\n')
    receipt = dict(source_sha256=model_source_hashes(tmp_path))
    assert verify_qualified_source(receipt, tmp_path) == receipt["source_sha256"]
    config.write_text('{"state": "bfloat16"}\n')
    with expect_error(ValueError, "differs"):
        verify_qualified_source(receipt, tmp_path)


def test_precision_environment_override_is_bound_to_qualification(tmp_path, monkeypatch, expect_error):
    (tmp_path / "tt").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "tt/model.py").write_text("# synthetic source\n")
    (tmp_path / "config/precision.json").write_text('{"policy": "native"}\n')
    override = tmp_path / "candidate.json"
    override.write_text('{"policy": "accurate"}\n')
    monkeypatch.delenv("QWEN_PRECISION_CONFIG", raising=False)
    default_receipt = dict(source_sha256=model_source_hashes(tmp_path))
    monkeypatch.setenv("QWEN_PRECISION_CONFIG", str(override))
    with expect_error(ValueError, "differs"):
        verify_qualified_source(default_receipt, tmp_path)
    receipt = dict(source_sha256=model_source_hashes(tmp_path))
    assert verify_qualified_source(receipt, tmp_path)
    override.write_text('{"policy": "changed"}\n')
    with expect_error(ValueError, "differs"):
        verify_qualified_source(receipt, tmp_path)
    monkeypatch.setenv("QWEN_PRECISION_CONFIG", "baseline")
    with expect_error(ValueError, "differs"):
        verify_qualified_source(receipt, tmp_path)
    baseline_receipt = dict(source_sha256=model_source_hashes(tmp_path))
    assert verify_qualified_source(baseline_receipt, tmp_path)


def test_qualified_override_reaches_server_workers(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_PRECISION_CONFIG", raising=False)
    assert "QWEN_PRECISION_CONFIG" not in qualified_runtime_environment()
    candidate = tmp_path / "candidate.json"
    monkeypatch.setenv("QWEN_PRECISION_CONFIG", str(candidate))
    assert qualified_runtime_environment()["QWEN_PRECISION_CONFIG"] == str(candidate.resolve())
    monkeypatch.setenv("QWEN_PRECISION_CONFIG", "baseline")
    assert qualified_runtime_environment()["QWEN_PRECISION_CONFIG"] == "baseline"


def test_changed_nested_kernel_requires_new_qualification(tmp_path, expect_error):
    (tmp_path / "tt/gdn_step").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/precision.json").write_text('{"policy": "single_step"}\n')
    kernel = tmp_path / "tt/gdn_step/compute.cpp"
    kernel.write_text("// first kernel\n")
    receipt = dict(source_sha256=model_source_hashes(tmp_path))
    assert "tt/gdn_step/compute.cpp" in receipt["source_sha256"]
    assert verify_qualified_source(receipt, tmp_path)
    kernel.write_text("// different kernel\n")
    with expect_error(ValueError, "differs"):
        verify_qualified_source(receipt, tmp_path)
