# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Protect the required gate and the boundary between model policy and CI plumbing."""

import runpy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
PREPARE = runpy.run_path(str(ROOT / ".github/scripts/utils/prepare_postprocess_matrix.py"))["prepare"]


def ltx_leg():
    entries = yaml.safe_load((ROOT / "tests/pipeline_reorg/models_e2e_tests.yaml").read_text())
    return next(entry for entry in entries if entry.get("model") == "ltx-2.3-fast-t2v")


def test_fast_ltx_ci_keeps_full_quality_gate():
    leg = ltx_leg()
    assert "RUN_VBENCH=1" in leg["cmd"]
    assert "VBENCH_SEEDS=5" in leg["cmd"]
    assert "RUN_VBENCH=0" not in leg["cmd"]
    assert leg["postprocess"]["shards"] == 5
    assert "vbench_bundle score" in leg["postprocess"]["cmd"]
    assert "vbench_bundle aggregate" in leg["postprocess"]["aggregate_cmd"]
    parent = yaml.safe_load((ROOT / ".github/workflows/models-e2e-tests-impl.yaml").read_text())["jobs"]
    hook = parent["models-e2e-postprocess"]
    assert "models-e2e-tests-multihost" in hook["needs"]
    assert hook["uses"] == "./.github/workflows/models-e2e-postprocess.yaml"
    assert "!cancelled()" in hook["if"]
    assert not hook.get("continue-on-error", False)
    jobs = yaml.safe_load((ROOT / ".github/workflows/models-e2e-postprocess.yaml").read_text())["jobs"]
    assert "check" in jobs["aggregate"]["needs"]
    assert "!cancelled()" in jobs["aggregate"]["if"]
    for name in ("check", "aggregate"):
        assert not jobs[name].get("continue-on-error", False)
        assert all(not step.get("continue-on-error", False) for step in jobs[name]["steps"])
    assert "aggregate_cmd" in jobs["aggregate"]["steps"][-1]["run"]


def test_shared_workflows_have_no_ltx_policy():
    for filename in (
        "models-e2e-tests-multihost-impl.yaml",
        "models-e2e-tests-impl.yaml",
        "models-e2e-postprocess.yaml",
    ):
        text = (ROOT / ".github/workflows" / filename).read_text().lower()
        assert "ltx" not in text
        assert "vbench" not in text


def test_postprocess_keeps_producer_indices_and_ignores_unconfigured_models():
    leg = ltx_leg()
    assert PREPARE([{"name": "unrelated"}]) == ([], [])
    legs, shards = PREPARE([{"name": "unrelated"}, leg, {"name": "another"}, leg])
    assert [entry["index"] for entry in legs] == [1, 3]
    assert [(entry["index"], entry["shard"]) for entry in shards] == [(i, s) for i in (1, 3) for s in range(5)]


@pytest.mark.parametrize(
    "key,value", [("shards", 0), ("shards", 257), ("shards", True), ("timeout", 0), ("cmd", ""), ("aggregate_cmd", "")]
)
def test_invalid_postprocess_cannot_silently_skip_gate(key, value, expect_error):
    leg = ltx_leg()
    leg["postprocess"][key] = value
    with expect_error(ValueError, "postprocess"):
        PREPARE([leg])


def test_lora_staging_runs_on_a_worker_before_offline_pytest():
    entries = yaml.safe_load((ROOT / "tests/pipeline_reorg/models_unit_tests.yaml").read_text())
    cmd = next(entry["cmd"] for entry in entries if entry.get("name") == "TT-DiT LTX-2.3 component unit tests")
    preflight = cmd[cmd.index("LORA_PATH=$(") : cmd.index("export LORA_PATH")]
    assert "mpirun -np 1" in preflight
    assert '--wdir "${TT_METAL_HOME}"' in preflight
    assert "env HF_HUB_OFFLINE=0 python3 -m models.tt_dit.utils.ltx_lora_asset" in preflight
    assert "--tag-output" not in preflight  # Captured stdout must remain a plain pathname.
    assert cmd.index("export HF_HUB_OFFLINE=1") < cmd.index("LORA_PATH=$(") < cmd.index("python3 -m pytest")
