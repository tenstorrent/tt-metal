# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Protect the handoff: exporting videos alone must never make the CI quality gate green."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]


def test_fast_ltx_ci_keeps_full_quality_gate():
    entries = yaml.safe_load((ROOT / "tests/pipeline_reorg/models_e2e_tests.yaml").read_text())
    leg = next(entry for entry in entries if entry.get("model") == "ltx-2.3-fast-t2v")
    assert "RUN_VBENCH=1" in leg["cmd"]
    assert "VBENCH_SEEDS=5" in leg["cmd"]
    assert "RUN_VBENCH=0" not in leg["cmd"]
    workflow = yaml.safe_load((ROOT / ".github/workflows/models-e2e-tests-multihost-impl.yaml").read_text())
    jobs = workflow["jobs"]
    assert "models-e2e-tests-multihost" in jobs["vbench-score"]["needs"]
    assert "vbench-score" in jobs["vbench-gate"]["needs"]
    assert "continue-on-error" not in jobs["vbench-score"]
    assert "continue-on-error" not in jobs["vbench-gate"]
    assert "aggregate" in jobs["vbench-gate"]["steps"][-1]["run"]
