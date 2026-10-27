# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Protect the required gate and the boundary between model policy and CI plumbing."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]


def ltx_leg():
    entries = yaml.safe_load((ROOT / "tests/pipeline_reorg/models_e2e_tests.yaml").read_text())
    return next(entry for entry in entries if entry.get("model") == "ltx-2.3-fast-t2v")


def test_fast_ltx_ci_keeps_all_seeds_and_required_quality_gate():
    leg = ltx_leg()
    assert "RUN_VBENCH=1" in leg["cmd"]
    assert "VBENCH_SEEDS=5" in leg["cmd"]
    assert "RUN_VBENCH=0" not in leg["cmd"]
    assert "VBENCH_TEMPORAL_WIDTH=960" in leg["cmd"]
    assert "postprocess" not in leg
    assert leg["skus"]["bh_sc1"]["timeout"] == 35
    cmd = leg["cmd"]
    assert cmd.index("python3 -m pytest") < cmd.index("vbench_bundle evaluate")
    assert "|| exit $?" in cmd[cmd.index("python3 -m pytest") : cmd.index("vbench_bundle evaluate")]
    assert "mpirun" in cmd[cmd.index("|| exit $?") :]


def test_shared_workflows_have_no_ltx_policy():
    for filename in (
        "models-e2e-tests-multihost-impl.yaml",
        "models-e2e-tests-impl.yaml",
    ):
        text = (ROOT / ".github/workflows" / filename).read_text().lower()
        assert "ltx" not in text
        assert "vbench" not in text


def test_lora_staging_runs_on_a_worker_before_offline_pytest():
    entries = yaml.safe_load((ROOT / "tests/pipeline_reorg/models_unit_tests.yaml").read_text())
    cmd = next(entry["cmd"] for entry in entries if entry.get("name") == "TT-DiT LTX-2.3 component unit tests")
    preflight = cmd[cmd.index("LORA_PATH=$(") : cmd.index("export LORA_PATH")]
    assert "mpirun -np 1" in preflight
    assert '--wdir "${TT_METAL_HOME}"' in preflight
    assert "env HF_HUB_OFFLINE=0 python3 -m models.tt_dit.utils.ltx_lora_asset" in preflight
    assert "--tag-output" not in preflight  # Captured stdout must remain a plain pathname.
    assert cmd.index("export HF_HUB_OFFLINE=1") < cmd.index("LORA_PATH=$(") < cmd.index("python3 -m pytest")
