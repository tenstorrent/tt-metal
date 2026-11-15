# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Protect the ordering of the LTX component leg's offline LoRA staging."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]


def test_lora_staging_runs_on_a_worker_before_offline_pytest():
    entries = yaml.safe_load((ROOT / "tests/pipeline_reorg/models_unit_tests.yaml").read_text())
    cmd = next(entry["cmd"] for entry in entries if entry.get("name") == "TT-DiT LTX-2.3 component unit tests")
    preflight = cmd[cmd.index("LORA_PATH=$(") : cmd.index("export LORA_PATH")]
    assert "mpirun -np 1" in preflight
    assert '--wdir "${TT_METAL_HOME}"' in preflight
    assert "env HF_HUB_OFFLINE=0 python3 -m models.tt_dit.utils.ltx_lora_asset" in preflight
    assert "--tag-output" not in preflight  # Captured stdout must remain a plain pathname.
    assert cmd.index("export HF_HUB_OFFLINE=1") < cmd.index("LORA_PATH=$(") < cmd.index("python3 -m pytest")
