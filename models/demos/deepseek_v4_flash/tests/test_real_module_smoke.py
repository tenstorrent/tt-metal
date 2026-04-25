# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from models.demos.deepseek_v4_flash.real_module_smoke import run_real_module_smoke
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_cpu_real_module_smoke_selects_router_norm_slice_and_reference(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_real_module_smoke(snapshot, layer=3, seq_len=4, max_bytes=2048, cpu_only=True)

    loaded_keys = [item["canonical_key"] for item in result["loaded_tensors"]]
    assert loaded_keys == [
        "layers.3.attn_norm.weight",
        "layers.3.ffn_norm.weight",
        "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",
    ]
    assert result["mode"] == "cpu-reference"
    assert result["payload_bytes"] == result["budget"]["selected_payload_bytes"]
    assert result["reference"]["rms_norm"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["router_weights"]["shape"] == [1, 4, 2]
    assert result["reference"]["router_indices"]["shape"] == [1, 4, 2]
    assert result["ttnn_ops"] == []
    assert result["passed"] is True


def test_cpu_real_module_smoke_refuses_budget_overrun(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    with pytest.raises(ValueError, match="byte budget"):
        run_real_module_smoke(snapshot, layer=3, seq_len=4, max_bytes=256, cpu_only=True)


def test_cpu_real_module_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_module_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--seq-len",
            "4",
            "--max-bytes",
            "2048",
            "--cpu-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["mode"] == "cpu-reference"
    assert payload["layer"] == 3
    assert payload["payload_bytes"] > 0
    assert payload["loaded_tensors"][2]["canonical_key"] == "layers.3.ffn.gate.weight"


def test_real_module_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_SMOKE", "0") == "1"
    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        if required:
            pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")
        pytest.skip(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    available, reason = _available_ttnn_devices()
    if available < 1:
        if required:
            pytest.fail(reason)
        pytest.skip(reason)

    result = run_real_module_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_SMOKE_LAYER", "3")),
        seq_len=32,
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)


def _available_ttnn_devices() -> tuple[int, str]:
    try:
        import ttnn
    except Exception as exc:
        return 0, f"Unable to import ttnn: {exc}"

    try:
        return int(ttnn.GetNumAvailableDevices()), "No TTNN devices available"
    except Exception as exc:
        return 0, f"Unable to query TTNN devices: {exc}"
