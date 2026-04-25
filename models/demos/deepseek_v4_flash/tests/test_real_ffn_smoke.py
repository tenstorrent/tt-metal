# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex
from models.demos.deepseek_v4_flash.real_expert_smoke import decode_real_expert_weights
from models.demos.deepseek_v4_flash.real_ffn_smoke import (
    build_torch_ffn_reference,
    layer_ffn_keys,
    load_real_ffn_slice,
    run_real_ffn_smoke,
)
from models.demos.deepseek_v4_flash.real_routed_moe_smoke import deterministic_routed_activation
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import decode_real_shared_expert_weights
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_ffn_selector_uses_norm_router_one_routed_expert_and_shared_expert(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_ffn_keys(index, layer=3, expert=0)

    assert keys == [
        "layers.3.ffn_norm.weight",
        "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",
        "layers.3.ffn.experts.0.w1.weight",
        "layers.3.ffn.experts.0.w1.scale",
        "layers.3.ffn.experts.0.w2.weight",
        "layers.3.ffn.experts.0.w2.scale",
        "layers.3.ffn.experts.0.w3.weight",
        "layers.3.ffn.experts.0.w3.scale",
        "layers.3.ffn.shared_experts.w1.weight",
        "layers.3.ffn.shared_experts.w1.scale",
        "layers.3.ffn.shared_experts.w2.weight",
        "layers.3.ffn.shared_experts.w2.scale",
        "layers.3.ffn.shared_experts.w3.weight",
        "layers.3.ffn.shared_experts.w3.scale",
    ]


def test_cpu_real_ffn_smoke_selects_decodes_and_references_one_layer(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_real_ffn_smoke(snapshot, layer=3, expert=0, seq_len=4, max_bytes=16384, cpu_only=True)

    loaded_keys = [item["canonical_key"] for item in result["loaded_tensors"]]
    assert loaded_keys[:3] == [
        "layers.3.ffn_norm.weight",
        "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",
    ]
    assert loaded_keys[-6:] == [
        "layers.3.ffn.shared_experts.w1.weight",
        "layers.3.ffn.shared_experts.w1.scale",
        "layers.3.ffn.shared_experts.w2.weight",
        "layers.3.ffn.shared_experts.w2.scale",
        "layers.3.ffn.shared_experts.w3.weight",
        "layers.3.ffn.shared_experts.w3.scale",
    ]
    assert result["mode"] == "cpu-reference"
    assert result["payload_bytes"] == {
        "norm": 128,
        "router": 272,
        "routed_expert": 1920,
        "shared_expert": 6156,
        "total": 8476,
    }
    assert result["selected_expert_token_count"] > 0
    assert result["reference"]["routed_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["shared_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["combined_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["residual_output"]["shape"] == [1, 1, 4, 32]
    assert "residual_output = activation +" in result["residual_semantics"]
    assert {boundary["name"] for boundary in result["host_boundaries"]} >= {
        "router_topk",
        "selected_expert_gather",
        "ffn_host_combine",
        "residual_host_add",
    }
    assert result["passed"] is True


def test_cpu_real_ffn_smoke_hash_layer_constructs_selected_input_ids(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_ffn_keys(index, layer=0, expert=3)
    assert keys[2] == "layers.0.ffn.gate.tid2eid"

    result = run_real_ffn_smoke(snapshot, layer=0, expert=3, seq_len=4, max_bytes=32768, cpu_only=True)

    assert result["inputs"]["input_ids"]["shape"] == [1, 4]
    assert result["reference"]["selected_route"]["selected_token_count"] == 4
    assert result["reference"]["selected_route"]["topk_slot_histogram"] == {"1": 4}


def test_cpu_real_ffn_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_ffn_smoke(snapshot, layer=3, expert=0, seq_len=4, max_tensors=14, max_bytes=16384, cpu_only=True)
    with pytest.raises(ValueError, match="byte budget"):
        run_real_ffn_smoke(snapshot, layer=3, expert=0, seq_len=4, max_tensors=15, max_bytes=8475, cpu_only=True)


def test_torch_ffn_reference_combines_routed_shared_and_residual(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    tensors, _ = load_real_ffn_slice(snapshot, layer=3, expert=0, max_bytes=16384)
    routed_weights = decode_real_expert_weights(tensors, config=config, layer=3, expert=0)
    shared_weights = decode_real_shared_expert_weights(tensors, config=config, layer=3)
    activation = deterministic_routed_activation(
        hidden_size=config.hidden_size,
        seq_len=4,
        gate_weight=tensors["layers.3.ffn.gate.weight"],
        expert=0,
    )

    reference = build_torch_ffn_reference(
        tensors,
        routed_weights,
        shared_weights,
        config=config,
        layer=3,
        expert=0,
        activation=activation,
    )

    expected_combined = (reference["routed_output"].float() + reference["shared_output"].float()).to(torch.bfloat16)
    expected_residual = (activation.float() + expected_combined.float()).to(torch.bfloat16)
    torch.testing.assert_close(reference["combined_output"], expected_combined)
    torch.testing.assert_close(reference["residual_output"], expected_residual)


def test_cpu_real_ffn_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_ffn_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--expert",
            "0",
            "--seq-len",
            "4",
            "--max-bytes",
            "16384",
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
    assert payload["expert"] == 0
    assert payload["payload_bytes"]["total"] == 8476
    assert payload["payload_bytes"]["shared_expert"] == 6156
    assert payload["reference"]["combined_output"]["shape"] == [1, 1, 4, 32]
    assert payload["reference"]["residual_output"]["shape"] == [1, 1, 4, 32]
    assert payload["ttnn_ops"] == []
    assert payload["host_boundaries"][-2]["name"] == "ffn_host_combine"
    assert payload["host_boundaries"][-1]["name"] == "residual_host_add"


def test_real_ffn_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_FFN_SMOKE", "0") == "1"
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

    result = run_real_ffn_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_FFN_LAYER", "3")),
        expert=int(os.environ.get("DSV4_FLASH_REAL_FFN_EXPERT", "0")),
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
