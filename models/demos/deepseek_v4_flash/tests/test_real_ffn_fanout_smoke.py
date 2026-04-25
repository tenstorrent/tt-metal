# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
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
from models.demos.deepseek_v4_flash.real_ffn_fanout_smoke import (
    _ordered_activated_expert_ids,
    build_torch_ffn_fanout_reference,
    build_torch_ffn_fanout_selector_reference,
    layer_ffn_fanout_keys,
    layer_ffn_selector_keys,
    run_real_ffn_fanout_smoke,
)
from models.demos.deepseek_v4_flash.real_routed_moe_smoke import deterministic_routed_activation
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import decode_real_shared_expert_weights
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_ffn_fanout_selector_loads_all_requested_experts_and_shared(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_ffn_fanout_keys(index, layer=3, experts=[2, 0])

    assert keys[:3] == [
        "layers.3.ffn_norm.weight",
        "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",
    ]
    assert "layers.3.ffn.experts.2.w1.weight" in keys
    assert "layers.3.ffn.experts.0.w3.scale" in keys
    assert keys[-6:] == [
        "layers.3.ffn.shared_experts.w1.weight",
        "layers.3.ffn.shared_experts.w1.scale",
        "layers.3.ffn.shared_experts.w2.weight",
        "layers.3.ffn.shared_experts.w2.scale",
        "layers.3.ffn.shared_experts.w3.weight",
        "layers.3.ffn.shared_experts.w3.scale",
    ]
    assert len(keys) == len(set(keys))


def test_cpu_real_ffn_fanout_smoke_loads_full_topk_and_combines(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_real_ffn_fanout_smoke(snapshot, layer=3, seq_len=1, anchor_expert=0, max_bytes=16384, cpu_only=True)

    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["decode_tokens"] == 1
    assert result["fanout_scope"]["topk"] == 2
    assert result["fanout_scope"]["activated_expert_count"] == 2
    assert result["reference"]["activated_experts"]["topk_is_full"] is True
    assert result["payload_bytes"] == {
        "norm": 128,
        "router": 272,
        "routed_experts": 3840,
        "routed_experts_by_id": {"0": 1920, "1": 1920},
        "shared_expert": 6156,
        "total": 10396,
    }
    assert result["reference"]["routed_output"]["shape"] == [1, 1, 1, 32]
    assert result["reference"]["shared_output"]["shape"] == [1, 1, 1, 32]
    assert result["reference"]["combined_output"]["shape"] == [1, 1, 1, 32]
    assert result["reference"]["residual_output"]["shape"] == [1, 1, 1, 32]
    assert "sum(route_weight_i" in result["residual_semantics"]
    assert result["performance_boundaries"][0]["name"] == "sequential_expert_execution"


def test_torch_ffn_fanout_reference_matches_generic_combine(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)
    selector_tensors, _ = index.load_tensors(layer_ffn_selector_keys(index, layer=3), max_bytes=4096)
    activation = deterministic_routed_activation(
        hidden_size=config.hidden_size,
        seq_len=1,
        gate_weight=selector_tensors["layers.3.ffn.gate.weight"],
        expert=0,
    )
    selector = build_torch_ffn_fanout_selector_reference(
        selector_tensors,
        config=config,
        layer=3,
        activation=activation,
        input_ids=None,
    )
    experts = _ordered_activated_expert_ids(selector["router_indices"])
    tensors, _ = index.load_tensors(
        layer_ffn_fanout_keys(index, layer=3, experts=experts),
        max_tensors=64,
        max_bytes=16384,
    )
    routed_weights_by_expert = {
        expert: decode_real_expert_weights(tensors, config=config, layer=3, expert=expert) for expert in experts
    }
    shared_weights = decode_real_shared_expert_weights(tensors, config=config, layer=3)

    reference = build_torch_ffn_fanout_reference(
        tensors,
        routed_weights_by_expert,
        shared_weights,
        config=config,
        layer=3,
        activation=activation,
    )

    torch.testing.assert_close(reference["routed_output"], reference["manual_routed_output"])
    expected_combined = (reference["routed_output"].float() + reference["shared_output"].float()).to(torch.bfloat16)
    expected_residual = (activation.float() + expected_combined.float()).to(torch.bfloat16)
    torch.testing.assert_close(reference["combined_output"], expected_combined)
    torch.testing.assert_close(reference["residual_output"], expected_residual)


def test_cpu_real_ffn_fanout_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_ffn_fanout_smoke(
            snapshot,
            layer=3,
            seq_len=1,
            anchor_expert=0,
            max_tensors=20,
            max_bytes=16384,
            cpu_only=True,
        )
    with pytest.raises(ValueError, match="byte budget"):
        run_real_ffn_fanout_smoke(
            snapshot,
            layer=3,
            seq_len=1,
            anchor_expert=0,
            max_tensors=64,
            max_bytes=10395,
            cpu_only=True,
        )


def test_cpu_real_ffn_fanout_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_ffn_fanout_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--seq-len",
            "1",
            "--anchor-expert",
            "0",
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
    assert payload["fanout_scope"]["activated_expert_count"] == 2
    assert payload["fanout_scope"]["topk_prefix_limit"] is None
    assert payload["payload_bytes"]["total"] == 10396
    assert payload["payload_bytes"]["routed_experts"] == 3840
    assert payload["reference"]["activated_experts"]["topk_is_full"] is True
    assert payload["ttnn_ops"] == []
    assert payload["host_boundaries"][-2]["name"] == "ffn_host_combine"
    assert payload["host_boundaries"][-1]["name"] == "residual_host_add"


def test_real_ffn_fanout_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_FFN_FANOUT_SMOKE", "0") == "1"
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

    result = run_real_ffn_fanout_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_FFN_FANOUT_LAYER", "3")),
        seq_len=1,
        anchor_expert=int(os.environ.get("DSV4_FLASH_REAL_FFN_FANOUT_ANCHOR_EXPERT", "1")),
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)
    assert result["fanout_scope"]["activated_expert_count"] == result["model"]["num_experts_per_tok"]
    assert result["ttnn"]["experts_executed"] == result["model"]["num_experts_per_tok"]


def _available_ttnn_devices() -> tuple[int, str]:
    try:
        import ttnn
    except Exception as exc:
        return 0, f"Unable to import ttnn: {exc}"

    try:
        return int(ttnn.GetNumAvailableDevices()), "No TTNN devices available"
    except Exception as exc:
        return 0, f"Unable to query TTNN devices: {exc}"
