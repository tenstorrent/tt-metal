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
from models.demos.deepseek_v4_flash.cpu_reference import swiglu_expert
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex
from models.demos.deepseek_v4_flash.real_expert_smoke import decode_real_expert_weights
from models.demos.deepseek_v4_flash.real_routed_moe_smoke import (
    build_torch_selected_expert_reference,
    deterministic_input_ids_for_expert,
    layer_routed_moe_keys,
    load_real_routed_moe_slice,
    run_real_routed_moe_smoke,
    selected_expert_route,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_routed_moe_selector_uses_norm_router_and_one_expert(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_routed_moe_keys(index, layer=3, expert=0)

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
    ]


def test_cpu_real_routed_moe_smoke_selects_and_references_one_expert(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_real_routed_moe_smoke(snapshot, layer=3, expert=0, seq_len=4, max_bytes=4096, cpu_only=True)

    loaded_keys = [item["canonical_key"] for item in result["loaded_tensors"]]
    assert loaded_keys == [
        "layers.3.ffn_norm.weight",
        "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",
        "layers.3.ffn.experts.0.w1.weight",
        "layers.3.ffn.experts.0.w1.scale",
        "layers.3.ffn.experts.0.w2.weight",
        "layers.3.ffn.experts.0.w2.scale",
        "layers.3.ffn.experts.0.w3.weight",
        "layers.3.ffn.experts.0.w3.scale",
    ]
    assert result["mode"] == "cpu-reference"
    assert result["payload_bytes"] == {"expert": 1920, "norm": 128, "router": 272, "total": 2320}
    assert result["reference"]["rms_norm"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["router_weights"]["shape"] == [1, 4, 2]
    assert result["reference"]["selected_route"]["expert"] == 0
    assert result["reference"]["selected_route"]["selected_token_count"] > 0
    assert result["reference"]["selected_expert_output"]["shape"][-1] == 32
    assert result["host_boundaries"][1]["name"] == "selected_expert_gather"
    assert result["passed"] is True


def test_cpu_real_routed_moe_smoke_hash_layer_constructs_selected_input_ids(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_routed_moe_keys(index, layer=0, expert=3)
    assert keys[2] == "layers.0.ffn.gate.tid2eid"

    result = run_real_routed_moe_smoke(snapshot, layer=0, expert=3, seq_len=4, max_bytes=8192, cpu_only=True)

    assert result["inputs"]["input_ids"]["shape"] == [1, 4]
    assert result["reference"]["selected_route"]["selected_token_count"] == 4
    assert result["reference"]["selected_route"]["topk_slot_histogram"] == {"1": 4}


def test_selected_expert_route_sums_matching_topk_slots_and_rejects_absent_expert() -> None:
    route_weights = torch.tensor([[[0.2, 0.8], [0.4, 0.6], [0.5, 0.25]]], dtype=torch.float32)
    route_indices = torch.tensor([[[2, 0], [0, 2], [2, 2]]], dtype=torch.int64)

    selection = selected_expert_route(route_weights, route_indices, expert=2)

    torch.testing.assert_close(selection.token_indices, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(selection.route_weight.float(), torch.tensor([[[0.2], [0.6], [0.75]]]))
    assert selection.hit_count == 4
    assert selection.topk_slot_histogram == {0: 2, 1: 2}

    with pytest.raises(ValueError, match="No tokens selected expert 3"):
        selected_expert_route(route_weights, route_indices, expert=3)


def test_torch_selected_expert_reference_matches_manual_scatter(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    tensors, _ = load_real_routed_moe_slice(snapshot, layer=3, expert=0, max_bytes=4096)
    weights = decode_real_expert_weights(tensors, config=config, layer=3, expert=0)
    hidden_states = torch.linspace(-0.2, 0.3, steps=3 * config.hidden_size, dtype=torch.float32)
    hidden_states = hidden_states.reshape(1, 1, 3, config.hidden_size).to(torch.bfloat16)
    route_weights = torch.tensor([[[0.2, 0.8], [0.4, 0.6], [0.5, 0.25]]], dtype=torch.float32)
    route_indices = torch.tensor([[[2, 0], [0, 2], [1, 2]]], dtype=torch.int64)
    selection = selected_expert_route(route_weights, route_indices, expert=0)

    reference = build_torch_selected_expert_reference(
        weights,
        config=config,
        hidden_states=hidden_states,
        selected_route=selection,
    )
    manual_selected = swiglu_expert(
        hidden_states[:, 0, selection.token_indices].reshape(-1, config.hidden_size),
        weights["w1"],
        weights["w2"],
        weights["w3"],
        route_weight=selection.route_weight.reshape(-1, 1),
        swiglu_limit=config.swiglu_limit,
    ).reshape(1, 1, selection.selected_token_count, config.hidden_size)
    manual_scattered = torch.zeros_like(hidden_states)
    manual_scattered[:, :, selection.token_indices, :] = manual_selected

    torch.testing.assert_close(reference["selected_expert_output"], manual_selected)
    torch.testing.assert_close(reference["expert_scattered_output"], manual_scattered)


def test_deterministic_input_ids_for_expert_reports_missing_hash_route() -> None:
    tid2eid = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int32)

    input_ids = deterministic_input_ids_for_expert(tid2eid=tid2eid, expert=2, seq_len=5)

    assert input_ids.tolist() == [[1, 2, 1, 2, 1]]
    with pytest.raises(ValueError, match="No input_ids in tid2eid route to expert 3"):
        deterministic_input_ids_for_expert(tid2eid=tid2eid, expert=3, seq_len=1)


def test_cpu_real_routed_moe_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_routed_moe_smoke(snapshot, layer=3, expert=0, seq_len=4, max_tensors=8, max_bytes=4096, cpu_only=True)
    with pytest.raises(ValueError, match="byte budget"):
        run_real_routed_moe_smoke(snapshot, layer=3, expert=0, seq_len=4, max_tensors=9, max_bytes=2319, cpu_only=True)


def test_cpu_real_routed_moe_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_routed_moe_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--expert",
            "0",
            "--seq-len",
            "4",
            "--max-bytes",
            "4096",
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
    assert payload["payload_bytes"]["total"] == 2320
    assert payload["loaded_tensors"][0]["canonical_key"] == "layers.3.ffn_norm.weight"
    assert payload["host_boundaries"][-1]["name"] == "selected_expert_scatter"


def test_real_routed_moe_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_ROUTED_MOE_SMOKE", "0") == "1"
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

    result = run_real_routed_moe_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_ROUTED_MOE_LAYER", "3")),
        expert=int(os.environ.get("DSV4_FLASH_REAL_ROUTED_MOE_EXPERT", "0")),
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
