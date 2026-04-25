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
from models.demos.deepseek_v4_flash.fp4 import FP4_E2M1_TABLE
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex, layer_expert_mlp_keys
from models.demos.deepseek_v4_flash.real_expert_smoke import (
    build_torch_expert_reference,
    decode_real_expert_weights,
    deterministic_expert_activation,
    deterministic_route_weight,
    load_real_expert_slice,
    run_real_expert_smoke,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_expert_mlp_selector_uses_explicit_projection_pairs(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_expert_mlp_keys(index, layer=3, expert=0)
    metadata = {item.canonical_key: item for item in index.metadata_for_keys(keys)}

    assert keys == [
        "layers.3.ffn.experts.0.w1.weight",
        "layers.3.ffn.experts.0.w1.scale",
        "layers.3.ffn.experts.0.w2.weight",
        "layers.3.ffn.experts.0.w2.scale",
        "layers.3.ffn.experts.0.w3.weight",
        "layers.3.ffn.experts.0.w3.scale",
    ]
    assert metadata["layers.3.ffn.experts.0.w1.weight"].source_key == "layers.3.ffn.experts.0.w1.weight"
    assert metadata["layers.3.ffn.experts.0.w1.weight"].dtype == "U8"
    assert metadata["layers.3.ffn.experts.0.w1.weight"].shape == (32, 16)
    assert metadata["layers.3.ffn.experts.0.w1.scale"].shape == (32, 1)


def test_cpu_real_expert_smoke_selects_decodes_and_references_one_expert(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_real_expert_smoke(snapshot, layer=3, expert=0, seq_len=4, max_bytes=4096, cpu_only=True)

    loaded_keys = [item["canonical_key"] for item in result["loaded_tensors"]]
    assert loaded_keys == [
        "layers.3.ffn.experts.0.w1.weight",
        "layers.3.ffn.experts.0.w1.scale",
        "layers.3.ffn.experts.0.w2.weight",
        "layers.3.ffn.experts.0.w2.scale",
        "layers.3.ffn.experts.0.w3.weight",
        "layers.3.ffn.experts.0.w3.scale",
    ]
    assert result["selected_source_keys"] == loaded_keys
    assert result["mode"] == "cpu-reference"
    assert result["payload_bytes"] == 1920
    assert result["decoded_tensors"]["w1"]["shape"] == [32, 32]
    assert result["decoded_tensors"]["w2"]["shape"] == [32, 32]
    assert result["decoded_tensors"]["w3"]["shape"] == [32, 32]
    assert result["reference"]["output"]["shape"] == [1, 1, 4, 32]
    assert result["ttnn_ops"] == []
    assert result["passed"] is True


def test_cpu_real_expert_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_expert_smoke(snapshot, layer=3, expert=0, seq_len=4, max_tensors=5, max_bytes=4096, cpu_only=True)
    with pytest.raises(ValueError, match="byte budget"):
        run_real_expert_smoke(snapshot, layer=3, expert=0, seq_len=4, max_tensors=6, max_bytes=1919, cpu_only=True)


def test_real_expert_decode_and_torch_reference_match_fp4_fixture(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    tensors, _ = load_real_expert_slice(snapshot, layer=3, expert=0, max_bytes=4096)
    weights = decode_real_expert_weights(tensors, config=config, layer=3, expert=0)

    expected_first_row = FP4_E2M1_TABLE[torch.arange(32) % 16].to(torch.bfloat16)
    torch.testing.assert_close(weights["w1"][0], expected_first_row)

    activation = deterministic_expert_activation(hidden_size=config.hidden_size, seq_len=4)
    route_weight = deterministic_route_weight(seq_len=4)
    reference = build_torch_expert_reference(
        weights,
        config=config,
        activation=activation,
        route_weight=route_weight,
    )
    manual = swiglu_expert(
        activation[:, 0].reshape(-1, config.hidden_size),
        weights["w1"],
        weights["w2"],
        weights["w3"],
        route_weight=route_weight.reshape(-1, 1),
        swiglu_limit=config.swiglu_limit,
    ).reshape(1, 4, config.hidden_size)

    torch.testing.assert_close(reference[:, 0], manual)


def test_cpu_real_expert_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_expert_smoke",
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
    assert payload["payload_bytes"] == 1920
    assert payload["decoded_tensors"]["w1"]["dtype"] == "torch.bfloat16"


def test_real_expert_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_EXPERT_SMOKE", "0") == "1"
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

    result = run_real_expert_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_EXPERT_LAYER", "3")),
        expert=int(os.environ.get("DSV4_FLASH_REAL_EXPERT_ID", "0")),
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
