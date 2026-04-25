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
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    RealCheckpointTensorIndex,
    layer_shared_expert_mlp_keys,
)
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import (
    build_torch_shared_expert_reference,
    decode_fp8_block_scaled_weight,
    decode_real_shared_expert_weights,
    deterministic_shared_expert_activation,
    load_real_shared_expert_slice,
    run_real_shared_expert_smoke,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_shared_expert_selector_uses_explicit_projection_pairs(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_shared_expert_mlp_keys(index, layer=3)
    metadata = {item.canonical_key: item for item in index.metadata_for_keys(keys)}

    assert keys == [
        "layers.3.ffn.shared_experts.w1.weight",
        "layers.3.ffn.shared_experts.w1.scale",
        "layers.3.ffn.shared_experts.w2.weight",
        "layers.3.ffn.shared_experts.w2.scale",
        "layers.3.ffn.shared_experts.w3.weight",
        "layers.3.ffn.shared_experts.w3.scale",
    ]
    assert metadata["layers.3.ffn.shared_experts.w1.weight"].source_key == "layers.3.ffn.shared_experts.w1.weight"
    assert metadata["layers.3.ffn.shared_experts.w1.weight"].dtype == "BF16"
    assert metadata["layers.3.ffn.shared_experts.w1.weight"].shape == (32, 32)
    assert metadata["layers.3.ffn.shared_experts.w1.scale"].shape == (1, 1)


def test_cpu_real_shared_expert_smoke_selects_decodes_and_references_one_layer(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_real_shared_expert_smoke(snapshot, layer=3, seq_len=4, max_bytes=8192, cpu_only=True)

    loaded_keys = [item["canonical_key"] for item in result["loaded_tensors"]]
    assert loaded_keys == [
        "layers.3.ffn.shared_experts.w1.weight",
        "layers.3.ffn.shared_experts.w1.scale",
        "layers.3.ffn.shared_experts.w2.weight",
        "layers.3.ffn.shared_experts.w2.scale",
        "layers.3.ffn.shared_experts.w3.weight",
        "layers.3.ffn.shared_experts.w3.scale",
    ]
    assert result["selected_source_keys"] == loaded_keys
    assert result["mode"] == "cpu-reference"
    assert result["payload_bytes"] == {"scales": 12, "total": 6156, "weights": 6144}
    assert result["decoded_tensors"]["w1"]["shape"] == [32, 32]
    assert result["decoded_tensors"]["w2"]["shape"] == [32, 32]
    assert result["decoded_tensors"]["w3"]["shape"] == [32, 32]
    assert result["shared_expert_format"]["source_formats"]["w1"]["format"] == "DIRECT_FLOAT_WEIGHT_SCALE_IGNORED"
    assert result["reference"]["output"]["shape"] == [1, 1, 4, 32]
    assert result["host_boundaries"][0]["name"] == "fp8_decode_to_bf16"
    assert result["ttnn_ops"] == []
    assert result["passed"] is True


def test_cpu_real_shared_expert_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_shared_expert_smoke(snapshot, layer=3, seq_len=4, max_tensors=5, max_bytes=8192, cpu_only=True)
    with pytest.raises(ValueError, match="byte budget"):
        run_real_shared_expert_smoke(snapshot, layer=3, seq_len=4, max_tensors=6, max_bytes=6155, cpu_only=True)


def test_real_shared_expert_decode_and_torch_reference_match_direct_fixture(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    tensors, _ = load_real_shared_expert_slice(snapshot, layer=3, max_bytes=8192)
    weights = decode_real_shared_expert_weights(tensors, config=config, layer=3)

    assert weights["w1"].dtype == torch.bfloat16
    torch.testing.assert_close(weights["w1"], tensors["layers.3.ffn.shared_experts.w1.weight"])

    activation = deterministic_shared_expert_activation(hidden_size=config.hidden_size, seq_len=4)
    reference = build_torch_shared_expert_reference(weights, config=config, activation=activation)
    manual = swiglu_expert(
        activation[:, 0].reshape(-1, config.hidden_size),
        weights["w1"],
        weights["w2"],
        weights["w3"],
        swiglu_limit=config.swiglu_limit,
    ).reshape(1, 4, config.hidden_size)

    torch.testing.assert_close(reference[:, 0], manual)


def test_fp8_block_scaled_decode_multiplies_ue8m0_scales() -> None:
    source = torch.tensor([[1.0, -2.0, 4.0, -8.0], [16.0, -32.0, 64.0, -128.0]], dtype=torch.float32)
    weight = source.to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 0.25]], dtype=torch.float8_e8m0fnu)

    decoded = decode_fp8_block_scaled_weight(weight, scale, block_size=(2, 2), dtype=torch.float32)
    expected = weight.float() * scale.float().repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

    torch.testing.assert_close(decoded, expected)


def test_real_shared_expert_decode_rejects_unexpected_weight_format(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    tensors: dict[str, torch.Tensor] = {}
    for projection in ("w1", "w2", "w3"):
        shape = (config.moe_intermediate_size, config.hidden_size)
        if projection == "w2":
            shape = (config.hidden_size, config.moe_intermediate_size)
        prefix = f"layers.3.ffn.shared_experts.{projection}"
        tensors[f"{prefix}.weight"] = torch.zeros(shape, dtype=torch.uint8)
        tensors[f"{prefix}.scale"] = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(TypeError, match="Unsupported shared expert format"):
        decode_real_shared_expert_weights(tensors, config=config, layer=3)


def test_cpu_real_shared_expert_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_shared_expert_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--seq-len",
            "4",
            "--max-bytes",
            "8192",
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
    assert payload["payload_bytes"]["total"] == 6156
    assert payload["decoded_tensors"]["w1"]["dtype"] == "torch.bfloat16"
    assert payload["shared_expert_format"]["source_formats"]["w1"]["format"] == "DIRECT_FLOAT_WEIGHT_SCALE_IGNORED"


def test_real_shared_expert_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_SHARED_EXPERT_SMOKE", "0") == "1"
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

    result = run_real_shared_expert_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_SHARED_EXPERT_LAYER", "3")),
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
