# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import rms_norm
from models.demos.deepseek_v4_flash.real_attention_projection_smoke import (
    ATTENTION_FP8_BLOCK_SIZE,
    build_torch_attention_projection_reference,
    decode_real_attention_projection_weights,
    deterministic_attention_activation,
    layer_attention_projection_keys,
    load_real_attention_projection_slice,
    run_real_attention_projection_smoke,
)
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_attention_projection_selector_uses_attn_norm_and_query_path(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_attention_projection_keys(index, layer=3)

    assert keys == [
        "layers.3.attn_norm.weight",
        "layers.3.attn.q_norm.weight",
        "layers.3.attn.wq_a.weight",
        "layers.3.attn.wq_b.weight",
    ]


def test_layer_attention_projection_selector_includes_fp8_scales(tmp_path: Path) -> None:
    snapshot = _make_tiny_fp8_attention_snapshot(tmp_path)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    keys = layer_attention_projection_keys(index, layer=3)
    metadata = {item.canonical_key: item for item in index.metadata_for_keys(keys)}

    assert keys == [
        "layers.3.attn_norm.weight",
        "layers.3.attn.q_norm.weight",
        "layers.3.attn.wq_a.weight",
        "layers.3.attn.wq_a.scale",
        "layers.3.attn.wq_b.weight",
        "layers.3.attn.wq_b.scale",
    ]
    assert metadata["layers.3.attn.wq_a.weight"].dtype == "F8_E4M3"
    assert metadata["layers.3.attn.wq_a.scale"].dtype == "F8_E8M0"
    assert metadata["layers.3.attn.wq_b.scale"].shape == (1, 1)


def test_cpu_real_attention_projection_smoke_selects_decodes_and_references_one_layer(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = run_real_attention_projection_smoke(snapshot, layer=3, seq_len=4, max_bytes=4096, cpu_only=True)

    loaded_keys = [item["canonical_key"] for item in result["loaded_tensors"]]
    assert loaded_keys == [
        "layers.3.attn_norm.weight",
        "layers.3.attn.q_norm.weight",
        "layers.3.attn.wq_a.weight",
        "layers.3.attn.wq_b.weight",
    ]
    assert result["selected_source_keys"] == loaded_keys
    assert result["mode"] == "cpu-reference"
    assert result["payload_bytes"] == {
        "attn_norm": 128,
        "q_norm": 64,
        "wq_a_weight": 1024,
        "wq_a_scale": 0,
        "wq_b_weight": 1024,
        "wq_b_scale": 0,
        "norms": 192,
        "q_low_rank": 1024,
        "q_output": 1024,
        "weights": 2048,
        "scales": 0,
        "total": 2240,
    }
    assert result["attention_projection_format"]["source_formats"]["wq_a"]["format"] == (
        "DIRECT_FLOAT_WEIGHT_SCALE_NOT_SELECTED"
    )
    assert result["attention_projection_format"]["decoded_tensors"]["wq_a"]["shape"] == [16, 32]
    assert result["attention_projection_format"]["decoded_tensors"]["wq_b"]["shape"] == [32, 16]
    assert result["reference"]["attn_norm_output"]["shape"] == [1, 1, 4, 32]
    assert result["reference"]["q_rank_norm"]["shape"] == [1, 1, 4, 16]
    assert result["reference"]["q_output"]["shape"] == [1, 1, 4, 32]
    assert result["projection_scope"]["not_full_attention"] is True
    assert result["projection_scope"]["output_projection"] == "excluded"
    assert {boundary["name"] for boundary in result["host_boundaries"]} >= {
        "attention_fp8_decode_to_bf16",
        "activation_host_to_device",
        "projection_output_readback",
    }
    assert result["ttnn_ops"] == []
    assert result["passed"] is True


def test_cpu_real_attention_projection_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_attention_projection_smoke(
            snapshot,
            layer=3,
            seq_len=4,
            max_tensors=3,
            max_bytes=4096,
            cpu_only=True,
        )
    with pytest.raises(ValueError, match="byte budget"):
        run_real_attention_projection_smoke(
            snapshot,
            layer=3,
            seq_len=4,
            max_tensors=4,
            max_bytes=2239,
            cpu_only=True,
        )


def test_attention_projection_decode_and_reference_match_manual_fixture(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    tensors, _ = load_real_attention_projection_slice(snapshot, layer=3, max_bytes=4096)
    weights = decode_real_attention_projection_weights(tensors, config=config, layer=3)
    activation = deterministic_attention_activation(hidden_size=config.hidden_size, seq_len=4)

    reference = build_torch_attention_projection_reference(
        tensors,
        weights,
        config=config,
        layer=3,
        activation=activation,
    )

    attn_norm = rms_norm(activation[:, 0], tensors["layers.3.attn_norm.weight"].to(torch.bfloat16), config.rms_norm_eps)
    q_rank = F.linear(attn_norm.float(), weights.wq_a.float()).to(torch.bfloat16)
    q_rank = rms_norm(q_rank, weights.q_norm, config.rms_norm_eps)
    q_output = F.linear(q_rank.float(), weights.wq_b.float()).unsqueeze(1)
    torch.testing.assert_close(reference["q_output"], q_output)


def test_fp8_attention_projection_decode_reports_source_formats(tmp_path: Path) -> None:
    snapshot = _make_tiny_fp8_attention_snapshot(tmp_path)

    result = run_real_attention_projection_smoke(snapshot, layer=3, seq_len=4, max_bytes=4096, cpu_only=True)

    assert result["payload_bytes"]["weights"] == 1024
    assert result["payload_bytes"]["scales"] == 2
    assert result["payload_bytes"]["total"] == 1218
    assert result["attention_projection_format"]["source_formats"]["wq_a"]["format"] == (
        "FP8_E4M3_WEIGHT_UE8M0_128x128_SCALE"
    )
    assert result["attention_projection_format"]["source_formats"]["wq_b"]["scale_dtype"] == ("torch.float8_e8m0fnu")
    assert result["attention_projection_format"]["decoded_tensors"]["wq_a"]["dtype"] == "torch.bfloat16"
    assert result["reference"]["q_output"]["shape"] == [1, 1, 4, 32]
    assert result["passed"] is True


def test_cpu_real_attention_projection_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_attention_projection_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
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
    assert payload["payload_bytes"]["total"] == 2240
    assert payload["projection_scope"]["path"] == "attention_norm -> wq_a -> q_norm -> wq_b"
    assert payload["reference"]["q_output"]["shape"] == [1, 1, 4, 32]
    assert payload["host_boundaries"][0]["name"] == "attention_fp8_decode_to_bf16"


def test_real_attention_projection_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_ATTENTION_PROJECTION_SMOKE", "0") == "1"
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

    result = run_real_attention_projection_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_ATTENTION_PROJECTION_LAYER", "3")),
        seq_len=32,
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)


def _make_tiny_fp8_attention_snapshot(tmp_path: Path) -> Path:
    snapshot = generate_tiny_hf_checkpoint(tmp_path / "hf", num_hidden_layers=4, num_routed_experts=4)
    shard_path = snapshot / "model-00001-of-00001.safetensors"
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key).contiguous() for key in handle.keys()}

    prefix = "layers.3.attn"
    for projection in ("wq_a", "wq_b"):
        weight_key = f"{prefix}.{projection}.weight"
        scale_key = f"{prefix}.{projection}.scale"
        weight = tensors[weight_key].float().to(torch.float8_e4m3fn)
        scale_shape = (
            math.ceil(weight.shape[0] / ATTENTION_FP8_BLOCK_SIZE[0]),
            math.ceil(weight.shape[1] / ATTENTION_FP8_BLOCK_SIZE[1]),
        )
        tensors[weight_key] = weight
        tensors[scale_key] = torch.ones(scale_shape, dtype=torch.float32).to(torch.float8_e8m0fnu)

    save_file(tensors, str(shard_path))
    index_path = snapshot / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    index["weight_map"] = {key: "model-00001-of-00001.safetensors" for key in sorted(tensors)}
    index["metadata"] = {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())}
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return snapshot


def _available_ttnn_devices() -> tuple[int, str]:
    try:
        import ttnn
    except Exception as exc:
        return 0, f"Unable to import ttnn: {exc}"

    try:
        return int(ttnn.GetNumAvailableDevices()), "No TTNN devices available"
    except Exception as exc:
        return 0, f"Unable to query TTNN devices: {exc}"
