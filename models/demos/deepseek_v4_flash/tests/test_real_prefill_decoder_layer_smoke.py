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

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex
from models.demos.deepseek_v4_flash.real_prefill_decoder_layer_smoke import (
    layer_prefill_decoder_layer_keys,
    layer_prefill_decoder_layer_selector_keys,
    run_real_prefill_decoder_layer_smoke,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_prefill_decoder_layer_selector_loads_layer3_attention_and_ffn_router(
    tmp_path: Path,
) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )
    config = DeepSeekV4FlashConfig.from_model_path(snapshot)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot)

    selector_keys = layer_prefill_decoder_layer_selector_keys(index, config=config, layer=3)
    full_keys = layer_prefill_decoder_layer_keys(index, config=config, layer=3, expert=0)

    assert selector_keys == [
        "layers.3.attn_norm.weight",
        "layers.3.attn.q_norm.weight",
        "layers.3.attn.wq_a.weight",
        "layers.3.attn.wq_b.weight",
        "layers.3.attn.kv_norm.weight",
        "layers.3.attn.wkv.weight",
        "layers.3.attn.attn_sink",
        "layers.3.attn.compressor.ape",
        "layers.3.attn.compressor.wkv.weight",
        "layers.3.attn.compressor.wgate.weight",
        "layers.3.attn.compressor.norm.weight",
        "layers.3.attn.wo_a.weight",
        "layers.3.attn.wo_b.weight",
        "layers.3.ffn_norm.weight",
        "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",
    ]
    assert all(".attn.indexer." not in key for key in selector_keys)
    assert all(".ffn.experts." not in key for key in selector_keys)
    assert all(".ffn.shared_experts." not in key for key in selector_keys)
    assert len(full_keys) == len(set(full_keys))
    assert "layers.3.ffn.experts.0.w1.weight" in full_keys
    assert "layers.3.ffn.shared_experts.w3.scale" in full_keys


def test_cpu_real_prefill_decoder_layer_smoke_composes_attention_ffn_and_residuals(
    tmp_path: Path,
) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_prefill_decoder_layer_smoke(
        snapshot,
        layer=3,
        seq_len=4,
        max_bytes=64 * 1024,
        cpu_only=True,
    )

    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["requested_expert"] is None
    assert result["input_id_anchor_expert"] is None
    assert result["model"]["compress_ratio"] == 128
    assert result["output_shapes"] == {
        "input_hidden_states": [1, 1, 4, 32],
        "attention_output": [1, 4, 4, 8],
        "attention_output_projected": [1, 1, 4, 32],
        "post_attention_residual": [1, 1, 4, 32],
        "ffn_norm": [1, 1, 4, 32],
        "per_expert_selected_output": {"2": [1, 1, 4, 32], "3": [1, 1, 4, 32]},
        "routed_output": [1, 1, 4, 32],
        "shared_output": [1, 1, 4, 32],
        "combined_ffn_output": [1, 1, 4, 32],
        "post_ffn_residual": [1, 1, 4, 32],
    }
    assert result["loaded_tensor_groups"]["attention_runtime"]["count"] == 13
    assert result["loaded_tensor_groups"]["ffn_selector"]["canonical_keys"] == [
        "layers.3.ffn_norm.weight",
        "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",
    ]
    assert result["loaded_tensor_groups"]["ffn"]["count"] == 33
    assert result["payload_bytes"]["total"] == 34476
    assert result["payload_bytes"]["attention_runtime"]["total"] == 20240
    assert result["payload_bytes"]["ffn"]["total"] == 14236
    assert result["prefill_fanout_info"]["source"] == "torch_router_full_topk_on_post_attention_prefill_residual"
    assert result["fanout_scope"] == {
        "full_expert_fanout": True,
        "activated_expert_ids": [3, 2],
        "activated_expert_count": 2,
        "loaded_expert_ids": [0, 1, 2, 3],
        "loaded_expert_count": 4,
        "loaded_extra_candidate_count": 2,
        "topk": 2,
        "routes_executed": 8,
        "tokens": 4,
        "unique_expert_cap": None,
    }
    assert result["sparse_attention_inputs"]["compressed_cache_length"] == 0
    assert {boundary["name"] for boundary in result["host_boundaries"]} >= {
        "attention_residual_host_add",
        "activated_expert_slice_selection",
        "activated_expert_scatter_add",
        "ffn_residual_host_add",
    }
    assert result["accuracy"]["cpu_reference"]["passed"] is True


def test_cpu_real_prefill_decoder_layer_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_prefill_decoder_layer_smoke(
            snapshot,
            layer=3,
            seq_len=4,
            max_tensors=15,
            max_bytes=64 * 1024,
            cpu_only=True,
        )
    with pytest.raises(ValueError, match="byte budget"):
        run_real_prefill_decoder_layer_smoke(
            snapshot,
            layer=3,
            seq_len=4,
            max_tensors=48,
            max_bytes=30635,
            cpu_only=True,
        )


def test_cpu_real_prefill_decoder_layer_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_prefill_decoder_layer_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--seq-len",
            "4",
            "--max-bytes",
            str(64 * 1024),
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
    assert payload["payload_bytes"]["total"] == 34476
    assert payload["output_shapes"]["post_attention_residual"] == [1, 1, 4, 32]
    assert payload["output_shapes"]["post_ffn_residual"] == [1, 1, 4, 32]
    assert payload["fanout_scope"]["full_expert_fanout"] is True
    assert payload["fanout_scope"]["routes_executed"] == 8
    assert payload["fanout_scope"]["loaded_expert_count"] == 4
    assert payload["fanout_scope"]["loaded_extra_candidate_count"] == 2
    assert payload["ttnn_ops"] == []
    assert payload["host_boundaries"][-1]["name"] == "ffn_residual_host_add"
    assert payload["accuracy"]["cpu_reference"]["passed"] is True


def test_real_prefill_decoder_layer_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_PREFILL_DECODER_LAYER_SMOKE", "0") == "1"
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

    layer = int(os.environ.get("DSV4_FLASH_REAL_PREFILL_DECODER_LAYER", "2"))
    expert_env = os.environ.get("DSV4_FLASH_REAL_PREFILL_DECODER_LAYER_EXPERT", "0" if layer == 2 else None)
    result = run_real_prefill_decoder_layer_smoke(
        snapshot,
        layer=layer,
        expert=None if expert_env is None else int(expert_env),
        seq_len=32,
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)
    assert result["fanout_scope"]["full_expert_fanout"] is True
    assert result["fanout_scope"]["routes_executed"] == 32 * result["model"]["num_experts_per_tok"]
    assert result["ttnn"]["experts_executed"] == result["fanout_scope"]["activated_expert_count"]
    assert result["output_shapes"]["post_ffn_residual"] == [1, 1, 32, result["model"]["hidden_size"]]


def _available_ttnn_devices() -> tuple[int, str]:
    try:
        import ttnn
    except Exception as exc:
        return 0, f"Unable to import ttnn: {exc}"

    try:
        return int(ttnn.GetNumAvailableDevices()), "No TTNN devices available"
    except Exception as exc:
        return 0, f"Unable to query TTNN devices: {exc}"
