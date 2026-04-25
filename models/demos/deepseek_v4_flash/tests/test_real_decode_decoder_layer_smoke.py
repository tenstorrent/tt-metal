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
from models.demos.deepseek_v4_flash.real_decode_decoder_layer_smoke import (
    layer_decode_decoder_layer_keys,
    layer_decode_decoder_layer_selector_keys,
    run_real_decode_decoder_layer_smoke,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_layer_decode_decoder_layer_selector_loads_layer3_attention_and_ffn_router(
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

    selector_keys = layer_decode_decoder_layer_selector_keys(index, config=config, layer=3)
    full_keys = layer_decode_decoder_layer_keys(index, config=config, layer=3, expert=0)

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
    assert len(full_keys) == len(set(full_keys))
    assert "layers.3.ffn.experts.0.w1.weight" in full_keys
    assert "layers.3.ffn.shared_experts.w3.scale" in full_keys


def test_cpu_real_decode_decoder_layer_smoke_composes_prefill_cache_decode_attention_ffn(
    tmp_path: Path,
) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_decode_decoder_layer_smoke(
        snapshot,
        layer=3,
        prefill_seq_len=4,
        max_bytes=64 * 1024,
        cpu_only=True,
    )

    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["requested_expert"] is None
    assert 0 <= result["expert"] < 4
    assert result["fanout_scope"]["activated_expert_count"] == 2
    assert result["fanout_scope"]["topk"] == 2
    assert result["ffn_fanout_info"]["activated_experts"]["topk_is_full"] is True
    assert result["prefill_sequence_length"] == 4
    assert result["decode_tokens"] == 1
    assert result["current_position"] == 4
    assert result["next_position"] == 5
    assert result["model"]["compress_ratio"] == 128
    assert result["cache_sizes"] == {
        "prefill_attention_tokens": 4,
        "current_token_cache_tokens": 1,
        "sliding_window_cache_before_decode": 4,
        "sliding_window_cache_after_decode": 5,
        "compressed_cache_length": 0,
        "attention_cache_length": 5,
        "runtime_topk_width": 5,
        "window_topk_valid_count": 5,
        "compress_topk_valid_count": 0,
    }
    assert result["output_shapes"] == {
        "prefill_input_hidden_states": [1, 1, 4, 32],
        "decode_input_hidden_states": [1, 1, 1, 32],
        "decode_attn_norm": [1, 1, 1, 32],
        "decode_q": [1, 1, 4, 8],
        "decode_kv_cache_ready": [1, 1, 8],
        "attention_cache": [1, 5, 8],
        "attention_output": [1, 1, 4, 8],
        "attention_output_projected": [1, 1, 1, 32],
        "post_attention_residual": [1, 1, 1, 32],
        "ffn_norm": [1, 1, 1, 32],
        "per_expert_selected_output": {
            str(expert): [1, 1, 1, 32] for expert in result["fanout_scope"]["activated_expert_ids"]
        },
        "routed_output": [1, 1, 1, 32],
        "shared_output": [1, 1, 1, 32],
        "combined_ffn_output": [1, 1, 1, 32],
        "post_ffn_residual": [1, 1, 1, 32],
    }
    assert result["loaded_tensor_groups"]["attention_runtime"]["count"] == 13
    assert result["loaded_tensor_groups"]["ffn_selector"]["canonical_keys"] == [
        "layers.3.ffn_norm.weight",
        "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",
    ]
    assert result["loaded_tensor_groups"]["ffn"]["count"] == 21
    assert result["payload_bytes"]["total"] == 30636
    assert result["payload_bytes"]["ffn"]["routed_experts"] == 3840
    assert result["sparse_attention_inputs"]["current_position_included"] is True
    assert result["sparse_attention_inputs"]["runtime_topk_idxs"]["shape"] == [1, 1, 5]
    assert all(route["selected_token_count"] == 1 for route in result["ffn_fanout_info"]["per_expert_routes"].values())
    assert {boundary["name"] for boundary in result["host_boundaries"]} >= {
        "prefill_cache_readback",
        "decode_rope_cache_prep_host",
        "decode_attention_cache_host_append",
        "attention_residual_host_add",
        "ffn_residual_host_add",
    }
    assert result["accuracy"]["cpu_reference"]["passed"] is True


def test_cpu_real_decode_decoder_layer_smoke_refuses_budget_overruns(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    with pytest.raises(ValueError, match="tensor budget"):
        run_real_decode_decoder_layer_smoke(
            snapshot,
            layer=3,
            prefill_seq_len=4,
            max_tensors=15,
            max_bytes=64 * 1024,
            cpu_only=True,
        )
    with pytest.raises(ValueError, match="byte budget"):
        run_real_decode_decoder_layer_smoke(
            snapshot,
            layer=3,
            prefill_seq_len=4,
            max_tensors=48,
            max_bytes=28715,
            cpu_only=True,
        )


def test_cpu_real_decode_decoder_layer_smoke_cli_outputs_json(tmp_path: Path) -> None:
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
            "models.demos.deepseek_v4_flash.real_decode_decoder_layer_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layer",
            "3",
            "--prefill-seq-len",
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
    assert payload["current_position"] == 4
    assert payload["next_position"] == 5
    assert payload["payload_bytes"]["total"] == 30636
    assert payload["cache_sizes"]["attention_cache_length"] == 5
    assert payload["output_shapes"]["decode_input_hidden_states"] == [1, 1, 1, 32]
    assert payload["output_shapes"]["post_ffn_residual"] == [1, 1, 1, 32]
    assert payload["fanout_scope"]["activated_expert_count"] == 2
    assert payload["ttnn_ops"] == []
    assert payload["host_boundaries"][-1]["name"] == "ffn_residual_host_add"
    assert payload["accuracy"]["cpu_reference"]["passed"] is True


def test_real_decode_decoder_layer_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_DECODE_DECODER_LAYER_SMOKE", "0") == "1"
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

    expert_env = os.environ.get("DSV4_FLASH_REAL_DECODE_DECODER_LAYER_EXPERT")
    result = run_real_decode_decoder_layer_smoke(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_REAL_DECODE_DECODER_LAYER", "3")),
        expert=None if expert_env is None else int(expert_env),
        prefill_seq_len=int(os.environ.get("DSV4_FLASH_REAL_DECODE_DECODER_LAYER_PREFILL_SEQ_LEN", "32")),
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)
    assert result["current_position"] == 32
    assert result["cache_sizes"]["attention_cache_length"] == 33
    assert result["sparse_attention_inputs"]["current_position_included"] is True
    assert result["fanout_scope"]["activated_expert_count"] == result["model"]["num_experts_per_tok"]
    assert all(route["selected_token_count"] == 1 for route in result["ffn_fanout_info"]["per_expert_routes"].values())
    assert result["output_shapes"]["post_ffn_residual"] == [1, 1, 1, result["model"]["hidden_size"]]


def _available_ttnn_devices() -> tuple[int, str]:
    try:
        import ttnn
    except Exception as exc:
        return 0, f"Unable to import ttnn: {exc}"

    try:
        return int(ttnn.GetNumAvailableDevices()), "No TTNN devices available"
    except Exception as exc:
        return 0, f"Unable to query TTNN devices: {exc}"
