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

from models.demos.deepseek_v4_flash.real_decode_stack_logits_smoke import run_real_decode_stack_logits_smoke
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.tests.test_real_decode_decoder_layer_smoke import _available_ttnn_devices

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_cpu_real_decode_stack_logits_smoke_composes_layer2_layer3_and_logits(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_decode_stack_logits_smoke(
        snapshot,
        layers=(2, 3),
        prefill_seq_len=4,
        top_k=3,
        max_bytes=256 * 1024,
        cpu_only=True,
    )

    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["layers"] == [2, 3]
    assert result["current_position"] == 4
    assert result["next_position"] == 5
    assert result["layer2_compressed_tokens_contributed"] is True
    assert result["layers_detail"][0]["decode_cache"]["compressed_cache_length"] == 1
    assert result["layers_detail"][0]["decode_cache"]["compress_topk_valid_count"] > 0
    assert result["layers_detail"][0]["decode_cache"]["attention_cache_length"] == 6
    assert result["layers_detail"][0]["fanout_scope"]["prefill_full_fanout_materialized"] is True
    assert result["layers_detail"][0]["fanout_scope"]["prefill_activated_expert_count"] == 2
    assert result["layers_detail"][0]["fanout_scope"]["prefill_routes_executed"] == 8
    assert result["layers_detail"][0]["fanout_scope"]["decode_activated_expert_count"] == 2
    assert result["layers_detail"][1]["fanout_scope"]["decode_activated_expert_count"] == 2
    assert result["layers_detail"][1]["prefill_cache"]["sliding_window_cache_length"] == 4
    assert result["layers_detail"][1]["decode_cache"]["attention_cache_length"] == 5
    assert result["cache_handoff"]["layer_3_prefill_cache_source"] == "layer_2_prefill_post_ffn_residual"
    assert result["output_shapes"]["stack_hidden"] == [1, 1, 1, 32]
    assert result["output_shapes"]["final_norm"] == [1, 1, 1, 32]
    assert result["output_shapes"]["logits"] == [1, 1, 1, 64]
    assert result["vocab"]["mode"] == "full"
    assert len(result["reference"]["top_k"]) == 3
    assert result["loaded_tensor_groups"]["layer_2"]["attention_runtime"]["count"] > 13
    assert any(
        ".attn.indexer." in key
        for key in result["loaded_tensor_groups"]["layer_2"]["attention_runtime"]["canonical_keys"]
    )
    assert result["final_norm_lm_head"]["loaded_keys"]["final_norm"] == "norm.weight"
    assert {boundary["name"] for boundary in result["host_boundaries"]} >= {
        "layer2_prefill_output_readback",
        "layer2_decode_output_readback",
        "layer2_decode_indexer_host_topk",
        "prefill_activated_expert_gather_scatter",
        "final_logits_readback",
    }
    assert result["ttnn_ops"] == []
    assert result["accuracy"]["cpu_reference"]["passed"] is True


def test_cpu_real_decode_stack_logits_smoke_cli_outputs_json(tmp_path: Path) -> None:
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
            "models.demos.deepseek_v4_flash.real_decode_stack_logits_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layers",
            "2",
            "3",
            "--prefill-seq-len",
            "4",
            "--vocab-mode",
            "slice",
            "--vocab-start",
            "8",
            "--vocab-size",
            "16",
            "--top-k",
            "2",
            "--max-bytes",
            str(256 * 1024),
            "--cpu-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["mode"] == "cpu-reference"
    assert payload["layers"] == [2, 3]
    assert payload["vocab"]["mode"] == "slice"
    assert payload["vocab"]["deterministic_slice"] == "[8, 24)"
    assert payload["output_shapes"]["logits"] == [1, 1, 1, 16]
    assert payload["reference"]["top_k"][0]["id"] >= 8
    assert payload["reference"]["top_k"][0]["id"] < 24
    assert payload["layer2_compressed_tokens_contributed"] is True
    assert payload["layers_detail"][0]["fanout_scope"]["prefill_full_fanout_materialized"] is True
    assert payload["layers_detail"][0]["fanout_scope"]["prefill_routes_executed"] == 8
    assert payload["layers_detail"][0]["fanout_scope"]["decode_activated_expert_count"] == 2
    assert payload["ttnn_ops"] == []
    assert payload["host_boundaries"][-1]["name"] == "lm_head_vocab_slice"
    assert payload["accuracy"]["cpu_reference"]["passed"] is True


def test_real_decode_stack_logits_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_DECODE_STACK_LOGITS_SMOKE", "0") == "1"
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

    vocab_mode = os.environ.get("DSV4_FLASH_REAL_DECODE_STACK_LOGITS_VOCAB_MODE", "full")
    vocab_size_env = os.environ.get("DSV4_FLASH_REAL_DECODE_STACK_LOGITS_VOCAB_SIZE")
    result = run_real_decode_stack_logits_smoke(
        snapshot,
        layers=tuple(
            int(layer)
            for layer in os.environ.get("DSV4_FLASH_REAL_DECODE_STACK_LAYERS", "2,3").replace(",", " ").split()
        ),
        prefill_seq_len=int(os.environ.get("DSV4_FLASH_REAL_DECODE_STACK_PREFILL_SEQ_LEN", "32")),
        vocab_mode=vocab_mode,  # type: ignore[arg-type]
        vocab_start=int(os.environ.get("DSV4_FLASH_REAL_DECODE_STACK_LOGITS_VOCAB_START", "0")),
        vocab_size=None if vocab_size_env is None else int(vocab_size_env),
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)
    assert result["layers"] == [2, 3]
    assert result["current_position"] == 32
    assert result["layer2_compressed_tokens_contributed"] is True
    assert result["layers_detail"][0]["decode_cache"]["compressed_cache_length"] > 0
    assert result["layers_detail"][0]["fanout_scope"]["prefill_full_fanout_materialized"] is True
    assert result["layers_detail"][0]["fanout_scope"]["prefill_routes_executed"] == (
        32 * result["model"]["num_experts_per_tok"]
    )
    assert (
        result["ttnn"]["layers"][0]["prefill_experts_executed"]
        == result["layers_detail"][0]["fanout_scope"]["prefill_activated_expert_count"]
    )
    assert all(
        layer["fanout_scope"]["decode_activated_expert_count"] == result["model"]["num_experts_per_tok"]
        for layer in result["layers_detail"]
    )
    assert result["output_shapes"]["stack_hidden"] == [1, 1, 1, result["model"]["hidden_size"]]
    assert result["output_shapes"]["logits"][-1] == result["vocab"]["vocab_size"]
    assert result["reference"]["top_k"][0]["id"] == result["ttnn"]["top_k"][0]["id"]
