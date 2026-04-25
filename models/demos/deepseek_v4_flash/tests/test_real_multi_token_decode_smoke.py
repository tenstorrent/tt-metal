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

from models.demos.deepseek_v4_flash.real_multi_token_decode_smoke import run_real_multi_token_decode_smoke
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.tests.test_real_decode_decoder_layer_smoke import _available_ttnn_devices

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_cpu_real_multi_token_decode_smoke_carries_cache_between_steps(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_multi_token_decode_smoke(
        snapshot,
        layers=(2, 3),
        prefill_seq_len=4,
        decode_steps=2,
        input_id_start=0,
        vocab_mode="slice",
        vocab_start=8,
        vocab_size=16,
        top_k=2,
        max_bytes=1024 * 1024,
        cpu_only=True,
    )

    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["decode_feed_mode"] == "supplied"
    assert result["input"]["token_ids"] == [0, 1, 2, 3, 4, 5]
    assert result["input"]["prefill_token_ids"] == [0, 1, 2, 3]
    assert result["input"]["supplied_decode_token_ids"] == [4, 5]
    assert result["current_positions"] == [4, 5]
    assert result["next_position"] == 6
    assert result["generated"]["reference_top1_ids"] == [
        result["steps"][0]["reference"]["top_k"][0]["id"],
        result["steps"][1]["reference"]["top_k"][0]["id"],
    ]
    assert result["generated"]["ttnn_top1_ids"] == []
    assert "reported only" in result["generated"]["feed_policy"]

    step0_layer2 = result["steps"][0]["layers"][0]
    step1_layer2 = result["steps"][1]["layers"][0]
    assert step0_layer2["cache_before"]["attention_input_tokens"] == 4
    assert step0_layer2["cache_after"]["attention_input_tokens"] == 5
    assert step1_layer2["cache_before"]["attention_input_tokens"] == 5
    assert step1_layer2["cache_after"]["attention_input_tokens"] == 6
    assert step0_layer2["decode_cache"]["attention_cache_length"] == 6
    assert step1_layer2["decode_cache"]["attention_cache_length"] == 7
    assert step0_layer2["decode_cache"]["compressed_cache_length_used"] == 1
    assert step1_layer2["decode_cache"]["compressed_cache_length_after_decode"] == 1
    assert step0_layer2["decode_cache"]["compressed_tokens_contributed"] is True

    step0_layer3 = result["steps"][0]["layers"][1]
    step1_layer3 = result["steps"][1]["layers"][1]
    assert step0_layer3["cache_before"]["sliding_window_cache_length"] == 4
    assert step0_layer3["cache_after"]["sliding_window_cache_length"] == 5
    assert step1_layer3["cache_before"]["sliding_window_cache_length"] == 5
    assert step1_layer3["cache_after"]["sliding_window_cache_length"] == 6
    assert step1_layer3["decode_cache"]["compressed_cache_length_used"] == 0

    assert result["steps"][0]["output_shapes"]["logits"] == [1, 1, 1, 16]
    assert result["steps"][1]["output_shapes"]["stack_hidden"] == [1, 1, 1, 32]
    assert result["prefill"][0]["fanout_scope"]["prefill_routes_executed"] == 8
    assert result["host_boundaries"][4]["name"] == "carried_decode_cache_host_state"
    assert result["ttnn_ops"] == []
    assert result["accuracy"]["cpu_reference"]["passed"] is True


def test_cpu_real_multi_token_decode_smoke_cli_outputs_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_multi_token_decode_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layers",
            "2",
            "3",
            "--prefill-seq-len",
            "4",
            "--decode-steps",
            "2",
            "--embedding-mode",
            "slice",
            "--vocab-mode",
            "slice",
            "--vocab-start",
            "8",
            "--vocab-size",
            "16",
            "--top-k",
            "2",
            "--max-bytes",
            str(1024 * 1024),
            "--cpu-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == 1
    assert payload["mode"] == "cpu-reference"
    assert payload["decode_steps"] == 2
    assert payload["current_positions"] == [4, 5]
    assert payload["vocab"]["mode"] == "slice"
    assert payload["vocab"]["deterministic_slice"] == "[8, 24)"
    assert payload["steps"][1]["feed_token_id"] == 5
    assert payload["steps"][1]["layers"][0]["cache_before"]["attention_input_tokens"] == 5
    assert payload["steps"][1]["layers"][0]["cache_after"]["attention_input_tokens"] == 6
    assert payload["steps"][1]["reference"]["top_k"][0]["id"] in payload["generated"]["reference_top1_ids"]
    assert payload["host_boundaries"][-1]["name"] == "lm_head_vocab_slice"
    assert payload["accuracy"]["cpu_reference"]["passed"] is True


def test_real_multi_token_decode_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_SMOKE", "0") == "1"
    if not required:
        pytest.skip("Set DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_SMOKE=1 to run the real Galaxy TTNN smoke")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    available, reason = _available_ttnn_devices()
    if available < 1:
        pytest.fail(reason)

    vocab_mode = os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_VOCAB_MODE", "slice")
    vocab_size_env = os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_VOCAB_SIZE")
    result = run_real_multi_token_decode_smoke(
        snapshot,
        layers=tuple(
            int(layer)
            for layer in os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_LAYERS", "2,3").replace(",", " ").split()
        ),
        prefill_seq_len=int(os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_PREFILL_SEQ_LEN", "32")),
        decode_steps=int(os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_STEPS", "2")),
        input_id_start=int(os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_INPUT_ID_START", "0")),
        embedding_mode=os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_EMBEDDING_MODE", "slice"),  # type: ignore[arg-type]
        vocab_mode=vocab_mode,  # type: ignore[arg-type]
        vocab_start=int(os.environ.get("DSV4_FLASH_REAL_MULTI_TOKEN_DECODE_VOCAB_START", "0")),
        vocab_size=None if vocab_size_env is None else int(vocab_size_env),
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(
        {
            "top_level": result["accuracy"],
            "steps": [step["accuracy"] for step in result["steps"]],
        },
        indent=2,
        sort_keys=True,
    )
    assert result["layers"] == [2, 3]
    assert result["current_positions"][0] == result["prefill_sequence_length"]
    assert result["next_position"] == result["prefill_sequence_length"] + result["decode_steps"]
    assert len(result["generated"]["reference_top1_ids"]) == result["decode_steps"]
    assert len(result["generated"]["ttnn_top1_ids"]) == result["decode_steps"]
    assert result["generated"]["top1_ids_match"] is True
    assert all(step["passed"] for step in result["steps"])
    assert result["steps"][1]["layers"][0]["cache_before"]["attention_input_tokens"] == (
        result["prefill_sequence_length"] + 1
    )
    assert result["steps"][-1]["output_shapes"]["logits"][-1] == result["vocab"]["vocab_size"]
