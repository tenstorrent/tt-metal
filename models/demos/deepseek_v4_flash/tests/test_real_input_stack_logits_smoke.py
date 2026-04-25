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

from models.demos.deepseek_v4_flash.real_input_stack_logits_smoke import run_real_input_stack_logits_smoke
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.tests.test_real_decode_decoder_layer_smoke import _available_ttnn_devices

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_cpu_real_input_stack_logits_smoke_starts_from_embedding_ids(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_input_stack_logits_smoke(
        snapshot,
        layers=(2, 3),
        prefill_seq_len=4,
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
    assert result["activation_source"] == "real_input_ids_host_embedding_lookup"
    assert result["input"]["token_ids"] == [0, 1, 2, 3, 4]
    assert result["input"]["prefill_token_ids"] == [0, 1, 2, 3]
    assert result["input"]["decode_token_ids"] == [4]
    assert result["embedding"]["mode"] == "slice"
    assert result["embedding"]["loaded_shape"] == [5, 32]
    assert result["embedding"]["deterministic_slice"] == "[0, 5)"
    assert result["output_shapes"]["initial_prefill_hidden"] == [1, 1, 4, 32]
    assert result["output_shapes"]["logits"] == [1, 1, 1, 16]
    assert result["layers_detail"][0]["selected_experts"]["prefill"]["activated_experts"][
        "topk_expert_ids_by_token"
    ] == [[0, 1], [2, 3], [0, 1], [2, 3]]
    assert result["layers_detail"][0]["selected_experts"]["decode"]["activated_experts"][
        "topk_expert_ids_by_token"
    ] == [[0, 1]]
    assert result["layers_detail"][0]["fanout_scope"]["prefill_routes_executed"] == 8
    assert result["layers_detail"][0]["fanout_scope"]["prefill_full_fanout_materialized"] is True
    assert result["layers_detail"][0]["fanout_scope"]["decode_routes_executed"] == 2
    assert result["loaded_tensor_groups"]["embedding"]["payload_bytes"]["embedding"] == 5 * 32 * 2
    assert result["host_boundaries"][0]["name"] == "embedding_weight_load"
    assert result["reference_ops"][0] == "torch.embedding(real_embed_weight)"
    assert result["ttnn_ops"] == []


def test_cpu_real_input_stack_logits_smoke_cli_outputs_json(tmp_path: Path) -> None:
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
            "models.demos.deepseek_v4_flash.real_input_stack_logits_smoke",
            "--snapshot-dir",
            str(snapshot),
            "--layers",
            "2",
            "3",
            "--prefill-seq-len",
            "4",
            "--input-ids",
            "0",
            "1",
            "2",
            "3",
            "4",
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
    assert payload["input"]["token_ids"] == [0, 1, 2, 3, 4]
    assert payload["embedding"]["mode"] == "slice"
    assert payload["vocab"]["mode"] == "slice"
    assert payload["reference"]["top_k"][0]["id"] >= 8
    assert payload["reference"]["top_k"][0]["id"] < 24
    assert payload["layers_detail"][0]["fanout_scope"]["prefill_routes_executed"] == 8
    assert payload["accuracy"]["cpu_reference"]["passed"] is True


def test_real_input_stack_logits_smoke_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_INPUT_STACK_LOGITS_SMOKE", "0") == "1"
    if not required:
        pytest.skip("Set DSV4_FLASH_REAL_INPUT_STACK_LOGITS_SMOKE=1 to run the real Galaxy TTNN smoke")

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

    vocab_mode = os.environ.get("DSV4_FLASH_REAL_INPUT_STACK_LOGITS_VOCAB_MODE", "slice")
    vocab_size_env = os.environ.get("DSV4_FLASH_REAL_INPUT_STACK_LOGITS_VOCAB_SIZE")
    result = run_real_input_stack_logits_smoke(
        snapshot,
        layers=tuple(
            int(layer)
            for layer in os.environ.get("DSV4_FLASH_REAL_INPUT_STACK_LAYERS", "2,3").replace(",", " ").split()
        ),
        prefill_seq_len=int(os.environ.get("DSV4_FLASH_REAL_INPUT_STACK_PREFILL_SEQ_LEN", "32")),
        input_id_start=int(os.environ.get("DSV4_FLASH_REAL_INPUT_STACK_INPUT_ID_START", "0")),
        embedding_mode=os.environ.get("DSV4_FLASH_REAL_INPUT_STACK_EMBEDDING_MODE", "slice"),  # type: ignore[arg-type]
        vocab_mode=vocab_mode,  # type: ignore[arg-type]
        vocab_start=int(os.environ.get("DSV4_FLASH_REAL_INPUT_STACK_LOGITS_VOCAB_START", "0")),
        vocab_size=None if vocab_size_env is None else int(vocab_size_env),
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["passed"], json.dumps(result["accuracy"], indent=2, sort_keys=True)
    assert result["layers"] == [2, 3]
    assert result["current_position"] == 32
    assert result["input"]["prefill_tokens"] == 32
    assert result["input"]["decode_tokens"] == 1
    assert result["embedding"]["mode"] in {"slice", "full"}
    assert result["layers_detail"][0]["fanout_scope"]["prefill_full_fanout_materialized"] is True
    assert result["layers_detail"][0]["fanout_scope"]["prefill_routes_executed"] == (
        32 * result["model"]["num_experts_per_tok"]
    )
    assert result["output_shapes"]["logits"][-1] == result["vocab"]["vocab_size"]
    assert result["reference"]["top_k"][0]["id"] == result["ttnn"]["top_k"][0]["id"]
