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

from models.demos.deepseek_v4_flash.real_generation_eval_runner import run_real_generation_eval_runner
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.tests.test_real_decode_decoder_layer_smoke import _available_ttnn_devices

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_cpu_real_generation_eval_runner_reports_compact_eval_json(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_generation_eval_runner(
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

    json.dumps(result, sort_keys=True)
    assert result["runner"] == "deepseek_v4_flash_real_generation_eval_runner"
    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["layers"] == [2, 3]
    assert result["positions"] == {"per_step": [4, 5], "next_position": 6}
    assert result["input"]["source"] == "deterministic_contiguous_input_ids"
    assert result["input"]["deterministic_notice"] == "deterministic contiguous input IDs were used"
    assert result["input"]["token_ids"] == [0, 1, 2, 3, 4, 5]
    assert result["generated"]["reference_top1_ids"] == [
        result["steps"][0]["reference_top_k"][0]["id"],
        result["steps"][1]["reference_top_k"][0]["id"],
    ]
    assert result["generated"]["ttnn_top1_ids"] == []
    assert result["top_k"]["k"] == 2
    assert result["vocab"]["deterministic_slice"] == "[8, 24)"
    assert result["payload_bytes"]["total"] > 0
    assert result["host_boundaries"][-1]["name"] == "lm_head_vocab_slice"
    assert result["ttnn_ops"] == []
    assert result["correctness"]["ttnn_compared_to_torch"] is False
    assert result["timing"]["setup_load_seconds"] >= 0.0
    assert result["timing"]["prefill_build_seconds"] >= 0.0
    assert result["timing"]["decode_build_total_seconds"] >= 0.0
    assert len(result["timing"]["decode_build_step_seconds"]) == 2
    assert result["timing"]["decode_tokens_per_sec_denominator"] == "torch_reference_decode_build_total_seconds"
    assert result["timing"]["decode_tokens_per_sec_per_user"] is not None
    assert "all-layer model eval" in result["limitations"]["serving_eval_boundaries"]


def test_cpu_real_generation_eval_runner_cli_accepts_input_ids_path(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )
    input_ids_path = tmp_path / "prompt_ids.json"
    input_ids_path.write_text(json.dumps({"input_ids": [7, 8, 9, 10, 11, 12]}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_generation_eval_runner",
            "--snapshot-dir",
            str(snapshot),
            "--layers",
            "2",
            "3",
            "--prefill-seq-len",
            "4",
            "--decode-steps",
            "2",
            "--input-ids-path",
            str(input_ids_path),
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
    assert payload["input"]["source"] == "prompt_ids_path"
    assert payload["input"]["source_path"] == str(input_ids_path)
    assert payload["input"]["prompt_label"] == "prompt_ids.json"
    assert payload["input"]["prefill_token_ids"] == [7, 8, 9, 10]
    assert payload["input"]["supplied_decode_token_ids"] == [11, 12]
    assert payload["steps"][1]["feed_token_id"] == 12
    assert payload["timing"]["runner_wall_seconds"] >= payload["timing"]["end_to_end_wall_seconds"]


def test_cpu_real_generation_eval_runner_supports_full_vocab_smoke(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )

    result = run_real_generation_eval_runner(
        snapshot,
        layers=(2, 3),
        prefill_seq_len=4,
        decode_steps=2,
        input_ids=[0, 1, 2, 3, 4, 5],
        vocab_mode="slice",
        full_vocab_smoke=True,
        top_k=3,
        max_bytes=1024 * 1024,
        cpu_only=True,
    )

    assert result["input"]["source"] == "explicit_input_ids"
    assert result["vocab"]["mode"] == "full"
    assert result["vocab"]["full_vocab_smoke_requested"] is True
    assert result["vocab"]["vocab_size"] == 64
    assert {boundary["name"] for boundary in result["host_boundaries"]}.isdisjoint({"lm_head_vocab_slice"})
    assert result["steps"][0]["output_shapes"]["logits"] == [1, 1, 1, 64]


def test_real_generation_eval_runner_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_RUNNER", "0") == "1"
    if not required:
        pytest.skip("Set DSV4_FLASH_REAL_GENERATION_EVAL_RUNNER=1 to run the real Galaxy TTNN eval runner")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    available, reason = _available_ttnn_devices()
    if available < 1:
        pytest.fail(reason)

    vocab_size_env = os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_VOCAB_SIZE")
    result = run_real_generation_eval_runner(
        snapshot,
        layers=tuple(
            int(layer)
            for layer in os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_LAYERS", "2,3").replace(",", " ").split()
        ),
        prefill_seq_len=int(os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_PREFILL_SEQ_LEN", "32")),
        decode_steps=int(os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_DECODE_STEPS", "2")),
        input_id_start=int(os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_INPUT_ID_START", "0")),
        vocab_mode=os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_VOCAB_MODE", "slice"),
        full_vocab_smoke=os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_FULL_VOCAB", "0") == "1",
        vocab_start=int(os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_VOCAB_START", "0")),
        vocab_size=None if vocab_size_env is None else int(vocab_size_env),
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["mode"] == "ttnn"
    assert result["passed"], json.dumps(result["correctness"], indent=2, sort_keys=True)
    assert result["correctness"]["ttnn_compared_to_torch"] is True
    assert result["correctness"]["top1_ids_match"] is True
    assert len(result["generated"]["ttnn_top1_ids"]) == result["decode_steps"]
    assert result["timing"]["ttnn_decode_total_seconds"] > 0.0
    assert result["timing"]["ttnn_logits_total_seconds"] > 0.0
    assert result["timing"]["decode_tokens_per_sec_denominator"] == "ttnn_decode_total_seconds"
    assert result["timing"]["decode_tokens_per_sec_per_user"] > 0.0
    assert result["ttnn_ops"]
