# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from models.demos.deepseek_v4_flash.real_traceable_decode_logits_runner import (
    RUNNER_NAME,
    run_real_traceable_decode_logits_runner,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.tests.test_real_decode_decoder_layer_smoke import _available_ttnn_devices

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def test_cpu_real_traceable_decode_logits_runner_reports_protected_layer_and_logits(tmp_path: Path) -> None:
    snapshot = generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=5,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128, 4],
    )

    result = run_real_traceable_decode_logits_runner(
        snapshot,
        layer=2,
        prefill_seq_len=4,
        decode_steps=4,
        input_id_start=3,
        vocab_mode="slice",
        vocab_start=8,
        vocab_size=16,
        top_k=2,
        max_bytes=16 * 1024 * 1024,
        cpu_only=True,
    )

    json.dumps(result, sort_keys=True)
    assert result["runner"] == RUNNER_NAME
    assert result["mode"] == "cpu-reference"
    assert result["passed"] is True
    assert result["layers"] == [2]
    assert result["decode_positions"] == [4, 5, 6, 7]
    assert result["input"]["source"] == "deterministic_contiguous_input_ids"
    assert result["input"]["token_ids"] == [3, 4, 5, 6, 7, 8, 9, 10]
    assert result["input"]["supplied_decode_token_ids"] == [7, 8, 9, 10]
    assert result["protected_decode"]["passed"] is True
    assert result["protected_decode"]["guard_status"]["ttnn_to_torch_guarded"] is True
    assert result["protected_decode"]["host_boundaries_inside_trace"] == []
    assert len(result["protected_decode"]["trace_capture"]["per_step_trace_variants"]) == 4
    assert len(result["protected_decode"]["selected_rows"]) == 4
    assert result["logits"]["mode"] == "torch-reference-only"
    assert result["logits"]["vocab"]["deterministic_slice"] == "[8, 24)"
    assert result["logits"]["steps"][0]["reference"]["logits"]["shape"] == [1, 1, 1, 16]
    assert result["logits"]["steps"][0]["reference"]["top_k"][0]["id"] >= 8
    assert result["host_boundaries"]["inside_protected_execution"] == []
    assert "final RMSNorm and LM head run after protected decode output readback" in result["limitations"][1]


def test_real_traceable_decode_logits_runner_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_RUNNER", "0") == "1"
    if not required:
        pytest.skip(
            "Set DSV4_FLASH_TRACEABLE_DECODE=1 and DSV4_FLASH_TRACEABLE_DECODE_LOGITS_RUNNER=1 "
            "to run the real Galaxy TTNN protected decode-to-logits smoke"
        )

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    available, reason = _available_ttnn_devices()
    if available < 1:
        pytest.fail(reason)

    vocab_size_env = os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_VOCAB_SIZE")
    result = run_real_traceable_decode_logits_runner(
        snapshot,
        layer=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_LAYER", "4")),
        prefill_seq_len=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_PREFILL_SEQ_LEN", "32")),
        decode_steps=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_DECODE_STEPS", "4")),
        input_id_start=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_INPUT_ID_START", "0")),
        vocab_mode=os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_VOCAB_MODE", "slice"),
        vocab_start=int(os.environ.get("DSV4_FLASH_TRACEABLE_DECODE_LOGITS_VOCAB_START", "0")),
        vocab_size=1024 if vocab_size_env is None else int(vocab_size_env),
        device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
    )

    assert result["mode"] == "ttnn-trace"
    assert result["passed"], json.dumps(
        {
            "protected": result["protected_decode"]["accuracy"],
            "logits": [step["accuracy"] for step in result["logits"]["steps"]],
        },
        indent=2,
        sort_keys=True,
    )
    assert result["protected_decode"]["host_boundaries_inside_trace"] == []
    assert result["logits"]["mode"] == "ttnn"
    assert result["logits"]["ttnn_compared_to_torch"] is True
    assert result["protected_decode"]["trace_capture"]["capture_count"] >= 1
    assert result["ttnn_ops"]
