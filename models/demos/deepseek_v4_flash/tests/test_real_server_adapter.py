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

from models.demos.deepseek_v4_flash.real_server_adapter import (
    REAL_SERVER_ADAPTER_NAME,
    RealServerRequest,
    ensure_real_server_request,
    run_real_server_request,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.tests.test_real_decode_decoder_layer_smoke import _available_ttnn_devices

REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")


def _generate_cpu_snapshot(tmp_path: Path) -> Path:
    return generate_tiny_hf_checkpoint(
        tmp_path / "hf",
        num_hidden_layers=4,
        num_routed_experts=4,
        compress_ratios=[0, 0, 4, 128],
    )


def _cpu_request(snapshot: Path, *, request_id: str = "unit-request") -> RealServerRequest:
    return RealServerRequest(
        request_id=request_id,
        snapshot_dir=snapshot,
        input_ids=[7, 8, 9, 10, 11, 12],
        prefill_seq_len=4,
        max_tokens=2,
        decode_steps=2,
        layers=(2, 3),
        top_k=2,
        vocab_mode="slice",
        vocab_start=8,
        vocab_size=16,
        max_bytes=1024 * 1024,
        cpu_only=True,
    )


def test_real_server_request_serializes_and_deserializes_json(tmp_path: Path) -> None:
    request = _cpu_request(tmp_path / "hf", request_id="json-roundtrip")

    payload = json.loads(json.dumps(request.to_mapping(), sort_keys=True))
    round_tripped = RealServerRequest.from_mapping(payload)

    assert round_tripped == request
    assert ensure_real_server_request(payload) == request
    assert round_tripped.max_tokens == 2
    assert round_tripped.decode_steps == 2
    assert round_tripped.vocab_mode == "slice"
    assert round_tripped.full_vocab is False

    full_vocab_request = RealServerRequest(
        request_id="full-vocab",
        snapshot_dir=tmp_path / "hf",
        input_ids=[0, 1, 2, 3, 4, 5],
        prefill_seq_len=4,
        max_tokens=2,
        layers=(2, 3),
        vocab_mode="full",
        cpu_only=True,
    )
    assert full_vocab_request.full_vocab is True
    assert full_vocab_request.vocab_mode == "full"

    with pytest.raises(ValueError, match="Unknown RealServerRequest"):
        RealServerRequest.from_mapping({"request_id": "bad", "batch_size": 4})
    with pytest.raises(ValueError, match="must match"):
        RealServerRequest(snapshot_dir=tmp_path, input_ids=[0, 1, 2], max_tokens=1, decode_steps=2)
    with pytest.raises(ValueError, match="Pass only one"):
        RealServerRequest(snapshot_dir=tmp_path, input_ids=[0, 1, 2, 3], prompt_path=tmp_path / "prompt.txt")


def test_real_server_adapter_cpu_tiny_checkpoint_summary(tmp_path: Path) -> None:
    snapshot = _generate_cpu_snapshot(tmp_path)

    response = run_real_server_request(_cpu_request(snapshot, request_id="cpu-adapter"))

    json.dumps(response, sort_keys=True)
    assert response["schema_version"] == 1
    assert response["request_id"] == "cpu-adapter"
    assert response["adapter"]["name"] == REAL_SERVER_ADAPTER_NAME
    assert response["adapter"]["batch_size"] == 1
    assert response["adapter"]["request"]["max_tokens"] == 2
    assert response["adapter"]["request"]["decode_steps"] == 2
    assert response["runner"]["name"] == "deepseek_v4_flash_real_generation_eval_runner"
    assert response["mode"] == "cpu-reference"
    assert response["passed"] is True
    assert response["input"]["token_ids"] == [7, 8, 9, 10, 11, 12]
    assert response["tokens"]["prefill_token_ids"] == [7, 8, 9, 10]
    assert response["tokens"]["supplied_token_ids"] == [11, 12]
    assert response["tokens"]["generated_token_ids"] == response["tokens"]["reference_generated_token_ids"]
    assert response["tokens"]["ttnn_generated_token_ids"] == []
    assert response["tokens"]["generated_ids_are_fed_back"] is False
    assert response["top_k"]["k"] == 2
    assert len(response["top_k"]["per_step"]) == 2
    assert response["correctness"]["passed"] is True
    assert response["timing"]["tokens_per_sec_per_user"] is not None
    assert response["payload_bytes"]["total"] > 0
    assert response["host_visible_boundaries"][-1]["name"] == "lm_head_vocab_slice"
    assert response["ttnn_ops"] == []
    assert response["limitation_flags"]["two_layer_stepping_stone"] is True
    assert response["limitation_flags"]["not_production_serving"] is True
    assert response["limitation_flags"]["generated_ids_not_fed_back"] is True


def test_real_server_adapter_cli_accepts_request_json(tmp_path: Path) -> None:
    snapshot = _generate_cpu_snapshot(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_cpu_request(snapshot, request_id="cli-json").to_mapping()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_server_adapter",
            "--request-json",
            str(request_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["request_id"] == "cli-json"
    assert payload["adapter"]["name"] == REAL_SERVER_ADAPTER_NAME
    assert payload["passed"] is True
    assert payload["input"]["source"] == "explicit_input_ids"
    assert payload["tokens"]["supplied_token_ids"] == [11, 12]


def test_real_server_adapter_cli_accepts_direct_flags(tmp_path: Path) -> None:
    snapshot = _generate_cpu_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_server_adapter",
            "--request-id",
            "cli-direct",
            "--snapshot-dir",
            str(snapshot),
            "--input-ids",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "--prefill-seq-len",
            "4",
            "--max-tokens",
            "2",
            "--layers",
            "2",
            "3",
            "--top-k",
            "2",
            "--vocab-start",
            "8",
            "--vocab-size",
            "16",
            "--max-bytes",
            str(1024 * 1024),
            "--cpu-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["request_id"] == "cli-direct"
    assert payload["adapter"]["request"]["max_tokens"] == 2
    assert payload["adapter"]["request"]["vocab_mode"] == "slice"
    assert payload["passed"] is True


def test_real_server_adapter_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER", "0") == "1"
    if not required:
        pytest.skip("Set DSV4_FLASH_REAL_SERVER_ADAPTER=1 to run the real Galaxy TTNN server adapter smoke")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    available, reason = _available_ttnn_devices()
    if available < 1:
        pytest.fail(reason)

    vocab_size_env = os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER_VOCAB_SIZE")
    response = run_real_server_request(
        RealServerRequest(
            request_id="galaxy-smoke",
            snapshot_dir=snapshot,
            input_ids=None,
            layers=tuple(
                int(layer)
                for layer in os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER_LAYERS", "2,3").replace(",", " ").split()
            ),
            prefill_seq_len=int(os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER_PREFILL_SEQ_LEN", "32")),
            decode_steps=int(os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER_DECODE_STEPS", "2")),
            top_k=int(os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER_TOP_K", "5")),
            vocab_mode=os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER_VOCAB_MODE", "slice"),
            full_vocab=os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER_FULL_VOCAB", "0") == "1",
            vocab_start=int(os.environ.get("DSV4_FLASH_REAL_SERVER_ADAPTER_VOCAB_START", "0")),
            vocab_size=None if vocab_size_env is None else int(vocab_size_env),
            device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
        )
    )

    assert response["mode"] == "ttnn"
    assert response["passed"], json.dumps(response["correctness"], indent=2, sort_keys=True)
    assert response["correctness"]["ttnn_compared_to_torch"] is True
    assert response["correctness"]["top1_ids_match"] is True
    assert response["tokens"]["generated_token_source"] == "ttnn_top1_ids"
    assert len(response["tokens"]["generated_token_ids"]) == response["adapter"]["request"]["decode_steps"]
    assert response["timing"]["tokens_per_sec_per_user"] > 0.0
    assert response["ttnn_ops"]
    assert response["limitation_flags"]["two_layer_stepping_stone"] is True
