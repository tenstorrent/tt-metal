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

from models.demos.deepseek_v4_flash.real_benchmark_eval import (
    REAL_BENCHMARK_EVAL_NAME,
    aggregate_real_benchmark_summaries,
    load_request_json,
    load_requests_jsonl,
    run_real_benchmark_eval,
)
from models.demos.deepseek_v4_flash.real_server_adapter import RealServerRequest
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


def _cpu_request(snapshot: Path, *, request_id: str = "benchmark-cpu") -> RealServerRequest:
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


def test_real_benchmark_eval_loads_request_json_and_jsonl(tmp_path: Path) -> None:
    first = _cpu_request(tmp_path / "hf", request_id="json-one")
    second = _cpu_request(tmp_path / "hf", request_id="jsonl-two")
    request_json = tmp_path / "request.json"
    request_json.write_text(json.dumps(first.to_mapping()), encoding="utf-8")
    requests_jsonl = tmp_path / "requests.jsonl"
    requests_jsonl.write_text(
        "\n".join(
            [
                json.dumps(first.to_mapping()),
                "",
                json.dumps(second.to_mapping()),
            ]
        ),
        encoding="utf-8",
    )

    assert load_request_json(request_json) == [first]
    assert load_requests_jsonl(requests_jsonl) == [first, second]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_request_json(bad_json)

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match=":1 must contain a JSON object"):
        load_requests_jsonl(bad_jsonl)


def test_real_benchmark_eval_aggregates_metrics() -> None:
    summaries = [
        {
            "request_id": "a",
            "passed": True,
            "top1_ids_match": True,
            "tokens": {"prompt_tokens": 4, "generated_tokens": 2},
            "timing": {
                "end_to_end_latency_seconds": 0.4,
                "decode_latency_seconds": 0.2,
            },
            "vocab": {"mode": "slice"},
            "payload_bytes": {"embedding": 10, "total": 100},
        },
        {
            "request_id": "b",
            "passed": False,
            "top1_ids_match": False,
            "tokens": {"prompt_tokens": 5, "generated_tokens": 1},
            "timing": {
                "end_to_end_latency_seconds": 0.2,
                "decode_latency_seconds": 0.1,
            },
            "vocab": {"mode": "full"},
            "payload_bytes": {"final_norm_lm_head": 5, "total": 50},
        },
        {
            "request_id": "c",
            "passed": True,
            "top1_ids_match": None,
            "tokens": {"prompt_tokens": 6, "generated_tokens": 1},
            "timing": {
                "end_to_end_latency_seconds": 0.3,
                "decode_latency_seconds": None,
            },
            "vocab": {"mode": "slice"},
            "payload_bytes": {"total": 25},
        },
    ]

    aggregate = aggregate_real_benchmark_summaries(summaries)

    assert aggregate["request_count"] == 3
    assert aggregate["passed_count"] == 2
    assert aggregate["pass_rate"] == 0.666667
    assert aggregate["top1_match_available_count"] == 2
    assert aggregate["top1_match_count"] == 1
    assert aggregate["top1_match_rate"] == 0.5
    assert aggregate["total_prompt_tokens"] == 15
    assert aggregate["total_generated_tokens"] == 4
    assert aggregate["mean_end_to_end_latency_seconds"] == 0.3
    assert aggregate["p50_end_to_end_latency_seconds"] == 0.3
    assert aggregate["max_end_to_end_latency_seconds"] == 0.4
    assert aggregate["mean_decode_latency_per_token_seconds"] == 0.1
    assert aggregate["aggregate_decode_tokens_per_sec_per_user"] == 10.0
    assert aggregate["vocab_modes_used"] == ["full", "slice"]
    assert aggregate["payload_byte_totals"] == {"embedding": 10, "final_norm_lm_head": 5, "total": 175}
    assert aggregate["metrics"]["payload_bytes_total"] == 175
    assert aggregate["metrics"]["aggregate_decode_tokens_s_per_user"] == 10.0


def test_real_benchmark_eval_cli_outputs_aggregate_json(tmp_path: Path) -> None:
    snapshot = _generate_cpu_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_benchmark_eval",
            "--request-id",
            "cli-benchmark",
            "--snapshot-dir",
            str(snapshot),
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
    assert payload["schema_version"] == 1
    assert payload["harness"]["name"] == REAL_BENCHMARK_EVAL_NAME
    assert payload["harness"]["request_source"] == "direct_flags"
    assert payload["aggregate"]["request_count"] == 1
    assert payload["aggregate"]["passed_count"] == 1
    assert payload["aggregate"]["pass_rate"] == 1.0
    assert payload["aggregate"]["total_prompt_tokens"] == 4
    assert payload["aggregate"]["total_generated_tokens"] == 2
    assert payload["aggregate"]["vocab_modes_used"] == ["slice"]
    assert payload["metrics"] == payload["aggregate"]["metrics"]
    assert len(payload["per_request"]) == 1
    assert payload["per_request"][0]["request_id"] == "cli-benchmark"
    assert payload["per_request"][0]["passed"] is True
    assert payload["per_request"][0]["generated_ids"] == payload["per_request"][0]["reference_generated_ids"]
    assert payload["per_request"][0]["tokens_per_sec_per_user"] > 0.0
    assert payload["per_request"][0]["limitation_flags"]["two_layer_stepping_stone"] is True


def test_real_benchmark_eval_cli_accepts_request_jsonl(tmp_path: Path) -> None:
    snapshot = _generate_cpu_snapshot(tmp_path)
    requests_jsonl = tmp_path / "requests.jsonl"
    requests_jsonl.write_text(
        json.dumps(_cpu_request(snapshot, request_id="jsonl-cli").to_mapping()),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "models.demos.deepseek_v4_flash.real_benchmark_eval",
            "--requests-jsonl",
            str(requests_jsonl),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["harness"]["request_source"] == "requests_jsonl"
    assert payload["aggregate"]["request_count"] == 1
    assert payload["per_request"][0]["request_id"] == "jsonl-cli"
    assert payload["per_request"][0]["top_k"]["k"] == 2


def test_real_benchmark_eval_ttnn_real_snapshot_matches_torch() -> None:
    required = os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL", "0") == "1"
    if not required:
        pytest.skip("Set DSV4_FLASH_REAL_BENCHMARK_EVAL=1 to run the real Galaxy TTNN benchmark/eval smoke")

    snapshot = Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(REAL_SNAPSHOT_DIR)))
    if not snapshot.is_dir():
        pytest.fail(f"Real DeepSeek V4 Flash snapshot is missing: {snapshot}")

    available, reason = _available_ttnn_devices()
    if available < 1:
        pytest.fail(reason)

    vocab_size_env = os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL_VOCAB_SIZE")
    result = run_real_benchmark_eval(
        [
            RealServerRequest(
                request_id="galaxy-benchmark-smoke",
                snapshot_dir=snapshot,
                layers=tuple(
                    int(layer)
                    for layer in os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL_LAYERS", "2,3")
                    .replace(",", " ")
                    .split()
                ),
                prefill_seq_len=int(os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL_PREFILL_SEQ_LEN", "32")),
                decode_steps=int(os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL_DECODE_STEPS", "2")),
                top_k=int(os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL_TOP_K", "5")),
                vocab_mode=os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL_VOCAB_MODE", "slice"),
                full_vocab=os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL_FULL_VOCAB", "0") == "1",
                vocab_start=int(os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL_VOCAB_START", "0")),
                vocab_size=None if vocab_size_env is None else int(vocab_size_env),
                device_id=int(os.environ.get("TTNN_DEVICE_ID", "0")),
            )
        ],
        request_source="env_gated_ttnn_smoke",
    )

    assert result["aggregate"]["request_count"] == 1
    assert result["aggregate"]["passed_count"] == 1
    assert result["aggregate"]["top1_match_count"] == 1
    assert result["aggregate"]["aggregate_decode_tokens_per_sec_per_user"] > 0.0
    assert result["per_request"][0]["mode"] == "ttnn"
    assert result["per_request"][0]["top1_ids_match"] is True
    assert result["per_request"][0]["ttnn_generated_ids"]
