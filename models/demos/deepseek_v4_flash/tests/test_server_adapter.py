# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v4_flash.demo import (
    DEMO_NAME,
    DemoTimings,
    GenerationTimings,
    PreparedCheckpoint,
    TinyGenerationResult,
    require_t3k_available,
)
from models.demos.deepseek_v4_flash.server_adapter import (
    ADAPTER_NAME,
    TinyServerRequest,
    ensure_tiny_server_request,
    run_tiny_server_request,
    summarize_tiny_server_result,
)


def test_tiny_server_request_normalizes_json_style_inputs(tmp_path) -> None:
    request = TinyServerRequest(
        input_ids=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        prompt="caller-owned prompt metadata",
        generate_steps=2,
        top_k=3,
        layer_ids=[0, 2],
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert request.input_ids == (0, 1, 2)
    assert request.prompt == "caller-owned prompt metadata"
    assert request.mode == "generate"
    assert request.generate_steps == 2
    assert request.decode_steps == 0
    assert request.top_k == 3
    assert request.layer_ids == (0, 2)
    assert request.artifact_dir == Path(tmp_path / "artifacts")
    assert request.input_ids_tensor(vocab_size=64).tolist() == [[0, 1, 2]]

    mapped = ensure_tiny_server_request(
        {
            "input_ids": [0, 1],
            "decode_steps": 2,
            "decode_input_ids": [2, 3],
            "top_k": 1,
        }
    )

    assert mapped.mode == "decode"
    assert mapped.decode_input_ids_tensor(vocab_size=64).tolist() == [[2, 3]]


def test_tiny_server_request_validation_rejects_unsupported_contracts(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one token"):
        TinyServerRequest(input_ids=[])
    with pytest.raises(TypeError, match=r"input_ids\[0\] must be an integer"):
        TinyServerRequest(input_ids=[True])
    with pytest.raises(ValueError, match="generate_steps must be non-negative"):
        TinyServerRequest(input_ids=[0], generate_steps=-1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        TinyServerRequest(input_ids=[0], generate_steps=1, decode_steps=1)
    with pytest.raises(ValueError, match="decode_input_ids length"):
        TinyServerRequest(input_ids=[0], decode_steps=2, decode_input_ids=[1])
    with pytest.raises(ValueError, match="top_k must be positive"):
        TinyServerRequest(input_ids=[0], top_k=0)
    with pytest.raises(ValueError, match="duplicates"):
        TinyServerRequest(input_ids=[0], layer_ids=[0, 0])
    with pytest.raises(ValueError, match="mutually exclusive"):
        TinyServerRequest(
            input_ids=[0],
            preprocessed_path=tmp_path / "tt_preprocessed",
            artifact_dir=tmp_path / "artifacts",
        )
    with pytest.raises(ValueError, match="Unknown TinyServerRequest"):
        ensure_tiny_server_request({"input_ids": [0], "batch_size": 2})
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        TinyServerRequest(input_ids=[4]).input_ids_tensor(vocab_size=4)


def test_tiny_server_summary_matches_demo_generation_metrics_and_is_json_serializable() -> None:
    request = TinyServerRequest(
        input_ids=[0, 1],
        prompt="metadata only",
        generate_steps=2,
        top_k=2,
        layer_ids=(1, 2),
    )
    prefill_logits = torch.tensor(
        [
            [
                [0.0, 0.25, 2.0, 1.0],
                [0.0, 3.0, 1.0, 2.0],
            ]
        ],
        dtype=torch.float32,
    )
    decode_logits = torch.tensor(
        [
            [
                [4.0, 0.0, 1.0, 2.0],
                [0.0, 1.0, 5.0, 2.0],
            ]
        ],
        dtype=torch.float32,
    )
    generation_result = TinyGenerationResult(
        prefill_logits=prefill_logits,
        decode_logits=decode_logits,
        generated_token_ids=torch.tensor([[1, 0]], dtype=torch.int64),
        cache_current_position=4,
        timings=GenerationTimings(prefill_s=0.2, decode_s=0.5, total_s=0.8),
    )

    summary = summarize_tiny_server_result(
        request=request,
        logits=prefill_logits,
        generation_result=generation_result,
        input_ids=torch.tensor([[0, 1]], dtype=torch.int64),
        timings=DemoTimings(setup_s=0.01, model_init_s=0.02, warmup_s=0.0, run_s=0.8, total_s=0.9),
        checkpoint=PreparedCheckpoint(Path("/tmp/tt_preprocessed"), generated_synthetic_checkpoint=False),
        layer_ids=(1, 2),
        device_ownership="unit-test",
    )

    json.dumps(summary, sort_keys=True)
    assert summary["demo"] == DEMO_NAME
    assert summary["adapter"]["name"] == ADAPTER_NAME
    assert summary["adapter"]["mode"] == "generate"
    assert summary["adapter"]["batch_size"] == 1
    assert summary["adapter"]["input_ids"] == [0, 1]
    assert summary["adapter"]["prompt"] == "metadata only"
    assert summary["adapter"]["device_ownership"] == "unit-test"
    assert "host-owned compressed decode cache" in summary["adapter"]["limitations"]
    assert "not final vLLM or production tt-inference-server integration" in summary["adapter"]["limitations"]
    assert summary["input"]["token_count"] == 2
    assert summary["input"]["layer_ids"] == [1, 2]
    assert summary["input"]["generate_steps"] == 2
    assert summary["generation"]["generated_token_ids"] == [1, 0]
    assert summary["generation"]["decode_cache_current_position"] == 4
    assert summary["generation"]["metrics"]["generated_tokens"] == 2
    assert summary["generation"]["metrics"]["per_token_decode_latency_s"] == 0.25
    assert summary["generation"]["metrics"]["effective_decode_tokens_s_per_user"] == 4.0


@pytest.mark.t3k_compat
def test_t3k_tiny_server_adapter_generation_smoke(tmp_path) -> None:
    ttnn = pytest.importorskip("ttnn")
    try:
        require_t3k_available(ttnn)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    summary = run_tiny_server_request(
        TinyServerRequest(
            input_ids=[0] * 8,
            generate_steps=1,
            top_k=3,
            artifact_dir=tmp_path,
        )
    )

    json.dumps(summary, sort_keys=True)
    assert summary["adapter"]["name"] == ADAPTER_NAME
    assert summary["adapter"]["mode"] == "generate"
    assert summary["adapter"]["device_ownership"] == "adapter-owned-t3k-mesh"
    assert summary["input"]["token_count"] == 8
    assert summary["input"]["generate_steps"] == 1
    assert summary["generation"]["generated_token_ids"]
    assert len(summary["generation"]["generated_token_ids"]) == 1
    assert summary["generation"]["decode_cache_current_position"] == 9
    assert summary["generation"]["metrics"]["prompt_tokens"] == 8
    assert summary["generation"]["metrics"]["generated_tokens"] == 1
    assert summary["generation"]["metrics"]["prefill_latency_s"] > 0.0
    assert summary["generation"]["metrics"]["total_decode_latency_s"] > 0.0
    assert summary["generation"]["metrics"]["per_token_decode_latency_s"] > 0.0
    assert summary["generation"]["metrics"]["effective_decode_tokens_s_per_user"] > 0.0
