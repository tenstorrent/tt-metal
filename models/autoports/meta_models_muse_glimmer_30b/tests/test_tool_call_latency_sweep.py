# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the tool-calling API latency sweep."""

from __future__ import annotations

import json

from transformers import AutoTokenizer

from models.autoports.meta_models_muse_glimmer_30b.doc.serving_perf.tool_call_latency_sweep import (
    DEFAULT_ISLS,
    DEFAULT_REPEATS,
    HF_MODEL_ID,
    SOURCE_CONTEXT_FILES,
    WEIGHT_REVISION,
    _append_tool_delta,
    exact_prompt,
    median_row,
    prompt_messages,
    prompt_tokens,
    source_context,
    source_context_sha256,
    write_artifact,
)


def test_exact_prompt_builds_the_requested_tool_template_length():
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID,
        revision=WEIGHT_REVISION,
        local_files_only=True,
    )
    for target in DEFAULT_ISLS:
        prompt = exact_prompt(tokenizer, target)
        assert prompt_tokens(tokenizer, prompt) == target
        assert "# Latency padding:" in prompt
        assert prompt_messages(prompt)[0]["role"] == "system"
        assert prompt_messages(prompt)[1] == {"role": "user", "content": prompt}


def test_prompt_corpus_is_large_tracked_source_with_a_stable_digest():
    corpus = source_context()
    assert len(SOURCE_CONTEXT_FILES) == 14
    assert len(corpus) > 500_000
    assert len(source_context_sha256()) == 64


def test_streamed_tool_fragments_are_reassembled_by_index():
    calls = {}
    _append_tool_delta(
        calls,
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_",
                    "function": {"name": "record_", "arguments": '{"payload":'},
                }
            ]
        },
    )
    _append_tool_delta(
        calls,
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "1",
                    "function": {"name": "latency_probe", "arguments": '"ready"}'},
                }
            ]
        },
    )
    assert calls[0]["id"] == "call_1"
    assert calls[0]["name"] == "record_latency_probe"
    assert json.loads(calls[0]["arguments"]) == {"payload": "ready"}


def test_median_row_retains_all_validation_samples():
    samples = [
        {
            "completion_tokens": 20,
            "ttft_ms": 10.0,
            "tpot_ms_derived": 2.0,
            "e2el_ms": 48.0,
            "tokens_per_second_per_user_derived": 500.0,
            "tool_call_pass": True,
        },
        {
            "completion_tokens": 22,
            "ttft_ms": 12.0,
            "tpot_ms_derived": 2.2,
            "e2el_ms": 58.2,
            "tokens_per_second_per_user_derived": 454.5,
            "tool_call_pass": True,
        },
        {
            "completion_tokens": 21,
            "ttft_ms": 11.0,
            "tpot_ms_derived": 2.1,
            "e2el_ms": 53.0,
            "tokens_per_second_per_user_derived": 476.2,
            "tool_call_pass": True,
        },
    ]
    row = median_row(512, samples)
    assert row["isl"] == 512
    assert row["repeats"] == 3
    assert row["ttft_ms"] == 11.0
    assert row["completion_tokens"] == 21.0
    assert row["tool_call_pass"] is True
    assert row["samples"] is samples


def test_release_sweep_defaults_to_three_samples_and_checkpoints(tmp_path):
    assert DEFAULT_REPEATS == 3
    out = tmp_path / "nested" / "sweep.json"
    artifact = {
        "status": "in_progress",
        "source_revision": "abc123",
        "image_digest": "sha256:123",
        "completed_isls": [512],
        "rows": [{"isl": 512, "tool_call_pass": True}],
    }
    write_artifact(out, artifact)
    assert json.loads(out.read_text()) == artifact
