# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the tool-calling API latency sweep."""

from __future__ import annotations

import json

from transformers import AutoTokenizer

from models.autoports.meta_models_muse_glimmer_30b.doc.serving_perf.tool_call_latency_sweep import (
    HF_MODEL_ID,
    WEIGHT_REVISION,
    _append_tool_delta,
    exact_prompt,
    median_row,
    prompt_tokens,
)


def test_exact_prompt_builds_the_requested_tool_template_length():
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID,
        revision=WEIGHT_REVISION,
        local_files_only=True,
    )
    for target in (512, 1024):
        assert prompt_tokens(tokenizer, exact_prompt(tokenizer, target)) == target


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
