# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from models.common.models.llama3_8b.hf_adaptor import Llama3RuntimeConfig, _trace_prefill_supported_seq_lens


def test_llama3_8b_trace_prefill_lengths_match_tttv1_devices():
    assert _trace_prefill_supported_seq_lens("N150", 4 * 1024, 128 * 1024) == (128, 1024)
    assert _trace_prefill_supported_seq_lens("N300", 64 * 1024, 128 * 1024) == (
        128,
        1024,
        2048,
        4096,
        8192,
    )
    assert _trace_prefill_supported_seq_lens("T3K", 128 * 1024, 128 * 1024) == (
        128,
        1024,
        2048,
        4096,
        8192,
    )


def test_llama3_8b_trace_prefill_lengths_respect_model_limits():
    assert _trace_prefill_supported_seq_lens("N300", 4 * 1024, 128 * 1024) == (128, 1024, 2048, 4096)
    assert _trace_prefill_supported_seq_lens("T3K", 128 * 1024, 2048) == (128, 1024, 2048)


def test_llama3_8b_runtime_trace_gate_uses_supported_lengths():
    runtime_config = Llama3RuntimeConfig(
        model_name="Llama-3.1-8B-Instruct",
        model_cache_path="model_cache",
        max_prefill_chunk_size=64 * 1024,
        max_context_len=128 * 1024,
        trace_prefill_supported_seq_lens=(128, 1024, 2048, 4096, 8192),
    )

    assert runtime_config.can_enable_trace(8192)
    assert not runtime_config.can_enable_trace(16384)
    assert not runtime_config.can_enable_trace(8192, num_cached_tokens=128)
