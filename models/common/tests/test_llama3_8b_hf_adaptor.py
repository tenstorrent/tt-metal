# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import errno
from pathlib import Path

from models.common.models.llama3_8b import hf_adaptor
from models.common.models.llama3_8b.hf_adaptor import (
    Llama3RuntimeConfig,
    _max_prefill_chunk_size,
    _model_cache_path,
    _trace_prefill_supported_seq_lens,
)


def test_model_cache_path_uses_configured_device_directory(monkeypatch, tmp_path):
    cache_root = tmp_path / "shared"
    monkeypatch.setenv("TT_CACHE_PATH", str(cache_root))
    monkeypatch.setattr(hf_adaptor, "get_device_name", lambda mesh_device: "N150")

    result = _model_cache_path("meta-llama/Llama-3.1-8B-Instruct", object())

    assert result == cache_root / "N150"
    assert result.is_dir()


def test_model_cache_path_falls_back_when_configured_cache_is_read_only(monkeypatch, tmp_path):
    cache_root = tmp_path / "shared"
    configured_path = cache_root / "N150"
    fallback_root = tmp_path / "fallback"
    real_mkdir = Path.mkdir

    def mkdir_with_read_only_cache(path, *args, **kwargs):
        if path == configured_path:
            raise OSError(errno.EROFS, "Read-only file system", str(path))
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setenv("TT_CACHE_PATH", str(cache_root))
    monkeypatch.setenv("TT_CACHE_FALLBACK_PATH", str(fallback_root))
    monkeypatch.setattr(hf_adaptor, "get_device_name", lambda mesh_device: "N150")
    monkeypatch.setattr(Path, "mkdir", mkdir_with_read_only_cache)

    result = _model_cache_path("meta-llama/Llama-3.1-8B-Instruct", object())

    assert result == fallback_root / "Llama-3.1-8B-Instruct" / "N150"
    assert result.is_dir()


def test_llama3_8b_n150x4_prefill_chunk_size_matches_compatibility_default(monkeypatch):
    monkeypatch.delenv("MAX_PREFILL_CHUNK_SIZE", raising=False)
    monkeypatch.setattr(hf_adaptor, "get_device_name", lambda mesh_device: "N150x4")

    assert _max_prefill_chunk_size(object()) == 4 * 1024


def test_llama3_8b_prefill_chunk_size_override_applies_to_n150x4(monkeypatch):
    monkeypatch.setenv("MAX_PREFILL_CHUNK_SIZE", "16")

    assert _max_prefill_chunk_size(object()) == 16 * 1024


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
    assert _trace_prefill_supported_seq_lens("N150x4", 4 * 1024, 128 * 1024) == (
        128,
        1024,
        2048,
        4096,
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
