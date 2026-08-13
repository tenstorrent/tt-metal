# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only policy for auto-enabling traced multi-chunk prefill."""

import os

from models.demos.gemma4.tt.generator_trace import (
    chunked_prefill_trace_enabled,
    maybe_auto_enable_chunked_prefill_trace,
)


def test_auto_enable_chunked_prefill_trace_policy(monkeypatch):
    monkeypatch.delenv("GEMMA4_CHUNKED_PREFILL_TRACE", raising=False)

    assert not maybe_auto_enable_chunked_prefill_trace(
        batch_size=1, max_seq_len=1024, prefill_chunk=2048, bounded_sliding=False
    )
    assert "GEMMA4_CHUNKED_PREFILL_TRACE" not in os.environ

    assert not maybe_auto_enable_chunked_prefill_trace(
        batch_size=8, max_seq_len=8192, prefill_chunk=2048, bounded_sliding=False
    )
    assert not maybe_auto_enable_chunked_prefill_trace(
        batch_size=1, max_seq_len=8192, prefill_chunk=2048, bounded_sliding=True
    )

    assert maybe_auto_enable_chunked_prefill_trace(
        batch_size=1, max_seq_len=4096, prefill_chunk=2048, bounded_sliding=False
    )
    assert chunked_prefill_trace_enabled()

    monkeypatch.setenv("GEMMA4_CHUNKED_PREFILL_TRACE", "0")
    assert not maybe_auto_enable_chunked_prefill_trace(
        batch_size=1, max_seq_len=4096, prefill_chunk=2048, bounded_sliding=False
    )
