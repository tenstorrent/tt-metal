# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for decode sample-writeback sync gating."""

from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin


class _G(ChunkedPrefillPageTableGuardMixin):
    pass


def test_should_sync_after_eager_sample_writeback_defaults_and_override(monkeypatch):
    g = _G()
    g.model_capabilities = {"supports_async_decode": False}
    monkeypatch.delenv("GEMMA4_DECODE_SAMPLE_SYNC", raising=False)
    assert g._should_sync_after_eager_sample_writeback() is False

    g.model_capabilities = {"supports_async_decode": True}
    assert g._should_sync_after_eager_sample_writeback() is True

    monkeypatch.setenv("GEMMA4_DECODE_SAMPLE_SYNC", "0")
    assert g._should_sync_after_eager_sample_writeback() is False

    monkeypatch.setenv("GEMMA4_DECODE_SAMPLE_SYNC", "1")
    g.model_capabilities = {"supports_async_decode": False}
    assert g._should_sync_after_eager_sample_writeback() is True
