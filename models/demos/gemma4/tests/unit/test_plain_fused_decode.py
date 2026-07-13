# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only unit tests for plain fused greedy decode helpers.

Device parity (token-identical vs host greedy) is covered by the demo A/B on QB2
— see Legacy_Gemma4_performance.md Phase B.
"""

from __future__ import annotations

from models.demos.gemma4.tt.plain_fused_decode import PlainFusedGreedyDecoder, fused_greedy_enabled


def test_fused_greedy_enabled_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_FUSED_GREEDY", raising=False)
    assert fused_greedy_enabled() is False
    monkeypatch.setenv("GEMMA4_FUSED_GREEDY", "1")
    assert fused_greedy_enabled() is True
    monkeypatch.setenv("GEMMA4_FUSED_GREEDY", "0")
    assert fused_greedy_enabled() is False


def test_release_fused_trace_idempotent():
    dec = object.__new__(PlainFusedGreedyDecoder)
    dec._fused_trace = None
    dec.mesh_device = None
    PlainFusedGreedyDecoder.release_fused_trace(dec)


def test_release_fused_trace_clears_buffers(monkeypatch):
    import models.demos.gemma4.tt.plain_fused_decode as mod

    dec = object.__new__(PlainFusedGreedyDecoder)
    released = {"trace": False}

    class _FakeTensor:
        def __init__(self):
            self.deallocated = False

        def deallocate(self, _force=True):
            self.deallocated = True

    class _Mesh:
        pass

    def _fake_release(mesh, tid):
        released["trace"] = True

    monkeypatch.setattr(mod.ttnn, "release_trace", _fake_release)

    dec.mesh_device = _Mesh()
    tok = _FakeTensor()
    idx = _FakeTensor()
    dec._fused_trace = {
        "id": object(),
        "tok": tok,
        "idx": idx,
        "pos_u": None,
        "pos_i": None,
        "pt": None,
    }

    PlainFusedGreedyDecoder.release_fused_trace(dec)
    assert released["trace"] is True
    assert dec._fused_trace is None
    assert tok.deallocated is True
    assert idx.deallocated is True


def test_sanitize_token_id(expect_error):
    dec = object.__new__(PlainFusedGreedyDecoder)

    class _T:
        vocab_size = 100

    dec.target = _T()
    assert dec._sanitize_token_id(7) == 7
    with expect_error(ValueError, "invalid token id"):
        dec._sanitize_token_id(100)
