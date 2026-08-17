# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only policy for opting single-chunk demo 4k into prefill Metal Trace."""

from types import SimpleNamespace

from models.demos.gemma4.tt import generator_trace as gt


def test_ensure_trace_prefill_seq_len_adds_once(monkeypatch):
    monkeypatch.setattr(gt, "GEMMA4_TRACE_PREFILL_SEQ_LENS", [96, 128, 512, 1024, 2048])
    assert gt.ensure_trace_prefill_seq_len(4096) is True
    assert 4096 in gt.GEMMA4_TRACE_PREFILL_SEQ_LENS
    assert gt.ensure_trace_prefill_seq_len(4096) is False
    assert gt.ensure_trace_prefill_seq_len(8192) is False  # above max bucket


def test_enable_single_chunk_demo_bucket_only_for_true_4k(monkeypatch):
    monkeypatch.setattr(gt, "GEMMA4_TRACE_PREFILL_SEQ_LENS", [96, 128, 512, 1024, 2048])
    patched = []

    def _fake_patch(args, *, prefill_trace_enabled=True):
        patched.append((args, prefill_trace_enabled))

    monkeypatch.setattr(gt, "patch_gemma4_trace_model_args", _fake_patch)
    args = SimpleNamespace(max_prefill_chunk_size=4096)

    assert not gt.enable_single_chunk_demo_prefill_trace_bucket(
        max_seq_len=1024, max_prefill_chunk_size=4096, model_args_list=[args]
    )
    assert 4096 not in gt.GEMMA4_TRACE_PREFILL_SEQ_LENS

    assert not gt.enable_single_chunk_demo_prefill_trace_bucket(
        max_seq_len=8192, max_prefill_chunk_size=2048, model_args_list=[args]
    )

    assert not gt.enable_single_chunk_demo_prefill_trace_bucket(
        max_seq_len=4096, max_prefill_chunk_size=4096, model_args_list=[args], batch_size=32
    )
    assert 4096 not in gt.GEMMA4_TRACE_PREFILL_SEQ_LENS

    assert gt.enable_single_chunk_demo_prefill_trace_bucket(
        max_seq_len=4096, max_prefill_chunk_size=4096, model_args_list=[args]
    )
    assert 4096 in gt.GEMMA4_TRACE_PREFILL_SEQ_LENS
    assert patched == [(args, True)]
