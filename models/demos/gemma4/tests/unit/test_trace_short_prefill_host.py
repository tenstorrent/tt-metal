# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the short-prefill trace opt-in (GEMMA4_TRACE_SHORT_PREFILL, tt/generator_trace.py): the
bounded-sliding trace policy keeps the trace only for a single short user with the flag on. No device.

    pytest models/demos/gemma4/tests/unit/test_trace_short_prefill_host.py --noconftest -p no:cacheprovider
"""

from types import SimpleNamespace

import pytest

from models.demos.gemma4.tt import generator_trace as gt


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, raising=False)
    monkeypatch.delenv("GEMMA4_TRACE_SHORT_PREFILL_MAX", raising=False)


def test_opt_in_off_by_default():
    assert not gt.trace_short_prefill_opt_in()
    assert not gt.trace_short_prefill_opt_in(128, 1)


def test_opt_in_short_single_user_only(monkeypatch):
    monkeypatch.setenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, "1")
    assert gt.trace_short_prefill_opt_in()
    assert gt.trace_short_prefill_opt_in(100, 1)  # pads to the 128 bucket
    assert gt.trace_short_prefill_opt_in(128, 1)
    assert not gt.trace_short_prefill_opt_in(129, 1)  # pads to the 1024 bucket (get_padded_prefill_len)
    assert not gt.trace_short_prefill_opt_in(128, 2)  # single user only
    monkeypatch.setenv("GEMMA4_TRACE_SHORT_PREFILL_MAX", "1024")
    assert gt.trace_short_prefill_opt_in(1000, 1)
    assert not gt.trace_short_prefill_opt_in(1025, 1)


def _resolve(enable_trace, seq_len, batch_size=1, can_batch=False, bounded=True):
    model = SimpleNamespace(bounded_sliding_kv_cache=bounded, hidden_size_per_layer_input=0)
    model_args = SimpleNamespace(max_batch_size=32, hidden_size_per_layer_input=0)
    return gt.resolve_gemma4_prefill_trace_enable(
        enable_trace,
        model,
        model_args,
        batch_size=batch_size,
        prefill_seq_lens=[seq_len],
        can_batch_prefill=can_batch,
        empty_slots=None,
    )


def test_bounded_policy_with_and_without_opt_in(monkeypatch):
    assert not _resolve(True, 128)  # bounded: eager by default
    monkeypatch.setenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, "1")
    assert _resolve(True, 128)  # the 128 bucket is a trace bucket
    assert not _resolve(True, 2048)  # longer prompts stay eager under bounded sliding
    assert not _resolve(True, 128, batch_size=2, can_batch=True)  # batched prefills stay eager
    assert not _resolve(False, 128)  # caller disabled
    assert _resolve(True, 128, bounded=False)  # unbounded: the usual policy


def test_short_prefill_stash_skipped_only_in_the_traced_bounded_short_bucket(monkeypatch):
    # the traced short single-user bucket under bounded sliding records no sliding-tail stash ...
    assert gt.short_prefill_stash_skipped(128, 1, bounded=True, trace_mode=True)
    assert gt.short_prefill_stash_skipped(64, 1, bounded=True, trace_mode=True)
    # ... every eager call keeps it (continuations are eager), as do unbounded, batched and long traced buckets
    assert not gt.short_prefill_stash_skipped(128, 1, bounded=True, trace_mode=False)
    assert not gt.short_prefill_stash_skipped(128, 1, bounded=False, trace_mode=True)
    assert not gt.short_prefill_stash_skipped(128, 2, bounded=True, trace_mode=True)
    assert not gt.short_prefill_stash_skipped(1024, 1, bounded=True, trace_mode=True)
    monkeypatch.setenv("GEMMA4_TRACE_SHORT_PREFILL_MAX", "1024")
    assert gt.short_prefill_stash_skipped(1024, 1, bounded=True, trace_mode=True)
