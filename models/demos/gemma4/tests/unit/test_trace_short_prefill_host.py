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


def test_opt_in_off_by_default_off_the_column_mesh():
    # no mesh shape (demo / host paths) or a row mesh: off unless GEMMA4_TRACE_SHORT_PREFILL=1
    assert not gt.trace_short_prefill_opt_in()
    assert not gt.trace_short_prefill_opt_in(128, 1)
    assert not gt.trace_short_prefill_opt_in(128, 1, mesh_shape=(1, 4))
    assert not gt.trace_short_prefill_opt_in(128, 1, mesh_shape=(8, 4))


def test_default_on_for_the_blackhole_column_mesh(monkeypatch):
    from models.common.utility_functions import is_blackhole

    on = gt.trace_short_prefill_default((4, 1))
    assert on == bool(is_blackhole())  # the validated 4-chip column mesh, Blackhole only
    monkeypatch.setenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, "0")
    assert not gt.trace_short_prefill_default((4, 1))
    assert not gt.trace_short_prefill_opt_in(128, 1, mesh_shape=(4, 1))
    monkeypatch.setenv(gt.GEMMA4_TRACE_SHORT_PREFILL_ENV, "1")
    assert gt.trace_short_prefill_default((1, 4))


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


def test_batched_prefill_trace_buckets_off_by_default(monkeypatch):
    monkeypatch.delenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_ENV, raising=False)
    monkeypatch.delenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_BUCKETS_ENV, raising=False)
    assert gt.batched_prefill_trace_buckets() == set()
    assert not gt.batched_prefill_trace_bucket(31, list(range(31)), 32, 128)


def test_batched_prefill_trace_buckets_opt_in(monkeypatch):
    monkeypatch.setenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_ENV, "1")
    monkeypatch.delenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_BUCKETS_ENV, raising=False)
    assert gt.batched_prefill_trace_buckets() == {(32, 128)}  # the QB2 32-user ramp bucket
    # a 31-user cold call at 128 tokens pads to 32 (batched_prefill_padded_batch) -> replays the bucket
    assert gt.batched_prefill_trace_bucket(31, list(range(31)), 32, 128)
    assert gt.batched_prefill_trace_bucket(32, None, 32, 128)
    # slots spanning the top of the batch pad to 32 as well
    assert gt.batched_prefill_trace_bucket(2, [0, 31], 32, 128)
    # other shapes keep the sequential single-user replays
    assert not gt.batched_prefill_trace_bucket(2, [0, 1], 32, 128)
    assert not gt.batched_prefill_trace_bucket(31, list(range(31)), 32, 1024)
    assert not gt.batched_prefill_trace_bucket(1, [0], 32, 128)
    assert not gt.batched_prefill_trace_bucket(31, list(range(31)), 32, None)
    monkeypatch.setenv(gt.GEMMA4_TRACE_BATCHED_PREFILL_BUCKETS_ENV, "32x128, 16x512,8x4096,3x128,32x100")
    # 8 x 4096 = 32768 virtual tokens is a legal bucket; 3 is not a padded batch, 100 is not a trace ISL
    assert gt.batched_prefill_trace_buckets() == {(32, 128), (16, 512), (8, 4096)}
