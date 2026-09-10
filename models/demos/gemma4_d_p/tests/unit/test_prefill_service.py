# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host validation for the service boundary, before any device work."""

import importlib
from types import SimpleNamespace

import pytest

from models.demos.gemma4_d_p.tt.runners.kv_caches import Gemma4KvCaches
from models.demos.gemma4_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig


@pytest.mark.parametrize(
    "slot,start,end,message",
    [
        (-1, 0, 8192, "slot_id"),
        (2, 0, 8192, "slot_id"),
        (0, -8192, 0, "32-token aligned"),
        (0, 7000, 9000, "32-token aligned"),
        (0, 0, 0, "invalid chunk range"),
        (0, 0, 8193, "invalid chunk range"),
        (0, 16384, 24576, "exceed"),
    ],
)
def test_invalid_requests_fail_before_device_work(slot, start, end, message, expect_error):
    runtime = TtPrefillRuntime(object(), "google/gemma-4-31B-it", TtPrefillRuntimeConfig(60, 16384, num_users=2))
    caches = Gemma4KvCaches([], (), 2, 16384, 8, 4)
    with expect_error(ValueError, message):
        runtime.prefill_chunk(None, caches, slot_id=slot, actual_start=start, actual_end=end)


def test_synthetic_producer_needs_no_golden_trace(monkeypatch, expect_error):
    monkeypatch.setenv("PREFILL_MODEL", "gemma4_31b")
    monkeypatch.setenv("PREFILL_PRODUCER_SYNTHETIC_TOKENS", "1")
    producer = importlib.import_module("models.demos.common.prefill.runners.prefill_producer")
    cfg = SimpleNamespace(verify=False, multi_turn_prob=0, chunks_max=2, num_users=2)
    slots, lengths, pools = producer._resolve_slot_prompts(cfg)
    assert set(slots) == {0, 1}
    assert lengths is None
    assert len(pools[slots[0]]) == 2 * producer.CHUNK_SIZE
    cfg.verify = True
    with expect_error(ValueError, "synthetic tokens cannot"):
        producer._resolve_slot_prompts(cfg)
