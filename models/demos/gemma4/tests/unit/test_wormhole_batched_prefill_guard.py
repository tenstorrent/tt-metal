# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wormhole keeps gemma4 on the sequential per-user prefill path.

Batched prefill buys nothing on WH and collapses at >=512 tokens per user
(T3K 12B, concurrency 32: 51.1 vs 198.6 aggregate tok/s, and the device wedged
outright on a repeat run). Blackhole must be untouched -- that is where the
batched-prefill win was measured.
"""

import models.common.utility_functions as uf
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


def test_wormhole_disables_batched_prefill(monkeypatch):
    monkeypatch.setattr(uf, "is_blackhole", lambda: False)
    assert Gemma4ModelArgs().disable_batched_prefill is True


def test_blackhole_keeps_batched_prefill(monkeypatch):
    monkeypatch.setattr(uf, "is_blackhole", lambda: True)
    assert Gemma4ModelArgs().disable_batched_prefill is False


def test_explicit_value_is_never_overridden(monkeypatch):
    """A caller opting in stays opted in -- test_batched_prefill_perf relies on this."""
    monkeypatch.setattr(uf, "is_blackhole", lambda: False)
    args = Gemma4ModelArgs()
    args.disable_batched_prefill = False  # post-construction opt back in
    assert args.disable_batched_prefill is False

    monkeypatch.setattr(uf, "is_blackhole", lambda: True)
    assert Gemma4ModelArgs(disable_batched_prefill=True).disable_batched_prefill is True


def test_guard_is_silent_when_arch_cannot_be_determined(monkeypatch):
    """Host-only environments must not crash; they keep the shared default."""

    def boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(uf, "is_blackhole", boom)
    assert Gemma4ModelArgs().disable_batched_prefill is False


def test_hatch_env_bypasses_guard_on_wormhole(monkeypatch):
    """G4_FORCE_BATCH_PREFILL=1 must reach the generator gates on Wormhole too:
    the guard steps aside so the hatch means the same thing on both arches."""
    monkeypatch.setattr(uf, "is_blackhole", lambda: False)
    monkeypatch.setenv("G4_FORCE_BATCH_PREFILL", "1")
    assert Gemma4ModelArgs().disable_batched_prefill is False


def test_hatch_env_off_keeps_guard(monkeypatch):
    monkeypatch.setattr(uf, "is_blackhole", lambda: False)
    monkeypatch.setenv("G4_FORCE_BATCH_PREFILL", "0")
    assert Gemma4ModelArgs().disable_batched_prefill is True
