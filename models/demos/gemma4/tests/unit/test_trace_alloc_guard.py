# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for the decode-trace allocation guard: a device allocation that appears after the
decode trace was captured (and is still alive at the next replay) makes the generator release and
re-capture its decode traces, a bounded number of times."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin


class _View:
    """Allocator view with a bottom-up run of ``allocated`` bytes, one free gap, and a top-down block."""

    def __init__(self, allocated, top_down=4096, total=1 << 20):
        self.total_bytes_allocated_per_bank = allocated + top_down
        self.block_table = [
            {"address": "0", "size": str(allocated), "allocated": "yes"},
            {"address": str(allocated), "size": str(total - allocated - top_down), "allocated": "no"},
            {"address": str(total - top_down), "size": str(top_down), "allocated": "yes"},
        ]


def _guard(monkeypatch, levels):
    """A bare mixin instance whose allocator view reports ``levels[buffer_type]`` bytes per bank."""
    monkeypatch.setattr(ttnn, "get_memory_view", lambda mesh, buffer_type: _View(levels[buffer_type]))
    g = ChunkedPrefillPageTableGuardMixin.__new__(ChunkedPrefillPageTableGuardMixin)
    g.model_args = [SimpleNamespace(mesh_device=object())]
    g.model = [SimpleNamespace(sampling=SimpleNamespace(reset_trace=lambda: None))]
    g.trace_ids_decode = defaultdict(lambda: None)
    g.trace_inputs_decode = defaultdict(lambda: None)
    g.trace_output_decode = defaultdict(lambda: None)
    g._g4_trace_alloc_baseline = {}
    g._g4_trace_guard_recaptures = 0
    g._g4_alloc_check_pending = True
    return g


def test_no_growth_no_recapture(monkeypatch):
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    assert not g._g4_trace_guard_should_recapture("k")  # nothing captured yet
    g._g4_trace_alloc_baseline["k"] = g._g4_alloc_levels()
    assert not g._g4_trace_guard_should_recapture("k")
    levels[ttnn.BufferType.DRAM] = 900  # something freed: still fine, and the baseline follows it down
    g._g4_alloc_check_pending = True
    assert not g._g4_trace_guard_should_recapture("k")
    assert g._g4_trace_alloc_baseline["k"][0] == 900
    levels[ttnn.BufferType.DRAM] = 950  # above the lowest level seen: a post-capture allocation
    g._g4_alloc_check_pending = True
    assert g._g4_trace_guard_should_recapture("k")


@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM, ttnn.BufferType.L1])
def test_growth_triggers_recapture_until_the_limit(monkeypatch, buffer_type):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MAX", "2")
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    g._g4_trace_alloc_baseline["k"] = g._g4_alloc_levels()
    levels[buffer_type] += 64
    g._g4_alloc_check_pending = True
    assert g._g4_trace_guard_should_recapture("k")
    g._g4_alloc_check_pending = True
    assert g._g4_trace_guard_should_recapture("k")
    g._g4_alloc_check_pending = True
    assert not g._g4_trace_guard_should_recapture("k")  # limit spent: logged, no more re-captures
    assert g._g4_trace_guard_recaptures == 3


def test_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD", "0")
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    g._g4_trace_alloc_baseline["k"] = g._g4_alloc_levels()
    levels[ttnn.BufferType.DRAM] += 64
    assert not g._g4_trace_guard_should_recapture("k")


def test_top_down_growth_and_replays_only_do_not_recapture(monkeypatch):
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    g._g4_trace_alloc_baseline["k"] = g._g4_alloc_levels()
    # kernel binaries: top-down, never inside the replay's bottom-up scratch
    monkeypatch.setattr(
        ttnn,
        "get_memory_view",
        lambda mesh, bt: _View(levels[bt], top_down=65536 if bt == ttnn.BufferType.DRAM else 4096),
    )
    g._g4_alloc_check_pending = True
    assert not g._g4_trace_guard_should_recapture("k")
    # bottom-up growth without a prefill in between is not checked (replays allocate nothing persistent)
    levels[ttnn.BufferType.DRAM] += 64
    g._g4_alloc_check_pending = False
    assert not g._g4_trace_guard_should_recapture("k")


def test_warmup_captured_trace_gets_its_baseline_at_the_first_replay(monkeypatch):
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    assert not g._g4_trace_guard_should_recapture("w")  # no baseline yet: recorded, no re-capture
    assert g._g4_trace_alloc_baseline["w"] == g._g4_alloc_levels()


def test_release_drops_every_trace_and_resets_sampling(monkeypatch):
    levels = {ttnn.BufferType.DRAM: 1000, ttnn.BufferType.L1: 10}
    g = _guard(monkeypatch, levels)
    released, resets = [], []
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: released.append(tid))
    g.model[0].sampling.reset_trace = lambda: resets.append(1)
    g.trace_ids_decode["a"] = {0: 11}
    g.trace_ids_decode["b"] = {0: 12}
    g._prev_decode_batch = 128  # the feedback buffer is found through this key: it must survive the release
    g.trace_output_decode["a"] = [SimpleNamespace()]  # dropped, never freed by the release
    g._g4_trace_alloc_baseline["a"] = (1, 2)
    g._g4_release_decode_traces("test")
    assert sorted(released) == [11, 12]
    assert resets == [1]
    assert not g.trace_ids_decode["a"] and not g.trace_output_decode["a"]
    assert g._g4_trace_alloc_baseline == {}
    assert g._g4_recapture_without_precompile  # the live request's sampler state must not be touched
    assert g._prev_decode_batch == 128
