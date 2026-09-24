# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A bounded sliding ring must not evict history a candidate query still needs.

A packed verify writes all candidate KV (p+1..p+K) BEFORE attention runs. With
a ring of exactly ``sliding_window`` slots, slot (p+j)%W holds position p+j-W,
which for j>=1 is still inside the live window [p-W+1, p]. Those writes destroy
positions an EARLIER candidate query in the same forward still needs, and
masking future candidates cannot restore them.

The arithmetic is checkable without a device, and it is the whole of the
defect: candidate j is safe exactly when p+j-R < p-W+1, i.e. j < R-W+1. A ring
equal to the window admits no j>=1 at all.
"""

import os

import pytest

pytest.importorskip("vllm")
from models.demos.gemma4.tt import generator_vllm as gv
from models.demos.gemma4.tt.attention import SPEC_RING_HEADROOM_ENV, bounded_ring_modulo

WINDOW = 1024


def _evicted_in_window(position, window, ring, candidates):
    """Candidates whose write lands on a slot holding a still-live position."""
    live_lo = position - window + 1
    hit = []
    for j in range(1, candidates + 1):
        displaced = (position + j) - ring  # what that slot held before the write
        if live_lo <= displaced <= position:
            hit.append((j, displaced))
    return hit


def test_an_exact_window_ring_evicts_from_the_first_candidate():
    """The defect, stated as arithmetic: this is what an unset headroom gives."""
    assert _evicted_in_window(131072, WINDOW, WINDOW, 6)[0] == (1, 130049)
    assert len(_evicted_in_window(131072, WINDOW, WINDOW, 6)) == 6


def test_the_reserved_ring_evicts_nothing(monkeypatch):
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW // 64))
    ring = bounded_ring_modulo(WINDOW)
    assert ring == 2 * WINDOW
    assert _evicted_in_window(131072, WINDOW, ring, 6) == []


def test_every_candidate_query_keeps_its_full_window(monkeypatch):
    """Victor's check: each candidate query's visible positions, not just the first."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW // 64))
    ring = bounded_ring_modulo(WINDOW)
    base = 131072
    for q in range(0, 6 + 1):  # query at each candidate position
        pos = base + q
        # positions this query must see, and the slots they occupy
        needed = range(pos - WINDOW + 1, pos + 1)
        slots = {p % ring: p for p in needed}
        assert len(slots) == WINDOW, f"query {pos} lost positions to slot aliasing"
        # no later candidate write in this same forward lands on those slots
        for j in range(q + 1, 6 + 1):
            assert (base + j) % ring not in slots or slots[(base + j) % ring] == base + j


def test_an_absent_headroom_refuses_instead_of_reserving_or_warning(monkeypatch, expect_error):
    """It must not reserve silently, and it must not continue either.

    Not reserving: doubling the ring doubles the bounded pool for every sliding
    layer (50 on 31B) and OOMs the shipped P150x8 config during KV allocation,
    so it cannot be switched on by default.

    Not continuing: this test used to assert that it warns and proceeds, on the
    grounds that "a server that cannot boot is worse" than wrong tokens. That
    judgement was wrong and tt-metal#57701 is the report -- a server on an exact
    ring produces plausible, wrong text with one startup warning as the only
    trace, which is worse than a boot failure that names the remedy.

    The ring itself is still left alone; the guard refuses rather than mutating
    a process-wide value the model and trace paths also read.
    """
    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
    with expect_error(RuntimeError, "exact-window ring"):
        gv._reserve_spec_ring_headroom(WINDOW, 6, "test")
    assert SPEC_RING_HEADROOM_ENV not in os.environ
    assert bounded_ring_modulo(WINDOW) == WINDOW


def test_an_operator_value_is_left_alone(monkeypatch):
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "48")
    gv._reserve_spec_ring_headroom(WINDOW, 6, "test")
    assert os.environ[SPEC_RING_HEADROOM_ENV] == "48"


def test_the_ring_stays_a_power_of_two(monkeypatch):
    """Chunk starts must be multiples of both the ring and SDPA's q_chunk_size."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW // 64))
    ring = bounded_ring_modulo(WINDOW)
    assert ring & (ring - 1) == 0


def test_the_pool_is_allocated_from_the_ring_not_the_window(monkeypatch):
    """Allocation, page tables and modulo have to move together.

    Sizing the pool from the bare window while wrapping modulo the ring writes
    past the pool, which is why setting the headroom env alone was never
    sufficient.
    """
    from types import SimpleNamespace

    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
    m = gv.Gemma4DFlashForCausalLM.__new__(gv.Gemma4DFlashForCausalLM)
    m._bounded_sliding_kv_cache = True
    m.model_args = [SimpleNamespace(max_batch_size=32)]
    monkeypatch.setattr(gv.Gemma4DFlashForCausalLM, "_text_config", lambda self: SimpleNamespace(sliding_window=WINDOW))
    window_blocks = (WINDOW // 64) * 32
    assert m._bounded_sliding_physical_blocks(64) == window_blocks, "exact ring by default"
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW // 64))
    assert m._bounded_sliding_physical_blocks(64) == window_blocks * 2, "follows the ring"
