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
from types import SimpleNamespace

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


def test_the_model_sizes_the_ring_rather_than_the_operator(monkeypatch):
    """With no headroom set, a speculating class gets a ring that fits its drafts.

    The premise of this test has been wrong twice, so it records what settled it.
    First it asserted the guard warns and continues, on the grounds that "a
    server that cannot boot is worse" than wrong tokens -- tt-metal#57701 is the
    report that this produces plausible, wrong text with one startup warning as
    the only trace. Then it asserted a refusal, on the grounds that doubling the
    ring OOMs the shipped P150x8 config. That was inherited from a comment
    written before tt-metal#57655 sized the bounded page tables from the ring;
    measured afterwards on P150x8 at max_num_seqs=32 with
    GEMMA4_MAX_TOKENS_ALL_USERS=262144, the doubled ring allocates and serves.

    So the headroom is neither warned about nor refused: the model sets it,
    because it alone knows the class, K, the window, and whether bounded sliding
    resolved on. An env var every launch must get right is the wrong interface.
    """
    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
    # _auto_size_spec_ring writes os.environ directly; monkeypatch cannot undo a
    # value it did not set, so restore it here or it leaks into later tests.
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "0")
    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV)
    gv._auto_size_spec_ring(
        SimpleNamespace(_SPEC_N=6, __name__="Gemma4DFlash"),
        SimpleNamespace(text_config=SimpleNamespace(sliding_window=WINDOW)),
        bounded_sliding=True,
    )
    assert bounded_ring_modulo(WINDOW) == 2 * WINDOW
    # and the guard that backstops it is satisfied by what the model chose
    gv._reserve_spec_ring_headroom(WINDOW, 6, "test")


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
