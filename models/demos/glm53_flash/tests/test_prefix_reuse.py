# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-blaze G53F-0311: prefill-side prefix reuse decisions (no device, no ttnn).

    pytest --noconftest models/demos/glm53_flash/tests/test_prefix_reuse.py

``FakeDevice`` models what the runner's device holds: a DSA row per position whose value depends on the chunk it was
computed in (the carries at the chunk's start, the chunk's start and its tokens so far), live KDA carries that depend on
the chunk partition, the decoders' contiguity check, checkpoint slots, and the export slabs the KV Manager ships (rows +
the KDA window). A resume from the wrong carries, a wrong partition, a stale slab row or a stale KDA window all change
what the KV Manager would ship. Every scenario compares that, and the model's own rows, to a cold prefill of the same
prompt on a fresh device.
"""

from __future__ import annotations

import random

import pytest
import torch

from models.demos.glm53_flash.tt.runners.prefix_reuse import (
    LAST_CHUNK_FLAG,
    MARKED_FLAG,
    CheckpointPool,
    PrefixRetention,
    split_slot_word,
)


def _h(*xs) -> int:
    return hash(xs) & 0xFFFFFFFFFFFF


class FakeDevice:
    def __init__(self, n_ckpts: int = 0):
        self.rows: dict[int, int] = {}
        self.carries = 0
        self.next_start = 0
        self.slots = [None] * n_ckpts
        self.slab: dict[int, int] = {}
        self.kda_slab = None
        self.model_chunks: list[tuple[int, int, str]] = []

    def prefill(self, ids, start: int, kind: str = "chunk") -> None:
        ids = [int(t) for t in ids]
        if start and start != self.next_start:
            raise ValueError(
                f"chunk start {start} is not contiguous with the previous chunk (expected {self.next_start})"
            )
        state = 0 if start == 0 else self.carries
        for i in range(len(ids)):
            self.rows[start + i] = _h(state, start, tuple(ids[: i + 1]))
        self.carries = _h(state, start, tuple(ids))
        self.next_start = start + len(ids)
        self.model_chunks.append((start, start + len(ids), kind))

    def export(self, start: int, end: int) -> None:
        for p in range(start, end):
            self.slab[p] = self.rows[p]
        self.kda_slab = self.carries

    def retention(self, chunk: int, n_ckpts: int, logs=None) -> PrefixRetention:
        def save(slot):
            self.slots[slot] = self.carries

        def restore(slot):
            assert self.slots[slot] is not None
            self.carries = self.slots[slot]

        def export_carries(slot):
            self.kda_slab = self.carries if slot is None else self.slots[slot]

        def set_next_start(pos):
            self.next_start = pos

        def silent(ids, start):
            self.prefill(ids.tolist(), start, kind="silent")

        return PrefixRetention(
            chunk=chunk,
            n_ckpts=n_ckpts,
            save=save,
            restore=restore,
            export_carries=export_carries,
            set_next_start=set_next_start,
            compute_silent=silent,
            log=(logs.append if logs is not None else (lambda m: None)),
        )


def chunks_of(S: int, C: int, cut: int | None = None):
    parts = [(s, min(S, s + C)) for s in range(0, S, C)]
    return parts[:cut] if cut else parts


def run_prompt(dev, pr, tokens, C, *, marks=True, cut=None):
    """The runtime's prefill_chunk for every chunk of one prompt; returns what each chunk did."""
    did = []
    parts = chunks_of(len(tokens), C, cut)
    for k, (s, e) in enumerate(parts):
        ids = tokens[s:e]
        last = (k == len(parts) - 1) if marks else None  # None: an unmarked header (any chunk may be the last)
        if pr is None:
            dev.prefill(ids, s)
            dev.export(s, e)
            did.append("C")
            continue
        if pr.begin_chunk(torch.tensor(ids), s, e, last):
            did.append("S")
            continue
        dev.prefill(ids, s)
        pr.end_compute(torch.tensor(ids), s, e)
        dev.export(s, e)
        did.append("C")
    return did


def shipped(dev, S):
    return tuple(dev.slab[p] for p in range(S)), dev.kda_slab


def cold(tokens, C, cut=None):
    d = FakeDevice()
    run_prompt(d, None, tokens, C, cut=cut)
    S = chunks_of(len(tokens), C, cut)[-1][1]
    return shipped(d, S), tuple(d.rows[p] for p in range(S))


def check_equal(dev, tokens, C, cut=None):
    S = chunks_of(len(tokens), C, cut)[-1][1]
    ref_ship, ref_rows = cold(tokens, C, cut)
    assert shipped(dev, S) == ref_ship, "the KV Manager would ship different bytes than a cold prefill"
    assert tuple(dev.rows[p] for p in range(S)) == ref_rows, "the model's own rows differ from a cold prefill"


def toks(n, seed):
    r = random.Random(seed)
    return [r.randrange(1, 50) for _ in range(n)]


C = 8


# ---------------------------------------------------------------------------------------------------- the slot word
def test_slot_word():
    assert split_slot_word(0) == (0, None)  # an engine without mark_last_chunk
    assert split_slot_word(3) == (3, None)
    assert split_slot_word(MARKED_FLAG) == (0, False)
    assert split_slot_word(MARKED_FLAG | LAST_CHUNK_FLAG) == (0, True)
    assert split_slot_word(3 | MARKED_FLAG | LAST_CHUNK_FLAG) == (3, True)
    assert split_slot_word(LAST_CHUNK_FLAG) == (LAST_CHUNK_FLAG, None)  # bit 30 means nothing without bit 29
    assert split_slot_word(-1) == (-1, None)  # the shutdown sentinel word is untouched


# ---------------------------------------------------------------------------------------------- the checkpoint pool
def test_pool_keeps_every_boundary_while_room():
    p = CheckpointPool(4, C)
    for k in range(1, 5):
        p.admit(k * C)
    assert sorted(p.by_pos) == [C, 2 * C, 3 * C, 4 * C] and p.evictions == 0


def test_pool_thins_evenly_and_keeps_the_two_newest():
    p = CheckpointPool(8, C)
    for k in range(1, 41):
        p.admit(k * C)
        held = sorted(q // C for q in p.by_pos)
        assert len(held) == min(k, 8)
        assert k in held and (k == 1 or k - 1 in held), held  # the two newest always survive
    held = sorted(q // C for q in p.by_pos)
    gaps = [b - a for a, b in zip([0] + held, held)]
    assert max(gaps) <= 8, (held, gaps)  # an even spread over 40 boundaries with 8 slots: no gap past 8 chunks
    assert p.evictions == 32


def test_pool_small_and_zero():
    assert CheckpointPool(0, C).admit(C) is None
    p = CheckpointPool(1, C)
    p.admit(C)
    p.admit(2 * C)
    assert sorted(p.by_pos) == [2 * C]
    p = CheckpointPool(2, C)
    for k in range(1, 6):
        p.admit(k * C)
    assert 5 * C in p.by_pos and len(p.by_pos) == 2


def test_pool_nearest_and_drop():
    p = CheckpointPool(8, C)
    for k in (1, 2, 4, 6):
        p.admit(k * C)
    assert p.nearest_at_or_below(5 * C)[0] == 4 * C
    assert p.nearest_at_or_below(C - 1) == (0, None)
    p.drop_above(2 * C)
    assert sorted(p.by_pos) == [C, 2 * C] and len(p.free) == 6


# ----------------------------------------------------------------------------------------------------- scenarios
def _setup(n_ckpts=8):
    dev = FakeDevice(n_ckpts)
    logs = []
    return dev, dev.retention(C, n_ckpts, logs=logs), logs


def test_extension_skips_the_shared_chunks_and_matches_cold():
    dev, pr, _ = _setup()
    P1 = toks(3 * C + 5, 1)
    P2 = P1 + toks(2 * C + 3, 2)  # a next turn: P1 + new content
    run_prompt(dev, pr, P1, C)
    check_equal(dev, P1, C)
    did = run_prompt(dev, pr, P2, C)
    assert did == ["S", "S", "S", "C", "C", "C"], did  # [0, 3C) skipped; [3C, 4C) resumed from the checkpoint at 3C
    check_equal(dev, P2, C)
    assert pr.stats.silent_chunks == 0 and pr.stats.skipped_tokens == 3 * C
    assert "silent" not in [k for _, _, k in dev.model_chunks]


def test_identical_resend_skips_everything_and_ships_the_live_carries():
    dev, pr, _ = _setup()
    P = toks(4 * C + 3, 3)
    run_prompt(dev, pr, P, C)
    dev.kda_slab = "stale"  # whatever another prompt left there
    did = run_prompt(dev, pr, P, C)
    assert did == ["S"] * 5
    check_equal(dev, P, C)


def test_full_chunk_resend_ends_on_a_boundary():
    dev, pr, _ = _setup()
    P = toks(3 * C, 4)
    run_prompt(dev, pr, P, C)
    dev.kda_slab = "stale"
    assert run_prompt(dev, pr, P, C) == ["S", "S", "S"]
    check_equal(dev, P, C)


def test_shorter_prompt_that_is_a_prefix():
    dev, pr, _ = _setup()
    P1 = toks(5 * C + 2, 5)
    run_prompt(dev, pr, P1, C)
    P2 = P1[: 2 * C + 3]  # ends inside the retained prompt, off a boundary: its last chunk is computed
    assert run_prompt(dev, pr, P2, C) == ["S", "S", "C"]
    check_equal(dev, P2, C)
    P3 = P2[: 2 * C]  # now ends ON a boundary whose carries are held: all skipped
    dev.kda_slab = "stale"
    assert run_prompt(dev, pr, P3, C) == ["S", "S"]
    check_equal(dev, P3, C)


def test_stale_ids_past_the_retained_end_are_never_reused():
    """A shorter prompt computed over a longer one leaves the longer one's ids in the buffer past ``length``; rows there
    were computed from the OLD history, so a prompt that happens to repeat those ids must recompute them."""
    dev, pr, _ = _setup()
    A = toks(4 * C, 30)
    B = toks(C, 31)
    run_prompt(dev, pr, A, C)
    run_prompt(dev, pr, A[:C] + B, C)
    assert pr.length == 2 * C
    P3 = A[:C] + B + A[2 * C :]
    did = run_prompt(dev, pr, P3, C)
    assert did == ["S", "S", "C", "C"], did
    check_equal(dev, P3, C)


def test_divergence_in_the_first_chunk_is_a_cold_prefill():
    dev, pr, _ = _setup()
    run_prompt(dev, pr, toks(3 * C, 6), C)
    Q = toks(2 * C + 1, 7)
    assert run_prompt(dev, pr, Q, C) == ["C", "C", "C"]
    check_equal(dev, Q, C)
    assert pr.length == len(Q)


def test_sibling_shares_a_system_prompt():
    dev, pr, _ = _setup()
    sysp = toks(C + 5, 8)
    A = sysp + toks(2 * C, 9)
    B = sysp + toks(3 * C, 10)
    run_prompt(dev, pr, A, C)
    assert run_prompt(dev, pr, B, C) == ["S", "C", "C", "C", "C"]
    check_equal(dev, B, C)


def test_thinned_pool_rebuilds_the_carries_silently():
    dev, pr, logs = _setup(n_ckpts=3)
    P1 = toks(12 * C + 1, 11)
    run_prompt(dev, pr, P1, C)
    held = sorted(p // C for p in pr.pool.by_pos)
    assert len(held) == 3 and 12 in held and 11 in held
    # a sibling diverging after 5 chunks: chunk 5's start (5C) has no checkpoint -> rebuilt from the nearest one below
    Q = P1[: 5 * C] + toks(2 * C, 12)
    did = run_prompt(dev, pr, Q, C)
    assert did == ["S"] * 5 + ["C", "C"], did
    check_equal(dev, Q, C)
    b = max([p for p in held if p <= 5] + [0])
    assert pr.stats.silent_chunks == 5 - b
    assert any("rebuilt the KDA carries at" in m for m in logs)


def test_unmarked_headers_mean_a_full_chunk_needs_its_end_carries():
    dev, pr, _ = _setup(n_ckpts=3)
    P1 = toks(10 * C + 1, 13)
    run_prompt(dev, pr, P1, C, marks=False)
    held = sorted(p // C for p in pr.pool.by_pos)
    first_gap = next(k for k in range(1, 11) if k not in held)
    P2 = P1 + toks(C, 14)
    did = run_prompt(dev, pr, P2, C, marks=False)
    # skipping stops at the first chunk whose end carries are not held (it might be the burst's last)
    assert did[: first_gap - 1] == ["S"] * (first_gap - 1) and did[first_gap - 1] == "C", (held, did)
    check_equal(dev, P2, C)


def test_marked_headers_skip_through_unheld_boundaries():
    dev, pr, _ = _setup(n_ckpts=2)
    P1 = toks(6 * C + 1, 15)
    run_prompt(dev, pr, P1, C)
    P2 = P1 + toks(C, 16)
    did = run_prompt(dev, pr, P2, C)
    assert did[:6] == ["S"] * 6, did  # mid chunks need no carries; resume at 6C (the newest checkpoint)
    check_equal(dev, P2, C)


def test_an_aborted_request_needs_nothing_from_the_next():
    """An abort ends a request without a last chunk; marks are per header, so the next prompt is unaffected."""
    dev, pr, _ = _setup(n_ckpts=2)
    P1 = toks(6 * C + 1, 32)
    run_prompt(dev, pr, P1, C)
    Q = toks(4 * C, 34)
    for s0, e0 in chunks_of(4 * C, C)[:2]:  # an aborted prompt: 2 chunks, neither marked last
        assert not pr.begin_chunk(torch.tensor(Q[s0:e0]), s0, e0, False)
        dev.prefill(Q[s0:e0], s0)
        pr.end_compute(torch.tensor(Q[s0:e0]), s0, e0)
        dev.export(s0, e0)
    P2 = Q[: 2 * C] + toks(C + 3, 35)
    did = run_prompt(dev, pr, P2, C)
    assert did == ["S", "S", "C", "C"], did
    check_equal(dev, P2, C)


def test_a_restarted_engine_that_stops_marking_is_safe():
    """Marks are read per header: an unmarked header is treated as a possible last chunk at once."""
    dev, pr, _ = _setup(n_ckpts=2)
    P1 = toks(6 * C + 1, 19)
    run_prompt(dev, pr, P1, C, marks=True)
    Pp = P1[: 3 * C]  # ends ON a boundary whose carries are thinned away, from an engine that no longer marks
    did = run_prompt(dev, pr, Pp, C, marks=False)
    check_equal(dev, Pp, C)
    assert did[-1] == "C" or pr.pool.held(3 * C) is not None, did


def test_declined_cut_and_empty_chunks_keep_state_consistent():
    dev, pr, _ = _setup()
    P1 = toks(5 * C, 20)
    run_prompt(dev, pr, P1, C, cut=2)  # declined after 2 chunks (the second carries the mark)
    pr.note_empty()  # a request declined before compute
    P2 = P1 + toks(3, 21)
    did = run_prompt(dev, pr, P2, C)
    assert did[:2] == ["S", "S"] and did[2:] == ["C"] * 4, did
    check_equal(dev, P2, C)


def test_failed_chunk_invalidates():
    dev, pr, logs = _setup()
    P1 = toks(3 * C, 22)
    run_prompt(dev, pr, P1, C)
    pr.invalidate("a chunk failed")
    assert pr.length == 0 and not pr.pool.by_pos
    did = run_prompt(dev, pr, P1, C)
    assert did == ["C"] * 3
    check_equal(dev, P1, C)


def test_non_contiguous_resend_resets():
    dev, pr, logs = _setup()
    P1 = toks(3 * C, 23)
    run_prompt(dev, pr, P1, C)
    # tt-d-gen #430: a final chunk moved back before the resident tip -- not a continuation of what the device holds
    assert not pr.begin_chunk(torch.tensor(P1[C : 2 * C]), C, 2 * C, True)
    assert pr.length == 0 and not pr.pool.by_pos and any("neither 0 nor" in m for m in logs)
    try:  # the model's own check refuses it, as without retention (no conftest here: plain try, not expect_error)
        dev.prefill(P1[C : 2 * C], C)
        refused = None
    except ValueError as e:
        refused = str(e)
    assert refused is not None and "not contiguous" in refused


def test_off_is_untouched():
    """With retention off the fake runtime computes every chunk (the runtime then never calls the retention)."""
    dev = FakeDevice()
    P = toks(3 * C + 1, 24)
    assert run_prompt(dev, None, P, C) == ["C"] * 4
    assert run_prompt(dev, None, P, C) == ["C"] * 4


@pytest.mark.parametrize("seed", range(40))
def test_fuzz_equals_cold(seed):
    """Random conversations, siblings, truncations, cuts, empty chunks, pool sizes and mark modes: whatever the
    retention skips or rebuilds, the KV Manager ships what a cold prefill would."""
    r = random.Random(seed)
    n_ckpts = r.choice([0, 1, 2, 3, 5, 8])
    dev, pr, _ = _setup(n_ckpts=n_ckpts)
    docs = [toks(r.randrange(1, 6 * C), 1000 + seed * 10 + k) for k in range(3)]
    history = []
    for _ in range(25):
        kind = r.random()
        if history and kind < 0.45:
            base = r.choice(history[-3:])
            P = base[: r.randrange(max(1, len(base) - 2 * C), len(base) + 1)] + toks(r.randrange(0, 3 * C), r.random())
        elif history and kind < 0.6:
            base = r.choice(history)
            P = base[: r.randrange(1, len(base) + 1)]
        else:
            P = r.choice(docs) + toks(r.randrange(0, 4 * C), r.random())
        if not P:
            continue
        if r.random() < 0.1:
            pr.note_empty()
        marks = r.random() < 0.7  # engines with and without the mark, switching (restarts) at random
        cut = r.randrange(1, len(chunks_of(len(P), C)) + 1) if r.random() < 0.15 else None
        run_prompt(dev, pr, P, C, marks=marks, cut=cut)
        S = chunks_of(len(P), C, cut)[-1][1]
        history.append(P[:S])
        check_equal(dev, P, C, cut)
    assert pr.stats.computed_chunks + pr.stats.skipped_chunks > 0
