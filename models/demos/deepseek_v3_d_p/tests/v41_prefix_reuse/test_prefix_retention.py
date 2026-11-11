# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-side prefix reuse decisions (``models/demos/common/prefill/prefix_reuse.py``), host only (no device, no ttnn).
GLM-5.3's G53F-0311 suite, ported to the V4.1 export semantics:

    python -m pytest -c /dev/null --noconftest -p no:cacheprovider models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse

``FakeDevice`` holds what the runner's device holds: a model row per position whose value depends on the chunk it was
computed in (the carries at the chunk's start, its start, its tokens so far; padding rows of a ragged chunk too), live carries
that depend on the chunk partition, the model's contiguity check, checkpoint slots -- and V4.1's export: a cold prompt's chunk 0
clears it, every chunk writes its FULL width of rows and the carry at its end into ONE of the window ring's two blocks (V4.1's
256-row ring: a chunk boundary -- even here -- writes block 1, a partial end writes block ``end % 2``, so after a partial last
chunk the ring holds two carries: device check 11-11, S.17). Every scenario compares the WHOLE export (rows past the prompt
included) and the model's rows with a cold prefill of the same prompt.
"""

from __future__ import annotations

import random

import pytest
import torch

from models.demos.common.prefill.prefix_reuse import (
    LAST_CHUNK_FLAG,
    MARKED_FLAG,
    CheckpointPool,
    PrefixRetention,
    split_slot_word,
)


def _h(*xs) -> int:
    return hash(xs) & 0xFFFFFFFFFFFF


C = 8


class FakeDevice:
    def __init__(self, n_ckpts: int = 0):
        self.rows: dict[int, int] = {}
        self.carries = 0
        self.next_start = 0
        self.slots = [None] * n_ckpts
        self.slab: dict[int, int] = {}  # the export's entry rows
        self.window = {0: None, 1: None}  # the export's window ring: two blocks
        self.model_chunks: list[tuple[int, int, str]] = []

    def prefill(self, ids, start: int, kind: str = "chunk") -> None:
        ids = [int(t) for t in ids]
        if start and start != self.next_start:
            raise ValueError(
                f"chunk start {start} is not contiguous with the previous chunk (expected {self.next_start})"
            )
        state = 0 if start == 0 else self.carries
        for i in range(C):  # the full chunk width: a ragged chunk's padding rows are computed too
            self.rows[start + i] = _h(state, start, tuple(ids[: i + 1]), i >= len(ids))
        self.carries = _h(state, start, tuple(ids))
        self.next_start = start + len(ids)
        self.model_chunks.append((start, start + len(ids), kind))

    def zero_slot(self) -> None:
        self.slab, self.window = {}, {0: None, 1: None}

    def export(self, start: int, end: int) -> None:
        for p in range(start, start + C):
            self.slab[p] = self.rows[p]
        self.window[end % 2] = self.carries  # C is even: a chunk boundary is always block 1

    def retention(self, chunk: int, n_ckpts: int, logs=None, skip_partial_last: bool = False) -> PrefixRetention:
        def save(slot):
            self.slots[slot] = self.carries

        def restore(slot):
            assert self.slots[slot] is not None
            self.carries = self.slots[slot]

        def export_carries(slot, end, final):  # V4.1's: the ring zeroed, the carry at `end` written (ONE block)
            self.window = {0: None, 1: None}
            self.window[end % 2] = self.carries if slot is None else self.slots[slot]
            if final:
                tail = -(-end // C) * C
                self.slab = {p: v for p, v in self.slab.items() if p < tail}
            return final

        def set_next_start(pos):
            self.next_start = pos

        def silent(ids, start):
            self.prefill(ids.tolist(), start, kind="silent")

        def on_resume(pos):
            assert self.next_start == pos
            self.window = {0: None, 1: None}
            self.window[pos % 2] = self.carries
            self.slab = {p: v for p, v in self.slab.items() if p < pos + C}

        return PrefixRetention(
            chunk=chunk,
            n_ckpts=n_ckpts,
            save=save,
            restore=restore,
            export_carries=export_carries,
            set_next_start=set_next_start,
            compute_silent=silent,
            on_resume=on_resume,
            log=(logs.append if logs is not None else (lambda m: None)),
            skip_partial_last=skip_partial_last,
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
        if pr is not None and pr.begin_chunk(torch.tensor(ids), s, e, last):
            did.append("S")
            continue
        if s == 0:
            dev.zero_slot()
        dev.prefill(ids, s)
        if pr is not None:
            pr.end_compute(torch.tensor(ids), s, e)
        dev.export(s, e)
        did.append("C")
    return did


def cold(tokens, C, cut=None):
    d = FakeDevice()
    run_prompt(d, None, tokens, C, cut=cut)
    S = chunks_of(len(tokens), C, cut)[-1][1]
    return dict(d.slab), d.window, tuple(d.rows[p] for p in range(S))


def check_equal(dev, tokens, C, cut=None, whole_export=True):
    S = chunks_of(len(tokens), C, cut)[-1][1]
    ref_slab, ref_window, ref_rows = cold(tokens, C, cut)
    if whole_export:
        assert dev.slab == ref_slab, "the export differs from a cold prefill's (rows past the prompt included)"
    else:  # the rows a burst of S tokens ships: [0, S) (a seal also carries the padded tail of the last chunk)
        top = -(-S // C) * C
        assert {p: v for p, v in dev.slab.items() if p < top} == {p: v for p, v in ref_slab.items() if p < top}
    assert dev.window == ref_window, "the export's window ring differs from a cold prefill's"
    assert tuple(dev.rows[p] for p in range(S)) == ref_rows, "the model's own rows differ from a cold prefill"


def toks(n, seed):
    r = random.Random(seed)
    return [r.randrange(1, 50) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------- the slot word
def test_slot_word():
    assert split_slot_word(0) == (0, None)  # an engine without mark_last_chunk
    assert split_slot_word(3) == (3, None)
    assert split_slot_word(MARKED_FLAG) == (0, False)
    assert split_slot_word(MARKED_FLAG | LAST_CHUNK_FLAG) == (0, True)
    assert split_slot_word(3 | MARKED_FLAG | LAST_CHUNK_FLAG) == (3, True)
    assert split_slot_word(LAST_CHUNK_FLAG) == (LAST_CHUNK_FLAG, None)  # bit 30 means nothing without bit 29
    assert split_slot_word(-1) == (-1, None)  # the shutdown sentinel word is untouched
    assert split_slot_word(0xFFFFFFFF) == (0xFFFFFFFF, None)  # ... also as an unsigned word (bit 31)
    assert split_slot_word(0x20000000) == (0, False) and split_slot_word(0x60000000) == (0, True)  # G53F-0326's words


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
        assert k in held and (k == 1 or k - 1 in held), held
    held = sorted(q // C for q in p.by_pos)
    gaps = [b - a for a, b in zip([0] + held, held)]
    assert max(gaps) <= 8, (held, gaps)
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
def _setup(n_ckpts=8, skip_partial_last=False):
    dev = FakeDevice(n_ckpts)
    logs = []
    return dev, dev.retention(C, n_ckpts, logs=logs, skip_partial_last=skip_partial_last), logs


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


def test_identical_resend_recomputes_only_its_partial_last_chunk():
    dev, pr, _ = _setup()
    P = toks(4 * C + 3, 3)  # ends at an odd position: a cold ring holds the carries at 4C AND at 4C + 3
    run_prompt(dev, pr, P, C)
    dev.window = {0: "stale", 1: "stale"}  # whatever another prompt left there
    did = run_prompt(dev, pr, P, C)
    assert did == ["S"] * 4 + ["C"], did  # resumed at 4C from its checkpoint, the partial chunk computed
    check_equal(dev, P, C)
    assert pr.stats.silent_chunks == 0


@pytest.mark.parametrize("marks", [True, False])
def test_skipping_a_partial_last_chunk_loses_the_rings_other_carry(marks):
    """Why V4.1 runs skip_partial_last=False (device check 11-11, S.17): skipping the partial last chunk of an exact
    re-send rebuilds the ring from the carry at its end only, while a cold ring also holds the previous boundary's."""
    dev, pr, _ = _setup(skip_partial_last=True)
    P = toks(4 * C + 3, 3)
    run_prompt(dev, pr, P, C, marks=marks)
    assert run_prompt(dev, pr, P, C, marks=marks) == ["S"] * 5
    with pytest.raises(AssertionError, match="window ring"):  # allow-pytest.raises: no ttnn conftest
        check_equal(dev, P, C)


def test_a_partial_last_chunk_ending_on_the_same_block_as_a_boundary_is_also_computed():
    dev, pr, _ = _setup()
    P = toks(4 * C + 4, 36)  # even end: its carry and a boundary's share block 1 (the device's E = 0 mod 256 case)
    run_prompt(dev, pr, P, C)
    assert run_prompt(dev, pr, P, C) == ["S"] * 4 + ["C"]
    check_equal(dev, P, C)


def test_full_chunk_resend_ends_on_a_boundary():
    dev, pr, _ = _setup()
    P = toks(3 * C, 4)
    run_prompt(dev, pr, P, C)
    dev.window = {0: "stale", 1: "stale"}
    assert run_prompt(dev, pr, P, C) == ["S", "S", "S"]
    check_equal(dev, P, C)


def test_shorter_prompt_that_is_a_prefix_clears_the_rows_past_it():
    dev, pr, _ = _setup()
    P1 = toks(5 * C + 2, 5)
    run_prompt(dev, pr, P1, C)
    P2 = P1[: 2 * C + 3]  # ends inside the retained prompt, off a boundary: its last chunk is computed
    assert run_prompt(dev, pr, P2, C) == ["S", "S", "C"]
    check_equal(dev, P2, C)
    P3 = P2[: 2 * C]  # now ends ON a boundary whose carries are held: all skipped, the export cleared past 2C
    dev.window = {0: "stale", 1: "stale"}
    assert run_prompt(dev, pr, P3, C) == ["S", "S"]
    check_equal(dev, P3, C)
    assert pr.length == 2 * C  # the export past 2C is gone: nothing there is retained any more


def test_after_a_cleared_tail_the_longer_prompt_is_recomputed_not_shipped_as_zeros():
    dev, pr, _ = _setup()
    P1 = toks(5 * C, 40)
    run_prompt(dev, pr, P1, C)
    assert run_prompt(dev, pr, P1[: 2 * C], C) == ["S", "S"]  # the strict prefix clears [2C, 5C) of the export
    did = run_prompt(dev, pr, P1, C)
    assert did == ["S", "S", "C", "C", "C"], did  # resumed at 2C (its checkpoint), never skipping a cleared chunk
    check_equal(dev, P1, C)


def test_stale_ids_past_the_retained_end_are_never_reused():
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


def test_shorter_sibling_clears_the_older_rows_on_resume():
    dev, pr, _ = _setup()
    A = toks(6 * C, 41)
    B = A[:C] + toks(C + 3, 42)  # 2C + 3 tokens: resumed at C, the rows [3C, 6C) of A must not survive in the export
    run_prompt(dev, pr, A, C)
    assert run_prompt(dev, pr, B, C) == ["S", "C", "C"]
    check_equal(dev, B, C)


def test_thinned_pool_rebuilds_the_carries_silently():
    dev, pr, logs = _setup(n_ckpts=3)
    P1 = toks(12 * C + 1, 11)
    run_prompt(dev, pr, P1, C)
    held = sorted(p // C for p in pr.pool.by_pos)
    assert len(held) == 3 and 12 in held and 11 in held
    Q = P1[: 5 * C] + toks(2 * C, 12)
    did = run_prompt(dev, pr, Q, C)
    assert did == ["S"] * 5 + ["C", "C"], did
    check_equal(dev, Q, C)
    b = max([p for p in held if p <= 5] + [0])
    assert pr.stats.silent_chunks == 5 - b
    assert any("rebuilt the carries at" in m for m in logs)


def test_unmarked_headers_mean_a_full_chunk_needs_its_end_carries():
    dev, pr, _ = _setup(n_ckpts=3)
    P1 = toks(10 * C + 1, 13)
    run_prompt(dev, pr, P1, C, marks=False)
    held = sorted(p // C for p in pr.pool.by_pos)
    first_gap = next(k for k in range(1, 11) if k not in held)
    P2 = P1 + toks(C, 14)
    did = run_prompt(dev, pr, P2, C, marks=False)
    assert did[: first_gap - 1] == ["S"] * (first_gap - 1) and did[first_gap - 1] == "C", (held, did)
    check_equal(dev, P2, C)


def test_unmarked_strict_prefix_ships_the_right_rows_but_keeps_older_ones_past_it():
    """Without the mark a skipped FULL last chunk cannot clear the export past it (the next chunk may still be skipped and
    need those rows): the rows a burst ships equal cold's, the rows past it are the older prompt's (documented)."""
    dev, pr, _ = _setup()
    P1 = toks(4 * C, 43)
    run_prompt(dev, pr, P1, C, marks=False)
    assert run_prompt(dev, pr, P1[: 2 * C], C, marks=False) == ["S", "S"]
    check_equal(dev, P1[: 2 * C], C, whole_export=False)
    with pytest.raises(AssertionError):  # allow-pytest.raises: no ttnn conftest
        check_equal(dev, P1[: 2 * C], C, whole_export=True)


def test_marked_headers_skip_through_unheld_boundaries():
    dev, pr, _ = _setup(n_ckpts=2)
    P1 = toks(6 * C + 1, 15)
    run_prompt(dev, pr, P1, C)
    P2 = P1 + toks(C, 16)
    did = run_prompt(dev, pr, P2, C)
    assert did[:6] == ["S"] * 6, did
    check_equal(dev, P2, C)


def test_an_aborted_request_needs_nothing_from_the_next():
    dev, pr, _ = _setup(n_ckpts=2)
    P1 = toks(6 * C + 1, 32)
    run_prompt(dev, pr, P1, C)
    Q = toks(4 * C, 34)
    for s0, e0 in chunks_of(4 * C, C)[:2]:  # an aborted prompt: 2 chunks, neither marked last
        assert not pr.begin_chunk(torch.tensor(Q[s0:e0]), s0, e0, False)
        if s0 == 0:
            dev.zero_slot()
        dev.prefill(Q[s0:e0], s0)
        pr.end_compute(torch.tensor(Q[s0:e0]), s0, e0)
        dev.export(s0, e0)
    P2 = Q[: 2 * C] + toks(C + 3, 35)
    did = run_prompt(dev, pr, P2, C)
    assert did == ["S", "S", "C", "C"], did
    check_equal(dev, P2, C)


def test_a_restarted_engine_that_stops_marking_is_safe():
    dev, pr, _ = _setup(n_ckpts=2)
    P1 = toks(6 * C + 1, 19)
    run_prompt(dev, pr, P1, C, marks=True)
    Pp = P1[: 3 * C]
    did = run_prompt(dev, pr, Pp, C, marks=False)
    check_equal(dev, Pp, C, whole_export=False)
    assert did[-1] == "C" or pr.pool.held(3 * C) is not None, did


def test_declined_cut_and_empty_chunks_keep_state_consistent():
    dev, pr, _ = _setup()
    P1 = toks(5 * C, 20)
    run_prompt(dev, pr, P1, C, cut=2)  # declined after 2 chunks (the second carries the mark)
    pr.note_empty()
    P2 = P1 + toks(3, 21)
    did = run_prompt(dev, pr, P2, C)
    assert did[:2] == ["S", "S"] and did[2:] == ["C"] * 4, did
    check_equal(dev, P2, C)


def test_failed_chunk_invalidates():
    dev, pr, logs = _setup()
    P1 = toks(3 * C, 22)
    run_prompt(dev, pr, P1, C)
    pr.invalidate("a chunk failed")
    assert pr.length == 0 and not pr.pool.by_pos and pr.stats.invalidations == 1
    did = run_prompt(dev, pr, P1, C)
    assert did == ["C"] * 3
    check_equal(dev, P1, C)


def test_non_contiguous_resend_resets():
    dev, pr, logs = _setup()
    P1 = toks(3 * C, 23)
    run_prompt(dev, pr, P1, C)
    assert not pr.begin_chunk(torch.tensor(P1[C : 2 * C]), C, 2 * C, True)
    assert pr.length == 0 and not pr.pool.by_pos and any("neither 0 nor" in m for m in logs)
    with pytest.raises(ValueError, match="not contiguous"):  # allow-pytest.raises: no ttnn conftest
        dev.prefill(P1[C : 2 * C], C)


def test_off_is_untouched():
    dev = FakeDevice()
    P = toks(3 * C + 1, 24)
    assert run_prompt(dev, None, P, C) == ["C"] * 4
    assert run_prompt(dev, None, P, C) == ["C"] * 4


@pytest.mark.parametrize("seed", range(60))
def test_fuzz_equals_cold(seed):
    """Random conversations, siblings, truncations, cuts, empty chunks, pool sizes and mark modes: whatever the retention
    skips or rebuilds, the export (whole, with marked headers; the shipped rows, unmarked) equals a cold prefill's."""
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
        marks = r.random() < 0.7
        cut = r.randrange(1, len(chunks_of(len(P), C)) + 1) if r.random() < 0.15 else None
        run_prompt(dev, pr, P, C, marks=marks, cut=cut)
        S = chunks_of(len(P), C, cut)[-1][1]
        history.append(P[:S])
        # a cut burst has no marked last chunk: like an unmarked one, its rows past the cut are not cleared
        check_equal(dev, P, C, cut, whole_export=marks and cut is None)
    assert pr.stats.computed_chunks + pr.stats.skipped_chunks > 0
