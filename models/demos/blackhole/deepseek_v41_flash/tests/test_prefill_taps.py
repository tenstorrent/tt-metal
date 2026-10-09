# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side bookkeeping of the drafter taps of the prefill (tt/prefill_taps.py): which stash row every (user, position) of a prefill chunk is scattered to, for ragged prompt
lengths, chunk windows and decode users that are not the prefill slots. CPU only (no device)."""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt.prefill_taps import (
    PIECES,
    SKIP,
    TAP_ROWS,
    PrefillTaps,
    drafter_chunk,
    taps_possible,
)


def _taps(rows=4, Ud=8):
    t = PrefillTaps.__new__(PrefillTaps)  # (no device: only the index math)
    t.rows, t.cols, t.Ud, t.Uc = rows, 8, Ud, drafter_chunk(Ud)
    return t


def _pieces(ids):
    """[rows * G, 10 * g] scatter ids -> {(mesh row, token row): stash token row} for the rows that are written."""
    out = {}
    rows = ids.shape[0]
    flat = ids.reshape(rows, -1, PIECES)
    for r in range(rows):
        for i in range(flat.shape[1]):
            if int(flat[r, i, 0]) != SKIP:
                assert [int(x) - int(flat[r, i, 0]) for x in flat[r, i]] == list(
                    range(PIECES)
                )  # the 10 pieces of one token row
                assert int(flat[r, i, 0]) % PIECES == 0
                out[(r, i)] = int(flat[r, i, 0]) // PIECES
    return out


def test_drafter_chunk_and_possible():
    assert drafter_chunk(1) == 1 and drafter_chunk(4) == 4 and drafter_chunk(6) == 6
    assert drafter_chunk(8) == 4 and drafter_chunk(16) == 4
    assert taps_possible(list(range(40))) and not taps_possible(list(range(38)))


def test_last_128_positions_land_in_distinct_stash_rows():
    t = _taps()
    U, C = 1, 512  # interleaved prefill: one prefill slot per mesh row, 512-token windows
    lens = torch.zeros(32, dtype=torch.long)
    users = {0: 68, 9: 300, 18: 128, 27: 520}  # (global decode user -> prompt length), one per mesh row
    for b, n in users.items():
        lens[b] = n
    user_of = lambda r, u: next((b for b in users if b // 8 == r), None)
    for s0 in range(0, 1024, C):
        ids = t.host_ids(s0, C, U, user_of, lens)
        got = _pieces(ids)
        for r in range(4):
            b = next(b for b in users if b // 8 == r)
            S, ud = users[b], b % 8
            for tpos in range(C):
                pos = s0 + tpos
                want = ud * TAP_ROWS + pos % TAP_ROWS if S - TAP_ROWS <= pos < S else None
                assert got.get((r, tpos)) == want, (r, pos)


def test_two_windows_cover_the_last_128_once_and_slots_do_not_collide():
    t = _taps()
    U, C, S = 1, 128, 200  # the last 128 positions [72, 200) straddle the windows [0,128) and [128,256)
    lens = torch.zeros(32, dtype=torch.long)
    lens[3] = S
    user_of = lambda r, u: 3 if r == 0 else None
    seen = {}
    for s0 in (0, 128):
        for (r, i), row in _pieces(t.host_ids(s0, C, U, user_of, lens)).items():
            assert r == 0
            assert row not in seen, f"stash row {row} written twice"
            seen[row] = s0 + i
    assert sorted(seen.values()) == list(range(S - TAP_ROWS, S))
    assert all(row // TAP_ROWS == 3 for row in seen)


def test_short_prompt_and_prefill_slots_mapped_to_decode_users():
    t = _taps()
    U, C = 2, 256  # two prefill slots per row; slot 1 of row 2 holds decode user 2*8+5, slot 0 is empty
    lens = torch.zeros(32, dtype=torch.long)
    lens[21] = 40
    user_of = lambda r, u: 21 if (r, u) == (2, 1) else None
    got = _pieces(t.host_ids(0, C, U, user_of, lens))
    assert sorted(got) == [(2, 256 + i) for i in range(40)]
    assert all(got[(2, 256 + i)] == 5 * TAP_ROWS + i for i in range(40))  # a prompt shorter than 128: every position


def test_ring_slot_selection_of_the_seeding():
    """the position held by stash slot j (the latest position < S with position % 128 == j) and its ring slot (position % 160): a bijection of the last 128 positions."""
    for S in (1, 5, 127, 128, 129, 300, 511):
        j = torch.arange(TAP_ROWS)
        p = (S - 1) - ((S - 1 - j) % TAP_ROWS)
        valid = p >= 0
        assert sorted(p[valid].tolist()) == list(range(max(0, S - TAP_ROWS), S))
        slots = (p[valid] % 160).tolist()
        assert len(set(slots)) == len(slots)
