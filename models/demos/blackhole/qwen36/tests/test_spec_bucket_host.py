# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host tests of tt/spec_bucket.py (pure helpers) and the tp_common.spec_buckets flag."""

import itertools
import random

import pytest

from models.demos.blackhole.qwen36.tt.spec_bucket import (
    apply_row_swap,
    lowest_free_row,
    parse_buckets,
    pick_bucket,
    plan_row_swap,
)


@pytest.mark.parametrize(
    "text, want",
    [
        ("8", (8,)),
        ("1,2,4,8", (1, 2, 4, 8)),
        ("8,4,2,1", (1, 2, 4, 8)),
        ("4,2,8", (2, 4, 8)),
        (" 2 , 8 ", (2, 8)),
        ("8,8,4", (4, 8)),
        ("", (8,)),
        ("  ", (8,)),
        (None, (8,)),
    ],
)
def test_parse_ok(text, want):
    assert parse_buckets(text, 8) == want


@pytest.mark.parametrize(
    "text, message",
    [
        ("4,2", "must contain the engine width"),
        ("1,2,4", "must contain the engine width"),
        ("3", "not in"),
        ("0,8", "not in"),
        ("-1,8", "not in"),
        ("16", "not in"),
        ("8,16", "not in"),
        ("a,8", "not an integer"),
        ("8,,4", "not an integer"),
        ("4.5,8", "not an integer"),
    ],
)
def test_parse_bad(text, message, expect_error):
    with expect_error(ValueError, message):
        parse_buckets(text, 8)


def test_parse_other_engine_width(expect_error):
    assert parse_buckets("1,2,4", 4) == (1, 2, 4)
    with expect_error(ValueError, "larger than the engine width"):
        parse_buckets("1,2,4,8", 4)


def test_flag_default_and_env(monkeypatch, expect_error):
    from models.demos.blackhole.qwen36.tt import tp_common

    monkeypatch.delenv("QWEN36_SPEC_BUCKETS", raising=False)
    assert tp_common.spec_buckets() == (1, 2, 4, 8)
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "8")
    assert tp_common.spec_buckets() == (8,)
    assert tp_common.engine_spec_buckets(8) == (8,)
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4,8")
    assert tp_common.spec_buckets() == (1, 2, 4, 8)
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2")
    with expect_error(ValueError, "must contain the engine width"):
        tp_common.spec_buckets()
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "3,8")
    with expect_error(ValueError, "not in"):
        tp_common.spec_buckets()


@pytest.mark.parametrize(
    "text, B, want",
    [
        ("1,2,4,8", 1, (1,)),
        ("1,2,4,8", 2, (1, 2)),
        ("1,2,4,8", 4, (1, 2, 4)),
        ("1,2,4,8", 8, (1, 2, 4, 8)),
        ("2,8", 4, (2, 4)),
        ("8,4", 4, (4,)),
    ],
)
def test_parse_clip(text, B, want):
    assert parse_buckets(text, B, clip=True) == want


def test_parse_clip_still_strict_on_bad_values(expect_error):
    with expect_error(ValueError, "not in"):
        parse_buckets("3,8", 4, clip=True)
    with expect_error(ValueError, "not an integer"):
        parse_buckets("a,8", 4, clip=True)


def test_engine_flag_clips_to_a_narrow_engine(monkeypatch, expect_error):
    from models.demos.blackhole.qwen36.tt import tp_common

    warnings = []
    monkeypatch.setattr(tp_common.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
    monkeypatch.delenv("QWEN36_SPEC_BUCKETS", raising=False)
    assert tp_common.engine_spec_buckets(1) == (1,)  # unset: clipped silently, no warning
    assert tp_common.engine_spec_buckets(8) == (1, 2, 4, 8)
    assert not warnings
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4,8")
    assert tp_common.engine_spec_buckets(8) == (1, 2, 4, 8)
    assert not warnings
    assert tp_common.engine_spec_buckets(1) == (1,)
    assert len(warnings) == 1 and "B=1" in warnings[0]
    assert tp_common.engine_spec_buckets(4) == (1, 2, 4)
    assert len(warnings) == 2
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2")  # nothing above B=2: no warning, B added
    assert tp_common.engine_spec_buckets(2) == (1, 2)
    assert len(warnings) == 2
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "3,8")  # bad values stay strict
    with expect_error(ValueError, "not in"):
        tp_common.engine_spec_buckets(1)
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "a,8")
    with expect_error(ValueError, "not an integer"):
        tp_common.engine_spec_buckets(4)
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4")  # B >= 8 stays strict
    with expect_error(ValueError, "must contain the engine width"):
        tp_common.engine_spec_buckets(8)


def test_pick_all_buckets_every_row():
    bs = (1, 2, 4, 8)
    want = {0: 1, 1: 2, 2: 4, 3: 4, 4: 8, 5: 8, 6: 8, 7: 8}
    for row, b in want.items():
        assert pick_bucket(bs, [row]) == b
        assert pick_bucket(bs, {row}) == b


def test_pick_sets_and_edges():
    bs = (1, 2, 4, 8)
    assert pick_bucket(bs, []) == 1
    assert pick_bucket(bs, set()) == 1
    assert pick_bucket(bs, [0, 1, 2, 3]) == 4
    assert pick_bucket(bs, range(8)) == 8
    assert pick_bucket(bs, [7]) == 8
    assert pick_bucket(bs, iter([0, 5, 1])) == 8
    assert pick_bucket((8,), []) == 8
    assert pick_bucket((8,), [0]) == 8
    assert pick_bucket((8,), range(8)) == 8
    assert pick_bucket((2, 8), [2]) == 8
    assert pick_bucket((2, 8), [1]) == 2
    assert pick_bucket((8, 2), [0]) == 2  # unsorted input


def test_pick_is_smallest_and_covers_row():
    for bs in ((8,), (4, 8), (1, 8), (2, 4, 8), (1, 2, 4, 8)):
        for rows in itertools.chain.from_iterable(itertools.combinations(range(8), n) for n in range(1, 4)):
            b = pick_bucket(bs, rows)
            assert b in bs and b >= max(rows) + 1
            assert all(not (c > max(rows)) or c >= b for c in bs)


def test_pick_bad(expect_error):
    with expect_error(ValueError, "outside the engine width"):
        pick_bucket((1, 2, 4, 8), [8])
    with expect_error(ValueError, "outside the engine width"):
        pick_bucket((8,), [0, 9])
    with expect_error(ValueError, "negative row"):
        pick_bucket((8,), [-1])
    with expect_error(ValueError, "no buckets"):
        pick_bucket((), [0])


def test_lowest_free_row():
    assert lowest_free_row(set()) is None
    assert lowest_free_row([]) is None
    assert lowest_free_row({5}) == 5
    assert lowest_free_row({7, 3, 5}) == 3
    assert lowest_free_row(iter([4, 0])) == 0
    assert lowest_free_row(range(8)) == 0


def test_swap_noops():
    phys = list(range(8))
    assert plan_row_swap(0, {0, 1, 2}, phys) is None  # already the lowest free row
    assert plan_row_swap(3, set(), phys) is None  # nothing free
    assert plan_row_swap(5, {0, 1, 2}, phys) is None  # chosen row live: re-prefill never moves
    assert plan_row_swap(2, {2}, phys) is None  # only free row is its own
    assert apply_row_swap(phys, None) == phys
    assert apply_row_swap(phys, None) is not phys


def test_swap_basic_and_sequence():
    phys = list(range(8))
    free = set(range(8))
    # admit 4: plugin picks slots 0..3, all already on the lowest rows
    for s in range(4):
        assert plan_row_swap(s, free, phys) is None
        free.discard(phys[s])
    # free 1 (slot 1 -> row 1)
    free.add(phys[1])
    # a new request picks slot 6 (row 6): must land on row 1
    sw = plan_row_swap(6, free, phys)
    assert sw == (6, 1)
    phys = apply_row_swap(phys, sw)
    assert phys[6] == 1 and phys[1] == 6
    free.discard(phys[6])
    assert sorted(set(range(8)) - free) == [0, 1, 2, 3]
    assert sorted(phys) == list(range(8))


def test_swap_unrelated_slot_map():
    phys = [3, 0, 7, 1, 2, 6, 5, 4]
    free = {4, 6, 7}
    sw = plan_row_swap(2, free, phys)  # slot 2 on row 7; lowest free row 4 sits in slot 7
    assert sw == (2, 7)
    new = apply_row_swap(phys, sw)
    assert new == [3, 0, 4, 1, 2, 6, 5, 7]


def test_swap_bad(expect_error):
    with expect_error(ValueError, "outside"):
        plan_row_swap(8, {0}, list(range(8)))
    with expect_error(ValueError, "outside"):
        plan_row_swap(-1, {0}, list(range(8)))
    with expect_error(ValueError, "not in the slot map"):
        plan_row_swap(0, {9}, list(range(8)))


def test_swap_random_properties():
    rng = random.Random(7)
    for _ in range(500):
        phys = list(range(8))
        rng.shuffle(phys)
        free = {r for r in range(8) if rng.random() < 0.5}
        slot = rng.randrange(8)
        sw = plan_row_swap(slot, free, phys)
        new = apply_row_swap(phys, sw)
        assert sorted(new) == list(range(8))  # still a permutation
        for s in range(8):  # an occupied row never moves
            if phys[s] not in free:
                assert new[s] == phys[s]
        if phys[slot] in free:
            assert new[slot] == min(free)  # new request on the lowest free row
        else:
            assert new == phys
        again = plan_row_swap(slot, free, new)  # idempotent
        assert again is None
        assert apply_row_swap(new, again) == new
