# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the pipeline positions of a looped model (pass_schedule.py) and the layer-ack layout over them
(layer_completion_layout.py), on a small looped schedule built by hand, no model adapter.

SCHEDULE is 5 layers whose layers 1-2 run twice: 7 positions,

    position  0      1      2      3      4      5      6
    (layer,k) (0,0)  (1,1)  (2,1)  (1,2)  (2,2)  (3,0)  (4,0)

so loop pass 1 is positions [1, 3), pass 2 is [3, 5), and pass 2 of layer L acks as L + 5.
"""

import pytest

from models.demos.common.prefill.runners.layer_completion_layout import layer_completion_layout
from models.demos.common.prefill.runners.pass_schedule import (
    PassSchedule,
    RankSlice,
    builds_kv_chunk_table,
    loop_ack_id,
)

SCHEDULE = PassSchedule(passes=((0, 0), (1, 1), (2, 1), (1, 2), (2, 2), (3, 0), (4, 0)), num_layers=5)
# Four ranks: prelude, loop pass 1, loop pass 2, coda (the GLM-5.4 placement in small).
SPLIT = [(0, 1), (1, 2), (3, 2), (5, 2)]


def test_a_schedule_runs_every_layer(expect_error):
    with expect_error(ValueError, "must run each of the 3 layers"):
        PassSchedule(passes=((0, 0), (2, 0)), num_layers=3)


def test_rank_starts_exclude_the_inside_of_a_loop_pass():
    assert len(SCHEDULE) == 7
    assert SCHEDULE.valid_starts() == {0, 1, 3, 5, 6}


@pytest.mark.parametrize(
    "first, count, expected",
    [
        (0, 1, RankSlice(0, 1, ())),
        (1, 2, RankSlice(1, 2, (1,))),
        (3, 2, RankSlice(1, 2, (2,))),
        (5, 2, RankSlice(3, 2, ())),
        (0, 3, RankSlice(0, 3, (1,))),  # prelude + pass 1
        (3, 4, RankSlice(1, 4, (2,))),  # pass 2 + coda
        (0, 7, RankSlice(0, 5, (1, 2))),  # the whole model on one rank
    ],
    ids=["prelude", "pass1", "pass2", "coda", "prelude-pass1", "pass2-coda", "whole"],
)
def test_rank_slice_holds_whole_loop_passes(first, count, expected):
    assert SCHEDULE.rank_slice(first, count) == expected


@pytest.mark.parametrize(
    "first, count, match",
    [
        (2, 2, "a rank holds whole loop passes"),  # starts inside pass 1
        (0, 2, "a rank holds whole loop passes"),  # ends inside pass 1
        (1, 3, "a rank holds whole loop passes"),  # pass 1 into pass 2
        (5, 3, "outside the 7-pass schedule"),
    ],
    ids=["start-in-pass1", "end-in-pass1", "pass1-into-pass2", "past-the-end"],
)
def test_rank_slice_rejects_a_cut_loop_pass(first, count, match, expect_error):
    with expect_error(ValueError, match):
        SCHEDULE.rank_slice(first, count)


def test_ack_ids_give_each_later_pass_its_own_ids():
    assert [loop_ack_id(1, k, 5) for k in (0, 1, 2, 3)] == [1, 1, 6, 11]
    assert SCHEDULE.ack_ids(0, 7) == [0, 1, 2, 6, 7, 3, 4]
    assert [SCHEDULE.ack_ids(*rank) for rank in SPLIT] == [[0], [1, 2], [6, 7], [3, 4]]
    # A completion reader takes layer ids in [0, bound): the bound covers pass 2's ids.
    assert SCHEDULE.ack_id_bound == 8


def test_a_looped_model_publishes_no_kv_table(expect_error):
    """A plain model always builds the KV chunk table; a looped one never does, and refuses a run that asks for
    KV migration rather than leaving it silently unmigrated."""
    assert builds_kv_chunk_table(None, migration_requested=True, model_name="plain")
    assert builds_kv_chunk_table(None, migration_requested=False, model_name="plain")
    assert not builds_kv_chunk_table(SCHEDULE, migration_requested=False, model_name="looped")
    with expect_error(RuntimeError, "looped: KV migration of a looped model is not supported"):
        builds_kv_chunk_table(SCHEDULE, migration_requested=True, model_name="looped")


@pytest.mark.parametrize("mtp_levels", [0, 1], ids=["trunk", "mtp1"])
def test_looped_layer_completion_layout_per_rank(mtp_levels):
    """Every position acks once per chunk, under its ack id, ranks in position order; the last rank's extra rows
    (MTP) take ids from num_layers on."""
    layouts = [layer_completion_layout(SPLIT, r, extra_layers=mtp_levels, pass_schedule=SCHEDULE) for r in range(4)]
    assert {lay.completions_per_chunk for lay in layouts} == {7 + mtp_levels}
    assert [lay.rank_first_record for lay in layouts] == [0, 1, 3, 5]
    assert [lay.rank_record_count for lay in layouts] == [1, 2, 2, 2 + mtp_levels]
    assert [lay.rank_layer_ids for lay in layouts] == [[0], [1, 2], [6, 7], [3, 4] + [5] * mtp_levels]


def test_extra_rows_must_not_reach_the_loop_ack_ids(expect_error):
    # Two extra rows take ids 5 and 6; 6 is pass 2 of layer 1.
    with expect_error(AssertionError, r"extra rows \[6\] collide with loop pass ack ids"):
        layer_completion_layout(SPLIT, 3, extra_layers=2, pass_schedule=SCHEDULE)


def test_a_looped_layout_acks_every_position(expect_error):
    with expect_error(AssertionError, "a looped model acks every pass position"):
        layer_completion_layout(SPLIT, 0, kv_slot_layer_ids=[0, 1], pass_schedule=SCHEDULE)
