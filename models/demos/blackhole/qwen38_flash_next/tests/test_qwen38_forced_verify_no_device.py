# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The teacher-forced verify gate's bookkeeping without a device: the accept schedule, the forced verdict, the pass
plan and :func:`run_forced_verify` on a scripted chain whose row j of a pass depends only on the stream up to row j
(causal), so every scored position must carry the argmax of its own prefix whatever the schedule, the start, the
end, the pad rows at the run's end and the 1-row fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    Qwen38ForcedVerifyHooks,
    forced_verify_decision,
    forced_verify_plan,
    forced_verify_tokens,
    parse_accept_schedule,
    run_forced_verify,
)

K = 4
ROWS = K + 1


def _argmax_of(history) -> int:
    return (sum((i + 1) * t for i, t in enumerate(history)) * 31 + len(history)) % 997


class _ScriptedChain:
    """The device side of the hooks: the committed length, the last pass, the calls made."""

    def __init__(self, stream, start, fits=lambda position: True):
        self.stream = stream
        self.position = start  # the device's committed length
        self.last = None  # (position, accept)
        self.fits = fits
        self.calls: list[tuple] = []
        self.rows = {}
        self.decide = None

    def _pass(self, tokens):
        position = self.position if self.last is None else self.last[0] + self.last[1] + 1
        history = list(self.stream[:position])
        head = SimpleNamespace(
            argmaxes=tuple(_argmax_of(history + tokens[: row + 1]) for row in range(ROWS)),
            candidate_rows=[("row", position + row) for row in range(ROWS)],
        )
        decision = self.decide(tokens, head)
        assert decision.alignment_tokens[decision.accepted] == decision.next_token
        assert len(decision.alignment_tokens) == ROWS
        self.last = (position, decision.accepted)
        self.calls.append(("pass", position, decision.accepted))
        return position

    def enter(self, tokens, decide):
        assert self.last is None, "enter while a pass loop is open"
        self.decide = decide
        assert tokens[0] == self.stream[self.position] or self.position >= len(self.stream)
        self._pass(list(tokens))

    def step(self, tokens):
        assert self.last is not None
        self.position = self.last[0] + self.last[1] + 1  # the previous pass's commit
        assert tokens == forced_verify_tokens(self.stream, self.position, ROWS)
        self._pass(list(tokens))

    def leave(self, position, rows, first):
        assert self.last is not None
        assert 1 <= rows <= self.last[1] + 1
        self.position = self.last[0] + rows
        assert self.position == position
        self.last = None
        self.calls.append(("leave", position, rows, first))

    def one_row(self, position):
        assert self.last is None and self.position == position
        self.position += 1
        self.rows[position] = (_argmax_of(self.stream[: position + 1]), ("row", position))
        self.calls.append(("one_row", position))

    def hooks(self):
        return Qwen38ForcedVerifyHooks(self.enter, self.step, self.leave, self.fits, self.one_row)


def _run(schedule, start, end, *, wanted=None, fits=lambda position: True, min_position=32, stream_len=None):
    stream = [(i * 13 + 5) % 997 for i in range(stream_len if stream_len is not None else end + 1)]
    chain = _ScriptedChain(stream, start, fits)
    wanted = list(range(start, end + 1)) if wanted is None else wanted
    rows = chain.rows
    passes, one_row = run_forced_verify(
        stream[: end + 1],
        start,
        end,
        schedule,
        drafts=K,
        wanted=wanted,
        rows_out=rows,
        hooks=chain.hooks(),
        parse_row=lambda row: row,
        min_position=min_position,
    )
    return stream, chain, rows, passes, one_row


def test_schedule_parsing(expect_error):
    assert parse_accept_schedule("4", K) == (4,)
    assert parse_accept_schedule("4,2,0,3,1", K) == (4, 2, 0, 3, 1)
    for bad in ("5", "-1", "", "a", "4,,2"):
        with expect_error(ValueError, ""):
            parse_accept_schedule(bad, K)


def test_forced_tokens_pad_with_the_last_token():
    assert forced_verify_tokens([1, 2, 3], 1, 5) == [2, 3, 3, 3, 3]
    assert forced_verify_tokens([1, 2, 3, 4, 5, 6, 7], 1, 5) == [2, 3, 4, 5, 6]


def test_forced_decision_lanes_are_the_next_reference_tokens():
    stream = list(range(100, 120))
    decision = forced_verify_decision(stream, 10, 2, ROWS)
    assert decision.accepted == 2
    assert decision.alignment_tokens == (111, 112, 113, 114, 115)
    assert decision.next_token == 113 == stream[10 + 2 + 1]


def test_plan():
    assert forced_verify_plan(10, 100, 4) == (91, 5, False)
    assert forced_verify_plan(10, 14, 4) == (5, 5, True)  # exactly full
    assert forced_verify_plan(10, 12, 4) == (3, 3, True)  # item end: v < a + 1
    assert forced_verify_plan(10, 12, 0) == (3, 1, False)
    assert forced_verify_plan(10, 10, 0) == (1, 1, True)


@pytest.mark.parametrize("schedule", [(4,), (4, 2, 0, 3, 1), (0,), (1, 3), (2,)])
@pytest.mark.parametrize("start,end", [(40, 40), (40, 43), (40, 44), (40, 45), (40, 71), (63, 130), (100, 101)])
def test_every_position_scored_once_with_its_own_prefix(schedule, start, end):
    stream, chain, rows, passes, one_row = _run(schedule, start, end)
    assert sorted(rows) == list(range(start, end + 1))
    for position, (argmax, row) in rows.items():
        assert argmax == _argmax_of(stream[: position + 1])
        assert row == ("row", position)
    assert one_row == 0
    assert chain.position == end + 1 and chain.last is None
    scored_by_pass = [call for call in chain.calls if call[0] == "pass"]
    assert len(scored_by_pass) == passes
    # a pass starts where the previous one's accepted rows end
    for before, after in zip(scored_by_pass, scored_by_pass[1:]):
        assert after[1] == before[1] + before[2] + 1
    assert chain.calls[-1][0] == "leave"


def test_wanted_subset_keeps_only_wanted_rows():
    stream, chain, rows, passes, one_row = _run((4, 2), 40, 80, wanted=[40, 47, 48, 80])
    assert sorted(rows) == [40, 47, 48, 80]


def test_item_end_commits_only_the_valid_rows():
    stream, chain, rows, passes, one_row = _run((4,), 40, 42)
    assert chain.calls == [("pass", 40, 4), ("leave", 43, 3, 40)]


def test_positions_below_the_minimum_go_one_row():
    stream, chain, rows, passes, one_row = _run((4,), 28, 50)
    assert [call for call in chain.calls if call[0] == "one_row"] == [("one_row", p) for p in range(28, 32)]
    assert one_row == 4 and sorted(rows) == list(range(28, 51))
    assert chain.calls[4][:2] == ("pass", 32)


def test_no_room_leaves_with_the_last_pass_whole_then_goes_one_row():
    stream, chain, rows, passes, one_row = _run((4,), 40, 70, fits=lambda position: position < 60)
    assert ("leave", 60, 5, 40) in chain.calls
    assert [call for call in chain.calls if call[0] == "one_row"][0] == ("one_row", 60)
    assert sorted(rows) == list(range(40, 71)) and chain.position == 71


def test_partial_accepts_leave_with_the_accepted_rows():
    stream, chain, rows, passes, one_row = _run((0,), 40, 50, fits=lambda position: position < 45)
    assert ("leave", 45, 1, 40) in chain.calls
    assert sorted(rows) == list(range(40, 51))


def test_module_exports():
    assert session_module.Qwen38TeacherForcedRows.__dataclass_fields__["verify_passes"].default == 0
