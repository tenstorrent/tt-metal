# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The lanes server's scheduler without a device (``tools/qwen38_lane_scheduler.py``): the lane geometry, the
per-lane stream rules against a single-stream reference on a deterministic model (every event: EOS, ``max_tokens``,
the forced ``</think>`` at every budget both inside a pass and on its last emitted token, a zero budget, an oversize
id, ``ignore_eos``), and the driver loop over a fake device (FIFO order, one admission per boundary while a lane
decodes and every waiting request while none does, the B+1th request waits, the FIFO's limit, a request ending
while the others continue, the accept-count vector the commits take, the rewound position of an override, cancels,
the stop, a device failure)."""

from __future__ import annotations

import hashlib
import threading

import pytest

from models.demos.blackhole.qwen38_flash_next.chat import EOS_TOKEN_IDS, IM_END_ID, TOKENIZER_SIZE
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_lane_scheduler as scheduler_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import THINK_END_ID
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler import (
    PLACEHOLDER_DRAFT_TOKEN,
    Qwen38LaneAdmitted,
    Qwen38LaneScheduler,
    Qwen38LaneSchedulerBusy,
    Qwen38LaneStream,
    Qwen38LaneTicket,
    lane_geometry,
)

CONTEXT = 4096
DRAFTS = 4
ROWS = DRAFTS + 1
EOS_PERIOD = 37  # the deterministic model emits an EOS where its hash lands on a multiple of this


# --------------------------------------------------------------------------- a deterministic model


def next_token(sequence: list[int]) -> int:
    """The model: the argmax after ``sequence`` is a hash of the whole sequence (so a forced token changes every
    later token, as it does on the device), an EOS now and then, never an oversize id."""

    digest = hashlib.sha256(",".join(str(token) for token in sequence).encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "little")
    if value % EOS_PERIOD == 0:
        return IM_END_ID
    return 1000 + value % 5000


def draft_token(sequence: list[int], index: int) -> int:
    """The fake drafter: right about half the time (the same hash's parity), wrong otherwise."""

    truth = next_token(sequence)
    digest = hashlib.sha256((",".join(str(token) for token in sequence) + f"|{index}").encode("utf-8")).digest()
    return truth if digest[0] % 2 == 0 else 1000 + (digest[1] * 7 + 3) % 5000


def reference_stream(
    prompt: list[int], *, max_tokens: int, stop_ids, think_budget: int | None
) -> tuple[list[int], str]:
    """``Qwen38ChatSession._generate_mtp``'s greedy rules on the model: the token stream a request gets and its
    finish.  The first token (the model's next token after the prompt, read from the row) takes no budget check
    (the single stream yields it and runs its pass; the check follows that pass's first emitted token); a zero budget
    drops it for the forced ``</think>``."""

    sequence = list(prompt)
    tokens: list[int] = []
    produced = 0
    reasoning = 0
    thinking_open = think_budget is not None

    def count(token: int) -> None:
        nonlocal produced, reasoning, thinking_open
        produced += 1
        if thinking_open:
            thinking_open = token != THINK_END_ID
            reasoning += 1

    first = True
    if thinking_open and reasoning >= think_budget:
        count(THINK_END_ID)
        tokens.append(THINK_END_ID)
        sequence.append(THINK_END_ID)
        first = False
        if produced == max_tokens:
            return tokens, "length"
    while True:
        token = next_token(sequence)
        count(token)
        tokens.append(token)
        if token >= TOKENIZER_SIZE:
            return tokens, "error"
        if token in stop_ids:
            return tokens, "stop"
        if produced == max_tokens:
            return tokens, "length"
        sequence.append(token)
        if not first and thinking_open and reasoning >= think_budget:
            count(THINK_END_ID)
            tokens.append(THINK_END_ID)
            sequence.append(THINK_END_ID)
            if produced == max_tokens:
                return tokens, "length"
        first = False


# --------------------------------------------------------------------------- the fake device


class FakePassRecord:
    def __init__(self, positions, committed, accepted, argmaxes):
        self.positions = tuple(positions)
        self.committed = tuple(tuple(block) for block in committed)
        self.accepted = tuple(accepted)
        self.argmaxes = tuple(tuple(row) for row in argmaxes)
        self.segments_ns = {}


class FakeLanesDevice:
    """The lane chain on the deterministic model with the device's commit semantics: a pass runs every active lane's
    block from its committed sequence; the previous pass's rows commit at the START of the next pass, ``counts[u] + 1``
    of them (the host-written or landed count) for the lanes whose commit mask is set (the mask of the pass being
    committed); an inactive lane's rows are junk and its position holds; an admission imports a fresh state."""

    def __init__(
        self, lanes: int, *, fail_on_pass: int | None = None, admit_seconds: float = 0.0, segments: int = 3, clock=None
    ):
        self.lanes = lanes
        self.segments = segments  # the admission's device segments; ``between`` is called after all but the last
        self.clock = clock  # a fake clock the admission advances by admit_seconds / segments per segment
        self.between_calls: list[tuple[str, int, int, float]] = []
        self.sequence = [[] for _ in range(lanes)]  # the committed rows per lane
        self.last_rows = [None] * lanes  # the previous pass's rows (the persistent buffers)
        self.counts = [-1] * lanes
        self.active = [0] * lanes
        self.commit_mask = [0] * lanes
        self.blocks = [None] * lanes
        self.positions = [0] * lanes
        self.calls: list[tuple] = []
        self.passes = 0
        self.fail_on_pass = fail_on_pass
        self.admit_seconds = admit_seconds

    def admit(self, lane: int, ticket: Qwen38LaneTicket, between=None) -> Qwen38LaneAdmitted:
        self.calls.append(("admit", lane, ticket.request_id))
        # the segments: the clock advances per segment, ``between`` runs after every segment but the last and its
        # seconds are excluded from the segment records (the session's rule)
        spent = 0.0
        for index in range(self.segments):
            if self.clock is not None:
                self.clock.advance(self.admit_seconds / self.segments)
            if between is not None and index + 1 < self.segments:
                seconds = between(("chunks", "prefilled", "evicted")[min(index, 2)], index + 1, self.segments)
                self.between_calls.append((ticket.request_id, index + 1, self.segments, seconds))
                spent += seconds
        self.sequence[lane] = list(ticket.prompt_ids)
        self.last_rows[lane] = None
        self.positions[lane] = len(ticket.prompt_ids)
        seconds = {"prefill": self.admit_seconds, "evict": 0.0, "import": 0.0}
        vision = None
        if ticket.images:  # an image prompt: the tower segment before the prefill, the request's record
            seconds = {"tower": 0.05 * len(ticket.images), **seconds}
            vision = {"images": len(ticket.images), "positions": ticket.vision_positions}
        return Qwen38LaneAdmitted(
            pending=next_token(self.sequence[lane]),
            position=len(ticket.prompt_ids),
            ple_context=(1, 2),
            seconds=seconds,
            prefill={"mode": "chunked"},
            vision=vision,
        )

    def write_counts(self, counts):
        self.calls.append(("write_counts", list(counts)))
        self.counts = list(counts)

    def override_commit(self, lane, record, committed_rows):
        self.calls.append(("override_commit", lane, committed_rows))
        self.positions[lane] = record.positions[lane] + committed_rows

    def set_blocks(self, blocks):
        self.calls.append(("set_blocks", [None if b is None else list(b) for b in blocks]))
        for lane, block in enumerate(blocks):
            if block is not None:
                assert len(block) == ROWS, block
                self.blocks[lane] = list(block)

    def set_active(self, mask):
        self.calls.append(("set_active", list(mask)))
        self.commit_mask = list(self.active)
        self.active = list(mask)

    def room(self, position: int) -> bool:
        return (position & ~31) + 64 <= CONTEXT

    def step(self):
        self.passes += 1
        if self.fail_on_pass is not None and self.passes == self.fail_on_pass:
            raise RuntimeError("the device failed")
        # the commit of the previous pass
        for lane in range(self.lanes):
            if self.commit_mask[lane] and self.last_rows[lane] is not None and self.counts[lane] >= 0:
                self.sequence[lane].extend(self.last_rows[lane][: self.counts[lane] + 1])
        self.commit_mask = list(self.active)  # the mask refresh right behind the commit
        positions = list(self.positions)
        committed, accepted, argmaxes = [], [], []
        for lane in range(self.lanes):
            block = self.blocks[lane] if self.blocks[lane] is not None else [0] * ROWS
            rows = [next_token(self.sequence[lane] + block[: j + 1]) for j in range(ROWS)]
            a = 0
            while a < DRAFTS and block[a + 1] == rows[a]:
                a += 1
            argmaxes.append(rows)
            if self.active[lane]:
                accepted.append(a)
                committed.append(rows[: a + 1])
                self.last_rows[lane] = list(block)
                self.counts[lane] = a  # the landed count
                self.positions[lane] += a + 1
                # the device-assembled next block: the model's next token, then the drafter's guesses
                consumed = self.sequence[lane] + block[: a + 1]
                next_block = [rows[a]]
                for index in range(DRAFTS):
                    next_block.append(draft_token(consumed + next_block, index))
                self.blocks[lane] = next_block
            else:
                accepted.append(0)
                committed.append(())
        return FakePassRecord(positions, committed, accepted, argmaxes)


def _ticket(
    name: str, prompt: list[int], *, max_tokens=64, stop_ids=EOS_TOKEN_IDS, think_budget=None
) -> Qwen38LaneTicket:
    return Qwen38LaneTicket(
        request_id=name, prompt_ids=prompt, max_tokens=max_tokens, stop_ids=tuple(stop_ids), think_budget=think_budget
    )


def _delivered(ticket: Qwen38LaneTicket) -> list[int]:
    tokens = []
    while True:
        token, finish = ticket.tokens.get_nowait()
        if token is None:
            assert finish == ticket.finish
            return tokens
        tokens.append(token)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _run(
    device: FakeLanesDevice, tickets, *, lanes: int, queue_limit: int = 16, stall_budget_seconds=None, clock=None
) -> Qwen38LaneScheduler:
    scheduler = Qwen38LaneScheduler(
        lanes=lanes,
        drafts=DRAFTS,
        queue_limit=queue_limit,
        stall_budget_seconds=stall_budget_seconds,
        **({} if clock is None else {"clock": clock}),
    )
    for ticket in tickets:
        scheduler.submit(ticket)
    scheduler.run(device, forever=False)
    return scheduler


# --------------------------------------------------------------------------- the geometry


def test_lane_geometry_admits_two_to_eight_lanes_inside_one_tile(expect_error) -> None:
    assert lane_geometry(4, 4) == (4, 5) and lane_geometry(8, 3) == (8, 4) and lane_geometry(6, 4) == (6, 5)
    for lanes, drafts in ((1, 4), (9, 3), (7, 4), (8, 4), (5, 5 + 1)):
        with expect_error(ValueError):  # allow-pytest.raises: B in 2..8 and B x (k + 1) <= 32
            lane_geometry(lanes, drafts)
    with expect_error(ValueError):  # allow-pytest.raises: the draft count is required
        lane_geometry(4, 0)
    assert scheduler_module.MIN_LANES == 2 and scheduler_module.MAX_LANES == 8 and scheduler_module.ONE_TILE_ROWS == 32


# --------------------------------------------------------------------------- the stream rules against the reference


PROMPTS = {
    "alpha": [11, 12, 13, 14, 15, 16],
    "beta": [21, 22, 23],
    "gamma": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    "delta": [41],
    "epsilon": [51, 52, 53, 54],
    "zeta": [61, 62],
}


@pytest.mark.parametrize("think_budget", [None, 0, 1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize("max_tokens", [1, 2, 3, 7, 40, 200])
def test_lane_streams_equal_the_single_stream_reference_for_every_budget_and_length(think_budget, max_tokens) -> None:
    """Four lanes over six prompts (the fifth and sixth wait): every delivered stream equals the reference's, with
    its finish; the forced </think> at every budget falls inside passes (j < a) and on last emitted tokens (j == a)
    across these prompts (both counted below)."""

    device = FakeLanesDevice(4)
    tickets = [
        _ticket(name, prompt, max_tokens=max_tokens, think_budget=think_budget) for name, prompt in PROMPTS.items()
    ]
    scheduler = _run(device, tickets, lanes=4)
    for ticket in tickets:
        expected, finish = reference_stream(
            ticket.prompt_ids, max_tokens=max_tokens, stop_ids=EOS_TOKEN_IDS, think_budget=think_budget
        )
        assert _delivered(ticket) == expected, (ticket.request_id, think_budget, max_tokens)
        assert ticket.finish == finish, (ticket.request_id, think_budget, max_tokens)
        assert len(expected) <= max_tokens
    assert scheduler.requests_done == len(tickets) and scheduler.active_count == 0 and scheduler.waiting_count == 0


def test_forced_think_end_covers_both_pass_positions_and_the_zero_budget() -> None:
    """Across the reference sweep the budget fires on an emitted token inside a pass (rows 0 .. j + 1 commit, the
    next pass opens on </think>) and on a pass's last emitted token (the forced pass [t', </think>, placeholders]
    follows): both forms occur and both hand the device a count override or a forced block."""

    inside = last = 0
    for budget in range(1, 14):
        for name, prompt in PROMPTS.items():
            device = FakeLanesDevice(2)
            ticket = _ticket(name, prompt, max_tokens=60, think_budget=budget)
            _run(device, [ticket], lanes=2)
            expected, finish = reference_stream(prompt, max_tokens=60, stop_ids=EOS_TOKEN_IDS, think_budget=budget)
            assert _delivered(ticket) == expected and ticket.finish == finish
            overrides = [call for call in device.calls if call[0] == "override_commit"]
            forced_blocks = [
                call
                for call in device.calls
                if call[0] == "set_blocks" and any(b is not None and THINK_END_ID in b for b in call[1])
            ]
            if THINK_END_ID in expected and ticket.stream.forced_think_ends:
                assert forced_blocks, (name, budget)
                for call in overrides:
                    assert call[2] in (2, *range(2, ROWS + 1))
                if any(b is not None and b[0] == THINK_END_ID for call in forced_blocks for b in call[1]):
                    inside += 1
                if any(b is not None and b[1] == THINK_END_ID for call in forced_blocks for b in call[1]):
                    last += 1
    assert inside > 0 and last > 0
    # the zero budget: the model's first token is dropped for </think>, the first pass opens on it
    device = FakeLanesDevice(2)
    ticket = _ticket("alpha", PROMPTS["alpha"], max_tokens=20, think_budget=0)
    _run(device, [ticket], lanes=2)
    expected, _finish = reference_stream(PROMPTS["alpha"], max_tokens=20, stop_ids=EOS_TOKEN_IDS, think_budget=0)
    assert _delivered(ticket) == expected and expected[0] == THINK_END_ID
    first_blocks = next(call for call in device.calls if call[0] == "set_blocks")
    assert first_blocks[1][0][0] == THINK_END_ID and first_blocks[1][0][1:] == [PLACEHOLDER_DRAFT_TOKEN] * DRAFTS


def test_ignore_eos_streams_through_eos_and_an_oversize_id_ends_with_error(expect_error) -> None:
    device = FakeLanesDevice(2)
    ticket = _ticket("beta", PROMPTS["beta"], max_tokens=300, stop_ids=())
    _run(device, [ticket], lanes=2)
    expected, finish = reference_stream(PROMPTS["beta"], max_tokens=300, stop_ids=(), think_budget=None)
    assert finish == "length" and _delivered(ticket) == expected and IM_END_ID in expected
    stream = Qwen38LaneStream(
        drafts=DRAFTS, max_tokens=10, stop_ids=EOS_TOKEN_IDS, think_budget=None, room=lambda p: True
    )
    step = stream.start(TOKENIZER_SIZE + 5, 10)
    assert step.finish == "error" and step.tokens == ((TOKENIZER_SIZE + 5, "error"),)
    stream = Qwen38LaneStream(
        drafts=DRAFTS, max_tokens=10, stop_ids=EOS_TOKEN_IDS, think_budget=None, room=lambda p: True
    )
    assert stream.start(7, 10).block == (7, 0, 0, 0, 0)
    step = stream.after_pass([8, 9, TOKENIZER_SIZE + 1, 10], [8, 9, TOKENIZER_SIZE + 1, 10, 11], 10)
    assert step.finish == "error" and [token for token, _ in step.tokens] == [8, 9, TOKENIZER_SIZE + 1]
    with expect_error(RuntimeError):  # allow-pytest.raises: an ended stream takes no pass
        stream.after_pass([1], [1, 2, 3, 4, 5], 14)


def test_stream_no_room_ends_with_length_and_names_it() -> None:
    stream = Qwen38LaneStream(
        drafts=DRAFTS, max_tokens=50, stop_ids=EOS_TOKEN_IDS, think_budget=None, room=lambda p: p < 100
    )
    step = stream.start(5, 99)
    assert step.block == (5, 0, 0, 0, 0)
    step = stream.after_pass([6, 7], [6, 7, 8, 9, 10], 99)
    assert step.finish == "length" and step.finish_detail == "no_room" and [t for t, _ in step.tokens] == [6, 7]
    stream = Qwen38LaneStream(
        drafts=DRAFTS, max_tokens=50, stop_ids=EOS_TOKEN_IDS, think_budget=None, room=lambda p: False
    )
    step = stream.start(5, 200)
    assert step.finish == "length" and step.finish_detail == "no_room" and step.tokens == ((5, "length"),)


# --------------------------------------------------------------------------- the driver loop


def test_fifo_order_one_admission_per_boundary_and_the_fifth_request_waits() -> None:
    device = FakeLanesDevice(4)
    tickets = [_ticket(name, prompt, max_tokens=12) for name, prompt in PROMPTS.items()]
    scheduler = _run(device, tickets, lanes=4)
    admits = [call for call in device.calls if call[0] == "admit"]
    # the first four fill the idle batch in arrival order, the fifth and sixth follow as lanes free (one per boundary)
    assert [call[2] for call in admits[:4]] == ["alpha", "beta", "gamma", "delta"]
    assert [call[2] for call in admits[4:]] == ["epsilon", "zeta"]
    assert all(call[1] in range(4) for call in admits)
    first_pass = next(index for index, call in enumerate(device.calls) if call[0] == "set_active" and sum(call[1]) == 4)
    assert all(call[0] != "admit" or index < first_pass for index, call in enumerate(device.calls[: first_pass + 1]))
    steps_between = 0
    seen_fifth = False
    for call in device.calls:
        if call[0] == "admit" and call[2] == "epsilon":
            seen_fifth = True
        if seen_fifth and call[0] == "admit" and call[2] == "zeta":
            break
    assert tickets[4].queue_wait >= 0 and tickets[4].lane is not None and tickets[5].lane is not None
    assert scheduler.admissions == 6 and scheduler.requests_done == 6
    # every request's stream is the reference's (a request ending while the others continue)
    for ticket in tickets:
        expected, finish = reference_stream(ticket.prompt_ids, max_tokens=12, stop_ids=EOS_TOKEN_IDS, think_budget=None)
        assert _delivered(ticket) == expected and ticket.finish == finish


def test_the_queue_limit_refuses_beyond_the_lanes_and_stop_refuses_everything(expect_error) -> None:
    # two lanes, two behind them: a burst of four is admitted whole, the fifth is busy (the lanes are free but the
    # driver admits at boundaries, so the FIFO counts them)
    scheduler = Qwen38LaneScheduler(lanes=2, drafts=DRAFTS, queue_limit=2)
    for name in ("a", "b", "c", "d"):
        scheduler.submit(_ticket(name, [1, 2]))
    with expect_error(Qwen38LaneSchedulerBusy, match="4 requests wait"):  # allow-pytest.raises: the FIFO's limit
        scheduler.submit(_ticket("e", [1, 2]))
    assert scheduler.withdraw(scheduler.waiting[0]) and scheduler.waiting_count == 3
    scheduler.withdraw(scheduler.waiting[0])
    scheduler.withdraw(scheduler.waiting[0])
    assert scheduler.waiting_count == 1
    scheduler.stop()
    with expect_error(Qwen38LaneSchedulerBusy, match="stopping"):  # allow-pytest.raises: the stop
        scheduler.submit(_ticket("d", [1, 2]))
    device = FakeLanesDevice(2)
    scheduler.run(device, forever=False)
    remaining = [t for t in scheduler.active if t is not None]
    assert not remaining and scheduler.waiting_count == 0
    status = scheduler.status()
    assert status["lanes"] == 2 and status["stopping"] and status["running"] is False and status["requests_done"] == 1
    # with both lanes held the limit is the FIFO alone
    busy = Qwen38LaneScheduler(lanes=2, drafts=DRAFTS, queue_limit=1)
    busy.active = [_ticket("x", [1]), _ticket("y", [1])]
    busy.submit(_ticket("z", [1]))
    with expect_error(Qwen38LaneSchedulerBusy, match="0 free"):  # allow-pytest.raises: the FIFO behind held lanes
        busy.submit(_ticket("w", [1]))
    for lanes, limit in ((1, 4), (4, 0)):
        with expect_error(ValueError):  # allow-pytest.raises: the bounds
            Qwen38LaneScheduler(lanes=lanes, drafts=DRAFTS, queue_limit=limit)


def test_counts_vector_carries_admissions_and_overrides_and_positions_rewind() -> None:
    """After an admission the coming commit takes -1 on that lane (nothing) and the device-landed counts on the
    others; after a forced </think> the overridden lane's count is its committed rows - 1 and its position is
    rewound to the pass's start plus those rows; the next block is written for the lanes the boundary changed."""

    device = FakeLanesDevice(3)
    alpha = _ticket("alpha", PROMPTS["alpha"], max_tokens=40, think_budget=4)
    beta = _ticket("beta", PROMPTS["beta"], max_tokens=40)
    gamma = _ticket("gamma", PROMPTS["gamma"], max_tokens=40)
    delta = _ticket("delta", PROMPTS["delta"], max_tokens=40)
    scheduler = _run(device, [alpha, beta, gamma, delta], lanes=3)
    writes = [call for call in device.calls if call[0] == "write_counts"]
    assert writes[0][1] == [-1, -1, -1]  # the idle batch's three admissions: nothing commits at their first pass
    overrides = [call for call in device.calls if call[0] == "override_commit"]
    assert overrides, "alpha's budget of 4 forces </think> at least once"
    for index, call in enumerate(device.calls):
        if call[0] == "override_commit":
            lane, rows = call[1], call[2]
            # the boundary's counts write follows the override with rows - 1 on that lane
            following = next(c for c in device.calls[index:] if c[0] == "write_counts")
            assert following[1][lane] == rows - 1
    for ticket in (alpha, beta, gamma, delta):
        expected, finish = reference_stream(
            ticket.prompt_ids, max_tokens=40, stop_ids=EOS_TOKEN_IDS, think_budget=ticket.think_budget
        )
        assert _delivered(ticket) == expected and ticket.finish == finish
    # the device's committed sequences are the prompt plus the stream but its last token (the pending, never fed)
    assert scheduler.requests_done == 4
    # a delta admitted into a freed lane took -1 while the other lanes kept the device's counts
    admit_delta = next(index for index, call in enumerate(device.calls) if call[0] == "admit" and call[2] == "delta")
    following = next(c for c in device.calls[admit_delta:] if c[0] == "write_counts")
    assert following[1][delta.lane] == -1 and all(v >= -1 for v in following[1])


def test_a_cancel_parks_the_lane_at_the_next_boundary_and_the_others_continue() -> None:
    device = FakeLanesDevice(2)
    long_one = _ticket("alpha", PROMPTS["alpha"], max_tokens=300, stop_ids=())
    other = _ticket("beta", PROMPTS["beta"], max_tokens=300, stop_ids=())
    scheduler = Qwen38LaneScheduler(lanes=2, drafts=DRAFTS, queue_limit=4)
    scheduler.submit(long_one)
    scheduler.submit(other)
    original_step = device.step

    def step_and_cancel():
        record = original_step()
        if device.passes == 3:
            long_one.cancel("stop")
        return record

    device.step = step_and_cancel
    scheduler.run(device, forever=False)
    assert long_one.finish == "stop" and long_one.passes == 3
    assert other.finish == "length" and other.passes > 3
    masks = [call[1] for call in device.calls if call[0] == "set_active"]
    assert [1, 1] in masks and masks[-1] == [0, 0] or masks[-1] == [0, 1] or [0, 1] in masks
    expected, _finish = reference_stream(PROMPTS["beta"], max_tokens=300, stop_ids=(), think_budget=None)
    assert _delivered(other) == expected


def test_stalls_are_charged_to_the_lanes_that_were_decoding() -> None:
    """A decoding lane is charged the admission's device segments (the device's record of them), never the wall
    the admission took on the host; the admitted request records both."""

    device = FakeLanesDevice(2, admit_seconds=0.6)
    clock = [0.0]

    def tick() -> float:
        clock[0] += 1.0
        return clock[0]

    scheduler = Qwen38LaneScheduler(lanes=2, drafts=DRAFTS, queue_limit=4, clock=tick, stall_budget_seconds=None)
    tickets = [_ticket(name, prompt, max_tokens=30, stop_ids=()) for name, prompt in list(PROMPTS.items())[:3]]
    for ticket in tickets:
        scheduler.submit(ticket)
    scheduler.run(device, forever=False)
    third = tickets[2]
    assert third.queue_wait > 0 and third.admission_seconds == pytest.approx(0.6)
    assert third.admission_wall_seconds > 0 and third.interleaved_passes == 0
    # the lane that was still decoding when the third was admitted carries that admission's segments as a stall
    stalled = [t for t in tickets[:2] if t.stalled_seconds > 0]
    assert stalled and all(
        t.stalled_seconds == pytest.approx(0.6) or t.stalled_seconds == pytest.approx(1.2) for t in stalled
    )
    assert third.stalled_seconds == 0
    assert scheduler.stalled_seconds_total == pytest.approx(sum(t.stalled_seconds for t in tickets))


@pytest.mark.parametrize("stall_budget_seconds", [0.0, 0.5, 2.0, None])
def test_admissions_interleave_the_decoding_lanes_passes_under_the_budget(stall_budget_seconds) -> None:
    """Three requests on two lanes: the first admission runs whole (no lane decodes), the second and the third
    interleave the decoding lane's passes between their segments once the admission work since the last pass reached
    the budget (0: every segment; 2.0 with 1.2 s admissions: never; None: never); every stream equals the reference
    whatever the budget; the stall a decoding lane carries is the admission's segments, never the passes it got;
    the admitted request's wall carries the passes."""

    clock = FakeClock()
    device = FakeLanesDevice(2, admit_seconds=1.2, segments=3, clock=clock)
    tickets = [_ticket(name, prompt, max_tokens=30, stop_ids=()) for name, prompt in list(PROMPTS.items())[:3]]
    scheduler = _run(device, tickets, lanes=2, stall_budget_seconds=stall_budget_seconds, clock=clock)
    for ticket in tickets:
        expected, finish = reference_stream(ticket.prompt_ids, max_tokens=30, stop_ids=(), think_budget=None)
        assert _delivered(ticket) == expected and ticket.finish == finish
    first, second, third = tickets
    assert first.interleaved_passes == 0  # no lane decoded during the first admission
    interleaved = second.interleaved_passes + third.interleaved_passes
    if stall_budget_seconds is None or stall_budget_seconds > 1.2:
        assert interleaved == 0 and scheduler.interleaved_passes == 0
        assert not device.between_calls if stall_budget_seconds is None else device.between_calls
    else:
        assert interleaved > 0 and scheduler.interleaved_passes == interleaved
        per_segment = 1.2 / 3
        # budget 0: a pass at both yield points of an admission with a decoding lane; 0.5: after the second segment
        expected_per_admission = 2 if stall_budget_seconds == 0.0 else 1
        assert second.interleaved_passes == expected_per_admission
        # the passes' seconds came back through ``between`` and were excluded from the segment records
        assert all(seconds >= 0.0 for _, _, _, seconds in device.between_calls)
        assert second.admission_wall_seconds >= second.admission_seconds
    for ticket in tickets:
        assert ticket.admission_seconds == pytest.approx(1.2)
    # the decoding lane's stall is the admission segments only, charged per segment (0.4 s each): the whole 1.2 s of
    # the second admission it lived through, plus whichever segments of the third it was still active for
    assert 1.2 - 1e-9 <= first.stalled_seconds <= 2.4 + 1e-9
    assert first.stalled_seconds == pytest.approx(0.4 * round(first.stalled_seconds / 0.4))
    assert scheduler.stalled_seconds_total == pytest.approx(sum(t.stalled_seconds for t in tickets))
    assert scheduler.status()["stall_budget_seconds"] == stall_budget_seconds
    assert scheduler.status()["interleaved_passes"] == scheduler.interleaved_passes


def test_the_budget_counts_admission_work_since_the_decoding_lanes_last_pass() -> None:
    """A boundary with several short admissions: none reaches the budget alone, together they do, so the decoding
    lane gets its pass inside the second or third admission instead of waiting for the whole boundary."""

    clock = FakeClock()
    device = FakeLanesDevice(4, admit_seconds=0.3, segments=3, clock=clock)
    tickets = [_ticket(name, prompt, max_tokens=12, stop_ids=()) for name, prompt in list(PROMPTS.items())[:4]]
    # 0.45 and not 0.5: the fake clock sums 0.1 s steps, and 0.8 - 0.3 in floats falls a hair under 0.5
    scheduler = _run(device, tickets, lanes=4, stall_budget_seconds=0.45, clock=clock)
    for ticket in tickets:
        expected, finish = reference_stream(ticket.prompt_ids, max_tokens=12, stop_ids=(), think_budget=None)
        assert _delivered(ticket) == expected and ticket.finish == finish
    # the first admission runs whole (no lane decodes) and the first lane's wait starts at its activation; the second
    # admission's segments reach 0.1 and 0.2 s of wait, the third's 0.4 then 0.5 >= 0.45: the pass lands inside the
    # third admission (a per-admission clock would never have run it: every admission is 0.3 s); the fake pass takes
    # no clock time, so the fourth admission's segments reach 0.2 only
    assert [t.interleaved_passes for t in tickets] == [0, 0, 1, 0]
    assert scheduler.interleaved_passes == 1


def test_a_lane_ending_inside_an_interleaved_pass_is_charged_the_segments_it_waited_for() -> None:
    """The stall charge is per segment: a decoding lane whose request ends in the pass run between two segments of
    another request's admission carries the segments before that pass and none after (the whole-admission charge at
    the end credited it nothing, an under-count of one admission's segments on that request's record)."""

    clock = FakeClock()
    device = FakeLanesDevice(2, admit_seconds=1.2, segments=3, clock=clock)
    # alpha's stream start delivers the row's token; its first pass runs inside beta's admission (budget 0: a pass
    # at every yield point) and ends it at max_tokens 2, so alpha waited for beta's first segment only
    alpha = _ticket("alpha", PROMPTS["alpha"], max_tokens=2, stop_ids=())
    beta = _ticket("beta", PROMPTS["beta"], max_tokens=12, stop_ids=())
    scheduler = _run(device, [alpha, beta], lanes=2, stall_budget_seconds=0.0, clock=clock)
    for ticket in (alpha, beta):
        expected, finish = reference_stream(
            ticket.prompt_ids, max_tokens=ticket.max_tokens, stop_ids=(), think_budget=None
        )
        assert _delivered(ticket) == expected and ticket.finish == finish
    assert alpha.finish == "length" and alpha.passes == 1 and beta.interleaved_passes >= 1
    assert alpha.stalled_seconds == pytest.approx(0.4)  # one of beta's three 0.4 s segments
    assert beta.stalled_seconds == 0.0 and beta.admission_seconds == pytest.approx(1.2)
    assert scheduler.stalled_seconds_total == pytest.approx(0.4)


def test_an_image_ticket_carries_its_images_to_the_admission_and_takes_the_tower_segment_and_record() -> None:
    """An image request's ticket carries the decoded images and the prompt's rotary positions to the device's
    admit (the driver thread runs the tower there); the admission's tower segment counts in the admission seconds
    the decoding lanes wait for, and its vision record reaches the ticket; a text ticket carries none."""

    device = FakeLanesDevice(2, admit_seconds=0.3)
    text = _ticket("alpha", PROMPTS["alpha"], max_tokens=12, stop_ids=())
    image = _ticket("beta", PROMPTS["beta"], max_tokens=12, stop_ids=())
    image.images = ["image-a", "image-b"]
    image.vision_positions = "positions"
    _run(device, [text, image], lanes=2)
    for ticket in (text, image):
        expected, finish = reference_stream(ticket.prompt_ids, max_tokens=12, stop_ids=(), think_budget=None)
        assert _delivered(ticket) == expected and ticket.finish == finish
    assert text.vision is None and "tower" not in text.admission_segments
    assert image.vision == {"images": 2, "positions": "positions"}
    assert image.admission_segments["tower"] == pytest.approx(0.1)
    assert image.admission_seconds == pytest.approx(0.3 + 0.1)
    # the lane decoding through the image admission waited for the tower too
    assert text.stalled_seconds == pytest.approx(0.4)
    assert Qwen38LaneTicket("x", [1], 1, (), None).images == [] and Qwen38LaneAdmitted(1, 1, None, {}).vision is None


def test_stall_budget_is_none_or_a_non_negative_number(expect_error) -> None:
    from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler import (
        DEFAULT_STALL_BUDGET_SECONDS,
        stall_budget_seconds_admitted,
    )

    assert stall_budget_seconds_admitted(None) is None and stall_budget_seconds_admitted(0) == 0.0
    assert stall_budget_seconds_admitted(0.25) == 0.25 and DEFAULT_STALL_BUDGET_SECONDS == 0.5
    assert (
        Qwen38LaneScheduler(lanes=2, drafts=DRAFTS, queue_limit=1).stall_budget_seconds == DEFAULT_STALL_BUDGET_SECONDS
    )
    for bad in (-0.1, True, "0.2", float("nan")):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            stall_budget_seconds_admitted(bad)


def test_a_device_failure_ends_every_request_with_error_and_is_kept_as_fatal(expect_error) -> None:
    device = FakeLanesDevice(2, fail_on_pass=2)
    tickets = [_ticket(name, prompt, max_tokens=50, stop_ids=()) for name, prompt in list(PROMPTS.items())[:3]]
    scheduler = Qwen38LaneScheduler(lanes=2, drafts=DRAFTS, queue_limit=4)
    for ticket in tickets:
        scheduler.submit(ticket)
    with expect_error(
        RuntimeError, match="the device failed"
    ):  # allow-pytest.raises: the failure is re-raised for the server
        scheduler.run(device, forever=False)
    assert scheduler.fatal is not None and scheduler.stopping and not scheduler.running
    for ticket in tickets:
        assert ticket.finish == "error" and ticket.done.is_set()
        _delivered(ticket)  # the end marker is there


def test_the_forever_loop_waits_for_submits_and_ends_on_stop() -> None:
    device = FakeLanesDevice(2)
    scheduler = Qwen38LaneScheduler(lanes=2, drafts=DRAFTS, queue_limit=4)
    thread = threading.Thread(target=scheduler.run, args=(device,), kwargs={"forever": True}, daemon=True)
    thread.start()
    ticket = _ticket("alpha", PROMPTS["alpha"], max_tokens=9)
    scheduler.submit(ticket)
    assert ticket.done.wait(timeout=10.0)
    expected, finish = reference_stream(PROMPTS["alpha"], max_tokens=9, stop_ids=EOS_TOKEN_IDS, think_budget=None)
    assert _delivered(ticket) == expected and ticket.finish == finish
    late = _ticket("beta", PROMPTS["beta"], max_tokens=500, stop_ids=())
    scheduler.submit(late)
    while late.lane is None and not late.done.is_set():
        pass
    scheduler.stop()
    thread.join(timeout=10.0)
    assert not thread.is_alive() and late.finish == "shutdown" and late.done.is_set()
    assert device.active == [0, 0]  # every lane parked on the way out
