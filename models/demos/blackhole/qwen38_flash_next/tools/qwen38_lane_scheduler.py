# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The lanes server's scheduler (``--lanes B``): ``B`` lane slots over one MTP lane chain, a FIFO of waiting
requests, and the per-lane stream rules of the single-stream pass loop, driven from ONE device thread at pass
boundaries.  No device import: the device is an object with the methods below, so the statics drive the scheduler
with a fake and the server with :class:`Qwen38LanesSession`.

The driver loop (:meth:`Qwen38LaneScheduler.run`): at every pass boundary (1) the lanes whose request ended are
parked and freed, (2) a waiting request is admitted into a free lane -- ONE per boundary while another lane decodes,
back to back while none does -- (3) the host writes the boundary needs (the accept counts of admitted or overridden
lanes, a rewound position, the next token blocks, the active mask) and (4) one pass runs; its committed ids reach
every active lane's request through its ticket queue, token by token, under the request's own rules (EOS,
``max_tokens``, the thinking budget, an oversize id), the handler thread's rules (a stop string, the deadline, a
hang-up) arriving as a cancel the driver reads at the boundary.

The stall policy: an admission's device work (its prefill, eviction and import) is what the decoding lanes wait for,
so the device runs it in segments (the chunk groups of the prefill, one per 128 prompt rows; the forced last step;
the eviction; the import) and between two segments the driver runs one pass for the decoding lanes whenever the
admission work since their last pass reached ``stall_budget_seconds`` (the admission's ``between`` yield point; the
clock is the decoding lanes' last pass, or the moment the first of them became active, so a boundary of several short
admissions counts them together): every other lane pauses at most the budget plus one segment instead of the whole
admission, and the admitted request's first token waits one pass per interleaved pass.  The total device work is unchanged, so this
moves latency, not throughput: ``stalled_seconds`` counts the admission segments a decoding lane waited for (never
the passes it got), charged PER SEGMENT to the lanes active while it ran -- the admission's seconds are split over
its yield-point intervals by their wall time, so a request that ends inside an interleaved pass carries the segments
before its end and none after -- ``admission_seconds`` those segments, ``admission_wall_seconds`` the admission from
its start to its end with the interleaved passes inside, ``interleaved_passes`` their count.  ``stall_budget_seconds``
None runs every admission whole (the first form of the mode).

A lane's request ends at a boundary; the lane is then free and the next admission overwrites every family of its
state (readmission into any lane is exact: the stage-4 lifecycle gate), so an event that ends a request costs the
lane nothing beyond the normal commit of its last pass.  The one event that continues a lane's stream is the forced
``</think>`` of the thinking budget (the single stream forces it through the 1-row chain, which the lanes do not
have): the host rewrites the count the coming commit takes, the lane's position and its next block between the
readback and the next pass -- the admission's own mechanism (a count of -1 commits nothing) generalised -- with the
device's ``k_eff`` mask untouched (:class:`Qwen38LaneStream`).  Greedy only: acceptance is exact, so the committed
stream is the argmax stream whatever the drafts, and the forced token costs one pass of placeholder drafts, as the
single stream's re-entry after its forced step does.

Device protocol (every call from the driver thread):

* ``admit(lane, ticket, between=None) -> Qwen38LaneAdmitted``: prefill the request's prompt on the single-lane
  chain (an image prompt's images through the resident tower first: the ``"tower"`` segment), evict its state into
  the lane's host slot, import it into ``lane`` (the lane's position, rotary shift and n-gram context follow);
  ``between(segment, done, total) -> seconds`` is called between the admission's device segments and runs
  the decoding lanes' pass when the stall budget says so (its segment records exclude the seconds it returns);
* ``write_counts(counts)``: the per-lane accept counts the coming commit takes (-1 commits nothing);
* ``override_commit(lane, record, committed_rows)``: lane ``lane`` commits ``committed_rows`` of the pass ``record``
  (fewer than the device accepted): its position mirror and device row rewound to ``start + committed_rows``, its
  n-gram context that of the rows kept;
* ``set_blocks(blocks)``: the next pass's token block per lane (``None`` keeps the device-assembled block);
* ``set_active(mask)``: the coming pass's active mask (the commit mask stays the pass being committed's);
* ``step() -> record``: one pass (commit, PLE rows, verify, draft, one readback) with ``positions``, ``committed``,
  ``accepted``, ``argmaxes`` per lane and ``segments_ns``;
* ``room(position) -> bool``: whether a verify pass fits at ``position``.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import THINK_END_ID

MIN_LANES = 2
MAX_LANES = 8  # B x (k + 1) <= 32 with k >= 3
ONE_TILE_ROWS = 32
# Placeholder drafts of a host-written block (the bootstrap's form; any exact ids would do: acceptance is a filter).
PLACEHOLDER_DRAFT_TOKEN = 0
IDLE_WAIT_SECONDS = 0.25  # the driver's wait on the condition between submits when no lane is active
# The stall budget: the admission work (seconds) the decoding lanes wait for, since their last pass, before the driver
# runs their pass between two admission segments; None runs every admission whole.  The default is the measured
# setting of the 2026-09-29 sweep (docs/NUMERICS.md "Served lanes": half the interleave cost of 0.25 s on a four-burst
# for a pause bounded near 0.7 s; docs/SERVER.md states the trade-off).
DEFAULT_STALL_BUDGET_SECONDS = 0.5
# Finish reasons the driver assigns; the handler's cancels arrive with their own ("stop", "deadline", "disconnected").
FINISH_SHUTDOWN = "shutdown"


class Qwen38LaneSchedulerBusy(RuntimeError):
    """The FIFO beyond the lanes is full, or the scheduler is stopping."""


def lane_geometry(lanes: int, drafts: int) -> tuple[int, int]:
    """``(lanes, rows)`` of an admitted ``--lanes B --mtp k`` form: B in [2, 8], B x (k + 1) <= 32."""

    if isinstance(lanes, bool) or type(lanes) is not int or not MIN_LANES <= lanes <= MAX_LANES:
        raise ValueError(f"--lanes takes an int in [{MIN_LANES}, {MAX_LANES}], got {lanes!r}")
    if isinstance(drafts, bool) or type(drafts) is not int or drafts < 1:
        raise ValueError(f"the lanes need the MTP draft count (--mtp), got {drafts!r}")
    rows = drafts + 1
    if lanes * rows > ONE_TILE_ROWS:
        raise ValueError(
            f"--lanes {lanes} --mtp {drafts}: B x (k + 1) = {lanes * rows} rows exceed the {ONE_TILE_ROWS}-row tile"
        )
    return lanes, rows


def stall_budget_seconds_admitted(value: Any) -> float | None:
    """The stall budget as the scheduler takes it: None (admissions run whole) or a non-negative number of seconds
    (0: a pass at every admission segment)."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
        raise ValueError(f"the stall budget is None or a non-negative number of seconds, got {value!r}")
    return float(value)


class _Boundary:
    """The host writes one pass boundary owes the device: the accept counts the coming commit takes (as the host last
    wrote or the device last landed them, -1 on the inactive lanes: an admission or an override rewrites the vector,
    so it persists across boundaries), the next token blocks (None: the device-assembled block), the active mask, and
    whether anything is to write."""

    def __init__(self, lanes: int) -> None:
        self.counts: list[int] = [-1] * lanes
        self.blocks: list = [None] * lanes
        self.mask: list[int] = [0] * lanes
        self.writes = False


@dataclass(frozen=True)
class Qwen38LaneAdmitted:
    """What the device reports for an admission: the model's next token after the prompt (the first pass's row 0),
    the position (the prompt length), the n-gram context, and the segments in seconds (prefill, evict, import)."""

    pending: int
    position: int
    ple_context: tuple[int, int] | None
    seconds: dict[str, float]
    prefill: dict[str, Any] = field(default_factory=dict)  # the prefill's shape (chunks, slabs, forced tokens)
    vision: dict[str, Any] | None = None  # an image prompt's record (the images, their tokens, the tower's time)


@dataclass(frozen=True)
class Qwen38LaneStreamStep:
    """What one lane needs after a pass: the tokens to deliver (``(token, finish)`` pairs), the request's end, the
    commit override (``committed_rows``: fewer rows than the device accepted) and the next block (``None``: the
    device-assembled one)."""

    tokens: tuple[tuple[int, str | None], ...]
    finish: str | None = None
    committed_rows: int | None = None
    block: tuple[int, ...] | None = None
    finish_detail: str | None = None


class Qwen38LaneStream:
    """One request's stream rules over the lane's passes: ``Qwen38ChatSession._generate_mtp``'s greedy loop with the
    forced ``</think>`` as a host count override instead of a 1-row forced step.

    Row 0 of a pass is a token the client already saw; a pass emits ``[d_1 .. d_a, t']`` (the device's committed ids),
    every one counted and checked in order: an EOS or an oversize id ends the request (``stop`` / ``error``),
    ``max_tokens`` ends it with ``length``, and the thinking budget forces ``</think>`` right after the token that
    reached it.  The forced token at emitted index ``j < a`` commits rows ``0 .. j + 1`` (the token that reached the
    budget is row ``j + 1``) and opens the next pass on ``[</think>, placeholders]``; at ``j == a`` (the budget reached
    on ``t'``, which is not a row of the pass) the next pass is the FORCED pass ``[t', </think>, placeholders]`` whose
    two rows commit whatever the argmax said and whose row-1 argmax is the model's token after ``</think>`` -- the
    single stream's ``next_pending`` after its forced steps -- delivered by the following boundary's checks.
    """

    def __init__(
        self,
        *,
        drafts: int,
        max_tokens: int,
        stop_ids: Sequence[int],
        think_budget: int | None,
        room: Callable[[int], bool],
    ) -> None:
        if isinstance(drafts, bool) or type(drafts) is not int or drafts < 1:
            raise ValueError(f"drafts must be a positive int, got {drafts!r}")
        if isinstance(max_tokens, bool) or type(max_tokens) is not int or max_tokens < 1:
            raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
        if think_budget is not None and (type(think_budget) is not int or think_budget < 0):
            raise ValueError(f"think_budget must be a non-negative int or None, got {think_budget!r}")
        self.drafts = drafts
        self.rows = drafts + 1
        self.max_tokens = max_tokens
        self.stop_ids = tuple(int(value) for value in stop_ids)
        self.think_budget = think_budget
        self.room = room
        self.produced = 0
        self.reasoning_tokens = 0
        self.thinking_open = think_budget is not None
        self.forced_pass = False  # the next pass is [x, </think>, placeholders]: its two rows commit, nothing streams
        self.finish: str | None = None
        self.finish_detail: str | None = None
        self.forced_think_ends = 0  # the budget events this stream took (the request's record)
        self.forced_inside = 0  # ... reached on an emitted token inside the pass (rows 0 .. j + 1 commit)
        self.forced_last = 0  # ... reached on a pass's last emitted token (the forced pass follows)
        self.passes = 0

    # -- the rules ---------------------------------------------------------------------------

    def _count(self, token: int) -> None:
        self.produced += 1
        if self.thinking_open:
            self.thinking_open = token != THINK_END_ID
            self.reasoning_tokens += 1

    def _budget_hit(self) -> bool:
        return self.thinking_open and self.reasoning_tokens >= self.think_budget

    def _placeholders(self, *head: int) -> tuple[int, ...]:
        block = (*head, *([PLACEHOLDER_DRAFT_TOKEN] * (self.rows - len(head))))
        if len(block) != self.rows:
            raise RuntimeError(f"a lane block needs {self.rows} tokens, got {block}")
        return block

    def _end(
        self, tokens: list[tuple[int, str | None]], finish: str, *, detail: str | None = None, committed_rows=None
    ):
        self.finish, self.finish_detail = finish, detail
        return Qwen38LaneStreamStep(tuple(tokens), finish=finish, committed_rows=committed_rows, finish_detail=detail)

    def _force_think_end(self, tokens: list[tuple[int, str | None]]) -> bool:
        """``</think>`` delivered as the budget's forced token; True when it ended the request (``length``)."""

        self._count(THINK_END_ID)
        self.forced_think_ends += 1
        done = self.produced == self.max_tokens
        tokens.append((THINK_END_ID, "length" if done else None))
        return done

    def _take_pending(self, pending: int, position: int, tokens: list[tuple[int, str | None]]) -> Qwen38LaneStreamStep:
        """The single stream's not-yet-yielded branch: the checks on ``pending`` (the model's next token, row 0 of the
        coming pass at ``position``), its delivery, then the budget check that may turn the coming pass into the
        forced one."""

        if self.produced + 1 == self.max_tokens:
            tokens.append((pending, "length"))
            return self._end(tokens, "length")
        if pending >= TOKENIZER_SIZE or pending in self.stop_ids:
            tokens.append((pending, "error" if pending >= TOKENIZER_SIZE else "stop"))
            return self._end(tokens, "error" if pending >= TOKENIZER_SIZE else "stop")
        if not self.room(position):
            # The single stream hands the rest of the request to its 1-row loop here; a lane has none.  Unreachable
            # under require_budget (the context limit keeps 64 rows of headroom: every position fits a pass).
            self._count(pending)
            tokens.append((pending, "length"))
            return self._end(tokens, "length", detail="no_room")
        self._count(pending)
        tokens.append((pending, None))
        # No budget check here: the single stream yields the pending token and runs the pass at once; a budget the
        # pending token reached fires after that pass's first emitted token (the emitted loop's check), as after_pass
        # does at index 0.
        return Qwen38LaneStreamStep(tuple(tokens), block=self._placeholders(pending))

    def start(self, pending: int, position: int) -> Qwen38LaneStreamStep:
        """The admission: ``pending`` is the model's token after the prompt (read from the row), ``position`` the
        prompt length.  Returns the first pass's block, or the end of a request the first token already ends."""

        if self.finish is not None or self.passes:
            raise RuntimeError("a lane stream starts once")
        tokens: list[tuple[int, str | None]] = []
        if self._budget_hit():
            # A zero budget: the single stream drops the unyielded pending token for the forced </think>, then reads the
            # model's next token after it; the lane's first pass opens on </think> and delivers that token as emitted[0].
            if self._force_think_end(tokens):
                return self._end(tokens, "length")
            if not self.room(position):
                return self._end(tokens, "length", detail="no_room")
            return Qwen38LaneStreamStep(tuple(tokens), block=self._placeholders(THINK_END_ID))
        return self._take_pending(pending, position, tokens)

    def after_pass(self, emitted: Sequence[int], argmaxes: Sequence[int], start_position: int) -> Qwen38LaneStreamStep:
        """One pass of this lane: ``emitted`` = the device's committed ids ``[d_1 .. d_a, t']``, ``argmaxes`` the R
        per-row argmaxes, ``start_position`` the lane's position when the pass began."""

        if self.finish is not None:
            raise RuntimeError("the lane stream already ended")
        self.passes += 1
        emitted = [int(token) for token in emitted]
        tokens: list[tuple[int, str | None]] = []
        if self.forced_pass:
            # The forced pass [x, </think>, placeholders]: both rows commit (the count override), nothing new streamed;
            # the model's token after </think> is row 1's argmax, taken as the pending token at start + 2.
            self.forced_pass = False
            step = self._take_pending(int(argmaxes[1]), start_position + 2, tokens)
            return Qwen38LaneStreamStep(step.tokens, step.finish, 2, step.block, step.finish_detail)
        accepted = len(emitted) - 1
        if accepted < 0 or len(emitted) > self.rows:
            raise RuntimeError(f"a pass emits 1 .. {self.rows} ids, got {emitted}")
        for index, token in enumerate(emitted):
            self._count(token)
            if token >= TOKENIZER_SIZE or token in self.stop_ids:
                tokens.append((token, "error" if token >= TOKENIZER_SIZE else "stop"))
                return self._end(tokens, "error" if token >= TOKENIZER_SIZE else "stop")
            if self.produced == self.max_tokens:
                tokens.append((token, "length"))
                return self._end(tokens, "length")
            tokens.append((token, None))
            if self._budget_hit():
                if index < accepted:
                    # The budget reached on d_{index+1} = row index + 1: rows 0 .. index + 1 commit, </think> follows
                    # as the next pass's row 0 (the token after it streams as that pass's emitted[0]).
                    committed_rows = index + 2
                    self.forced_inside += 1
                    if self._force_think_end(tokens):
                        return self._end(tokens, "length", committed_rows=committed_rows)
                    if not self.room(start_position + committed_rows):
                        return self._end(tokens, "length", detail="no_room", committed_rows=committed_rows)
                    return Qwen38LaneStreamStep(
                        tuple(tokens), committed_rows=committed_rows, block=self._placeholders(THINK_END_ID)
                    )
                # The budget reached on t' (not a row of this pass): the whole pass commits and the next pass is the
                # forced one [t', </think>, placeholders].
                self.forced_last += 1
                if self._force_think_end(tokens):
                    return self._end(tokens, "length")
                if not self.room(start_position + accepted + 1):
                    return self._end(tokens, "length", detail="no_room")
                self.forced_pass = True
                return Qwen38LaneStreamStep(tuple(tokens), block=self._placeholders(token, THINK_END_ID))
        if not self.room(start_position + accepted + 1):
            return self._end(tokens, "length", detail="no_room")
        return Qwen38LaneStreamStep(tuple(tokens))


# --------------------------------------------------------------------------- the ticket


@dataclass
class Qwen38LaneTicket:
    """One request in the lanes server: its prompt and rules, the queue its tokens reach the handler thread by
    (``(token, finish)`` items; ``(None, finish)`` ends it), and its record."""

    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    stop_ids: tuple[int, ...]
    think_budget: int | None
    submitted: float = field(default_factory=time.perf_counter)
    sampling: Any = None  # wave D: a sampled request's parameters; None = greedy (the only served form today)
    # An image prompt: the decoded images in prompt order and the prompt's (t, h, w) rotary positions (the handler
    # thread decodes and renders; the driver thread runs the tower inside the admission); the admission's record.
    images: list = field(default_factory=list)
    vision_positions: Any = None
    vision: dict[str, Any] | None = None
    tokens: "queue.SimpleQueue[tuple[int | None, str | None]]" = field(default_factory=queue.SimpleQueue)
    done: threading.Event = field(default_factory=threading.Event)
    lane: int | None = None
    stream: Qwen38LaneStream | None = None
    admitted_at: float | None = None
    first_pass_at: float | None = None
    finished_at: float | None = None
    finish: str | None = None
    finish_detail: str | None = None
    queue_wait: float = 0.0
    admission_seconds: float = 0.0  # the admission's device segments (what the decoding lanes waited for)
    admission_wall_seconds: float = 0.0  # the admission from its start to its end, the interleaved passes inside
    admission_segments: dict[str, float] = field(default_factory=dict)
    interleaved_passes: int = 0  # the decoding lanes' passes run between this admission's segments
    prefill: dict[str, Any] = field(default_factory=dict)
    stalled_seconds: float = 0.0  # other requests' admission segments while this one decoded
    passes: int = 0
    committed: int = 0  # ids delivered to the queue
    position: int = 0
    ple_context: tuple[int, int] | None = None
    _cancel: str | None = None

    def cancel(self, reason: str) -> None:
        """The handler thread's end of the request (a stop string, the deadline, a hang-up): read by the driver at
        the next pass boundary."""

        if not isinstance(reason, str) or not reason:
            raise ValueError("a cancel needs a reason")
        self._cancel = reason

    @property
    def cancelled(self) -> str | None:
        return self._cancel

    def items(self, poll_seconds: float = 0.5, on_idle: Callable[[], None] | None = None):
        """The delivered ``(token, finish)`` items until the end marker; ``on_idle`` runs every ``poll_seconds`` the
        queue stays empty (the handler polls its client there)."""

        while True:
            try:
                token, finish = self.tokens.get(timeout=poll_seconds)
            except queue.Empty:
                if on_idle is not None:
                    on_idle()
                continue
            if token is None:
                return finish
            yield token, finish


# --------------------------------------------------------------------------- the scheduler


class Qwen38LaneScheduler:
    """``B`` lane slots, the FIFO beyond them (``queue_limit`` waiting tickets, then busy), the driver loop."""

    def __init__(
        self,
        *,
        lanes: int,
        drafts: int,
        queue_limit: int,
        clock: Callable[[], float] = time.perf_counter,
        stall_budget_seconds: float | None = DEFAULT_STALL_BUDGET_SECONDS,
    ) -> None:
        self.lanes, self.rows = lane_geometry(lanes, drafts)
        self.drafts = drafts
        if isinstance(queue_limit, bool) or type(queue_limit) is not int or queue_limit < 1:
            raise ValueError(f"queue_limit must be a positive int, got {queue_limit!r}")
        self.queue_limit = queue_limit
        self.stall_budget_seconds = stall_budget_seconds_admitted(stall_budget_seconds)
        self.clock = clock
        self.interleaved_passes = 0
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.waiting: collections.deque[Qwen38LaneTicket] = collections.deque()
        self.active: list[Qwen38LaneTicket | None] = [None] * lanes
        self.stopping = False
        self.running = False
        self.fatal: BaseException | None = None
        self.passes = 0
        self.admissions = 0
        self.requests_done = 0
        self.stalled_seconds_total = 0.0
        self.last_progress: float = clock()  # the last completed device call (a pass, an admission)
        # When the decoding lanes last progressed: the end of the last pass, or the activation of the first lane
        # after an idle boundary (the stall budget counts admission work from here).
        self.lanes_progressed_at: float = clock()
        self.last_pass_seconds: float | None = None
        self.pass_seconds: collections.deque[float] = collections.deque(maxlen=200)

    # -- the handler threads' side -----------------------------------------------------------

    def submit(self, ticket: Qwen38LaneTicket) -> Qwen38LaneTicket:
        """A place in the FIFO: the free lanes take the first arrivals at the next boundary and ``queue_limit`` may
        wait behind the B lanes; a request beyond that is busy (a burst of B + queue_limit is admitted whole)."""

        with self.lock:
            if self.stopping:
                raise Qwen38LaneSchedulerBusy("the server is stopping")
            free = sum(1 for held in self.active if held is None)
            if len(self.waiting) >= self.queue_limit + free:
                raise Qwen38LaneSchedulerBusy(
                    f"{len(self.waiting)} requests wait for the {self.lanes} lanes ({free} free), "
                    f"the limit is {self.queue_limit} behind the lanes"
                )
            ticket.submitted = self.clock()
            self.waiting.append(ticket)
            self.wake.notify_all()
        return ticket

    def withdraw(self, ticket: Qwen38LaneTicket) -> bool:
        """A waiting request whose client left gives up its place (False: it was already admitted)."""

        with self.lock:
            try:
                self.waiting.remove(ticket)
            except ValueError:
                return False
            return True

    def stop(self) -> None:
        with self.lock:
            self.stopping = True
            self.wake.notify_all()

    @property
    def active_count(self) -> int:
        return sum(1 for ticket in self.active if ticket is not None)

    @property
    def waiting_count(self) -> int:
        return len(self.waiting)

    def free_lanes(self) -> list[int]:
        return [lane for lane, ticket in enumerate(self.active) if ticket is None]

    def status(self) -> dict[str, Any]:
        with self.lock:
            active = [None if ticket is None else ticket.request_id for ticket in self.active]
            return {
                "lanes": self.lanes,
                "drafts": self.drafts,
                "rows": self.rows,
                "stall_budget_seconds": self.stall_budget_seconds,
                "interleaved_passes": self.interleaved_passes,
                "active": sum(1 for ticket in self.active if ticket is not None),
                "active_requests": active,
                "free": sum(1 for ticket in self.active if ticket is None),
                "waiting": len(self.waiting),
                "queue_limit": self.queue_limit,
                "passes": self.passes,
                "admissions": self.admissions,
                "requests_done": self.requests_done,
                "stalled_seconds_total": round(self.stalled_seconds_total, 3),
                "last_pass_seconds": None if self.last_pass_seconds is None else round(self.last_pass_seconds, 4),
                "seconds_since_progress": round(self.clock() - self.last_progress, 3),
                "stopping": self.stopping,
                "running": self.running,
            }

    # -- the driver thread --------------------------------------------------------------------

    def _finish(self, ticket: Qwen38LaneTicket, finish: str, *, detail: str | None = None) -> None:
        ticket.finish, ticket.finish_detail = finish, detail
        ticket.finished_at = self.clock()
        if ticket.lane is not None and self.active[ticket.lane] is ticket:
            self.active[ticket.lane] = None
        ticket.tokens.put((None, finish))
        ticket.done.set()
        self.requests_done += 1

    def _deliver(self, ticket: Qwen38LaneTicket, step: Qwen38LaneStreamStep) -> None:
        for token, finish in step.tokens:
            ticket.committed += 1
            ticket.tokens.put((token, finish))

    def _admit(self, device, ticket: Qwen38LaneTicket, lane: int, boundary: _Boundary) -> None:
        """One admission into ``lane``: the device's prefill / evict / import in segments, the decoding lanes' pass
        between two segments once the admission work since their last pass reached the stall budget, then the
        stream's start and the boundary's writes for the new lane."""

        started = self.clock()
        ticket.queue_wait = started - ticket.submitted
        ticket.lane = lane
        budget = self.stall_budget_seconds
        was_idle = self.active_count == 0
        # The stall charge per segment: at every yield point (and at the end) the wall since the last mark and the
        # decoding lanes that waited through it; the admission's device seconds are split over these intervals.
        waited: list[tuple[float, tuple[Qwen38LaneTicket, ...]]] = []
        mark = [started]  # the clock at the end of the last segment (a pass moves it: its seconds count for nobody)

        def segment_done(now: float) -> None:
            waited.append((now - mark[0], tuple(t for t in self.active if t is not None and t is not ticket)))
            mark[0] = now

        def between(segment: str, done: int, total: int) -> float:
            now = self.clock()
            segment_done(now)
            if budget is None or self.active_count == 0 or now - self.lanes_progressed_at < budget:
                return 0.0
            self._pass(device, boundary)
            ticket.interleaved_passes += 1
            self.interleaved_passes += 1
            after = self.clock()
            mark[0] = after
            return after - now

        interleave = budget is not None and self.active_count > 0
        admitted = device.admit(lane, ticket, between=between if interleave else None)
        ticket.admitted_at = self.clock()
        segment_done(ticket.admitted_at)
        ticket.admission_wall_seconds = ticket.admitted_at - started
        ticket.admission_segments = dict(admitted.seconds)
        ticket.admission_seconds = sum(float(value) for value in admitted.seconds.values())
        ticket.prefill = dict(admitted.prefill)
        ticket.vision = admitted.vision
        ticket.position = admitted.position
        ticket.ple_context = admitted.ple_context
        self.admissions += 1
        self.last_progress = ticket.admitted_at
        # The device's segment seconds (the passes excluded) over the yield-point intervals by wall share: each lane
        # carries the intervals it was active for, so a lane whose request ended inside an interleaved pass is
        # credited the segments it waited for and none after.
        walls = [max(0.0, wall) for wall, _ in waited]
        total_wall = sum(walls)
        for wall, others in zip(walls, (lanes for _, lanes in waited)):
            share = ticket.admission_seconds * (wall / total_wall if total_wall > 0 else 1.0 / len(waited))
            for other in others:
                other.stalled_seconds += share
                self.stalled_seconds_total += share
        ticket.stream = Qwen38LaneStream(
            drafts=self.drafts,
            max_tokens=ticket.max_tokens,
            stop_ids=ticket.stop_ids,
            think_budget=ticket.think_budget,
            room=device.room,
        )
        step = ticket.stream.start(admitted.pending, admitted.position)
        self._deliver(ticket, step)
        if step.finish is not None:
            # The first token ended the request: the lane was never activated and stays free.
            ticket.lane = None
            self._finish(ticket, step.finish, detail=step.finish_detail)
            return
        self.active[lane] = ticket
        boundary.counts[lane] = -1  # the admitted lane's first commit takes nothing
        boundary.blocks[lane] = list(step.block)
        boundary.mask[lane] = 1
        boundary.writes = True
        if was_idle:
            self.lanes_progressed_at = self.clock()  # the first decoding lane's wait starts here

    def _pass(self, device, boundary: _Boundary) -> None:
        """One pass for the active lanes: the boundary's pending writes, the active mask, the pass, then every
        active lane's stream over its committed ids (the deliveries, the ends, the overrides and next blocks the
        following boundary writes)."""

        if boundary.writes:
            device.write_counts(list(boundary.counts))
            device.set_blocks(boundary.blocks)
        device.set_active(list(boundary.mask))
        started = self.clock()
        record = device.step()
        now = self.clock()
        self.passes += 1
        self.last_pass_seconds = now - started
        self.pass_seconds.append(self.last_pass_seconds)
        self.last_progress = now
        self.lanes_progressed_at = now
        boundary.counts = [-1 if self.active[u] is None else int(record.accepted[u]) for u in range(self.lanes)]
        boundary.blocks = [None] * self.lanes
        boundary.writes = False
        for lane, ticket in enumerate(self.active):
            if ticket is None:
                continue
            ticket.passes += 1
            if ticket.first_pass_at is None:
                ticket.first_pass_at = now
            step = ticket.stream.after_pass(record.committed[lane], record.argmaxes[lane], record.positions[lane])
            self._deliver(ticket, step)
            if step.committed_rows is not None:
                boundary.counts[lane] = step.committed_rows - 1
                device.override_commit(lane, record, step.committed_rows)
                ticket.position = record.positions[lane] + step.committed_rows
                boundary.writes = True
            else:
                ticket.position = record.positions[lane] + len(record.committed[lane])
            if step.block is not None:
                boundary.blocks[lane] = list(step.block)
                boundary.writes = True
            finish = step.finish if step.finish is not None else ticket.cancelled
            if finish is not None:
                boundary.mask[lane] = 0
                self._finish(ticket, finish, detail=step.finish_detail)
        if boundary.writes:
            device.write_counts(list(boundary.counts))
            device.set_blocks(boundary.blocks)
            boundary.blocks = [None] * self.lanes
            boundary.writes = False

    def run(self, device, *, forever: bool = True) -> None:
        """The driver loop; ``forever`` waits for submits when idle, else returns once every ticket is done (the
        startup replay's form).  A device failure ends the loop with every request answered ``error`` and is kept
        in ``fatal`` for the server (a poisoned model owner cannot serve)."""

        self.running = True
        boundary = _Boundary(self.lanes)
        try:
            while True:
                with self.lock:
                    if self.stopping:
                        break
                    if not self.waiting and self.active_count == 0:
                        if not forever:
                            break
                        self.wake.wait(timeout=IDLE_WAIT_SECONDS)
                        continue
                    admit_now = list(self.waiting) if self.active_count == 0 else list(self.waiting)[:1]
                    for ticket in admit_now:
                        self.waiting.remove(ticket)
                # -- the boundary's host work: admissions (one while a lane decodes, every one while none does; from
                # the second on, the decoding lanes' passes run between an admission's segments under the budget)
                for index, ticket in enumerate(admit_now):
                    if ticket.cancelled is not None:
                        self._finish(ticket, ticket.cancelled)
                        continue
                    free = self.free_lanes()
                    if not free:
                        with self.lock:
                            self.waiting.extendleft(reversed(admit_now[index:]))
                        break
                    self._admit(device, ticket, free[0], boundary)
                if self.active_count == 0:
                    continue
                # -- the pass and every active lane's stream
                self._pass(device, boundary)
        except BaseException as error:  # noqa: BLE001  the device failed: every request ends, the server learns
            self.fatal = error
            with self.lock:
                self.stopping = True
                waiting = list(self.waiting)
                self.waiting.clear()
            for ticket in waiting:
                self._finish(ticket, "error")
            for ticket in list(self.active):
                if ticket is not None:
                    self._finish(ticket, "error")
            raise
        finally:
            if self.stopping and self.fatal is None:
                with self.lock:
                    waiting = list(self.waiting)
                    self.waiting.clear()
                for ticket in waiting:
                    self._finish(ticket, FINISH_SHUTDOWN)
                for ticket in list(self.active):
                    if ticket is not None:
                        self._finish(ticket, FINISH_SHUTDOWN)
                if any(boundary.mask):
                    try:
                        device.set_active([0] * self.lanes)
                    except BaseException as error:  # noqa: BLE001
                        self.fatal = error
            self.running = False
