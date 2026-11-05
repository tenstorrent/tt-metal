# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP lane width buckets: the width pick, the commit-at-previous-width rule, the free-DRAM
fallback, the env, and the lowest-free-lane admission (no device).

The lane chain runs on scripted stand-ins (the replay log names the trace that ran), the session on a stub with the
device calls replaced; the adapter tests reuse the lane fake of ``test_vllm_mtp_no_device``.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_mtp_no_device import (  # noqa: F401 (fixtures, helpers)
    LROWS,
    PLACEHOLDER,
    _hybrid_model,
    _hybrid_prefill,
    _prompt,
    _run_hybrid_steps,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_lanes_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm_mtp as adapter_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes

LANES, DRAFTS = 8, 3
ROWS = DRAFTS + 1
VARIABLE = "QWEN38_MTP_LANE_BUCKETS"


# --------------------------------------------------------------------------- env


@pytest.mark.parametrize(
    "environ,expected",
    (
        ({}, (2, 4, 8)),
        ({VARIABLE: ""}, (2, 4, 8)),
        ({VARIABLE: "2,4,8"}, (2, 4, 8)),
        ({VARIABLE: "8,2,4"}, (2, 4, 8)),  # sorted here
        ({VARIABLE: " 4 , 8 "}, (4, 8)),
        ({VARIABLE: "2,8"}, (2, 8)),
        ({VARIABLE: "8"}, (8,)),
        ({VARIABLE: "off"}, (8,)),
        ({VARIABLE: "OFF"}, (8,)),
        ({VARIABLE: "0"}, (8,)),
    ),
)
def test_buckets_from_env_of_an_eight_lane_session(environ, expected) -> None:
    assert mtp_lanes.mtp_lane_buckets_from_env(LANES, DRAFTS, environ) == expected


@pytest.mark.parametrize("text", ("3,2,8", "9", "2,4", "a,b", "2,2,8", "1,8", "16", "2;8", "-8", "4,8,"))
def test_buckets_from_env_refuses_by_name(expect_error, text: str) -> None:
    with expect_error(ValueError, VARIABLE):
        mtp_lanes.mtp_lane_buckets_from_env(LANES, DRAFTS, {VARIABLE: text})


def test_buckets_must_fit_one_verify_tile(expect_error) -> None:
    with expect_error(ValueError, "do not fit one 32-row tile"):
        mtp_lanes.mtp_lane_buckets_from_env(LANES, 4, {VARIABLE: "2,4,8"})  # 8 x 5 rows


@pytest.mark.parametrize("lanes", (2, 3, 4, 6))
def test_other_lane_counts_have_one_width_and_ignore_the_variable(lanes: int) -> None:
    for environ in ({}, {VARIABLE: "2,4,8"}, {VARIABLE: "garbage"}, {VARIABLE: "off"}):
        assert mtp_lanes.mtp_lane_buckets_from_env(lanes, DRAFTS, environ) == (lanes,)


def test_the_flag_name_is_one_constant_in_the_library_and_the_adapter() -> None:
    assert mtp_lanes.MTP_LANE_BUCKETS_VARIABLE == adapter_module.MTP_LANE_BUCKETS_VARIABLE == VARIABLE
    assert session_module.mtp_lane_buckets_from_env is mtp_lanes.mtp_lane_buckets_from_env


# --------------------------------------------------------------------------- the lane chain over width sets


class _World:
    """A scripted lane chain: 8 lanes, width sets 2/4/8 whose trace ids are ``100 * width + {1: verify, 2: commit, 3: draft}``.
    ``log`` holds ``enqueue:<name>`` for every launched trace, the host writes and the readback."""

    def __init__(self, monkeypatch, widths=(2, 4, 8), enqueue=True, positions=None):
        self.calls: list[str] = []
        self.committed_counts: list[list[int]] = []
        self.writes: list[tuple[str, int]] = []
        self.ple_views: list[int | None] = []
        owner = object()
        positions_row = mtp_lanes.Qwen38TTNNMTPLanePositions.__new__(mtp_lanes.Qwen38TTNNMTPLanePositions)
        positions_row.positions, positions_row.lanes = (
            list(positions or [10 * (lane + 1) for lane in range(LANES)]),
            LANES,
        )
        self.positions = positions_row
        self.parent = SimpleNamespace(lanes=LANES, drafts=DRAFTS, rows=ROWS, positions=positions_row, _owner=owner)
        self.model = SimpleNamespace(_state_owner=owner, allocated_context=32_768)
        self.names: dict[int, str] = {}

        def traces(width):
            ids = [100 * width + 1, 100 * width + 2, 100 * width + 3]
            for tid, kind in zip(ids, ("verify", "commit", "draft")):
                self.names[tid] = f"{kind}{width}"
            return mtp_lanes.Qwen38TTNNMTPLaneTraces(*ids)

        self.sets = {}
        for width in widths:
            if width == LANES:
                continue
            view = SimpleNamespace(lanes=width, parent=self.parent, width=width)
            self.sets[width] = mtp_lanes.Qwen38TTNNMTPLaneWidthSet(
                width, view, SimpleNamespace(width=width), traces(width)
            )
        self.draft = SimpleNamespace(width=LANES)
        self.full_traces = traces(LANES)
        self.readback_width: list[int] = []
        accepted_by_lane = (1, 0, 2, 3, 1, 0, 2, 1)

        def read_row(verify_, draft_):
            width = verify_.lanes if verify_ is not self.parent else LANES
            self.readback_width.append(width)
            self.calls.append("readback")
            accepted = accepted_by_lane[:width]
            argmaxes = tuple(tuple(100 * u + j for j in range(ROWS)) for u in range(width))
            readback = mtp_lanes.Qwen38TTNNLaneVerifyReadback(
                accepted=tuple(accepted),
                next_token=tuple(argmaxes[u][accepted[u]] for u in range(width)),
                first_draft=tuple(7 + u for u in range(width)),
                argmaxes=argmaxes,
                alignment_argmaxes=argmaxes,
            )
            blocks = tuple(
                (readback.next_token[u], readback.first_draft[u], *range(500 + u, 500 + u + DRAFTS - 1))
                for u in range(width)
            )
            return readback, blocks

        def commit_host(model_, verify_, counts):
            assert verify_ is self.parent  # every host mirror lives on the parent
            self.committed_counts.append(list(counts))
            positions_row.positions = [p + c for p, c in zip(positions_row.positions, counts)]

        def write_tokens(model_, verify_, tokens):
            assert verify_ is self.parent and len(tokens) == LANES
            self.calls.append("write_tokens")

        def write_ple(model_, verify_, tokens, *, view=None):
            assert verify_ is self.parent and len(tokens) == LANES
            self.ple_views.append(None if view is None else view.lanes)
            self.calls.append("ple_rows")

        def write_active(model_, verify_, flags, *, commit):
            assert verify_ is self.parent and len(flags) == LANES
            self.calls.append("write_active")

        monkeypatch.setattr(mtp_lanes, "_validate_lane_draft_state", lambda *args: None)
        monkeypatch.setattr(mtp_lanes, "read_lane_pass_row", read_row)
        monkeypatch.setattr(mtp_lanes, "commit_lane_verify_host", commit_host)
        monkeypatch.setattr(mtp_lanes, "write_lane_tokens", write_tokens)
        monkeypatch.setattr(mtp_lanes, "write_lane_ple_rows", write_ple)
        monkeypatch.setattr(mtp_lanes, "write_lane_active", write_active)
        monkeypatch.setattr(
            mtp_lanes, "write_lane_verify_inputs", lambda model_, verify_, tokens: self.calls.append("host_inputs")
        )
        launch = lambda tid: self.calls.append(f"launch:{self.names[tid]}")  # noqa: E731
        held_blocks = tuple((900 + lane, 910 + lane, 1, 2) for lane in range(LANES))
        self.held_blocks = held_blocks
        self.chain = mtp_lanes.Qwen38TTNNMTPLaneChain(
            self.model,
            self.parent,
            self.draft,
            self.full_traces,
            replay=launch,
            enqueue=launch if enqueue else None,
            next_tokens=held_blocks,
            width_sets=self.sets,
        )

    def step(self, active, **kwargs):
        self.calls.clear()
        self.chain.active = list(active)
        self.chain.commit_mask = list(active)  # the mask refresh is not under test here
        return self.chain.step(**kwargs)

    def launched(self):
        return [call.removeprefix("launch:") for call in self.calls if call.startswith("launch:")]


def _mask(*lanes: int) -> list[int]:
    return [1 if lane in lanes else 0 for lane in range(LANES)]


@pytest.mark.parametrize(
    "active,held,expected",
    (
        ((), None, 2),  # none: the smallest bucket
        ((0,), None, 2),
        ((0, 1), None, 2),
        ((0, 2), None, 4),
        ((2,), None, 4),
        ((0, 3), None, 4),
        ((3,), None, 4),
        ((0, 4), None, 8),
        ((0, 5), None, 8),
        ((7,), None, 8),
        ((0,), (0, 5), 8),  # a held lane counts although it is parked
        ((0, 1), (0, 1, 2), 4),
        ((), (3,), 4),
        ((1,), (0,), 2),
    ),
)
def test_width_pick_covers_the_highest_active_or_held_lane(monkeypatch, active, held, expected) -> None:
    world = _World(monkeypatch)
    held_mask = None if held is None else _mask(*held)
    assert world.chain.width_for(_mask(*active), held_mask) == expected


def test_width_pick_uses_only_the_buckets_there_are(monkeypatch) -> None:
    world = _World(monkeypatch, widths=(4, 8))
    assert world.chain.buckets == (4, 8)
    assert world.chain.width_for(_mask(0)) == 4 and world.chain.width_for(_mask(4)) == 8
    only = _World(monkeypatch, widths=(8,))
    assert only.chain.buckets == (8,) and only.chain.width_for(_mask(0)) == 8


def test_held_mask_is_remembered_and_validated(monkeypatch, expect_error) -> None:
    world = _World(monkeypatch)
    world.chain.set_held(_mask(0, 6))
    assert world.chain.width_for(_mask(0)) == 8
    world.chain.set_held(None)
    assert world.chain.width_for(_mask(0)) == 2
    with expect_error(ValueError, "held needs 8 0/1 flags"):
        world.chain.set_held([1, 0])
    with expect_error(ValueError, "held needs 8 0/1 flags"):
        world.chain.set_held([2] * LANES)


def test_the_commit_replays_at_the_previous_passes_width_verify_and_draft_at_the_current(monkeypatch) -> None:
    world = _World(monkeypatch)
    # the chain starts after an eager 8-lane pass: the first commit is the 8-lane commit
    schedule = [
        ((0, 1), ["commit8", "verify2", "draft2"], 2),
        ((0, 1), ["commit2", "verify2", "draft2"], 2),
        ((0, 1, 2), ["commit2", "verify4", "draft4"], 4),
        ((0, 1, 2, 5), ["commit4", "verify8", "draft8"], 8),
        ((0,), ["commit8", "verify2", "draft2"], 2),
        ((0,), ["commit2", "verify2", "draft2"], 2),
    ]
    for index, (active, launched, width) in enumerate(schedule):
        record = world.step(_mask(*active))
        assert world.launched() == launched, (index, world.calls)
        assert record.width == width and record.index == index
        assert world.chain.prev_set.width == width


def test_a_forced_width_runs_that_width_and_refuses_a_lane_outside_it(monkeypatch, expect_error) -> None:
    world = _World(monkeypatch)
    record = world.step(_mask(0, 1), force_width=8)
    assert record.width == 8 and world.launched() == ["commit8", "verify8", "draft8"]
    record = world.step(_mask(0, 1), force_width=4)
    assert world.launched() == ["commit8", "verify4", "draft4"] and record.width == 4
    with expect_error(RuntimeError, "outside the width 2"):
        world.step(_mask(0, 2), force_width=2)
    with expect_error(RuntimeError, "outside the width 2"):  # a parked but held lane is outside it too
        world.chain.set_held(_mask(3))
        world.step(_mask(0), force_width=2)
    with expect_error(ValueError, "no trace set of width 6"):
        world.step(_mask(0), force_width=6)


def test_a_narrow_readback_is_padded_to_eight_lanes_and_the_mirror_follows_the_parent(monkeypatch) -> None:
    world = _World(monkeypatch, positions=[100, 200, 300, 400, 500, 600, 700, 800])
    record = world.step(_mask(0, 1))
    assert world.readback_width == [2]
    assert record.accepted == (1, 0, -1, -1, -1, -1, -1, -1)
    assert [len(block) for block in record.argmaxes] == [ROWS, ROWS, 0, 0, 0, 0, 0, 0]
    assert record.committed == ((0, 1), (100,), (), (), (), (), (), ())
    assert record.next_token[:2] == (1, 100) and record.next_token[2:] == tuple(900 + lane for lane in range(2, 8))
    assert record.first_draft[2:] == tuple(910 + lane for lane in range(2, 8))
    # the lanes past the width keep their held blocks, the others the pass's assembled ones
    assert record.next_tokens[2:] == world.held_blocks[2:] and len(record.next_tokens) == LANES
    assert record.next_tokens[0][0] == 1 and record.next_tokens[1][0] == 100
    assert world.chain.next_tokens == record.next_tokens
    # the host commit gets 8 counts, zero above the width; the mirror advances only the lanes that ran
    assert world.committed_counts == [[2, 1, 0, 0, 0, 0, 0, 0]]
    assert world.positions.positions == [102, 201, 300, 400, 500, 600, 700, 800]
    assert record.positions == (100, 200, 300, 400, 500, 600, 700, 800) and len(record.tokens) == LANES


def test_a_full_width_pass_is_not_padded(monkeypatch) -> None:
    world = _World(monkeypatch)
    record = world.step(_mask(*range(LANES)))
    assert record.width == LANES and world.readback_width == [LANES]
    assert all(len(block) == ROWS for block in record.argmaxes) and -1 not in record.accepted


def test_the_held_blocks_are_rewritten_only_when_the_width_grows(monkeypatch) -> None:
    world = _World(monkeypatch)
    world.step(_mask(0, 1))  # 8 -> 2: the draft's sentinel clobbers lanes 2..7: nothing to restore yet
    assert "write_tokens" not in world.calls
    world.step(_mask(0, 1))  # 2 -> 2
    assert "write_tokens" not in world.calls
    world.step(_mask(0, 1, 2))  # 2 -> 4: lanes 2..3 are about to run: the host copy goes back first
    assert world.calls.index("write_tokens") < world.calls.index("ple_rows") < world.calls.index("launch:verify4")
    world.step(_mask(0, 1, 2))  # 4 -> 4
    assert "write_tokens" not in world.calls
    world.step(_mask(0))  # 4 -> 2
    assert "write_tokens" not in world.calls
    world.step(_mask(*range(LANES)))  # 2 -> 8
    assert world.calls.count("write_tokens") == 1


def test_host_writes_of_a_pass_go_to_the_parent_with_eight_lanes(monkeypatch) -> None:
    world = _World(monkeypatch)
    world.chain.commit_mask = [0] * LANES  # the mask refresh write runs once
    world.chain.active = _mask(0)
    world.chain.step()
    assert "write_active" in world.calls and "ple_rows" in world.calls  # the stubs assert verify_ is the parent


def test_bootstrap_has_no_commit_and_picks_the_width(monkeypatch) -> None:
    world = _World(monkeypatch)
    world.chain.next_tokens = None
    world.chain.active = _mask(0, 1, 2)
    record = world.chain.bootstrap([[1, 0, 0, 0]] * LANES)
    assert world.launched() == ["verify4", "draft4"] and record.width == 4
    assert world.calls[0] == "host_inputs" and world.chain.prev_set.width == 4
    assert len(world.committed_counts) == 1 and len(world.committed_counts[0]) == LANES


def test_a_width_cannot_be_dropped_while_it_holds_the_pending_commit(monkeypatch, expect_error) -> None:
    world = _World(monkeypatch)
    world.step(_mask(0, 1, 2))
    assert world.chain.prev_set.width == 4
    with expect_error(RuntimeError, "pending commit"):
        world.chain.drop_width(4)
    with expect_error(ValueError, "cannot drop width 8"):
        world.chain.drop_width(8)
    dropped = world.chain.drop_width(2)
    assert dropped.width == 2 and world.chain.buckets == (4, 8)
    assert world.chain.width_for(_mask(0)) == 4
    with expect_error(ValueError, "cannot drop width 2"):
        world.chain.drop_width(2)


def test_the_chain_refuses_a_set_that_is_not_a_view_of_its_parent(monkeypatch, expect_error) -> None:
    world = _World(monkeypatch)
    stranger = mtp_lanes.Qwen38TTNNMTPLaneWidthSet(
        2, SimpleNamespace(lanes=2, parent=object()), SimpleNamespace(), world.full_traces
    )
    with expect_error(ValueError, "not a width view"):
        mtp_lanes.Qwen38TTNNMTPLaneChain(
            world.model, world.parent, world.draft, world.full_traces, replay=lambda t: None, width_sets={2: stranger}
        )


def test_flag_off_the_chain_runs_todays_calls_on_the_one_full_set(monkeypatch) -> None:
    world = _World(monkeypatch, widths=(8,))
    assert world.chain.buckets == (8,) and world.chain.width_sets.keys() == {8}
    for active in (_mask(0), _mask(0, 1, 2), _mask(*range(LANES))):
        record = world.step(active)
        assert world.launched() == ["commit8", "verify8", "draft8"] and record.width == 8
        assert world.calls.count("write_tokens") == 0 and "readback" in world.calls
        assert world.readback_width[-1] == LANES and -1 not in record.accepted
    assert world.chain.traces is world.full_traces and world.chain.verify is world.parent


def test_flag_off_builds_no_view_and_no_narrow_capture() -> None:
    """The session builds and captures only ``buckets[:-1]``: with ``(8,)`` those loops are empty (source facts)."""

    prepare = inspect.getsource(session_module.Qwen38LanesSession.prepare)
    capture = inspect.getsource(session_module.Qwen38LanesSession.capture)
    assert "for width in self.buckets[:-1]:" in prepare and "allocate_lane_width_view" in prepare
    assert "for width in reversed(self.buckets[:-1]):" in capture
    assert mtp_lanes.mtp_lane_buckets_from_env(8, 3, {VARIABLE: "off"}) == (8,)
    # the full-width eager passes stay as they were and run after every narrow width
    assert prepare.index("self._eager_passes_at(self.width_sets[width])") < prepare.index(
        "mtp_lanes.write_lane_accepted(model, self.verify, [-1] * self.lanes)"
    )


# --------------------------------------------------------------------------- the session: fallback and warm replays


class _Dram:
    def __init__(self, free_values):
        self.free_values = list(free_values)
        self.reads = 0

    def __call__(self, mesh):
        value = self.free_values[min(self.reads, len(self.free_values) - 1)]
        self.reads += 1
        return {
            "num_banks": 8,
            "free_bytes_per_bank": value,
            "largest_contiguous_bytes_free_per_bank": value,
            "allocated_bytes_per_bank": 0,
        }


def _session(monkeypatch, free_values, widths=(2, 4, 8)):
    released_traces, released_views = [], []
    monkeypatch.setattr(session_module, "_dram_view", _Dram(free_values))
    monkeypatch.setattr(session_module.ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(session_module.ttnn, "release_trace", lambda mesh, tid: released_traces.append(tid))
    monkeypatch.setattr(
        session_module.mtp_lanes, "release_lane_width_view", lambda model, view: released_views.append(view.width)
    )
    world = _World(monkeypatch, widths=widths)
    session = session_module.Qwen38LanesSession(lanes=LANES, drafts=DRAFTS, marker=lambda text: None)
    session.mesh, session.model = object(), world.model
    session.buckets = tuple(sorted(widths))
    for width, width_set in world.sets.items():
        width_set.view = SimpleNamespace(width=width)
        width_set.output = None
    session.width_sets = dict(world.sets)
    session.lane_chain = world.chain
    return session, world, released_traces, released_views


MIB = 2**20


def test_enough_free_dram_keeps_every_width(monkeypatch) -> None:
    session, world, traces, views = _session(monkeypatch, [200 * MIB])
    assert session.enforce_free_dram() == [] and session.buckets == (2, 4, 8)
    assert traces == [] and views == [] and world.chain.buckets == (2, 4, 8)


def test_free_dram_floor_is_150_mib_per_bank() -> None:
    assert session_module.FREE_DRAM_FLOOR_BYTES_PER_BANK == 150 * MIB


def test_low_free_dram_drops_the_four_lane_width_first_then_the_two_lane_width(monkeypatch) -> None:
    session, world, traces, views = _session(monkeypatch, [100 * MIB, 120 * MIB, 140 * MIB])
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        dropped = session.enforce_free_dram()
    finally:
        logger.remove(sink)
    assert dropped == [4, 2] and session.buckets == (8,) and session.dropped_widths == [4, 2]
    assert views == [4, 2] and sorted(traces) == [201, 202, 203, 401, 402, 403]
    assert world.chain.buckets == (8,) and set(session.width_sets) == set()
    assert any("dropping the 4-lane width" in m for m in messages) and any(
        "dropping the 2-lane width" in m for m in messages
    )
    assert any("widths [4, 2] dropped" in m for m in messages)


def test_dropping_the_four_lane_width_may_be_enough(monkeypatch) -> None:
    session, world, traces, views = _session(monkeypatch, [100 * MIB, 160 * MIB])
    assert session.enforce_free_dram() == [4] and session.buckets == (2, 8) and views == [4]
    assert world.chain.buckets == (2, 8)


def test_the_full_width_always_stays(monkeypatch) -> None:
    session, world, traces, views = _session(monkeypatch, [1 * MIB])
    assert session.enforce_free_dram() == [4, 2] and session.buckets == (8,)
    assert world.chain.buckets == (8,) and world.chain.traces is world.full_traces
    assert session.enforce_free_dram() == []  # nothing narrow left to drop


def test_a_missing_four_lane_width_is_skipped(monkeypatch) -> None:
    session, world, traces, views = _session(monkeypatch, [1 * MIB], widths=(2, 8))
    assert session.enforce_free_dram() == [2] and views == [2]


def test_warm_widths_replays_every_width_and_the_smallest_twice(monkeypatch) -> None:
    session, world, *_ = _session(monkeypatch, [200 * MIB])
    world.chain.active = _mask(0, 1)
    assert session.warm_widths() == [8, 4, 2]
    assert [record.width for record in world.chain.records] == [8, 4, 2, 2]
    launched = world.launched()
    world.calls.clear()
    # every trace of every set ran: commit of a width follows the pass at it
    session2, world2, *_ = _session(monkeypatch, [200 * MIB])
    world2.chain.active = _mask(0, 1)
    seen = set()
    original = world2.chain.enqueue
    world2.chain.enqueue = lambda tid: (seen.add(world2.names[tid]), original(tid))[1]
    session2.warm_widths()
    assert seen == {f"{kind}{width}" for kind in ("verify", "commit", "draft") for width in (2, 4, 8)}
    del launched


def test_warm_widths_stops_at_the_widest_lane_in_use(monkeypatch) -> None:
    session, world, *_ = _session(monkeypatch, [200 * MIB])
    world.chain.active = _mask(0, 2)
    assert session.warm_widths() == [8, 4]
    assert [record.width for record in world.chain.records] == [8, 4, 4]


def test_close_releases_the_narrow_widths_before_the_parent(monkeypatch) -> None:
    session, world, traces, views = _session(monkeypatch, [200 * MIB])
    order = []
    monkeypatch.setattr(session_module.ttnn, "release_trace", lambda mesh, tid: (order.append(tid), traces.append(tid)))
    session.traces = world.full_traces
    session.output = None
    session.draft = session.verify = None
    session.backbone_pager = session.alignment_pager = None
    session.close()
    narrow = [tid for tid in order if tid not in (801, 802, 803)]
    assert order[: len(narrow)] == narrow and order[-3:] == [801, 802, 803]
    assert sorted(views) == [2, 4] and session.closed and session.width_sets == {}


def test_the_free_dram_check_runs_after_the_captures_and_the_chain_gets_the_width_sets() -> None:
    capture = inspect.getsource(session_module.Qwen38LanesSession.capture)
    assert "width_sets=self.width_sets" in capture
    assert capture.index("self.lane_chain = ") < capture.index("self.enforce_free_dram()")
    prepare = inspect.getsource(session_module.Qwen38LanesSession.prepare)
    for name in ("alignment.residual", "ple_rows.embedding_rows", "accepted_lanes", "commit_lanes", "pass_row"):
        assert name in prepare  # the per-width host-written / device-read tensors are acknowledged


# ---------------------------------------------------------------------------- lowest free lane


def _perm(model) -> bool:
    return sorted(model.lane_of_slot) == list(range(len(model.lane_of_slot)))


def test_admissions_take_the_lowest_free_lanes_regardless_of_the_slot(monkeypatch) -> None:
    model, device, chain = _hybrid_model(monkeypatch, 8)
    model.prefill_forward(
        tokens=torch.tensor([_prompt(4), _prompt(4, 300), _prompt(4, 500)], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[4, 4, 4],
        start_pos=[0, 0, 0],
        empty_slots=[5, 3, 6],
    )
    # in the order of the slots: each takes the lowest free lane
    assert model.lane_driver.holders == [0, 1, 2]
    assert [model.lane_of_slot[slot] for slot in (5, 3, 6)] == [0, 1, 2]
    assert _perm(model)


def test_a_released_lane_is_taken_again_and_the_map_stays_a_permutation(monkeypatch) -> None:
    model, device, chain = _hybrid_model(monkeypatch, 8)
    for slot in (4, 7, 2):
        model.prefill_forward(
            tokens=torch.tensor([_prompt(4, 100 * slot)] * 2, dtype=torch.int32)[:1],
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[4],
            start_pos=[0],
            empty_slots=[slot],
        )
        assert _perm(model)
    # the single hybrid starts on the chain for the first lone prompt; settle all three on lanes
    assert model.lane_driver.holders in ([0, 1], [0, 1, 2], [1, 2], [0, 1])
    lanes_before = sorted(model.lane_driver.holders)
    model.release_request(model.lane_of_slot.index(1))
    assert 1 not in model.lane_driver.holders
    taken = model._take_lane(6)
    assert taken == min(lane for lane in range(8) if model.lane_driver.runs[lane] is None)
    assert _perm(model) and model.lane_of_slot[6] == taken and lanes_before


def test_take_lane_swaps_entries_so_the_map_is_a_permutation(monkeypatch, expect_error) -> None:
    model, device, chain = _hybrid_model(monkeypatch, 8)
    model.lane_of_slot = [3, 2, 1, 0, 7, 6, 5, 4]
    runs = model.lane_driver.runs
    runs[0] = runs[1] = object()  # lanes 0 and 1 hold requests
    assert model._take_lane(7) == 2  # lowest free lane
    assert model.lane_of_slot[7] == 2 and _perm(model)
    assert model.lane_of_slot[1] == 4  # slot 1 held lane 2: it takes the lane slot 7 gave up
    runs[2] = object()
    assert model._take_lane(0) == 3 and model.lane_of_slot[0] == 3 and _perm(model)
    assert model._take_lane(0) == 3 and model.lane_of_slot[0] == 3  # already there: no swap
    for lane in range(8):
        runs[lane] = object()
    with expect_error(adapter_module.Qwen38ChatChainError, "no free lane"):
        model._take_lane(5)


def test_the_hybrid_move_takes_the_lowest_free_lane(monkeypatch) -> None:
    model, device, chain = _hybrid_model(monkeypatch, 3)
    prompt = _prompt(6)
    committed, drafts, accepted, position = _run_hybrid_steps(model, 3, 2, prompt, 3)  # the lone request sits in slot 2
    assert model.mtp_hybrid_slot == 2 and model.lane_of_slot == [0, 1, 2]
    _hybrid_prefill(model, _prompt(5, 900), 1)  # a second request: the chain request moves to a lane first
    assert model.mtp_hybrid_slot is None
    assert device.imports[0][0] == 0, "the hybrid request must take lane 0, not its slot's lane 2"
    assert model.lane_of_slot[2] == 0 and _perm(model)
    assert model.lane_of_slot[1] == 1  # the new request takes the next lowest lane
    assert model.lane_driver.holders == [0, 1]


def test_admission_with_a_stale_lane_releases_it_then_takes_the_lowest(monkeypatch) -> None:
    model, device, chain = _hybrid_model(monkeypatch, 3, flag="0")
    _hybrid_prefill(model, _prompt(5), 2)
    assert model.lane_driver.holders == [0] and model.lane_of_slot[2] == 0 and _perm(model)
    _hybrid_prefill(model, _prompt(5, 900), 2)  # the plugin re-prefills slot 2 whose lane still holds a request
    assert model.lane_driver.holders == [0] and model.lane_of_slot[2] == 0 and _perm(model)


def test_the_driver_tells_the_session_which_lanes_hold_a_request(monkeypatch) -> None:
    source = inspect.getsource(adapter_module.Qwen38VLLMMTPLanesDriver.verify)
    assert 'getattr(self.device, "set_held", None)' in source
    assert source.index("set_held(") < source.index("self.device.set_active(") < source.index("self.device.step()")


def test_no_compaction_in_this_phase() -> None:
    for module in (adapter_module, session_module):
        source = inspect.getsource(module)
        assert "move_lane" not in source and "compaction_moves" not in source


def test_ple_rows_are_written_for_the_width_of_the_pass_only(monkeypatch) -> None:
    world = _World(monkeypatch)
    for active, expected in ((_mask(0, 1), 2), (_mask(0, 1, 2), 4), (_mask(0, 5), None), (_mask(0), 2)):
        world.step(active)
        assert world.ple_views[-1] == expected, (active, world.ple_views)


def test_flag_off_ple_rows_call_is_the_old_three_argument_call(monkeypatch) -> None:
    world = _World(monkeypatch, widths=(8,))
    world.step(_mask(0))
    assert world.ple_views == [None]
    source = inspect.getsource(mtp_lanes.Qwen38TTNNMTPLaneChain.step)
    assert "write_lane_ple_rows(self.model, self.verify, tokens))" in source


class _Ple:
    def __init__(self):
        self.calls = []

    def host_rows_lanes(self, tokens_by_lane, contexts):
        self.calls.append((len(tokens_by_lane), len(contexts)))
        lanes = len(tokens_by_lane)
        return torch.zeros(1, 1, lanes * ROWS, 4), tuple(
            ((lane, 0), (lane, 1), (lane, 2), (lane, 3), (lane, 4)) for lane in range(lanes)
        )


def _ple_world(monkeypatch):
    ple = _Ple()
    owner = object()
    target_parent, target_view = object(), object()
    state = SimpleNamespace(token_contexts=tuple([None] * LANES))
    layers = [SimpleNamespace(ple=None)] * mtp_lanes.PLE_CHECKPOINT_LAYER + [SimpleNamespace(ple=ple)]
    model = SimpleNamespace(layers=layers, mesh_device="mesh", _state_owner=owner)
    parent_layers = [SimpleNamespace(ple=None)] * mtp_lanes.PLE_CHECKPOINT_LAYER + [SimpleNamespace(ple=state)]
    parent = SimpleNamespace(
        lanes=LANES,
        rows=ROWS,
        is_view=False,
        parent=None,
        layers=parent_layers,
        pass_contexts=[("old",) * (ROWS + 1) for _ in range(LANES)],
        ple_rows=SimpleNamespace(embedding_rows=target_parent),
    )
    view = SimpleNamespace(lanes=2, is_view=True, parent=parent, ple_rows=SimpleNamespace(embedding_rows=target_view))
    written = []
    monkeypatch.setattr(mtp_lanes, "_validate_lane_verify_state", lambda model_, verify_: None)
    monkeypatch.setattr(mtp_lanes, "_lane_tokens", lambda verify_, tokens: [list(t) for t in tokens])
    monkeypatch.setattr(mtp_lanes.ttnn, "from_torch", lambda host, **kw: host)
    monkeypatch.setattr(
        mtp_lanes.ttnn, "copy_host_to_device_tensor", lambda host, target: written.append((tuple(host.shape), target))
    )
    monkeypatch.setattr(mtp_lanes.ttnn, "ShardTensor2dMesh", lambda *a, **k: None)
    return model, parent, view, ple, written, target_parent, target_view


def test_write_lane_ple_rows_looks_up_and_uploads_only_the_views_lanes(monkeypatch) -> None:
    model, parent, view, ple, written, target_parent, target_view = _ple_world(monkeypatch)
    tokens = [[lane] * ROWS for lane in range(LANES)]
    mtp_lanes.write_lane_ple_rows(model, parent, tokens, view=view)
    assert ple.calls == [(2, 2)] and written == [((1, 1, 2 * ROWS, 4), target_view)]
    assert [c[0][0] for c in parent.pass_contexts[:2]] == [0, 1]  # lanes < W took their new contexts
    assert all(c == ("old",) * (ROWS + 1) for c in parent.pass_contexts[2:])  # lanes >= W untouched


def test_write_lane_ple_rows_without_a_view_is_the_eight_lane_call(monkeypatch) -> None:
    model, parent, view, ple, written, target_parent, target_view = _ple_world(monkeypatch)
    mtp_lanes.write_lane_ple_rows(model, parent, [[1] * ROWS] * LANES)
    assert ple.calls == [(LANES, LANES)] and written == [((1, 1, LANES * ROWS, 4), target_parent)]
    assert all(c[0][0] == lane for lane, c in enumerate(parent.pass_contexts))


def test_write_lane_ple_rows_refuses_a_foreign_view(monkeypatch, expect_error) -> None:
    model, parent, view, ple, written, *_ = _ple_world(monkeypatch)
    view.parent = object()
    with expect_error(ValueError, "width view of this verify state"):
        mtp_lanes.write_lane_ple_rows(model, parent, [[1] * ROWS] * LANES, view=view)
    assert ple.calls == [] and written == []
