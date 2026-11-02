# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The chunked prefill driver's sequence without a device: alignment steps, the seed, full chunks, the padded tail,
the event cadence, the hand-off and the n-gram context threading, recorded through fakes of the model and ttnn."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tests.helpers.prefill import prefill_context_after
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_prefill_driver import (
    CHUNK_PAD_TOKEN_ID,
    Qwen38ChunkPrefill,
    alignment_steps,
    chunk_accepts,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS

ALLOCATED_CONTEXT = 500  # not a tile multiple, so the padded tail's end can overrun it in the budget test
TRACE_ID = 77
SHORT = SimpleNamespace(rows=CHUNK_ROWS)  # the 32-row chunk state


@pytest.fixture
def harness(prefill_fakes):
    prefill_fakes.model.allocated_context = ALLOCATED_CONTEXT
    prefill_fakes.prefill = Qwen38ChunkPrefill(
        prefill_fakes.model, "mesh", "state", SHORT, TRACE_ID, forced_step=prefill_fakes.forced_step
    )
    return prefill_fakes


@pytest.mark.parametrize("start", (0, 5, 32, 60, 95))
@pytest.mark.parametrize("count", (0, 1, 3, 31, 32, 33, 64, 96, 100, 137, 160))
def test_run_sequences_alignment_chunks_tail_and_handoff(harness, start: int, count: int) -> None:
    tokens = [1000 + index for index in range(count)]
    aligned = alignment_steps(start, count)
    assert aligned == min(count, (32 - start % 32) % 32) and (aligned == count or (start + aligned) % 32 == 0)
    remaining = tokens[aligned:]
    accepts = chunk_accepts(len(remaining))
    assert accepts == [31] * (len(remaining) // 32) + ([len(remaining) % 32 - 1] if len(remaining) % 32 else [])

    result = harness.prefill.run(tokens, start_position=start, ple_context=None)

    assert result.position == start + count
    assert result.ple_context == prefill_context_after(tokens, None)
    timing = result.timing
    assert (timing.alignment_steps, timing.chunks, timing.tail_rows) == (aligned, len(accepts), len(remaining) % 32)
    assert timing.chunk_replay_ms == () and timing.chunk_host_ms == () and timing.wall_ms >= 0.0
    assert timing.slab_prepare_ms == () and timing.slab_upload_ms == () and timing.slab_wait_ms == ()  # no slabs

    log = harness.log
    forced = [entry for entry in log if entry[0] == "forced_step"]
    assert forced == [
        ("forced_step", token, prefill_context_after(tokens[:i], None)) for i, token in enumerate(tokens[:aligned])
    ]
    if not accepts:
        assert log[len(forced) :] == [] and timing.verify_ms is None and timing.handoff_ms == 0.0
        return

    context = prefill_context_after(tokens[:aligned], None)
    chunks = [remaining[offset : offset + 32] for offset in range(0, len(remaining), 32)]
    padded = [tuple(chunk + [CHUNK_PAD_TOKEN_ID] * (32 - len(chunk))) for chunk in chunks]
    # The host half of chunk i + 1 is prepared right after chunk i's replay is queued (under it), before the event;
    # only the first chunk's preparation precedes its own upload.
    expected = [("synchronize",), ("reset", "state", 32), ("verify", TRACE_ID), ("prepare", 32, padded[0], context)]
    for index, (chunk, accepted) in enumerate(zip(chunks, accepts, strict=True)):
        if accepted != 31:  # only before the padded tail; finish_prefill restores the full-chunk value
            expected.append(("accepted", CHUNK_ROWS, accepted))
        context = prefill_context_after(chunk, context)  # the pad rows never enter the committed context
        expected += [("upload", 32, padded[index]), ("replay", TRACE_ID, False)]
        if index + 1 < len(chunks):
            expected.append(("prepare", 32, padded[index + 1], context))
        if (index + 1) % 4 == 0:
            expected += [("record", index // 4), ("wait", index // 4)]
    expected += [("synchronize",), ("finish", CHUNK_ROWS, start + count), ("synchronize",)]
    assert log[len(forced) :] == expected
    assert timing.verify_ms is not None and timing.handoff_ms >= 0.0
    assert timing.traced is True and timing.gdn_step_anchor is False


@pytest.mark.parametrize("anchor", (False, True))
def test_eager_chunks_pass_the_gdn_step_anchor_per_chunk_and_skip_the_trace_verification(
    expect_error, harness, anchor: bool
) -> None:
    """Without a captured trace the driver runs the chunk body itself: one ``forward_prefill_chunk_generic`` per chunk
    with the re-anchor flag, no ``verify_before_replay``, the blocking form synchronizing after every chunk."""

    model, forced_step = harness.model, harness.forced_step
    prefill = Qwen38ChunkPrefill(model, "mesh", "state", SHORT, None, forced_step=forced_step, gdn_step_anchor=anchor)
    result = prefill.run(list(range(1, 101)), start_position=32, ple_context=None)
    assert result.position == 132 and result.timing.chunks == 4 and result.timing.tail_rows == 4
    assert result.timing.traced is False and result.timing.gdn_step_anchor is anchor and result.timing.verify_ms is None
    chunks = [entry for entry in harness.log if entry[0] in ("eager", "replay", "verify")]
    assert chunks == [("eager", 32, anchor, None)] * 4
    synchronizes = [entry for entry in harness.log if entry[0] == "synchronize"]
    assert len(synchronizes) == 3  # the seed and the two around the hand-off; the non-blocking chunks add none
    # The blocking (timed) form synchronizes after every eager chunk so the wall is the chunk's.
    harness.log.clear()
    timed = prefill.run(list(range(64)), start_position=0, ple_context=None, time_each_chunk=True)
    assert len(timed.timing.chunk_replay_ms) == 2 and timed.timing.traced is False
    assert [entry[0] for entry in harness.log if entry[0] in ("eager", "synchronize")] == [
        "synchronize",
        "eager",
        "synchronize",
        "eager",
        "synchronize",
        "synchronize",
        "synchronize",
    ]
    # The traced form carries the flag its capture had (the chain passes its own); a replay never takes it per chunk.
    harness.log.clear()
    traced = Qwen38ChunkPrefill(model, "mesh", "state", SHORT, TRACE_ID, forced_step=forced_step, gdn_step_anchor=True)
    result = traced.run(list(range(32)), start_position=0, ple_context=None)
    assert result.timing.traced is True and result.timing.gdn_step_anchor is True
    assert [entry for entry in harness.log if entry[0] in ("eager", "replay")] == [("replay", TRACE_ID, False)]
    for bad in ("1", 1, None):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            Qwen38ChunkPrefill(model, "mesh", "s", "c", TRACE_ID, forced_step=forced_step, gdn_step_anchor=bad)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        Qwen38ChunkPrefill(model, "mesh", "s", "c", "77", forced_step=forced_step)


def test_timed_replays_block_and_skip_the_events(harness) -> None:
    result = harness.prefill.run(list(range(1, 161)), start_position=0, ple_context=None, time_each_chunk=True)
    assert result.timing.chunks == 5 and len(result.timing.chunk_replay_ms) == 5
    assert len(result.timing.chunk_host_ms) == 5 and all(ms >= 0.0 for ms in result.timing.chunk_host_ms)
    assert all(ms >= 0.0 for ms in result.timing.chunk_replay_ms)
    replays = [entry for entry in harness.log if entry[0] == "replay"]
    assert replays == [("replay", TRACE_ID, True)] * 5
    assert not any(entry[0] in ("record", "wait") for entry in harness.log)


def test_verify_can_be_skipped_and_the_budget_is_checked(expect_error, harness) -> None:
    harness.prefill.verify_allocations = False
    result = harness.prefill.run(list(range(40)), start_position=0, ple_context=(3, 4))
    assert result.timing.verify_ms is None
    assert not any(entry[0] == "verify" for entry in harness.log)
    assert result.ple_context == prefill_context_after(list(range(40)), (3, 4))
    with expect_error(ValueError):  # allow-pytest.raises: the prompt does not fit the allocated context
        harness.prefill.run(list(range(ALLOCATED_CONTEXT + 1)), start_position=0, ple_context=None)
    with expect_error(ValueError):  # allow-pytest.raises: the padded tail chunk would end past the allocated context
        harness.prefill.run(list(range(ALLOCATED_CONTEXT - 3)), start_position=0, ple_context=None)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        harness.prefill.run([1], start_position=-1, ple_context=None)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        Qwen38ChunkPrefill(harness.model, "mesh", "s", "c", TRACE_ID, forced_step=lambda t, c: c, event_interval=0)


@pytest.mark.parametrize("count", (100, 128, 300, 479))
def test_event_rows_cadence_syncs_per_rows_and_calls_between_chunks(expect_error, harness, count: int) -> None:
    """The lanes admission's form: an event once the chunk rows since the last one reach ``event_rows`` (128 here:
    every four 32-row chunks), ``between_chunks`` right after it with the rows consumed so far; the padded tail
    ends without one (the hand-off synchronizes)."""

    yields: list[tuple[int, int]] = []
    prefill = Qwen38ChunkPrefill(
        harness.model, "mesh", "state", SHORT, TRACE_ID, forced_step=harness.forced_step, event_rows=128
    )
    tokens = list(range(1, count + 1))
    result = prefill.run(tokens, start_position=0, ple_context=None, between_chunks=lambda d, t: yields.append((d, t)))
    assert result.position == count and result.ple_context == prefill_context_after(tokens, None)
    full = count // 32
    expected_syncs = full // 4
    events = [entry for entry in harness.log if entry[0] == "wait"]
    assert len(events) == expected_syncs
    assert yields == [(128 * (k + 1), count) for k in range(expected_syncs)]
    # every yield follows its event and precedes the next chunk's upload; the next chunk's inputs were prepared before
    log = harness.log
    for done, _total in yields:
        sync = [i for i, e in enumerate(log) if e[0] == "wait"][yields.index((done, count))]
        before = [e[0] for e in log[:sync]]
        assert before.count("prepare") == before.count("upload") + (1 if done < full * 32 or count % 32 else 0)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test (a positive multiple of 32 or None)
        Qwen38ChunkPrefill(harness.model, "mesh", "s", "c", TRACE_ID, forced_step=lambda t, c: c, event_rows=100)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        Qwen38ChunkPrefill(harness.model, "mesh", "s", "c", TRACE_ID, forced_step=lambda t, c: c, event_rows=0)


def test_between_chunks_is_not_called_when_stopped_or_without_events(harness) -> None:
    """A stop at the event ends the prefill before the yield point; the chunk cadence without ``event_rows`` is the
    single stream's (every four chunks) and the yield point follows each of its events."""

    yields: list[tuple[int, int]] = []
    result = harness.prefill.run(
        list(range(300)),
        start_position=0,
        ple_context=None,
        should_stop=lambda: "deadline",
        between_chunks=lambda d, t: yields.append((d, t)),
    )
    assert result.stopped == "deadline" and result.position == 128 and yields == []
    harness.log.clear()
    result = harness.prefill.run(
        list(range(300)), start_position=0, ple_context=None, between_chunks=lambda d, t: yields.append((d, t))
    )
    assert result.stopped is None and yields == [(128, 300), (256, 300)]


def test_driver_source_pins() -> None:
    import inspect

    source = inspect.getsource(driver_module)
    assert source.count("ttnn._ttnn_execute_trace(") == 1 and "ttnn.execute_trace(" not in source
    assert source.count("TraceAllocationTracker.verify_before_replay(self.mesh, trace_id)") == 1
    run = inspect.getsource(Qwen38ChunkPrefill.run)
    order = (
        "self.forced_step(token, ple_context)",
        "ttnn.synchronize_device(self.mesh)",
        "self.model.reset_chunk_state_inplace(self.state, self.chunk_state)",
        "verify_before_replay",
        "pending = self._prepare(plan, starts, 0, ple_context, position, positions, features, feature_cursor)",
        "self.model.write_chunk_accepted(self.chunk_state, accepted)",
        "prepared, prepare_ns, feature_cursor = pending",
        "ple_context = prepared.contexts[real_rows]",
        "self.model.upload_chunk_inputs(chunk_state, prepared)",
        "self._run_chunk(blocking=True, kind=kind)",
        "self._run_chunk(blocking=False, kind=kind)",
        "pending = self._prepare(\n                    plan, starts, index + 1, ple_context, position, positions, features, feature_cursor\n                )",
        "ttnn.event_synchronize(ttnn.record_event(self.mesh, cq_id=0))",
        "between_chunks(row_offset, len(remaining))",
        "self.model.finish_prefill(self.state, self.chunk_state, position, rope_shift=rope_shift)",
    )
    positions = [run.index(fragment) for fragment in order]
    assert positions == sorted(positions)
    # The next chunk's host half is prepared under the running replay, before any event wait or yield point.
    assert run.index(
        "pending = self._prepare(\n                    plan, starts, index + 1, ple_context, position, positions, features, feature_cursor\n                )"
    ) < run.index("ttnn.event_synchronize(slab_event)")
    prepare = inspect.getsource(Qwen38ChunkPrefill._prepare)
    assert (
        "prepared = self.model.prepare_chunk_inputs(" in prepare
        and "positions=chunk_positions, features=chunk_features" in prepare
    )
    assert "chunk_features, feature_cursor = vision_splice.split_features(rows, features, feature_cursor)" in prepare
    assert "rows = rows + [self.pad_token_id] * (CHUNK_ROWS - len(rows))" in prepare
    # The host half (the lookup and the packing) is timed apart from the copies' enqueue and the slab's event wait.
    assert run.index("slab_prepare_ms.append(prepare_ns / 1_000_000)") < run.index(
        "slab_upload_ms.append((replay_started_ns - upload_started_ns)"
    )
    assert "slab_wait_ms.append((time.perf_counter_ns() - wait_started_ns)" in run
    # A slab waits for the previous slab's event once its own replay is queued (the host runs one slab ahead).
    assert run.index("self._run_chunk(blocking=False, kind=kind)") < run.index("ttnn.event_synchronize(slab_event)")
    assert run.index("ttnn.event_synchronize(slab_event)") < run.index(
        "slab_event = ttnn.record_event(self.mesh, cq_id=0)"
    )
    assert run.count("ttnn.record_event(") == 2 and run.count("ttnn.event_synchronize(") == 2
    chunk = inspect.getsource(Qwen38ChunkPrefill._run_chunk)
    assert "ttnn._ttnn_execute_trace(self.mesh, trace_id, cq_id=0, blocking=blocking)" in chunk
    assert (
        "self.model.forward_prefill_chunk_generic(\n"
        "            chunk_state, self.state, gdn_step_anchor=self.gdn_step_anchor and short, mtp=self._extension(kind)\n"
        "        )"
    ) in chunk
    extension = inspect.getsource(Qwen38ChunkPrefill._extension)
    assert 'return {"short": self.mtp, "long": self.long_mtp, "slab": self.long_mtp}[kind]' in extension
    assert 'short = kind == "short"' in chunk
    assert CHUNK_PAD_TOKEN_ID == 0 and driver_module.CHUNK_EVENT_INTERVAL == 4


# -- image prompts -------------------------------------------------------------------------------------------------


def _image_prompt(committed: int = 0):
    """A 48-token prefill after ``committed`` text tokens the device already holds: 10 text tokens,
    <|vision_start|>, a 16-pad image (grid 8x8: span 4, shift 12), <|vision_end|>, 20 text tokens.  The positions
    cover the whole sequence by device index (the committed prefix included), as the session builds them."""

    import torch

    from models.demos.blackhole.qwen38_flash_next import mrope

    grid = mrope.Qwen38ImageGrid(1, 8, 8)
    tokens = (
        [1000 + i for i in range(10)]
        + [mrope.VISION_START_TOKEN_ID]
        + [mrope.IMAGE_TOKEN_ID] * grid.merged_tokens
        + [mrope.VISION_END_TOKEN_ID]
        + [2000 + i for i in range(20)]
    )
    positions = mrope.mrope_positions([3000 + i for i in range(committed)] + tokens, [grid])
    features = torch.zeros((grid.merged_tokens, 2560), dtype=torch.bfloat16)
    features[:, 0] = torch.arange(grid.merged_tokens, dtype=torch.float32).to(torch.bfloat16)
    return tokens, positions, features


def test_image_prompt_rows_take_their_positions_and_features_and_the_handoff_sets_the_shift(harness) -> None:
    import torch

    tokens, positions, features = _image_prompt()
    result = harness.prefill.run(tokens, start_position=0, ple_context=None, positions=positions, features=features)
    assert result.position == len(tokens) == 48
    prepared = [entry for entry in harness.model.vision if entry[0] != "finish"]
    assert len(prepared) == 2  # a full chunk and the 16-row padded tail
    first_positions, first_features = prepared[0]
    assert torch.equal(first_positions, positions.rows(0, 32))
    assert first_features is not None and torch.equal(first_features, features)  # the 16 pads sit at rows 11..26
    tail_positions, tail_features = prepared[1]
    assert tail_features is None and tuple(tail_positions.shape) == (3, 32)
    assert torch.equal(tail_positions[:, :16], positions.rows(32, 48))
    # the padded rows continue the plain positions past the last real row (never read past the accept count)
    last = int(positions.rows(32, 48)[:, -1].max())
    assert tail_positions[:, 16:].tolist() == [[last + 1 + i for i in range(16)]] * 3
    assert ("finish", positions.shift) in harness.model.vision and positions.shift == 12
    assert positions.shift <= (48 & ~3)


def test_image_pads_never_take_the_alignment_steps(expect_error, harness) -> None:
    """From position 5 the driver would force 27 tokens through the 1-row body: the pads (the prefill's tokens
    11..26) are among them, so the prefill is refused before any device call (the session prefills such a prompt
    from position 0)."""

    import torch

    tokens, positions, features = _image_prompt(committed=5)
    with expect_error(ValueError, match="image pads in the 27 alignment steps"):
        harness.prefill.run(tokens, start_position=5, ple_context=None, positions=positions, features=features)
    assert harness.log == []
    # text before the pads may take the alignment steps as long as no pad does: from position 22, 10 steps
    # cover the prefill's tokens 0..9 (text); the pads start at its token 11.
    tokens, positions, features = _image_prompt(committed=22)
    result = harness.prefill.run(tokens, start_position=22, ple_context=None, positions=positions, features=features)
    assert result.position == 22 + 48 and result.timing.alignment_steps == 10
    forced = [entry for entry in harness.log if entry[0] == "forced_step"]
    assert [entry[1] for entry in forced] == tokens[:10]
    first_positions, first_features = [entry for entry in harness.model.vision if entry[0] != "finish"][0]
    assert torch.equal(first_positions, positions.rows(32, 64)) and first_features is not None
    assert ("finish", positions.shift_at(70)) in harness.model.vision


def test_image_prompt_refusals(expect_error, harness) -> None:
    tokens, positions, features = _image_prompt()
    with expect_error(ValueError, match="16 image pads in the prefill vs 15 feature rows"):
        harness.prefill.run(tokens, start_position=0, ple_context=None, positions=positions, features=features[:15])
    with expect_error(ValueError, match="rotary positions"):
        harness.prefill.run(tokens, start_position=0, ple_context=None, features=features)
    with expect_error(ValueError, match="rotary positions for a prefill ending at"):
        harness.prefill.run(tokens + [5], start_position=0, ple_context=None, positions=positions, features=features)
    assert harness.log == []
