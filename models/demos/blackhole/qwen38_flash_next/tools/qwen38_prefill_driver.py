# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Chunked prefill of one prompt on the generic chain: alignment steps, traced chunks (the slab rows, then 128, then
32), the padded tail, the hand-off.

The chain owner (the chat session, a runner) hands over the model, its generic and chunk states, the captured chunk
traces and ``forced_step``, which teacher-forces one token through the decode traces at the current position and
returns the next n-gram context.  :meth:`Qwen38ChunkPrefill.run` then prefills ``token_ids`` from ``start_position``:
``(32 - P % 32) % 32`` forced steps so the first chunk starts at ``P % 32 == 0``; the chunk carry seeded from the
decode buffers (``reset_chunk_state_inplace``, the 32-row state first: the 128-row and slab states share its
histories); ``N // S`` slabs of ``S`` rows when the chain captured a slab trace (``--prefill-slab``), then
``M // 128`` long chunks of the remainder when the chain captured the 128-row trace; then ``M // 32`` full 32-row
chunks (accept scalar 31) over the remainder ``M``; if ``M % 32 = r > 0`` one padded chunk whose rows ``r .. 31`` are
:data:`CHUNK_PAD_TOKEN_ID` with accept scalar ``r - 1``; ``finish_prefill`` from the 32-row state.  The host work of
chunk i + 1 (``prepare_chunk_inputs``: the n-gram lookups from the running context and the row packing; then
``upload_chunk_inputs``: the token-row and PLE-row copies) runs under chunk i's replay: the host half is prepared as
soon as chunk i is queued and the copies are queued behind it, so only the first chunk's preparation is exposed
whatever the event cadence.  An event every ``event_interval`` chunks (or, with ``event_rows``, once the chunk rows
since the last event reach that count: the lanes' admission form, one event per 128 prompt rows) bounds the run-ahead
(a slab: the previous slab's event, waited for once the next replay is queued); ``run``'s ``between_chunks`` hook is
called after every such event, with the prompt rows consumed so far, so a caller may run other device work between
two chunk groups (the lanes server runs the decoding lanes' passes there); the eager seed and hand-off run only after
a device synchronize (their transients must not land in a running trace's addresses).

Timing (the rule every measurement here follows): ``verify_before_replay`` once per trace, outside the window; the raw
``ttnn._ttnn_execute_trace`` per chunk is timed only with ``time_each_chunk`` (blocking replays, no host overlap); the
end-to-end wall (forced steps, host writes, replays, hand-off) is reported separately.

``gdn_step_anchor`` (default off) is the GDN state re-anchor of every 32-row chunk (``forward_prefill_chunk_generic``'s
keyword: the committed GDN state through the 1-row FP32 step arithmetic instead of the chunk kernel's).  With a
captured trace the driver replays what the capture baked in, so the chain owner passes the flag its capture carried;
without one (``chunk_trace_id=None``, the eager form of the micro-tests) the driver runs the chunk body itself and
passes the flag per chunk.  The 128-row form has no anchor: a chain with the anchor on runs 32-row chunks only.

``mtp`` / ``long_mtp`` are the MTP-drafting chain's chunk extensions (``mtp_v2.Qwen38TTNNMTPChunkExtension``, the
32-row and the 128-row twin): the chunk traces were captured with them, so every chunk also takes its MTP tokens (the
chunk's rows one position ahead, ``following_token`` after the last) through the extension of its form, and the
hand-off includes the MTP layer (the 32-row twin's ``finish_chunk``).  A chain with the 128-row chunk state and MTP
drafting needs both twins; a slab runs the 128-row twin's slab form (``forward_slab_rows``: the slab as 128-row slices
inside the slab body), so a slab chain with drafting needs the twin allocated with ``slab_rows``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

import ttnn
from models.demos.blackhole.qwen38_flash_next import mrope, vision_splice
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS, LONG_CHUNK_ROWS, is_slab_rows

# Rows past the prompt in the padded tail chunk: any in-vocabulary id; their state is never read after the hand-off.
CHUNK_PAD_TOKEN_ID = 0
CHUNK_EVENT_INTERVAL = 4


def alignment_steps(start_position: int, count: int) -> int:
    """Tokens teacher-forced through the decode traces before the first chunk: up to the next multiple of 32."""

    return min(count, (CHUNK_ROWS - start_position % CHUNK_ROWS) % CHUNK_ROWS)


def long_chunk_count(count: int, *, long_chunks: bool) -> int:
    """The 128-row chunks of a ``count``-token chunked prefill: ``count // 128`` when the chain has the long trace."""

    return count // LONG_CHUNK_ROWS if long_chunks else 0


def slab_count(count: int, *, slab_rows: int | None) -> int:
    """The slabs of a ``count``-token chunked prefill: ``count // slab_rows`` when the chain has a slab trace."""

    return count // slab_rows if slab_rows else 0


def chunk_accepts(count: int) -> list[int]:
    """The accept scalar of every 32-row chunk of a ``count``-token chunked prefill: 31 per full chunk, r - 1 for the tail."""

    full, tail = divmod(count, CHUNK_ROWS)
    return [CHUNK_ROWS - 1] * full + ([tail - 1] if tail else [])


@dataclass(frozen=True)
class Qwen38PrefillTiming:
    alignment_steps: int
    chunks: int  # the 32-row chunks (full and the padded tail)
    tail_rows: int
    chunk_replay_ms: tuple[float, ...]  # raw blocking replays, one per 32-row chunk; empty unless time_each_chunk
    chunk_host_ms: tuple[float, ...]  # the host writes ahead of each 32-row chunk (lookups, rows); empty unless timed
    verify_ms: float | None
    handoff_ms: float
    wall_ms: float
    traced: bool = True  # chunks replayed the captured traces (False: the eager chunk body ran per chunk)
    gdn_step_anchor: bool = False  # the GDN state re-anchor the 32-row chunks ran with
    long_chunks: int = 0  # the 128-row chunks ahead of the 32-row ones
    long_chunk_replay_ms: tuple[float, ...] = ()  # raw blocking replays, one per 128-row chunk; empty unless timed
    long_chunk_host_ms: tuple[float, ...] = ()
    slabs: int = 0  # the slabs ahead of the 128-row chunks
    slab_rows: int = 0
    slab_replay_ms: tuple[float, ...] = ()  # raw blocking replays, one per slab; empty unless timed
    slab_host_ms: tuple[float, ...] = ()
    # The split of every slab's host work (every run): the input preparation (the n-gram lookup and the row
    # packing), the input copies' enqueue, and the wait at the slab's event sync (non-blocking runs).
    slab_prepare_ms: tuple[float, ...] = ()
    slab_upload_ms: tuple[float, ...] = ()
    slab_wait_ms: tuple[float, ...] = ()


@dataclass(frozen=True)
class Qwen38PrefillResult:
    position: int  # P after the prefill: the count of positions consumed
    ple_context: tuple[int, int] | None  # the n-gram context of the committed stream
    timing: Qwen38PrefillTiming
    stopped: str | None = None  # the should_stop reason that ended the prefill after the chunks already replayed


class Qwen38ChunkPrefill:
    """One prompt's chunked prefill on a chain whose chunk traces are captured (see the module docstring)."""

    def __init__(
        self,
        model: Any,
        mesh: Any,
        state: Any,
        chunk_state: Any,
        chunk_trace_id: Any | None,
        *,
        forced_step: Callable[[int, tuple[int, int] | None], tuple[int, int] | None],
        pad_token_id: int = CHUNK_PAD_TOKEN_ID,
        event_interval: int = CHUNK_EVENT_INTERVAL,
        event_rows: int | None = None,
        verify_allocations: bool = True,
        gdn_step_anchor: bool = False,
        long_chunk_state: Any | None = None,
        long_chunk_trace_id: Any | None = None,
        mtp: Any = None,
        slab_state: Any | None = None,
        slab_trace_id: Any | None = None,
        long_mtp: Any = None,
    ) -> None:
        if isinstance(event_interval, bool) or type(event_interval) is not int or event_interval <= 0:
            raise ValueError(f"event interval must be a positive int, got {event_interval!r}")
        if event_rows is not None and (
            isinstance(event_rows, bool) or type(event_rows) is not int or event_rows <= 0 or event_rows % CHUNK_ROWS
        ):
            raise ValueError(f"event rows must be a positive multiple of {CHUNK_ROWS} or None, got {event_rows!r}")
        for name, trace_id in (("chunk", chunk_trace_id), ("long chunk", long_chunk_trace_id), ("slab", slab_trace_id)):
            if isinstance(trace_id, (bool, str, float)):  # the runtime's trace handle (MeshTraceId) or None
                raise ValueError(f"{name} trace id must be a trace handle or None (the eager body), got {trace_id!r}")
        if type(gdn_step_anchor) is not bool:
            raise ValueError(f"gdn_step_anchor must be a bool, got {gdn_step_anchor!r}")
        if long_chunk_state is not None and gdn_step_anchor:
            raise ValueError("the GDN step anchor is a 32-row chunk option: no long chunks with the anchor on")
        if (long_chunk_trace_id is not None) and (long_chunk_state is None or chunk_trace_id is None):
            raise ValueError("a long chunk trace needs the long chunk state and the 32-row chunk trace")
        if long_chunk_state is not None and mtp is not None and long_mtp is None:
            raise ValueError("a long chunk state with MTP drafting needs the 128-row MTP chunk extension (long_mtp)")
        if long_mtp is not None and (mtp is None or long_chunk_state is None):
            raise ValueError("the 128-row MTP chunk extension needs the 32-row extension and the long chunk state")
        if slab_state is not None and (long_chunk_state is None or not is_slab_rows(getattr(slab_state, "rows", None))):
            raise ValueError("a slab needs the 128-row chunk state (its remainder) and a slab row count")
        if slab_trace_id is not None and (slab_state is None or long_chunk_trace_id is None):
            raise ValueError("a slab trace needs the slab state and the long chunk trace")
        if slab_state is not None and mtp is not None:
            # The slab's MTP rows run through the 128-row twin's slab form (mtp_v2: forward_slab_rows), captured in
            # the slab trace; a slab with drafting but without that form is refused.
            if long_mtp is None or int(getattr(long_mtp, "slab_rows", 0)) != int(slab_state.rows):
                raise ValueError(
                    "a slab with MTP drafting needs the 128-row MTP chunk extension allocated with the slab form "
                    f"(slab_rows={slab_state.rows}), got {None if long_mtp is None else getattr(long_mtp, 'slab_rows', None)!r}"
                )
        self.model = model
        self.mesh = mesh
        self.state = state
        self.chunk_state = chunk_state
        self.chunk_trace_id = chunk_trace_id
        self.long_chunk_state = long_chunk_state
        self.long_chunk_trace_id = long_chunk_trace_id
        self.slab_state = slab_state
        self.slab_trace_id = slab_trace_id
        self.slab_rows = 0 if slab_state is None else int(slab_state.rows)
        self.forced_step = forced_step
        self.pad_token_id = pad_token_id
        self.event_interval = event_interval
        # The event cadence in chunk rows instead of chunks (None: every ``event_interval`` chunks, the chain owner's
        # form): the lanes' admission takes an event, and a ``between_chunks`` call, per 128 prompt rows.
        self.event_rows = event_rows
        self.verify_allocations = verify_allocations
        self.gdn_step_anchor = gdn_step_anchor
        # The MTP-drafting chain's chunk extensions (mtp_v2.Qwen38TTNNMTPChunkExtension): the 32-row chunk trace was
        # captured with ``mtp`` and the 128-row one with ``long_mtp``, so every chunk also takes its MTP tokens (the
        # tokens one position ahead) through the extension of its form, and the hand-off includes the MTP layer.
        self.mtp = mtp
        self.long_mtp = long_mtp

    def _run_chunk(self, *, blocking: bool, kind: str = "short") -> None:
        """One chunk at the device position: the captured trace's replay, or the eager chunk body with the
        re-anchor flag and the MTP extension of the chunk's form passed per chunk (a blocking eager chunk
        synchronizes so its wall is the chunk's).  ``kind`` is ``slab``, ``long`` (128 rows) or ``short`` (32 rows)."""

        chunk_state, trace_id = {
            "slab": (self.slab_state, self.slab_trace_id),
            "long": (self.long_chunk_state, self.long_chunk_trace_id),
            "short": (self.chunk_state, self.chunk_trace_id),
        }[kind]
        if trace_id is not None:
            ttnn._ttnn_execute_trace(self.mesh, trace_id, cq_id=0, blocking=blocking)
            return
        short = kind == "short"
        self.model.forward_prefill_chunk_generic(
            chunk_state, self.state, gdn_step_anchor=self.gdn_step_anchor and short, mtp=self._extension(kind)
        )
        if blocking:
            ttnn.synchronize_device(self.mesh)

    @staticmethod
    def _chunk_positions(
        positions: mrope.Qwen38MRoPEPositions | None, chunk_start: int, rows: Sequence[int]
    ) -> torch.Tensor:
        """The (t, h, w) rotary positions of a chunk's rows (int64 ``[3, rows]``): the prompt's from ``positions``
        for the real rows, the plain continuation for a padded tail (its rows are never read past the accept
        count); the plain positions ``chunk_start ..`` without ``positions``."""

        count = len(rows)
        if positions is None:
            plain = torch.arange(chunk_start, chunk_start + count, dtype=torch.int64)
            return plain.reshape(1, -1).expand(3, -1).contiguous()
        real = max(0, min(count, positions.length - chunk_start))
        axes = torch.empty((3, count), dtype=torch.int64)
        if real:
            axes[:, :real] = positions.rows(chunk_start, chunk_start + real)
        if real < count:
            after = int(axes[:, real - 1].max().item()) + 1 if real else chunk_start - positions.shift_at(chunk_start)
            axes[:, real:] = torch.arange(count - real, dtype=torch.int64) + after
        return axes

    def _extension(self, kind: str) -> Any:
        """The MTP chunk extension of a chunk kind: the 32-row one for ``short``, the 128-row twin for ``long`` and, in
        its slab form, for ``slab`` (none at all on a plain chain)."""

        return {"short": self.mtp, "long": self.long_mtp, "slab": self.long_mtp}[kind]

    def _prepare(
        self,
        plan: list,
        starts: Sequence[int],
        index: int,
        ple_context: tuple[int, int] | None,
        position: int,
        positions,
        features,
        feature_cursor: int,
    ):
        """The host half of chunk ``index`` of the plan (its rows padded when it is the tail; ``starts[index]`` its
        first row within the prefill, so its rotary positions start at ``position + starts[index]`` and its feature
        rows follow ``feature_cursor``): ``(inputs, ns, feature_cursor after it)``, or None past the last chunk.
        Host work only: it runs while the previous chunk replays."""

        if index >= len(plan):
            return None
        chunk_state, _kind, rows, accepted = plan[index]
        if accepted is not None and accepted != CHUNK_ROWS - 1:
            rows = rows + [self.pad_token_id] * (CHUNK_ROWS - len(rows))
        started_ns = time.perf_counter_ns()
        chunk_positions = self._chunk_positions(positions, position + starts[index], rows)
        chunk_features, feature_cursor = vision_splice.split_features(rows, features, feature_cursor)
        prepared = self.model.prepare_chunk_inputs(
            chunk_state, rows, ple_context=ple_context, positions=chunk_positions, features=chunk_features
        )
        return prepared, time.perf_counter_ns() - started_ns, feature_cursor

    def run(
        self,
        token_ids: Sequence[int],
        *,
        start_position: int,
        ple_context: tuple[int, int] | None,
        time_each_chunk: bool = False,
        following_token: int | None = None,
        should_stop: Callable[[], str | None] | None = None,
        between_chunks: Callable[[int, int], None] | None = None,
        positions: mrope.Qwen38MRoPEPositions | None = None,
        features: torch.Tensor | None = None,
    ) -> Qwen38PrefillResult:
        """Prefill ``token_ids`` at positions ``start_position ..``; the caller's first decode replay consumes the
        token after them (``following_token``, the MTP token of the last prefilled position when the chain drafts).
        Returns the position after the prefill and the committed stream's n-gram context.  ``should_stop`` is
        polled at every event sync: a reason ends the prefill after the chunks already replayed (the hand-off runs
        at that position; ``stopped`` carries the reason, ``position`` what was consumed).  ``between_chunks(done,
        total)`` runs after every chunk event sync that did not stop (the chunk rows consumed so far, the chunk rows
        of the whole prefill): the device is idle at that point and the next chunk's inputs are prepared but not yet
        uploaded, so the caller may run other device work there (never after a slab's event: that waits for the slab
        before it while the slab itself still replays).

        An image prompt passes ``positions`` (the whole sequence's (t, h, w) rotary positions, ``mrope``; the rows of
        ``start_position ..`` are this prefill's) and ``features`` (the tower's rows, one per ``<|image_pad|>`` of
        ``token_ids`` in order): every chunk writes its rows' RoPE rows and its pads' feature rows, and the hand-off
        sets the rotary shift the decode subtracts.  Image pads never take the alignment steps (the 1-row body has
        no feature input): a prefill whose alignment window holds one is refused, so the caller prefills from a
        32-aligned position (a reset).  Without ``positions`` the rows are plain text positions."""

        tokens = [int(token) for token in token_ids]
        if isinstance(start_position, bool) or type(start_position) is not int or start_position < 0:
            raise ValueError(f"start position must be a non-negative int, got {start_position!r}")
        if self.mtp is not None and following_token is None:
            raise ValueError("an MTP-drafting chain's chunked prefill needs the token following the prefilled ones")
        if start_position + len(tokens) > self.model.allocated_context:
            raise ValueError(
                f"prefill of {len(tokens)} tokens from position {start_position} exceeds the allocated context "
                f"{self.model.allocated_context}"
            )
        started_ns = time.perf_counter_ns()
        aligned = alignment_steps(start_position, len(tokens))
        image_pads = len(vision_splice.image_lanes(tokens))
        feature_count = 0 if features is None else int(features.shape[0])
        if image_pads != feature_count:
            raise ValueError(f"{image_pads} image pads in the prefill vs {feature_count} feature rows")
        if positions is None:
            if image_pads:
                raise ValueError("image pads need the prompt's rotary positions (mrope) and their feature rows")
        else:
            if positions.length < start_position + len(tokens):
                raise ValueError(
                    f"{positions.length} rotary positions for a prefill ending at {start_position + len(tokens)}"
                )
            if vision_splice.image_lanes(tokens[:aligned]):
                raise ValueError(
                    f"image pads in the {aligned} alignment steps from position {start_position}: an image prompt "
                    "prefills from a 32-aligned position"
                )
        feature_cursor = 0
        for token in tokens[:aligned]:
            ple_context = self.forced_step(token, ple_context)
        position = start_position + aligned
        remaining = tokens[aligned:]
        slabs = slab_count(len(remaining), slab_rows=self.slab_rows or None)
        after_slabs = len(remaining) - slabs * self.slab_rows
        long_chunks = long_chunk_count(after_slabs, long_chunks=self.long_chunk_state is not None)
        accepts = chunk_accepts(after_slabs - long_chunks * LONG_CHUNK_ROWS)
        # The chunk sequence: (chunk state, kind, rows, accept scalar or None): the slabs, then the long chunks,
        # then the 32-row chunks and the padded tail.
        plan: list[tuple[Any, str, list[int], int | None]] = [
            (self.slab_state, "slab", remaining[self.slab_rows * index : self.slab_rows * (index + 1)], None)
            for index in range(slabs)
        ]
        offset = slabs * self.slab_rows
        plan += [
            (
                self.long_chunk_state,
                "long",
                remaining[offset + LONG_CHUNK_ROWS * index : offset + LONG_CHUNK_ROWS * (index + 1)],
                None,
            )
            for index in range(long_chunks)
        ]
        offset += long_chunks * LONG_CHUNK_ROWS
        plan += [
            (
                self.chunk_state,
                "short",
                remaining[offset + CHUNK_ROWS * index : offset + CHUNK_ROWS * (index + 1)],
                accepted,
            )
            for index, accepted in enumerate(accepts)
        ]
        verify_ms = None
        handoff_ms = 0.0
        replay_ms: list[float] = []
        host_ms: list[float] = []
        long_replay_ms: list[float] = []
        long_host_ms: list[float] = []
        slab_replay_ms: list[float] = []
        slab_host_ms: list[float] = []
        slab_prepare_ms: list[float] = []
        slab_upload_ms: list[float] = []
        slab_wait_ms: list[float] = []
        stopped = None
        consumed = len(remaining)
        chunks = len(accepts)
        long_done = long_chunks
        slabs_done = slabs
        if plan:
            if position % CHUNK_ROWS:
                raise AssertionError(f"chunks must start at P % {CHUNK_ROWS} == 0, got P = {position}")
            end = position + slabs * self.slab_rows + long_chunks * LONG_CHUNK_ROWS + CHUNK_ROWS * len(accepts)
            if end > self.model.allocated_context:
                raise ValueError(
                    f"the padded tail chunk ends at {end}, past the allocated context {self.model.allocated_context}"
                )
            ttnn.synchronize_device(self.mesh)
            self.model.reset_chunk_state_inplace(self.state, self.chunk_state)
            if long_chunks or slabs:
                self.model.reset_chunk_state_inplace(self.state, self.long_chunk_state)
            if slabs:
                self.model.reset_chunk_state_inplace(self.state, self.slab_state)
            if self.mtp is not None:
                self.mtp.reset_chunk()
                if (long_chunks or slabs) and self.long_mtp is not None:
                    self.long_mtp.reset_chunk()
            if self.verify_allocations and self.chunk_trace_id is not None:
                verify_started_ns = time.perf_counter_ns()
                for trace_id in (
                    (self.chunk_trace_id,)
                    + ((self.long_chunk_trace_id,) if long_chunks or slabs else ())
                    + ((self.slab_trace_id,) if slabs else ())
                ):
                    if trace_id is not None:
                        TraceAllocationTracker.verify_before_replay(self.mesh, trace_id)
                verify_ms = (time.perf_counter_ns() - verify_started_ns) / 1_000_000
            # The MTP layer's tokens sit one position ahead: the chunk's rows shifted by one, then the following token.
            following = remaining[1:] + [following_token if following_token is not None else self.pad_token_id]
            # Their feature rows: ``pads_before[i]`` pads among ``remaining[:i]`` index the prefill's features (the
            # alignment steps hold no pads), so the tokens ahead of rows ``start .. start + w`` take the features of
            # the pads among ``remaining[start + 1 : start + 1 + w]`` (the following token is text: none).
            pads_before = [0]
            for token in remaining:
                pads_before.append(pads_before[-1] + (token == vision_splice.IMAGE_TOKEN_ID))

            def features_ahead(start: int, width: int):
                if features is None:
                    return None
                first, last = (
                    pads_before[min(start + 1, len(remaining))],
                    pads_before[min(start + 1 + width, len(remaining))],
                )
                return features[first:last] if last > first else None

            row_offset = 0
            timings = {
                "slab": (slab_host_ms, slab_replay_ms),
                "long": (long_host_ms, long_replay_ms),
                "short": (host_ms, replay_ms),
            }
            slab_event = None  # recorded after the last slab replay; waited for after the next one is queued
            rows_since_event = 0  # the chunk rows replayed since the last event (the event_rows cadence)
            # The next chunk's host half (the n-gram lookup and the row packing) is prepared while the current chunk
            # replays and consumed at the next iteration: only the first chunk's preparation is exposed.
            starts = [0]
            for _state, _kind, plan_rows, _accepted in plan[:-1]:
                starts.append(starts[-1] + len(plan_rows))
            pending = self._prepare(plan, starts, 0, ple_context, position, positions, features, feature_cursor)
            for index, (chunk_state, kind, rows, accepted) in enumerate(plan):
                real_rows = len(rows)
                start = row_offset
                row_offset += real_rows
                host_started_ns = time.perf_counter_ns()
                if accepted is not None and accepted != CHUNK_ROWS - 1:
                    self.model.write_chunk_accepted(self.chunk_state, accepted)
                prepared, prepare_ns, feature_cursor = pending
                ple_context = prepared.contexts[real_rows]
                upload_started_ns = time.perf_counter_ns()
                self.model.upload_chunk_inputs(chunk_state, prepared)
                extension = self._extension(kind)
                if extension is not None:
                    # The extension of the chunk's form takes the chunk's rows one position ahead (a long chunk and a
                    # slab are always full; the padded 32-row tail pads its MTP tokens too).
                    if kind == "slab":
                        extension.write_slab_tokens(
                            self.model,
                            following[start : start + self.slab_rows],
                            features=features_ahead(start, self.slab_rows),
                        )
                    else:
                        width = CHUNK_ROWS if kind == "short" else LONG_CHUNK_ROWS
                        ahead = following[start : start + width]
                        extension.write_tokens(
                            self.model,
                            ahead + [self.pad_token_id] * (width - len(ahead)),
                            features=features_ahead(start, width),
                        )
                replay_started_ns = time.perf_counter_ns()
                if kind == "slab":
                    slab_prepare_ms.append(prepare_ns / 1_000_000)
                    slab_upload_ms.append((replay_started_ns - upload_started_ns) / 1_000_000)
                if time_each_chunk:
                    timings[kind][0].append((prepare_ns + replay_started_ns - host_started_ns) / 1_000_000)
                    self._run_chunk(blocking=True, kind=kind)
                    timings[kind][1].append((time.perf_counter_ns() - replay_started_ns) / 1_000_000)
                else:
                    self._run_chunk(blocking=False, kind=kind)
                # The next chunk's host half under this chunk's replay (its copies are queued behind the replay on the
                # same command queue at the next iteration, so the device finishes reading the input buffers before
                # they change); a slab's wait for the previous slab's event follows, so the host runs one slab ahead.
                pending = self._prepare(
                    plan, starts, index + 1, ple_context, position, positions, features, feature_cursor
                )
                if time_each_chunk:
                    continue
                if kind == "slab":
                    wait_started_ns = time.perf_counter_ns()
                    if slab_event is not None:
                        ttnn.event_synchronize(slab_event)
                    slab_event = ttnn.record_event(self.mesh, cq_id=0)
                    slab_wait_ms.append((time.perf_counter_ns() - wait_started_ns) / 1_000_000)
                    rows_since_event = 0
                else:
                    rows_since_event += real_rows
                    if self.event_rows is None:
                        due = (index + 1) % self.event_interval == 0
                    else:
                        due = rows_since_event >= self.event_rows
                    if not due:
                        continue
                    ttnn.event_synchronize(ttnn.record_event(self.mesh, cq_id=0))
                    rows_since_event = 0
                stopped = None if should_stop is None else should_stop()
                if stopped is not None:
                    done = plan[: index + 1]
                    consumed = sum(len(done_rows) for _, _, done_rows, _ in done)
                    chunks = sum(1 for _, done_kind, _, _ in done if done_kind == "short")
                    long_done = sum(1 for _, done_kind, _, _ in done if done_kind == "long")
                    slabs_done = sum(1 for _, done_kind, _, _ in done if done_kind == "slab")
                    break
                if between_chunks is not None and kind != "slab":
                    between_chunks(row_offset, len(remaining))
            position += consumed
            handoff_started_ns = time.perf_counter_ns()
            ttnn.synchronize_device(self.mesh)
            # The decode's shift after the prefill; a prompt that ends in text (Qwen38VisionPrompt.validate_prompt)
            # keeps it within the next position's block start.  A stop inside an image leaves a state no request
            # decodes from (an image prompt always resets), so its shift is clamped to a valid value.
            rope_shift = 0 if positions is None else min(positions.shift_at(position), position & ~3)
            self.model.finish_prefill(self.state, self.chunk_state, position, rope_shift=rope_shift)
            if self.mtp is not None:
                self.mtp.finish_chunk(self.model, prefilled=position)
            ttnn.synchronize_device(self.mesh)
            handoff_ms = (time.perf_counter_ns() - handoff_started_ns) / 1_000_000
        timing = Qwen38PrefillTiming(
            alignment_steps=aligned,
            chunks=chunks,
            tail_rows=consumed % CHUNK_ROWS,
            chunk_replay_ms=tuple(replay_ms),
            chunk_host_ms=tuple(host_ms),
            verify_ms=verify_ms,
            handoff_ms=handoff_ms,
            wall_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            traced=self.chunk_trace_id is not None,
            gdn_step_anchor=self.gdn_step_anchor,
            long_chunks=long_done,
            long_chunk_replay_ms=tuple(long_replay_ms),
            long_chunk_host_ms=tuple(long_host_ms),
            slabs=slabs_done,
            slab_rows=self.slab_rows,
            slab_replay_ms=tuple(slab_replay_ms),
            slab_host_ms=tuple(slab_host_ms),
            slab_prepare_ms=tuple(slab_prepare_ms),
            slab_upload_ms=tuple(slab_upload_ms),
            slab_wait_ms=tuple(slab_wait_ms),
        )
        return Qwen38PrefillResult(position, ple_context, timing, stopped)


__all__ = [
    "CHUNK_EVENT_INTERVAL",
    "CHUNK_PAD_TOKEN_ID",
    "Qwen38ChunkPrefill",
    "Qwen38PrefillResult",
    "Qwen38PrefillTiming",
    "alignment_steps",
    "chunk_accepts",
    "long_chunk_count",
    "slab_count",
]
