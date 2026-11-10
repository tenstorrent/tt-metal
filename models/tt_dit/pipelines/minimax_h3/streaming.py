# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded, synchronous H3 reference-feedback streaming contracts.

This is an opt-in experiment, not a claim of temporally pinned continuation.
The current ref2va API conditions on references; it does not pin an output prefix.
Consequently a backend may only remove overlap when it verifies an exact repeated
prefix. A sink must finish mux/export/transport and acknowledge client readiness
before another render starts. No background queues or cross-request latents are
owned here. Pipeline caches remain the caller's responsibility to measure/bound.

The module deliberately imports neither torch nor ttnn: control-plane tests must
not open a device or require a checkpoint.
"""

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Optional, Tuple


@dataclass(frozen=True)
class Identity:
    variant: str
    source_sha: str
    build: str
    checkpoint_sha256: str
    config_sha256: str
    adapter_sha256: Optional[str] = None

    def __post_init__(self):
        if self.variant not in ("h3-r2v-high", "h3-r2v-turbo4", "h3-r2v-hyperflow8"):
            raise ValueError("unknown H3 variant")
        for value, length in ((self.source_sha, 40), (self.checkpoint_sha256, 64), (self.config_sha256, 64)):
            if not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
                raise ValueError("identities require full lowercase hexadecimal digests")
        if not self.build:
            raise ValueError("a build identity is required")
        if self.variant == "h3-r2v-high":
            if self.adapter_sha256 is not None:
                raise ValueError("base route must not silently select an adapter")
        elif not re.fullmatch(r"[0-9a-f]{64}", self.adapter_sha256 or ""):
            raise ValueError("speed variants require an explicit adapter digest")


@dataclass(frozen=True)
class StreamConfig:
    identity: Identity
    raw_frames: int = 124
    context_frames: int = 24
    fps: int = 24
    sample_rate: int = 32000
    initial_buffer_s: float = 0.0
    max_buffer_s: float = 10.5
    measurement_mode: str = "paced"
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        if self.measurement_mode not in ("paced", "capacity") or not self.session_id:
            raise ValueError("explicit paced/capacity mode and session identity required")
        if self.fps != 24 or self.sample_rate != 32000:
            raise ValueError("H3 uses 24 fps and 32000 Hz audio")
        if self.raw_frames < 22 or (self.raw_frames - 5) % 17:
            raise ValueError("raw_frames must be a supported 17n+5 H3 length")
        if not 1 <= self.context_frames < self.raw_frames:
            raise ValueError("context must be positive and smaller than one chunk")
        if not math.isfinite(self.initial_buffer_s) or self.initial_buffer_s < 0:
            raise ValueError("initial buffer must be finite and nonnegative")
        if (
            not math.isfinite(self.max_buffer_s)
            or self.max_buffer_s < self.raw_frames / self.fps
            or self.initial_buffer_s > self.max_buffer_s
        ):
            raise ValueError("buffer cap must hold a whole chunk and the initial buffer")


@dataclass(frozen=True)
class Media:
    """One owned media buffer, with backend-defined slicing and exact comparison.

    slice_copy(start, stop, sample_start, sample_stop) must copy, not retain a view
    of the whole output. A request owns its latents; only this bounded decoded
    context survives. `prefix_is_previous_tail` must compare actual video AND
    audio bytes, not rely on a mode or an assumed overlap count.
    """

    video: Any
    audio: Any
    frames: int
    samples: int
    slice_copy: Callable[[int, int, int, int], "Media"]
    prefix_is_previous_tail: Callable[["Media", int, int], bool]
    repeated_prefix_frames: int = 0


@dataclass(frozen=True)
class Request:
    index: int
    prompt: str
    seed: int
    fixed_references: Tuple[Any, ...]
    previous: Optional[Media]
    config: StreamConfig


@dataclass(frozen=True)
class Chunk:
    index: int
    media: Media
    video_pts: int
    audio_pts: int
    video_time_base: Fraction
    audio_time_base: Fraction
    trimmed_frames: int
    identity: Identity


@dataclass(frozen=True)
class Acknowledgement:
    """Returned only after the client can decode this exact chunk.

    played_frames is the actual playhead in paced mode. In an explicitly labelled
    capacity experiment it is the count decoded and discarded by the client;
    that experiment is not evidence of sustained real-time playback.
    """

    index: int
    playable_frames: int
    playable_samples: int
    played_frames: int = 0


class Backpressure(RuntimeError):
    """No render admitted; the caller may retry after this explicit pacing delay."""

    def __init__(self, retry_after_s):
        super().__init__("playout buffer cap: pace before admitting another chunk")
        self.retry_after_s = retry_after_s


class Session:
    """One chain and one synchronous in-flight chunk; failures poison the chain.

    receipt(row) must persist rows without retaining media. Render and sink must
    be bounded by the owner's job timeout. The controller cannot interrupt a
    hung device call; the broker/driver owns that watchdog and mesh teardown.
    """

    def __init__(self, config, references, render, sink, receipt, *, clock=time.monotonic):
        if not 1 <= len(references) <= 11:
            raise ValueError("reserve one of twelve reference slots for rolling context")
        self.config = config
        self.references = tuple(references)
        self.render, self.sink, self.receipt, self.clock = render, sink, receipt, clock
        self.previous = None
        self.index = self.frames = self.samples = 0
        self.played_frames = 0
        self.started = self.last_ready = self.last_clock = None
        self.service_wall_s = 0.0
        self.underruns = 0
        self.buffer_s = config.initial_buffer_s
        self.busy = self.closed = False

    def _now(self):
        value = self.clock()
        if not math.isfinite(value) or (self.last_clock is not None and value < self.last_clock):
            raise ValueError("clock must be finite and monotonic")
        self.last_clock = value
        return value

    def close(self):
        self.previous = None
        self.closed = True

    def update_playout(self, played_frames):
        """Accept a fresh client playhead; wall time alone never frees capacity."""
        if self.busy or self.closed:
            raise RuntimeError("playout update requires an open, idle session")
        self._validate_playhead(played_frames, self.frames)
        self.played_frames = played_frames

    def _validate_playhead(self, played_frames, available_frames):
        if not isinstance(played_frames, int) or not self.played_frames <= played_frames <= available_frames:
            raise ValueError("client playhead regressed or exceeds delivered media")

    def step(self, prompt, seed):
        if self.closed or self.busy:
            raise RuntimeError("closed session or concurrent/reentrant admission")
        self.busy = True
        admitted_work = False
        try:
            admitted = self._now()
            remaining = (self.frames - self.played_frames) / self.config.fps
            delay = remaining + self.config.raw_frames / self.config.fps - self.config.max_buffer_s
            if delay > 1e-9:
                raise Backpressure(delay)
            admitted_work = True
            return self._step(prompt, seed, admitted)
        except Backpressure:
            if admitted_work:
                self.close()
            raise
        except BaseException:
            # A failed sink can have emitted some bytes. Never retry/advance that
            # chain implicitly: otherwise output and timestamps may duplicate.
            self.close()
            raise
        finally:
            self.busy = False

    def _step(self, prompt, seed, admitted):
        cfg = self.config
        if self.started is None:
            self.started = admitted
        raw = self.render(Request(self.index, prompt, seed, self.references, self.previous, cfg))
        rendered = self._now()
        if raw.frames != cfg.raw_frames:
            raise ValueError("backend frame count differs from configured shape")
        overlap = raw.repeated_prefix_frames
        if not isinstance(overlap, int) or not 0 <= overlap <= cfg.context_frames:
            raise ValueError("invalid repeated prefix")
        trim_samples = overlap * cfg.sample_rate // cfg.fps
        if overlap and (self.previous is None or not raw.prefix_is_previous_tail(self.previous, overlap, trim_samples)):
            raise ValueError("unverified overlap: ref2va conditioning does not pin output")
        new_frames = raw.frames - overlap
        # Global rational boundary rounding prevents per-chunk 1/3-sample drift.
        next_samples = (self.frames + new_frames) * cfg.sample_rate // cfg.fps
        new_samples = next_samples - self.samples
        if raw.samples < trim_samples + new_samples:
            raise ValueError("audio is shorter than emitted video; padding is not implicit")
        playable = raw.slice_copy(overlap, raw.frames, trim_samples, trim_samples + new_samples)
        if playable.frames != new_frames or playable.samples != new_samples:
            raise ValueError("backend slicing returned incorrect lengths")
        # Copy the retained tail before releasing the full generated buffer.
        keep = min(cfg.context_frames, new_frames)
        keep_samples = keep * cfg.sample_rate // cfg.fps
        tail = playable.slice_copy(new_frames - keep, new_frames, new_samples - keep_samples, new_samples)
        if tail.frames != keep or tail.samples != keep_samples:
            raise ValueError("backend context copy returned incorrect lengths")
        chunk = Chunk(
            self.index,
            playable,
            self.frames,
            self.samples,
            Fraction(1, cfg.fps),
            Fraction(1, cfg.sample_rate),
            overlap,
            cfg.identity,
        )
        ack = self.sink(chunk)
        ready = self._now()  # includes copying, mux, export, transport and acknowledgement
        if not isinstance(ack, Acknowledgement) or (ack.index, ack.playable_frames, ack.playable_samples) != (
            self.index,
            new_frames,
            new_samples,
        ):
            raise ValueError("sink did not acknowledge this exact playable chunk")
        self._validate_playhead(ack.played_frames, self.frames + new_frames)
        self.played_frames = ack.played_frames
        service = ready - admitted
        self.service_wall_s += service
        if self.last_ready is not None:
            self.buffer_s -= ready - self.last_ready
            if self.buffer_s < 0:
                self.underruns += 1
        minimum_buffer = self.buffer_s
        self.buffer_s = max(0.0, self.buffer_s) + new_frames / cfg.fps
        self.previous = tail
        self.frames += new_frames
        self.samples = next_samples
        row = {
            "index": self.index,
            "identity": cfg.identity.__dict__,
            "session_id": cfg.session_id,
            "measurement_mode": cfg.measurement_mode,
            "chunk": self.index,
            "ready_s": ready - self.started,
            "media_start_s": chunk.video_pts / cfg.fps,
            "media_end_s": self.frames / cfg.fps,
            "emitted_frames": raw.frames,
            "discarded_frames": overlap,
            "quality_pass": False,
            "continuity_pass": False,
            "measurement_kind": "control_plane_incomplete",
            "host_rss_bytes": None,
            "device_allocated_bytes": None,
            "stages_s": None,
            "continuation": "experimental-reference-feedback",
            "quality_validated": False,
            "raw_frames": raw.frames,
            "trimmed_frames": overlap,
            "new_frames": new_frames,
            "video_pts": chunk.video_pts,
            "audio_pts": chunk.audio_pts,
            "new_samples": new_samples,
            "request_s": admitted - self.started,
            "render_done_s": rendered - self.started,
            "client_ready_s": ready - self.started,
            "service_wall_s": service,
            "client_interval_s": None if self.last_ready is None else ready - self.last_ready,
            "startup_s": service if self.index == 0 else None,
            "total_playable_s": self.frames / cfg.fps,
            "elapsed_rtf": (ready - self.started) / (self.frames / cfg.fps),
            "service_rtf": self.service_wall_s / (self.frames / cfg.fps),
            "buffer_s": self.buffer_s,
            "minimum_buffer_s": minimum_buffer,
            "underruns": self.underruns,
            "buffer_evidence": "ideal continuous playout model; client telemetry still required",
            "client_unplayed_frames": self.frames - self.played_frames,
            "retained_context_frames": keep,
            "pending_chunks": 0,
        }
        self.receipt(row)
        self.last_ready = ready
        self.index += 1
        return row
