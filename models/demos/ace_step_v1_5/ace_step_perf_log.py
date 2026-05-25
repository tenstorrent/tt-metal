# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock perf logging for ACE-Step v1.5 demos and E2E generate().

Enable module-level timing with either:

- ``ACE_STEP_DEMO_PERF_LOG=1`` (or ``ACE_STEP_PERF_LOG=1``), or
- multi-device mesh runs (on by default).

On multi-device mesh (e.g. ``BH_QB``), perf logging is **on by default** unless
``ACE_STEP_DEMO_PERF_LOG=0``.

Logs go to stdout (``[ace_step_v1_5][perf]``) and loguru at INFO.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from loguru import logger


@dataclass
class SessionPassSnapshot:
    """One pass worth of timings captured for a multi-pass demo session."""

    label: str
    session_pass: int
    is_warmup: bool
    total_ms: float
    modules_ms: List[Tuple[str, float]] = field(default_factory=list)

    def accounted_ms(self) -> float:
        return sum(ms for _, ms in self.modules_ms)


@dataclass
class SessionPerfState:
    """Accumulated perf data across ``main(session_pass=...)`` invocations."""

    session_t0: float | None = None
    init_timings_ms: List[Tuple[str, float]] = field(default_factory=list)
    pass_snapshots: List[SessionPassSnapshot] = field(default_factory=list)

    def note_init(self, label: str, elapsed_ms: float) -> None:
        self.init_timings_ms.append((label, elapsed_ms))

    def add_pass_snapshot(self, snap: SessionPassSnapshot) -> None:
        self.pass_snapshots.append(snap)


def ace_step_perf_logging_enabled(*, explicit: Optional[bool] = None) -> bool:
    """Return True when demo/E2E perf logging is active."""
    if explicit is not None:
        return bool(explicit)
    env = os.environ.get("ACE_STEP_DEMO_PERF_LOG", os.environ.get("ACE_STEP_PERF_LOG", ""))
    return env.lower() in ("1", "true", "yes")


def sync_device(device: Any) -> None:
    """Block until device work queued before *device* is complete."""
    if device is None:
        return
    try:
        import ttnn

        ttnn.synchronize_device(device)
    except Exception:
        pass


def _emit_line(label: str, elapsed_ms: float, *, extra: str = "") -> None:
    suffix = f" {extra}" if extra else ""
    line = f"[ace_step_v1_5][perf] {label:40s} {elapsed_ms:10.2f} ms{suffix}"
    print(line, flush=True)
    logger.info("ACE-Step perf: {}: {:.2f} ms{}", label, elapsed_ms, suffix)


def _perf_banner(title: str) -> None:
    bar = "=" * 72
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    print(f"[ace_step_v1_5][perf] {title}", flush=True)
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)


class AceStepPerfRecorder:
    """Collect module timings and run parameters; emit a summary table at the end."""

    def __init__(self, *, enabled: Optional[bool] = None, params: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = ace_step_perf_logging_enabled(explicit=enabled)
        self.params: Dict[str, Any] = dict(params or {})
        self.timings_ms: List[Tuple[str, float]] = []
        self._t0 = time.perf_counter()
        if self.enabled:
            _perf_banner("wall-clock timing enabled (module lines stream as each stage completes)")

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def begin_run(self, *, summary_label: str = "demo_total", record: bool = True) -> None:
        """Start a new timed pass (e.g. warmup vs steady-state) in the same process."""
        self._summary_label = summary_label
        self._run_recording = bool(record)
        self.timings_ms = []
        self._t0 = time.perf_counter()
        if self.enabled and record:
            _perf_banner(f"timing pass: {summary_label}")

    def begin_run_disabled(self, *, summary_label: str = "warmup_total") -> None:
        """Warmup pass: skip module lines but still allow a summary label."""
        self.begin_run(summary_label=summary_label, record=False)

    def record(self, label: str, elapsed_ms: float) -> None:
        self.timings_ms.append((label, elapsed_ms))
        if self.enabled and getattr(self, "_run_recording", True):
            _emit_line(label, elapsed_ms)

    @contextmanager
    def timed(self, label: str, *, device: Any = None) -> Iterator[None]:
        """Time a block; sync *device* before/after when provided."""
        active = self.enabled and getattr(self, "_run_recording", True)
        if active and device is not None:
            sync_device(device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if active and device is not None:
                sync_device(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if active:
                self.record(label, elapsed_ms)

    def record_init_once(self, label: str, elapsed_ms: float) -> None:
        """Record one-time init cost shown before the first inference pass summary."""
        if not hasattr(self, "_init_timings_ms"):
            self._init_timings_ms: List[Tuple[str, float]] = []
        self._init_timings_ms.append((label, elapsed_ms))
        if self.enabled:
            _emit_line(label, elapsed_ms)

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def export_pass_snapshot(
        self,
        *,
        label: str,
        session_pass: int,
        is_warmup: bool,
    ) -> SessionPassSnapshot:
        """Capture module + wall times for session rollup (call before ``emit_summary``)."""
        total_ms = self.total_ms()
        modules = list(self.timings_ms)
        return SessionPassSnapshot(
            label=str(label),
            session_pass=int(session_pass),
            is_warmup=bool(is_warmup),
            total_ms=float(total_ms),
            modules_ms=modules,
        )

    def emit_summary(self, *, label: str | None = None) -> None:
        if not self.enabled:
            return
        label = label or getattr(self, "_summary_label", "e2e_total")
        total_ms = self.total_ms()
        self.timings_ms.append((label, total_ms))

        _perf_banner(f"RUN SUMMARY ({label})")

        param_parts = [f"{k}={v}" for k, v in sorted(self.params.items())]
        if param_parts:
            print(f"[ace_step_v1_5][perf] parameters:", flush=True)
            for part in param_parts:
                print(f"[ace_step_v1_5][perf]   {part}", flush=True)
            logger.info("ACE-Step perf parameters: {}", " ".join(param_parts))

        init_timings = getattr(self, "_init_timings_ms", None)
        if init_timings:
            print("[ace_step_v1_5][perf] one-time init (amortized across session passes):", flush=True)
            for name, ms in init_timings:
                print(f"[ace_step_v1_5][perf]   {name:40s} {ms:10.2f} ms", flush=True)

        if not self.timings_ms:
            return

        print("[ace_step_v1_5][perf] module breakdown:", flush=True)
        accounted = 0.0
        for name, ms in self.timings_ms:
            if name == label:
                continue
            accounted += ms
            pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
            row = f"[ace_step_v1_5][perf]   {name:40s} {ms:10.2f} ms  ({pct:5.1f}%)"
            print(row, flush=True)
        unaccounted = max(0.0, total_ms - accounted)
        if unaccounted > 0.5:
            pct = unaccounted / total_ms * 100.0 if total_ms > 0 else 0.0
            print(
                f"[ace_step_v1_5][perf]   {'(other/overhead)':40s} {unaccounted:10.2f} ms  ({pct:5.1f}%)",
                flush=True,
            )
        print(f"[ace_step_v1_5][perf]   {'TOTAL (wall)':40s} {total_ms:10.2f} ms", flush=True)
        _perf_banner("end perf summary")


def emit_session_summary(state: SessionPerfState) -> None:
    """Print a rollup across init + every pass in a ``--warmup`` / ``--repeat`` session."""
    if not ace_step_perf_logging_enabled():
        return
    if not state.pass_snapshots and not state.init_timings_ms:
        return

    _perf_banner("SESSION SUMMARY")

    if state.init_timings_ms:
        print("[ace_step_v1_5][perf] one-time init (paid once per process):", flush=True)
        init_total = 0.0
        for name, ms in state.init_timings_ms:
            init_total += ms
            print(f"[ace_step_v1_5][perf]   {name:40s} {ms:10.2f} ms", flush=True)
        print(f"[ace_step_v1_5][perf]   {'init subtotal':40s} {init_total:10.2f} ms", flush=True)

    if state.pass_snapshots:
        print("[ace_step_v1_5][perf] pass wall times:", flush=True)
        passes_wall = 0.0
        for snap in state.pass_snapshots:
            passes_wall += snap.total_ms
            tag = "warmup" if snap.is_warmup else "timed"
            print(
                f"[ace_step_v1_5][perf]   [{snap.session_pass}] {snap.label:24s} " f"{snap.total_ms:10.2f} ms  ({tag})",
                flush=True,
            )
        print(f"[ace_step_v1_5][perf]   {'passes subtotal':40s} {passes_wall:10.2f} ms", flush=True)

        # Aggregate modules across passes (same label → sum).
        merged: Dict[str, float] = {}
        for snap in state.pass_snapshots:
            for name, ms in snap.modules_ms:
                merged[name] = merged.get(name, 0.0) + float(ms)
        if merged:
            merge_total = sum(merged.values())
            print("[ace_step_v1_5][perf] module rollup (summed across all passes):", flush=True)
            for name, ms in sorted(merged.items(), key=lambda kv: -kv[1]):
                pct = (ms / merge_total * 100.0) if merge_total > 0 else 0.0
                print(f"[ace_step_v1_5][perf]   {name:40s} {ms:10.2f} ms  ({pct:5.1f}%)", flush=True)

        timed = [s for s in state.pass_snapshots if not s.is_warmup]
        if timed:
            last = timed[-1]
            print(
                f"[ace_step_v1_5][perf] steady-state (last timed pass '{last.label}'): " f"{last.total_ms:.2f} ms",
                flush=True,
            )
        warmup = [s for s in state.pass_snapshots if s.is_warmup]
        if warmup and timed:
            print(
                "[ace_step_v1_5][perf] note: timed pass(es) after warmup may skip LM/preprocess "
                "when prompt+duration+seed match (see 'reusing cached preprocess' in log).",
                flush=True,
            )

    init_total = sum(ms for _, ms in state.init_timings_ms)
    passes_wall = sum(s.total_ms for s in state.pass_snapshots)
    session_wall = (time.perf_counter() - state.session_t0) * 1000.0 if state.session_t0 else init_total + passes_wall
    grand = init_total + passes_wall
    print(f"[ace_step_v1_5][perf]   {'SESSION (init + passes)':40s} {grand:10.2f} ms", flush=True)
    print(f"[ace_step_v1_5][perf]   {'SESSION (process wall)':40s} {session_wall:10.2f} ms", flush=True)
    _perf_banner("end session summary")


@contextmanager
def perf_timer(
    label: str,
    *,
    device: Any = None,
    recorder: Optional[AceStepPerfRecorder] = None,
    enabled: Optional[bool] = None,
) -> Iterator[None]:
    """Standalone timer when you do not need a full :class:`AceStepPerfRecorder`."""
    active = ace_step_perf_logging_enabled(explicit=enabled)
    if active and device is not None:
        sync_device(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if active and device is not None:
            sync_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if recorder is not None:
            recorder.record(label, elapsed_ms)
        elif active:
            _emit_line(label, elapsed_ms)


def make_denoise_progress_fn(
    recorder: Optional[AceStepPerfRecorder],
    *,
    num_steps: int,
) -> Optional[Any]:
    """Reserved for optional per-step denoise logging (disabled)."""
    del recorder, num_steps
    return None


def log_denoise_step_summary(step_times_ms: List[float]) -> None:
    del step_times_ms
