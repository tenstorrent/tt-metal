# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock perf logging for ACE-Step v1.5 demos and E2E generate().

Enable module-level timing with either:

- ``ACE_STEP_DEMO_PERF_LOG=1`` (or ``ACE_STEP_PERF_LOG=1``), or
- ``--perf-log`` on ``run_prompt_to_wav.py``.

Optional per-Euler-step lines: ``ACE_STEP_DEMO_PERF_LOG_STEPS=1``.

Logs go to stdout (``[ace_step_v1_5][perf]``) and loguru at INFO.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from loguru import logger


def ace_step_perf_logging_enabled(*, explicit: Optional[bool] = None) -> bool:
    """Return True when demo/E2E perf logging is active."""
    if explicit is not None:
        return bool(explicit)
    env = os.environ.get("ACE_STEP_DEMO_PERF_LOG", os.environ.get("ACE_STEP_PERF_LOG", ""))
    return env.lower() in ("1", "true", "yes")


def ace_step_perf_log_steps_enabled() -> bool:
    env = os.environ.get("ACE_STEP_DEMO_PERF_LOG_STEPS", "")
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
    line = f"[ace_step_v1_5][perf] {label}: {elapsed_ms:.2f} ms{suffix}"
    print(line, flush=True)
    logger.info("ACE-Step perf: {}: {:.2f} ms{}", label, elapsed_ms, suffix)


class AceStepPerfRecorder:
    """Collect module timings and run parameters; emit a summary table at the end."""

    def __init__(self, *, enabled: Optional[bool] = None, params: Optional[Dict[str, Any]] = None) -> None:
        self.enabled = ace_step_perf_logging_enabled(explicit=enabled)
        self.params: Dict[str, Any] = dict(params or {})
        self.timings_ms: List[Tuple[str, float]] = []
        self._t0 = time.perf_counter()

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def record(self, label: str, elapsed_ms: float) -> None:
        self.timings_ms.append((label, elapsed_ms))
        if self.enabled:
            _emit_line(label, elapsed_ms)

    @contextmanager
    def timed(self, label: str, *, device: Any = None) -> Iterator[None]:
        """Time a block; sync *device* before/after when provided."""
        if self.enabled and device is not None:
            sync_device(device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.enabled and device is not None:
                sync_device(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.record(label, elapsed_ms)

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def emit_summary(self, *, label: str = "e2e_total") -> None:
        if not self.enabled:
            return
        total_ms = self.total_ms()
        self.record(label, total_ms)

        param_parts = [f"{k}={v}" for k, v in sorted(self.params.items())]
        if param_parts:
            print(f"[ace_step_v1_5][perf] parameters: {' '.join(param_parts)}", flush=True)
            logger.info("ACE-Step perf parameters: {}", " ".join(param_parts))

        if not self.timings_ms:
            return

        print("[ace_step_v1_5][perf] module breakdown:", flush=True)
        accounted = 0.0
        for name, ms in self.timings_ms:
            if name == label:
                continue
            accounted += ms
            pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
            row = f"  {name:40s} {ms:10.2f} ms  ({pct:5.1f}%)"
            print(row, flush=True)
        unaccounted = max(0.0, total_ms - accounted)
        if unaccounted > 0.5:
            pct = unaccounted / total_ms * 100.0 if total_ms > 0 else 0.0
            print(f"  {'(other/overhead)':40s} {unaccounted:10.2f} ms  ({pct:5.1f}%)", flush=True)
        print(f"  {'TOTAL':40s} {total_ms:10.2f} ms", flush=True)


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
    """Build a ``progress_fn`` for :func:`run_ttnn_denoise_loop` when per-step logging is on."""
    if not ace_step_perf_log_steps_enabled():
        return None
    step_times: List[float] = []
    step_t0 = time.perf_counter()

    def _progress(step_idx: int, _num_steps: int, t_curr: float, dt: float) -> None:
        nonlocal step_t0
        elapsed_ms = (time.perf_counter() - step_t0) * 1000.0
        step_times.append(elapsed_ms)
        line = (
            f"[ace_step_v1_5][perf] denoise_step {step_idx + 1}/{num_steps} "
            f"t={t_curr:.5f} dt={dt:.5f} wall={elapsed_ms:.2f} ms"
        )
        print(line, flush=True)
        logger.info(
            "ACE-Step denoise step {}/{} t={:.5f} dt={:.5f} {:.2f} ms",
            step_idx + 1,
            num_steps,
            t_curr,
            dt,
            elapsed_ms,
        )
        step_t0 = time.perf_counter()

    return _progress


def log_denoise_step_summary(step_times_ms: List[float]) -> None:
    if not ace_step_perf_log_steps_enabled() or not step_times_ms:
        return
    total = sum(step_times_ms)
    avg = total / len(step_times_ms)
    best = min(step_times_ms)
    worst = max(step_times_ms)
    print(
        f"[ace_step_v1_5][perf] denoise_steps n={len(step_times_ms)} "
        f"sum={total:.2f} ms avg={avg:.2f} ms best={best:.2f} ms worst={worst:.2f} ms",
        flush=True,
    )
