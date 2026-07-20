# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock perf logging for ACE-Step v1.5 demos and E2E generate().

Enable module-level timing with either:

- **Default on** for all runs (single-card and mesh), or
- ``ACE_STEP_DEMO_PERF_LOG=0`` (or ``ACE_STEP_PERF_LOG=0``) to disable, or
- ``ACE_STEP_DEMO_PERF_LOG=1`` to force on explicitly.

Prints per-module wall times, KEY METRICS (LM / DiT / **VAE decode**), and upstream-style
``RTF`` (``audio_duration / wall``, higher = faster; same as
https://github.com/ace-step/ACE-Step#hardware-performance).

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
    """Return True when demo/E2E perf logging is active (default **on**)."""
    if explicit is not None:
        return bool(explicit)
    env = os.environ.get("ACE_STEP_DEMO_PERF_LOG", os.environ.get("ACE_STEP_PERF_LOG", ""))
    if env.lower() in ("0", "false", "no", "off"):
        return False
    if env.lower() in ("1", "true", "yes", "on"):
        return True
    return True


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

    def apply_preprocess_handoff(self, preprocess_perf: Optional[Dict[str, Any]]) -> None:
        """Restore Phase-A LM/preprocess timings + params after BH mesh re-exec handoff."""
        self._handoff_preprocess_perf = preprocess_perf
        ace_step_apply_preprocess_handoff_perf(self, preprocess_perf)

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
        handoff_perf = getattr(self, "_handoff_preprocess_perf", None)
        phase_a_ms = ace_step_phase_a_wall_ms(handoff_perf)
        effective_wall_ms = ace_step_effective_wall_ms(total_ms, handoff_perf)
        wall_denominator_ms = effective_wall_ms if phase_a_ms > 0.0 else total_ms
        accounted = 0.0
        for name, ms in self.timings_ms:
            if name == label:
                continue
            accounted += ms
            pct = (ms / wall_denominator_ms * 100.0) if wall_denominator_ms > 0 else 0.0
            row = f"[ace_step_v1_5][perf]   {name:40s} {ms:10.2f} ms  ({pct:5.1f}%)"
            print(row, flush=True)
        unaccounted = max(0.0, wall_denominator_ms - accounted)
        if unaccounted > 0.5:
            pct = unaccounted / wall_denominator_ms * 100.0 if wall_denominator_ms > 0 else 0.0
            print(
                f"[ace_step_v1_5][perf]   {'(other/overhead)':40s} {unaccounted:10.2f} ms  ({pct:5.1f}%)",
                flush=True,
            )
        print(f"[ace_step_v1_5][perf]   {'TOTAL (wall)':40s} {effective_wall_ms:10.2f} ms", flush=True)
        if phase_a_ms > 0.0:
            print(
                f"[ace_step_v1_5][perf]   {'  Phase-A (LM+preprocess)':40s} {phase_a_ms:10.2f} ms",
                flush=True,
            )
            print(
                f"[ace_step_v1_5][perf]   {'  Phase-B (DiT+VAE)':40s} {total_ms:10.2f} ms",
                flush=True,
            )
        _emit_rtf_metrics(wall_ms=effective_wall_ms, params=self.params)
        module_timings = [pair for pair in self.timings_ms if pair[0] != label]
        module_timings = ace_step_merge_handoff_module_timings(module_timings, handoff_perf)
        summary_params = dict(self.params)
        if isinstance(handoff_perf, dict):
            handoff_params = handoff_perf.get("params") or {}
            if handoff_params:
                summary_params.update(handoff_params)
        emit_benchmark_wall_breakdown(
            module_timings,
            wall_ms=effective_wall_ms,
            params=summary_params,
        )
        emit_key_metrics(
            module_timings,
            wall_ms=effective_wall_ms,
            params=summary_params,
        )
        _perf_banner("end perf summary")


def _ms_lookup(timings_ms: List[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, ms in timings_ms:
        out[name] = out.get(name, 0.0) + float(ms)
    return out


# Module labels → ACE-Step ``profile_inference.py`` TIME COSTS buckets (see BENCHMARK.md).
_BENCHMARK_LLM_LABELS = frozenset({"five_hz_lm_generate", "handler_preprocess", "preprocess_readback"})
_BENCHMARK_CONDITION_LABELS = frozenset({"condition_encoder"})
_BENCHMARK_DIT_LABELS = frozenset({"dit_mask_prep", "dit_denoise_loop"})
_BENCHMARK_VAE_LABELS = frozenset({"vae_decode", "vae_decode_torch"})
_BENCHMARK_AUDIO_SAVE_LABELS = frozenset({"audio_save"})
_BENCHMARK_INIT_LABELS = frozenset(
    {"handler_init", "qwen_encoder_init", "dit_pipeline_init", "vae_init", "five_hz_lm_init"}
)
_BENCHMARK_WALL_LABELS = frozenset({"demo_total", "e2e_total"})
_PHASE_A_HANDOFF_EXCLUDE_LABELS = (
    _BENCHMARK_WALL_LABELS | _BENCHMARK_DIT_LABELS | _BENCHMARK_VAE_LABELS | _BENCHMARK_AUDIO_SAVE_LABELS
)
_PREPROCESS_HANDOFF_TIMING_LABELS = frozenset(_BENCHMARK_LLM_LABELS)
_PREPROCESS_HANDOFF_PARAM_KEYS = (
    "lm_num_tokens",
    "lm_gen_time_s",
    "lm_phase1_time_s",
    "lm_phase2_time_s",
)


def ace_step_phase_a_wall_ms(preprocess_perf: Optional[Dict[str, Any]]) -> float:
    """Return Phase-A wall time carried across BH mesh re-exec handoff (0 when absent)."""
    if not preprocess_perf:
        return 0.0
    try:
        return max(0.0, float(preprocess_perf.get("phase_a_wall_ms") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def ace_step_effective_wall_ms(
    phase_b_wall_ms: float,
    preprocess_perf: Optional[Dict[str, Any]] = None,
) -> float:
    """End-to-end wall = Phase-A handoff wall + Phase-B measured wall."""
    return float(phase_b_wall_ms) + ace_step_phase_a_wall_ms(preprocess_perf)


def ace_step_build_preprocess_handoff_perf(
    *,
    timings_ms: List[Tuple[str, float]],
    params: Optional[Dict[str, Any]] = None,
    lm_perf: Optional[Dict[str, Any]] = None,
    phase_a_wall_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Serialize Phase-A preprocess perf for :func:`ace_step_reexec_for_dit_mesh` handoff."""
    lookup = _ms_lookup(timings_ms)
    out_timings = [
        (label, lookup[label])
        for label in sorted(lookup.keys())
        if label not in _PHASE_A_HANDOFF_EXCLUDE_LABELS and lookup.get(label, 0.0) > 0.0
    ]
    lm_lookup_ms = sum(float(lookup.get(lbl, 0.0)) for lbl in _PREPROCESS_HANDOFF_TIMING_LABELS)
    out_params: Dict[str, Any] = {}
    if params:
        for key in _PREPROCESS_HANDOFF_PARAM_KEYS:
            if key in params and params[key] is not None:
                out_params[key] = params[key]
    if isinstance(lm_perf, dict):
        for key in _PREPROCESS_HANDOFF_PARAM_KEYS:
            if key in lm_perf and lm_perf[key] is not None:
                out_params[key] = lm_perf[key]
    if lm_lookup_ms > 0.0:
        try:
            gen_s = float(out_params.get("lm_gen_time_s") or 0.0)
        except (TypeError, ValueError):
            gen_s = 0.0
        if gen_s <= 0.0:
            out_params["lm_gen_time_s"] = lm_lookup_ms / 1000.0
    out: Dict[str, Any] = {"timings_ms": out_timings, "params": out_params}
    if phase_a_wall_ms is not None:
        try:
            wall_ms = float(phase_a_wall_ms)
            if wall_ms > 0.0:
                out["phase_a_wall_ms"] = wall_ms
        except (TypeError, ValueError):
            pass
    return out


def ace_step_apply_preprocess_handoff_perf(
    recorder: AceStepPerfRecorder,
    preprocess_perf: Optional[Dict[str, Any]],
) -> None:
    """Inject handoff preprocess stats into the Phase-B perf recorder (silent, no re-print)."""
    if preprocess_perf is None:
        return
    handoff_params = preprocess_perf.get("params") or {}
    if handoff_params:
        recorder.set_params(**handoff_params)
    for label, ms in preprocess_perf.get("timings_ms") or []:
        recorder.timings_ms.append((str(label), float(ms)))


def ace_step_merge_handoff_module_timings(
    module_timings: List[Tuple[str, float]],
    preprocess_perf: Optional[Dict[str, Any]],
) -> List[Tuple[str, float]]:
    """Merge Phase-A handoff module lines into a Phase-B module list (dedupe by label, keep max ms)."""
    if not preprocess_perf:
        return module_timings
    merged = _ms_lookup(module_timings)
    for label, ms in preprocess_perf.get("timings_ms") or []:
        lbl = str(label)
        merged[lbl] = max(float(merged.get(lbl, 0.0)), float(ms))
    return sorted(merged.items(), key=lambda kv: kv[0])


def ace_step_rtf(
    *,
    wall_s: float,
    duration_sec: float,
) -> Optional[float]:
    """Upstream ACE-Step RTF: ``audio_duration / wall`` (higher = faster).

    Matches https://github.com/ace-step/ACE-Step hardware-performance: e.g. 27.27× means
    60 s of audio in ``60 / 27.27 ≈ 2.2`` s wall time.
    """
    if float(wall_s) <= 0.0 or float(duration_sec) <= 0.0:
        return None
    return float(duration_sec) / float(wall_s)


# Upstream ACE-Step README "Hardware Performance" table
# (https://github.com/ace-step/ACE-Step#%EF%B8%8F-hardware-performance).
# Single GPU, batch size 1. RTF = audio_duration / wall.
_UPSTREAM_RTF_REFERENCE: Tuple[Dict[str, Any], ...] = (
    {
        "device": "NVIDIA A100",
        "rtf_27": 27.27,
        "time_1min_27_s": 2.20,
        "rtf_60": 12.27,
        "time_1min_60_s": 4.89,
    },
    {
        "device": "NVIDIA RTX 3090",
        "rtf_27": 12.76,
        "time_1min_27_s": 4.70,
        "rtf_60": 6.48,
        "time_1min_60_s": 9.26,
    },
)

# Fair-comparison protocol matching upstream hardware-performance style inputs
# (duration / steps / CFG / Euler). Use ``--upstream-benchmark`` in the demo.
UPSTREAM_HARDWARE_BENCHMARK: Dict[str, Any] = {
    "duration_sec": 170.64,
    "infer_steps": 60,
    "guidance_scale": 15.0,
    "sampler_mode": "euler",
    "variant": "acestep-v15-base",
    "prompt": ("electronic music with a strong beat, warm pads, clear melody, " "high quality studio production"),
}


def ace_step_matches_upstream_hardware_benchmark(params: Optional[Dict[str, Any]]) -> bool:
    """True when this run uses the upstream hardware-performance style inputs."""
    if not params:
        return False
    try:
        duration = float(params.get("duration_sec"))
        steps = int(params.get("infer_steps"))
        gs = float(params.get("guidance_scale"))
    except (TypeError, ValueError):
        return False
    sampler = str(params.get("sampler_mode") or "euler").lower()
    return (
        abs(duration - float(UPSTREAM_HARDWARE_BENCHMARK["duration_sec"])) < 0.05
        and steps == int(UPSTREAM_HARDWARE_BENCHMARK["infer_steps"])
        and abs(gs - float(UPSTREAM_HARDWARE_BENCHMARK["guidance_scale"])) < 0.05
        and sampler == "euler"
    )


def ace_step_time_to_render_1min_s(rtf: float) -> Optional[float]:
    """Wall seconds to synthesize 60 s of audio at the given upstream-style RTF."""
    if float(rtf) <= 0.0:
        return None
    return 60.0 / float(rtf)


def emit_upstream_rtf_reference(
    *,
    our_rtf: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Print A100 / RTX 3090 upstream RTF rows next to this run for direct comparison."""
    if not ace_step_perf_logging_enabled():
        return

    infer_steps = None
    duration_sec = None
    if params:
        if params.get("infer_steps") is not None:
            try:
                infer_steps = int(params["infer_steps"])
            except (TypeError, ValueError):
                infer_steps = None
        if params.get("duration_sec") is not None:
            try:
                duration_sec = float(params["duration_sec"])
            except (TypeError, ValueError):
                duration_sec = None

    bar = "=" * 88
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    print(
        "[ace_step_v1_5][perf] RTF COMPARISON vs upstream ACE-Step (A100 / RTX 3090)",
        flush=True,
    )
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    print(
        "[ace_step_v1_5][perf]   Source: https://github.com/ace-step/ACE-Step#hardware-performance",
        flush=True,
    )
    print(
        "[ace_step_v1_5][perf]   Upstream protocol: single GPU, batch=1; "
        "RTF = audio_duration / wall (higher = faster).",
        flush=True,
    )
    matched = ace_step_matches_upstream_hardware_benchmark(params)
    if matched:
        print(
            "[ace_step_v1_5][perf]   This run: MATCHES upstream-style inputs "
            f"(duration={UPSTREAM_HARDWARE_BENCHMARK['duration_sec']}s, "
            f"steps={UPSTREAM_HARDWARE_BENCHMARK['infer_steps']}, "
            f"guidance={UPSTREAM_HARDWARE_BENCHMARK['guidance_scale']}, "
            f"sampler={UPSTREAM_HARDWARE_BENCHMARK['sampler_mode']}).",
            flush=True,
        )
    elif duration_sec is not None or infer_steps is not None:
        bits = []
        if duration_sec is not None:
            bits.append(f"duration={duration_sec:g}s")
        if infer_steps is not None:
            bits.append(f"infer_steps={infer_steps}")
        if params and params.get("guidance_scale") is not None:
            bits.append(f"guidance_scale={params['guidance_scale']}")
        if params and params.get("sampler_mode") is not None:
            bits.append(f"sampler={params['sampler_mode']}")
        print(
            f"[ace_step_v1_5][perf]   This run: {', '.join(bits)} "
            "(not upstream hardware-performance inputs; "
            "use --upstream-benchmark for a fairer comparison).",
            flush=True,
        )
    print(
        f"[ace_step_v1_5][perf]   {'Device / config':<36s} {'RTF':>10s}  {'Time for 1 min audio':>20s}",
        flush=True,
    )
    print(
        f"[ace_step_v1_5][perf]   {'─' * 36} {'─' * 10}  {'─' * 20}",
        flush=True,
    )
    for row in _UPSTREAM_RTF_REFERENCE:
        label_27 = f"{row['device']} (27 steps)"
        label_60 = f"{row['device']} (60 steps)"
        print(
            f"[ace_step_v1_5][perf]   {label_27:<36s} " f"{row['rtf_27']:9.2f}×  {row['time_1min_27_s']:19.2f} s",
            flush=True,
        )
        print(
            f"[ace_step_v1_5][perf]   {label_60:<36s} " f"{row['rtf_60']:9.2f}×  {row['time_1min_60_s']:19.2f} s",
            flush=True,
        )

    if our_rtf is not None:
        t_1min = ace_step_time_to_render_1min_s(float(our_rtf))
        steps_lbl = f"{infer_steps} steps" if infer_steps is not None else "this run"
        label = f"This run TTNN ({steps_lbl})"
        t_1min_str = f"{t_1min:19.2f} s" if t_1min is not None else f"{'n/a':>20s}"
        print(
            f"[ace_step_v1_5][perf]   {'─' * 36} {'─' * 10}  {'─' * 20}",
            flush=True,
        )
        print(
            f"[ace_step_v1_5][perf]   {label:<36s} {float(our_rtf):9.2f}×  {t_1min_str}",
            flush=True,
        )
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    logger.info(
        "ACE-Step RTF comparison: this_run={} | A100@27/60={}/{} | RTX3090@27/60={}/{}",
        f"{our_rtf:.2f}×" if our_rtf is not None else "n/a",
        27.27,
        12.27,
        12.76,
        6.48,
    )


def ace_step_rtf_from_params(
    *,
    wall_ms: float,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    if not params:
        return None
    duration_sec = params.get("duration_sec")
    if duration_sec is None:
        return None
    try:
        return ace_step_rtf(wall_s=float(wall_ms) / 1000.0, duration_sec=float(duration_sec))
    except (TypeError, ValueError):
        return None


def ace_step_rtf_per_step(
    *,
    wall_s: float,
    duration_sec: float,
    infer_steps: int,
) -> Optional[float]:
    """Legacy diagnostic: ``wall / (audio_duration × infer_steps)``.

    **Not** the upstream ACE-Step RTF. Prefer :func:`ace_step_rtf`
    (``audio_duration / wall``, higher = faster).
    """
    denom = float(duration_sec) * float(infer_steps)
    if denom <= 0.0 or wall_s < 0.0:
        return None
    return float(wall_s) / denom


def ace_step_rtf_per_step_from_params(
    *,
    wall_ms: float,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    if not params:
        return None
    duration_sec = params.get("duration_sec")
    infer_steps = params.get("infer_steps")
    if duration_sec is None or infer_steps is None:
        return None
    try:
        return ace_step_rtf_per_step(
            wall_s=float(wall_ms) / 1000.0,
            duration_sec=float(duration_sec),
            infer_steps=int(infer_steps),
        )
    except (TypeError, ValueError):
        return None


def _emit_rtf_metrics(*, wall_ms: float, params: Optional[Dict[str, Any]] = None) -> None:
    """Print upstream RTF (``audio_duration / wall``) when params allow."""
    duration_sec = None
    if params and params.get("duration_sec") is not None:
        try:
            duration_sec = float(params["duration_sec"])
        except (TypeError, ValueError):
            duration_sec = None

    rtf = ace_step_rtf_from_params(wall_ms=wall_ms, params=params)
    if rtf is not None and duration_sec is not None:
        wall_s = float(wall_ms) / 1000.0
        print(
            f"[ace_step_v1_5][perf]   {'RTF':40s} {rtf:10.2f}×  "
            f"(audio {duration_sec:g}s / wall {wall_s:.2f}s; higher = faster)",
            flush=True,
        )
        logger.info(
            "ACE-Step RTF: {:.2f}× (audio {:.3g}s / wall {:.2f}s)",
            rtf,
            duration_sec,
            wall_s,
        )


def ace_step_extract_key_metrics(
    timings_ms: List[Tuple[str, float]],
    *,
    wall_ms: float,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return BENCHMARK.md *Key Metrics* (wall / LM / DiT / VAE / optional tokens/sec)."""
    lookup = _ms_lookup(timings_ms)
    for wall_label in _BENCHMARK_WALL_LABELS:
        lookup.pop(wall_label, None)

    def _sum(labels: frozenset[str]) -> float:
        return sum(lookup.get(lbl, 0.0) for lbl in labels)

    lm_ms = _sum(_BENCHMARK_LLM_LABELS)
    dit_ms = _sum(_BENCHMARK_DIT_LABELS)
    vae_ms = _sum(_BENCHMARK_VAE_LABELS)

    lm_time_s = lm_ms / 1000.0
    if params and params.get("lm_gen_time_s") is not None:
        try:
            gen_s = float(params["lm_gen_time_s"])
            if gen_s > 0.0:
                lm_time_s = gen_s
        except (TypeError, ValueError):
            pass

    metrics: Dict[str, Any] = {
        "wall_time_s": float(wall_ms) / 1000.0,
        "lm_total_time_s": lm_time_s,
        "dit_total_time_s": dit_ms / 1000.0,
        "vae_decode_time_s": vae_ms / 1000.0,
    }
    rtf = ace_step_rtf_from_params(wall_ms=wall_ms, params=params)
    if rtf is not None:
        metrics["rtf"] = rtf

    if params:
        num_tokens = params.get("lm_num_tokens")
        lm_gen_time_s = params.get("lm_gen_time_s")
        if num_tokens is not None and lm_gen_time_s is not None:
            try:
                n_tok = int(num_tokens)
                gen_s = float(lm_gen_time_s)
                if n_tok > 0 and gen_s > 0.0:
                    metrics["lm_num_tokens"] = n_tok
                    metrics["tokens_per_sec"] = float(n_tok) / gen_s
            except (TypeError, ValueError):
                pass
        if metrics.get("tokens_per_sec") is None and num_tokens is not None:
            try:
                n_tok = int(num_tokens)
                if n_tok > 0 and lm_time_s > 0.0:
                    metrics["lm_num_tokens"] = n_tok
                    metrics["tokens_per_sec"] = float(n_tok) / float(lm_time_s)
            except (TypeError, ValueError):
                pass
    return metrics


def emit_key_metrics(
    timings_ms: List[Tuple[str, float]],
    *,
    wall_ms: float,
    params: Optional[Dict[str, Any]] = None,
    show_tokens_per_sec: bool = True,
) -> None:
    """Print the ACE-Step BENCHMARK.md *Key Metrics* table."""
    if not ace_step_perf_logging_enabled():
        return

    metrics = ace_step_extract_key_metrics(timings_ms, wall_ms=wall_ms, params=params)
    bar = "=" * 72
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    print("[ace_step_v1_5][perf] KEY METRICS (ACE-Step BENCHMARK.md)", flush=True)
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    print(f"[ace_step_v1_5][perf]   {'Metric':<22s} {'Value':>12s}  Description", flush=True)
    print(f"[ace_step_v1_5][perf]   {'─' * 22} {'─' * 12}  {'─' * 30}", flush=True)

    rows: List[Tuple[str, str, str]] = [
        ("Wall Time", f"{metrics['wall_time_s']:.2f} s", "End-to-end time from start to finish"),
        ("LM Total Time", f"{metrics['lm_total_time_s']:.2f} s", "LLM planning (generation + parsing)"),
        ("DiT Total Time", f"{metrics['dit_total_time_s']:.2f} s", "Diffusion (all steps combined)"),
        ("VAE Decode Time", f"{metrics['vae_decode_time_s']:.2f} s", "Decode latents to audio waveform"),
    ]

    rtf = metrics.get("rtf")
    if rtf is not None:
        rows.append(
            (
                "RTF",
                f"{rtf:.2f}×",
                "audio_duration / wall (upstream ACE-Step; higher = faster)",
            )
        )
    elif params and params.get("duration_sec") is not None:
        rows.append(("RTF", "n/a", "Need wall > 0 and duration_sec > 0"))

    tps = metrics.get("tokens_per_sec")
    if show_tokens_per_sec:
        if tps is not None:
            n_tok = metrics.get("lm_num_tokens", "?")
            rows.append(("Tokens/sec", f"{tps:.1f}", f"LLM token generation throughput ({n_tok} new tokens)"))
        else:
            rows.append(("Tokens/sec", "n/a", "LM token stats unavailable (cached preprocess or LM skipped?)"))

    for name, value, desc in rows:
        print(f"[ace_step_v1_5][perf]   {name:<22s} {value:>12s}  {desc}", flush=True)

    print(f"[ace_step_v1_5][perf] {bar}", flush=True)

    log_extra = ""
    if rtf is not None:
        log_extra += f" RTF={rtf:.2f}×"
    if tps is not None:
        log_extra += f" tokens_per_sec={tps:.1f}"
    logger.info(
        "ACE-Step key metrics: wall={:.2f}s LM={:.2f}s DiT={:.2f}s VAE={:.2f}s{}",
        metrics["wall_time_s"],
        metrics["lm_total_time_s"],
        metrics["dit_total_time_s"],
        metrics["vae_decode_time_s"],
        log_extra,
    )
    emit_upstream_rtf_reference(our_rtf=rtf if isinstance(rtf, (int, float)) else None, params=params)


def emit_benchmark_wall_breakdown(
    timings_ms: List[Tuple[str, float]],
    *,
    wall_ms: float,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Print ACE-Step BENCHMARK.md-style wall-time table (seconds + % of total).

    Compatible with the ``profile_inference.py`` *profile* mode breakdown documented at
    https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/BENCHMARK.md
    """
    lookup = _ms_lookup(timings_ms)
    for wall_label in _BENCHMARK_WALL_LABELS:
        lookup.pop(wall_label, None)

    def _sum(labels: frozenset[str]) -> float:
        return sum(lookup.get(lbl, 0.0) for lbl in labels)

    llm_ms = _sum(_BENCHMARK_LLM_LABELS)
    cond_ms = _sum(_BENCHMARK_CONDITION_LABELS)
    dit_ms = _sum(_BENCHMARK_DIT_LABELS)
    vae_ms = _sum(_BENCHMARK_VAE_LABELS)
    save_ms = _sum(_BENCHMARK_AUDIO_SAVE_LABELS)
    init_ms = _sum(_BENCHMARK_INIT_LABELS)

    bucketed = {
        "LLM Planning (total)": llm_ms,
        "Condition encode": cond_ms,
        "DiT Diffusion (total)": dit_ms,
        "VAE Decode": vae_ms,
        "Audio Save": save_ms,
    }
    used_labels = (
        _BENCHMARK_LLM_LABELS
        | _BENCHMARK_CONDITION_LABELS
        | _BENCHMARK_DIT_LABELS
        | _BENCHMARK_VAE_LABELS
        | _BENCHMARK_AUDIO_SAVE_LABELS
        | _BENCHMARK_INIT_LABELS
    )
    other_ms = sum(ms for lbl, ms in lookup.items() if lbl not in used_labels)
    accounted_ms = sum(bucketed.values()) + other_ms + init_ms
    overhead_ms = max(0.0, float(wall_ms) - accounted_ms)
    other_ms += overhead_ms

    wall_s = float(wall_ms) / 1000.0
    bar = "=" * 100

    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    print("[ace_step_v1_5][perf] TIME COSTS BREAKDOWN (ACE-Step BENCHMARK.md compatible)", flush=True)
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    if params:
        brief = ", ".join(
            f"{k}={v}"
            for k, v in sorted(params.items())
            if k in ("duration_sec", "infer_steps", "guidance_scale", "variant", "use_trace", "torch_vae")
        )
        if brief:
            print(f"[ace_step_v1_5][perf] {brief}", flush=True)
    print(f"[ace_step_v1_5][perf]   {'Component':<34s} {'Time (s)':>10s}  {'% of Total':>12s}", flush=True)
    print(f"[ace_step_v1_5][perf]   {'─' * 34} {'─' * 10}  {'─' * 12}", flush=True)

    def _row(name: str, ms: float, *, indent: str = "") -> None:
        if ms <= 0.0:
            return
        sec = ms / 1000.0
        pct = (ms / float(wall_ms) * 100.0) if wall_ms > 0 else 0.0
        print(
            f"[ace_step_v1_5][perf]   {indent}{name:<34s} {sec:10.2f}  {pct:11.1f}%",
            flush=True,
        )

    for name, ms in bucketed.items():
        _row(name, ms)
        if name == "LLM Planning (total)" and ms > 0:
            for sub, sub_labels in (
                ("├─ LM token generation", frozenset({"five_hz_lm_generate"})),
                ("├─ Handler preprocess", frozenset({"handler_preprocess", "preprocess_readback"})),
            ):
                sub_ms = _sum(sub_labels)
                if sub_ms > 0:
                    _row(sub, sub_ms, indent="  ")
            if params:
                num_tokens = params.get("lm_num_tokens")
                lm_gen_time_s = params.get("lm_gen_time_s")
                n_tok = int(num_tokens) if num_tokens is not None else 0
                gen_s = float(lm_gen_time_s) if lm_gen_time_s is not None else 0.0
                if n_tok > 0 and gen_s > 0.0:
                    tps = float(n_tok) / gen_s
                    print(
                        f"[ace_step_v1_5][perf]     {'├─ LM throughput':<34s} {tps:10.1f}  " f"{'tokens/sec':>12s}",
                        flush=True,
                    )
        if name == "DiT Diffusion (total)" and ms > 0:
            for sub, sub_labels in (
                ("├─ Mask prep", frozenset({"dit_mask_prep"})),
                ("├─ Denoise loop", frozenset({"dit_denoise_loop"})),
            ):
                sub_ms = _sum(sub_labels)
                if sub_ms > 0:
                    _row(sub, sub_ms, indent="  ")

    _row("Other / Overhead", other_ms)
    if init_ms > 0:
        _row("One-time init (excluded above)", init_ms)
    print(f"[ace_step_v1_5][perf]   {'─' * 34} {'─' * 10}  {'─' * 12}", flush=True)
    print(
        f"[ace_step_v1_5][perf]   {'Wall Time (total)':<34s} {wall_s:10.2f}  {'100.0':>11s}%",
        flush=True,
    )
    rtf = ace_step_rtf_from_params(wall_ms=wall_ms, params=params)
    if rtf is not None and params is not None:
        duration_sec = float(params["duration_sec"])
        print(
            f"[ace_step_v1_5][perf]   {'RTF':<34s} {rtf:10.2f}×  "
            f"(audio {duration_sec:g}s / wall {wall_s:.2f}s; higher = faster)",
            flush=True,
        )
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    rtf_log = f" RTF={rtf:.2f}×" if rtf is not None else ""
    logger.info(
        "ACE-Step benchmark wall: {:.2f}s (LLM {:.2f}s DiT {:.2f}s VAE {:.2f}s){}",
        wall_s,
        llm_ms / 1000.0,
        dit_ms / 1000.0,
        vae_ms / 1000.0,
        rtf_log,
    )


def emit_session_summary(state: SessionPerfState, *, params: Optional[Dict[str, Any]] = None) -> None:
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
            _emit_rtf_metrics(wall_ms=last.total_ms, params=params)
            emit_key_metrics(
                last.modules_ms,
                wall_ms=last.total_ms,
                params=params,
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
    _ = recorder, num_steps
    return None


def log_denoise_step_summary(step_times_ms: List[float]) -> None:
    _ = step_times_ms
