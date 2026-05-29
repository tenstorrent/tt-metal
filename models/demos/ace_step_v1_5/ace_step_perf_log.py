# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock perf logging for ACE-Step v1.5 demos and E2E generate().

Enable module-level timing with either:

- **Default on** for all runs (single-card and mesh), or
- ``ACE_STEP_DEMO_PERF_LOG=0`` (or ``ACE_STEP_PERF_LOG=0``) to disable, or
- ``ACE_STEP_DEMO_PERF_LOG=1`` to force on explicitly.

Prints per-module wall times, KEY METRICS (LM / DiT / **VAE decode**), and ``RTF_per_step``.

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
        _emit_rtf_per_step(wall_ms=total_ms, params=self.params)
        module_timings = [pair for pair in self.timings_ms if pair[0] != label]
        emit_benchmark_wall_breakdown(
            module_timings,
            wall_ms=total_ms,
            params=self.params,
        )
        emit_key_metrics(
            module_timings,
            wall_ms=total_ms,
            params=self.params,
            show_tokens_per_sec=bool(self.params.get("llm_debug")),
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


def ace_step_rtf_per_step(
    *,
    wall_s: float,
    duration_sec: float,
    infer_steps: int,
) -> Optional[float]:
    """Real-time factor per diffusion step: wall / (audio_duration × infer_steps)."""
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


def _emit_rtf_per_step(*, wall_ms: float, params: Optional[Dict[str, Any]] = None) -> None:
    rtf = ace_step_rtf_per_step_from_params(wall_ms=wall_ms, params=params)
    if rtf is None:
        return
    duration_sec = float(params["duration_sec"])  # type: ignore[index]
    infer_steps = int(params["infer_steps"])  # type: ignore[index]
    print(
        f"[ace_step_v1_5][perf]   {'RTF_per_step':40s} {rtf:10.6f}  "
        f"(wall / ({duration_sec:g}s × {infer_steps} steps))",
        flush=True,
    )
    logger.info(
        "ACE-Step RTF_per_step: {:.6f} (wall {:.2f}s / ({}s × {} steps))",
        rtf,
        float(wall_ms) / 1000.0,
        duration_sec,
        infer_steps,
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
            lm_time_s = float(params["lm_gen_time_s"])
        except (TypeError, ValueError):
            pass

    metrics: Dict[str, Any] = {
        "wall_time_s": float(wall_ms) / 1000.0,
        "lm_total_time_s": lm_time_s,
        "dit_total_time_s": dit_ms / 1000.0,
        "vae_decode_time_s": vae_ms / 1000.0,
    }

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
    return metrics


def emit_key_metrics(
    timings_ms: List[Tuple[str, float]],
    *,
    wall_ms: float,
    params: Optional[Dict[str, Any]] = None,
    show_tokens_per_sec: bool = False,
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

    tps = metrics.get("tokens_per_sec")
    if show_tokens_per_sec and tps is not None:
        n_tok = metrics.get("lm_num_tokens", "?")
        rows.append(("Tokens/sec", f"{tps:.1f}", f"LLM throughput ({n_tok} new tokens)"))
    elif show_tokens_per_sec:
        rows.append(("Tokens/sec", "n/a", "LM token stats unavailable (was preprocess cached?)"))

    for name, value, desc in rows:
        print(f"[ace_step_v1_5][perf]   {name:<22s} {value:>12s}  {desc}", flush=True)

    print(f"[ace_step_v1_5][perf] {bar}", flush=True)

    log_extra = ""
    if show_tokens_per_sec and tps is not None:
        log_extra = f" tokens_per_sec={tps:.1f}"
    logger.info(
        "ACE-Step key metrics: wall={:.2f}s LM={:.2f}s DiT={:.2f}s VAE={:.2f}s{}",
        metrics["wall_time_s"],
        metrics["lm_total_time_s"],
        metrics["dit_total_time_s"],
        metrics["vae_decode_time_s"],
        log_extra,
    )


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
    rtf = ace_step_rtf_per_step_from_params(wall_ms=wall_ms, params=params)
    if rtf is not None:
        duration_sec = float(params["duration_sec"])  # type: ignore[index]
        infer_steps = int(params["infer_steps"])  # type: ignore[index]
        print(
            f"[ace_step_v1_5][perf]   {'RTF_per_step':<34s} {rtf:10.6f}  "
            f"(wall / ({duration_sec:g}s × {infer_steps} steps))",
            flush=True,
        )
    print(f"[ace_step_v1_5][perf] {bar}", flush=True)
    logger.info(
        "ACE-Step benchmark wall: {:.2f}s (LLM {:.2f}s DiT {:.2f}s VAE {:.2f}s){}",
        wall_s,
        llm_ms / 1000.0,
        dit_ms / 1000.0,
        vae_ms / 1000.0,
        f" RTF_per_step={rtf:.6f}" if rtf is not None else "",
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
            _emit_rtf_per_step(wall_ms=last.total_ms, params=params)
            emit_key_metrics(
                last.modules_ms,
                wall_ms=last.total_ms,
                params=params,
                show_tokens_per_sec=bool((params or {}).get("llm_debug")),
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
