# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Diagnostic timing utilities for finding hidden time sinks in pipelines.

This module provides comprehensive timing instrumentation to identify
where time is being spent in the pipeline, including:
- Wall clock vs accounted time verification
- Sync vs enqueue timing for device operations
- Detailed region breakdowns
- Execution count verification
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class DiagnosticTimingData:
    """Extended timing data with detailed region breakdowns."""

    # Original timing fields for compatibility
    clip_encoding_time: float = 0.0
    t5_encoding_time: float = 0.0
    total_encoding_time: float = 0.0
    denoising_step_times: List[float] = field(default_factory=list)
    vae_decoding_time: float = 0.0
    total_time: float = 0.0

    # New diagnostic fields for finding hidden time
    encoder_reload_time: float = 0.0
    encoder_deallocate_time: float = 0.0
    transformer_load_time: float = 0.0
    scheduler_init_time: float = 0.0
    latents_init_time: float = 0.0
    rope_init_time: float = 0.0
    tensor_transfer_time: float = 0.0
    trace_capture_time: float = 0.0
    trace_execute_time: float = 0.0
    cfg_combine_time: float = 0.0
    transformer_deallocate_time: float = 0.0
    vae_reload_time: float = 0.0
    vae_gather_time: float = 0.0
    vae_readback_time: float = 0.0
    vae_unpatchify_time: float = 0.0
    postprocess_time: float = 0.0
    pil_convert_time: float = 0.0

    # Sync-specific timing (enqueue vs actual execution)
    denoising_enqueue_times: List[float] = field(default_factory=list)
    denoising_sync_times: List[float] = field(default_factory=list)

    # Per-step breakdowns
    step_inner_times: List[float] = field(default_factory=list)
    step_cfg_times: List[float] = field(default_factory=list)
    step_tensor_copy_times: List[float] = field(default_factory=list)
    step_sync_times: List[float] = field(default_factory=list)

    # Execution counts for verification
    pipeline_call_count: int = 0
    denoising_loop_count: int = 0
    vae_decode_count: int = 0
    trace_execute_count: int = 0
    device_sync_count: int = 0

    # Additional diagnostic info
    region_times: Dict[str, float] = field(default_factory=dict)
    region_counts: Dict[str, int] = field(default_factory=dict)


class DiagnosticTimingCollector:
    """
    Enhanced timing collector that tracks all regions and identifies hidden time.

    Usage:
        timer = DiagnosticTimingCollector()
        with timer.time_section("region_name"):
            # code to time

        # Get breakdown
        data = timer.get_diagnostic_data()
        timer.print_breakdown()
    """

    def __init__(self, enable_sync_timing: bool = True, enable_profiler: bool = False):
        self.timings: Dict[str, float] = {}
        self.step_timings: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}
        self._enable_sync_timing = enable_sync_timing
        self._enable_profiler = enable_profiler
        self._profiler: Optional[cProfile.Profile] = None
        self._profile_stats: Optional[str] = None
        self._active_sections: List[str] = []

        # Perf counter for higher resolution
        self._wall_start: Optional[float] = None
        self._wall_end: Optional[float] = None

    def reset(self) -> "DiagnosticTimingCollector":
        """Reset all timings for a new run."""
        self.timings = {}
        self.step_timings = {}
        self.counts = {}
        self._profile_stats = None
        self._wall_start = None
        self._wall_end = None
        return self

    @contextmanager
    def time_section(self, name: str) -> Generator[None, None, None]:
        """Time a named section (accumulates if called multiple times)."""
        start = time.perf_counter()
        self._active_sections.append(name)
        try:
            yield
        finally:
            self._active_sections.pop()
            elapsed = time.perf_counter() - start
            if name in self.timings:
                self.timings[name] += elapsed
            else:
                self.timings[name] = elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    @contextmanager
    def time_step(self, name: str) -> Generator[None, None, None]:
        """Time a step (appends to a list for per-step analysis)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.step_timings:
                self.step_timings[name] = []
            self.step_timings[name].append(elapsed)

    def record_time(self, name: str, elapsed: float) -> None:
        """Manually record a time measurement."""
        if name in self.timings:
            self.timings[name] += elapsed
        else:
            self.timings[name] = elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

    def record_step_time(self, name: str, elapsed: float) -> None:
        """Manually record a step time."""
        if name not in self.step_timings:
            self.step_timings[name] = []
        self.step_timings[name].append(elapsed)

    def increment_count(self, name: str, amount: int = 1) -> None:
        """Increment a counter for verification."""
        self.counts[name] = self.counts.get(name, 0) + amount

    def start_wall_clock(self) -> None:
        """Start the wall clock timer."""
        self._wall_start = time.perf_counter()

    def stop_wall_clock(self) -> None:
        """Stop the wall clock timer."""
        self._wall_end = time.perf_counter()

    @property
    def wall_clock_time(self) -> float:
        """Get the wall clock time."""
        if self._wall_start is None or self._wall_end is None:
            return 0.0
        return self._wall_end - self._wall_start

    @contextmanager
    def profile_section(self) -> Generator[None, None, None]:
        """Profile a section with cProfile."""
        if not self._enable_profiler:
            yield
            return

        self._profiler = cProfile.Profile()
        self._profiler.enable()
        try:
            yield
        finally:
            self._profiler.disable()
            # Capture stats
            s = io.StringIO()
            ps = pstats.Stats(self._profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats(50)  # Top 50 functions
            self._profile_stats = s.getvalue()

    def get_profile_stats(self) -> Optional[str]:
        """Get the profiler output."""
        return self._profile_stats

    def get_timing_data(self) -> DiagnosticTimingData:
        """Get timing data in the format compatible with original TimingData."""
        return DiagnosticTimingData(
            # Original fields
            clip_encoding_time=self.timings.get("clip_encoding", 0.0),
            t5_encoding_time=self.timings.get("t5_encoding", 0.0),
            total_encoding_time=self.timings.get("total_encoding", 0.0),
            denoising_step_times=self.step_timings.get("denoising_step", []),
            vae_decoding_time=self.timings.get("vae_decoding", 0.0),
            total_time=self.timings.get("total", 0.0),
            # Diagnostic fields
            encoder_reload_time=self.timings.get("encoder_reload", 0.0),
            encoder_deallocate_time=self.timings.get("encoder_deallocate", 0.0),
            transformer_load_time=self.timings.get("transformer_load", 0.0),
            scheduler_init_time=self.timings.get("scheduler_init", 0.0),
            latents_init_time=self.timings.get("latents_init", 0.0),
            rope_init_time=self.timings.get("rope_init", 0.0),
            tensor_transfer_time=self.timings.get("tensor_transfer", 0.0),
            trace_capture_time=self.timings.get("trace_capture", 0.0),
            trace_execute_time=self.timings.get("trace_execute", 0.0),
            cfg_combine_time=self.timings.get("cfg_combine", 0.0),
            transformer_deallocate_time=self.timings.get("transformer_deallocate", 0.0),
            vae_reload_time=self.timings.get("vae_reload", 0.0),
            vae_gather_time=self.timings.get("vae_gather", 0.0),
            vae_readback_time=self.timings.get("vae_readback", 0.0),
            vae_unpatchify_time=self.timings.get("vae_unpatchify", 0.0),
            postprocess_time=self.timings.get("postprocess", 0.0),
            pil_convert_time=self.timings.get("pil_convert", 0.0),
            # Sync-specific timing
            denoising_enqueue_times=self.step_timings.get("denoising_enqueue", []),
            denoising_sync_times=self.step_timings.get("denoising_sync", []),
            # Per-step breakdowns
            step_inner_times=self.step_timings.get("step_inner", []),
            step_cfg_times=self.step_timings.get("step_cfg", []),
            step_tensor_copy_times=self.step_timings.get("step_tensor_copy", []),
            step_sync_times=self.step_timings.get("step_sync", []),
            # Execution counts
            pipeline_call_count=self.counts.get("pipeline_call", 0),
            denoising_loop_count=self.counts.get("denoising_loop", 0),
            vae_decode_count=self.counts.get("vae_decode", 0),
            trace_execute_count=self.counts.get("trace_execute", 0),
            device_sync_count=self.counts.get("device_sync", 0),
            # Additional info
            region_times=dict(self.timings),
            region_counts=dict(self.counts),
        )

    def get_accounted_time(self) -> float:
        """Calculate total accounted time from major regions."""
        # Sum up major non-overlapping regions
        major_regions = [
            # Encoder/transformer/VAE weight management
            "encoder_reload",
            "encoder_deallocate",
            "transformer_load",
            "transformer_deallocate",
            "vae_reload",
            "vae_deallocate",
            # Encoding
            "total_encoding",
            # Prep
            "scheduler_init",
            "latents_init",
            "rope_init",
            "tensor_transfer",
            # Denoising overhead
            "trace_capture",
            # VAE decode (wrapper includes sub-regions)
            "vae_decoding",
        ]

        # For denoising, sum up per-step times (includes step_cfg which is the hidden sync)
        denoising_total = sum(self.step_timings.get("denoising_step", []))

        region_total = sum(self.timings.get(r, 0.0) for r in major_regions)

        return region_total + denoising_total

    def get_other_time(self) -> float:
        """Calculate unaccounted 'OTHER' time."""
        total = self.timings.get("total", 0.0)
        accounted = self.get_accounted_time()
        return total - accounted

    def print_breakdown(self) -> None:
        """Print a detailed timing breakdown table."""
        total = self.timings.get("total", 0.0)
        if total == 0:
            total = 1.0  # Avoid division by zero

        print("\n" + "=" * 100)
        print("DIAGNOSTIC TIMING BREAKDOWN")
        print("=" * 100)

        # Header
        print(f"{'Region':<45} | {'Time (s)':>12} | {'Count':>8} | {'% of Total':>10}")
        print("-" * 100)

        # Sort regions by time descending
        sorted_regions = sorted(self.timings.items(), key=lambda x: -x[1])

        for name, elapsed in sorted_regions:
            count = self.counts.get(name, 1)
            pct = (elapsed / total) * 100
            print(f"{name:<45} | {elapsed:>12.4f} | {count:>8} | {pct:>9.1f}%")

        # Step timings summary
        if self.step_timings:
            print("-" * 100)
            print("STEP TIMINGS (per-step)")
            print("-" * 100)
            for name, times in self.step_timings.items():
                if times:
                    avg = sum(times) / len(times)
                    total_step = sum(times)
                    pct = (total_step / total) * 100
                    print(
                        f"{name:<35} | Total: {total_step:>8.4f}s | Steps: {len(times):>4} | Avg: {avg:>8.4f}s | {pct:>5.1f}%"
                    )

        # Summary
        print("=" * 100)
        accounted = self.get_accounted_time()
        other = total - accounted

        print(f"{'Total Time':<45} | {total:>12.4f}s")
        print(f"{'Accounted Time':<45} | {accounted:>12.4f}s | {(accounted/total)*100:>9.1f}%")
        print(f"{'OTHER (unaccounted)':<45} | {other:>12.4f}s | {(other/total)*100:>9.1f}%")
        print("=" * 100)

        # Execution counts
        if self.counts:
            print("\nEXECUTION COUNTS:")
            for name, count in sorted(self.counts.items()):
                if not name.endswith("_time"):
                    print(f"  {name}: {count}")

        # Profile stats
        if self._profile_stats:
            print("\n" + "=" * 100)
            print("CPROFILE TOP FUNCTIONS (by cumulative time)")
            print("=" * 100)
            print(self._profile_stats)

    def get_breakdown_dict(self) -> Dict[str, float]:
        """Get breakdown as a dictionary for programmatic access."""
        total = self.timings.get("total", 0.0)
        accounted = self.get_accounted_time()
        other = total - accounted

        result = dict(self.timings)
        result["_accounted"] = accounted
        result["_other"] = other
        result["_total"] = total

        # Add step totals
        for name, times in self.step_timings.items():
            result[f"{name}_total"] = sum(times)
            result[f"{name}_count"] = len(times)
            if times:
                result[f"{name}_avg"] = sum(times) / len(times)

        return result


def time_with_sync(device, name: str, timer: Optional[DiagnosticTimingCollector] = None):
    """
    Context manager that times a region with explicit device sync.

    This separates 'enqueue' time from 'sync' time to detect hidden sync costs.

    Usage:
        with time_with_sync(device, "denoising_step", timer) as sync_timer:
            # code to time
            sync_timer.mark_enqueue_done()  # Optional: mark when enqueuing is done
    """
    return _SyncTimer(device, name, timer)


class _SyncTimer:
    """Helper class for time_with_sync context manager."""

    def __init__(self, device, name: str, timer: Optional[DiagnosticTimingCollector]):
        self._device = device
        self._name = name
        self._timer = timer
        self._start: float = 0.0
        self._enqueue_done: Optional[float] = None

    def __enter__(self) -> "_SyncTimer":
        self._start = time.perf_counter()
        return self

    def mark_enqueue_done(self) -> None:
        """Mark when enqueueing is complete (before sync)."""
        self._enqueue_done = time.perf_counter()

    def __exit__(self, *args) -> None:
        import ttnn

        # Sync the device
        sync_start = time.perf_counter()
        ttnn.synchronize_device(self._device)
        sync_end = time.perf_counter()

        total = sync_end - self._start
        sync_time = sync_end - sync_start

        if self._enqueue_done is not None:
            enqueue_time = self._enqueue_done - self._start
        else:
            enqueue_time = total - sync_time

        if self._timer:
            self._timer.record_step_time(f"{self._name}_total", total)
            self._timer.record_step_time(f"{self._name}_enqueue", enqueue_time)
            self._timer.record_step_time(f"{self._name}_sync", sync_time)
            self._timer.increment_count("device_sync")
