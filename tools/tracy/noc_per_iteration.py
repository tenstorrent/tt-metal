#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Split a NoC trace into per-iteration slices and run tt-npe on each.

Blaze fused ops run as a single persistent program that loops N times internally, so a
whole-run NoC trace collapses every iteration into one averaged tt-npe result. This module
splits the trace on ``ITERATION`` zone boundaries so each iteration can be simulated on its
own.

Why this works without cross-referencing ``profile_log_device.csv``:

* The profiler interleaves zone markers and NoC events into the *same* flat JSON array, on
  the same clock, sorted by ``(sx,sy)`` then ``proc`` then ``timestamp``. The ``ITERATION``
  zone is therefore already sitting next to the NoC events it brackets.
* Zone markers survive tt-npe's fabric merge with ``src_device_id`` intact, so a merged
  multi-device trace can be sliced the same way a per-device one can.
* tt-npe rebases every workload to ``t0 = min(timestamp)`` and derives ``golden_cycles`` from
  the file's own min/max, so a sliced file simulates correctly with no timestamp fixups.

Iteration identity is the *occurrence index* of the zone on a given (device, core, risc), not
wall-clock time: cores drift apart in absolute cycles but stay in logical lockstep, so the
Nth ``ITERATION`` on every core is the same logical iteration.
"""

import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

ITERATION_ZONE = "ITERATION"

# Key identifying one independent instruction stream: a single RISC on a single core on a
# single device. Zone nesting is only well-defined within such a stream.
StreamKey = Tuple[Any, Any, Any, str]


@dataclass
class IterationWindow:
    """One occurrence of the target zone on one (device, core, risc) stream.

    ``truncated`` means the ZONE_END was never observed. Such a window is still bounded by the
    next window's start when one exists -- otherwise it would swallow every later iteration's
    events. Only a genuinely final unclosed window stays open-ended.
    """

    index: int
    start_ts: int
    end_ts: Optional[int] = None  # None => open-ended (final, unclosed)
    truncated: bool = False

    def contains(self, ts: int) -> bool:
        if ts < self.start_ts:
            return False
        return self.end_ts is None or ts <= self.end_ts


@dataclass
class SliceStats:
    """Bookkeeping so callers can tell truncation from genuine absence of traffic."""

    total_records: int = 0
    assigned_records: int = 0
    unassigned_records: int = 0
    unclosed_windows: int = 0
    streams: int = 0
    per_iteration_events: Dict[int, int] = field(default_factory=dict)


def stream_key(record: Dict[str, Any]) -> StreamKey:
    """Identify the (device, core, risc) stream a record belongs to."""
    return (record.get("src_device_id"), record.get("sx"), record.get("sy"), record.get("proc"))


def is_zone_marker(record: Dict[str, Any]) -> bool:
    return "zone" in record


def build_iteration_windows(
    records: Iterable[Dict[str, Any]],
    zone_name: str = ITERATION_ZONE,
) -> Tuple[Dict[StreamKey, List[IterationWindow]], int]:
    """Pair ZONE_START/ZONE_END of ``zone_name`` per stream, numbering occurrences from 0.

    Returns ``(windows_by_stream, unclosed_count)``.

    A ZONE_START arriving while one is already open is treated as the previous window ending
    unclosed rather than as nesting -- the persistent loop cannot legitimately nest, so this
    means markers were dropped (typically a mid-run profiler buffer flush).
    """
    by_stream: Dict[StreamKey, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if is_zone_marker(rec) and rec.get("zone") == zone_name:
            by_stream[stream_key(rec)].append(rec)

    windows: Dict[StreamKey, List[IterationWindow]] = {}
    unclosed = 0

    for key, markers in by_stream.items():
        markers.sort(key=lambda r: r["timestamp"])
        stream_windows: List[IterationWindow] = []
        open_window: Optional[IterationWindow] = None
        next_index = 0

        for marker in markers:
            phase = marker.get("zone_phase")
            ts = marker["timestamp"]

            if phase == "ZONE_START":
                if open_window is not None:
                    # Dropped ZONE_END; keep the window but flag it.
                    unclosed += 1
                    open_window.truncated = True
                    stream_windows.append(open_window)
                open_window = IterationWindow(index=next_index, start_ts=ts)
                next_index += 1
            elif phase == "ZONE_END":
                if open_window is None:
                    # Dropped ZONE_START; nothing to attach this end to.
                    continue
                open_window.end_ts = ts
                stream_windows.append(open_window)
                open_window = None

        if open_window is not None:
            # Loop terminated mid-iteration, or capture truncated.
            unclosed += 1
            open_window.truncated = True
            stream_windows.append(open_window)

        # Bound any truncated window by the next window's start, so a dropped ZONE_END cannot
        # absorb subsequent iterations' events. The final one legitimately stays open-ended.
        stream_windows.sort(key=lambda w: w.start_ts)
        for current, following in zip(stream_windows, stream_windows[1:]):
            if current.end_ts is None:
                current.end_ts = following.start_ts - 1

        if stream_windows:
            windows[key] = stream_windows

    return windows, unclosed


def slice_records_by_iteration(
    records: List[Dict[str, Any]],
    zone_name: str = ITERATION_ZONE,
    include_unclosed: bool = True,
) -> Tuple[Dict[int, List[Dict[str, Any]]], SliceStats]:
    """Bucket every record into the iteration window enclosing it on its own stream.

    Records on a stream that has no windows at all (e.g. a core that never ran the loop) are
    dropped -- they cannot be attributed to any iteration.
    """
    windows, unclosed = build_iteration_windows(records, zone_name)

    stats = SliceStats(total_records=len(records), unclosed_windows=unclosed, streams=len(windows))
    slices: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    # Windows on a stream are sorted and non-overlapping, so the owning window can be found
    # by binary search. A linear scan would be O(records x iterations), and num_iterations
    # runs to 100 in the test matrix.
    starts = {key: [w.start_ts for w in ws] for key, ws in windows.items()}

    for rec in records:
        key = stream_key(rec)
        stream_windows = windows.get(key)
        if not stream_windows:
            stats.unassigned_records += 1
            continue

        ts = rec["timestamp"]
        idx = bisect.bisect_right(starts[key], ts) - 1
        window = stream_windows[idx] if idx >= 0 else None

        if window is None or not window.contains(ts):
            # Falls between iterations (loop epilogue/prologue) or outside the loop entirely.
            stats.unassigned_records += 1
        elif window.truncated and not include_unclosed:
            stats.unassigned_records += 1
        else:
            slices[window.index].append(rec)
            stats.assigned_records += 1

    for idx, recs in slices.items():
        stats.per_iteration_events[idx] = sum(1 for r in recs if not is_zone_marker(r))

    return dict(slices), stats


def write_iteration_slices(
    slices: Dict[int, List[Dict[str, Any]]],
    output_dir: Path,
    device_id: int = 0,
    op_name: str = "iter",
    program_runtime_id: int = 0,
) -> Dict[int, Path]:
    """Write one JSON file per iteration, named so tt-npe's filename parser accepts it.

    tt-npe matches ``noc_trace_dev(\\d+)_((\\w*)_)?ID(\\d+)(_traceID(\\d+))?\\.json``; ``\\w*``
    permits underscores, so the iteration index rides along inside the op-name field.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[int, Path] = {}

    for idx in sorted(slices):
        records = sorted(slices[idx], key=lambda r: (r.get("sx"), r.get("sy"), str(r.get("proc")), r["timestamp"]))
        path = output_dir / f"noc_trace_dev{device_id}_{op_name}{idx}_ID{program_runtime_id}.json"
        with open(path, "w") as f:
            json.dump(records, f)
        written[idx] = path

    return written


def _clean(value: Any) -> Optional[float]:
    """tt-npe returns NaN for averages over zero events; surface that as absent, not 'nan'."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def summarize_npe_result(
    result: Any,
    devices_present: Optional[Iterable[int]] = None,
) -> Dict[int, Dict[str, Optional[float]]]:
    """Flatten a tt-npe Stats object into ``{device_id: {metric: value|None}}``.

    Metrics live on ``Stats.per_device_stats`` (a dict); device id -1 is the aggregate
    pseudo-device and is skipped.

    tt-npe reports a row for every device in the topology, including ones with no events in
    this slice. Those rows are meaningless -- their ``golden_cycles`` is
    ``numeric_limits<Cycle>::max()`` underflowed to ~1.8e19. Pass ``devices_present`` (the
    device ids actually appearing in the slice) to drop them.
    """
    out: Dict[int, Dict[str, Optional[float]]] = {}
    per_device = getattr(result, "per_device_stats", None)
    if not per_device:
        return out

    allowed = None if devices_present is None else set(devices_present)

    for device_id, stats in per_device.items():
        if device_id == -1:
            continue
        if allowed is not None and device_id not in allowed:
            continue
        out[device_id] = {
            "noc_util": _clean(getattr(stats, "overall_avg_link_util", None)),
            "mcast_noc_util": _clean(getattr(stats, "overall_avg_mcast_write_link_util", None)),
            "dram_bw_util": _clean(getattr(stats, "dram_bw_util", None)),
            "congestion_impact": _clean(_call(stats, "getCongestionImpact")),
            "golden_cycles": _clean(getattr(stats, "golden_cycles", None)),
        }
    return out


def _call(obj: Any, name: str) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def import_npe():
    """Import tt-npe lazily, mirroring process_ops_logs.analyzeNoCTraces' soft-failure.

    tt-npe is a separate repo, not vendored in tt-metal. It must be built and its install
    dirs placed on PYTHONPATH (``source tt-npe/ENV_SETUP``). NOTE: npe main rejects
    ``FABRIC_2D_TORUS_X``, which blaze fused-op tests use -- the ``snadeem/blaze_layer_support_v2``
    branch is required.
    """
    try:
        import tt_npe_pybind as npe
        from fabric_post_process import TopologyGraph

        return npe, TopologyGraph
    except ImportError as e:
        logger.warning(
            "Could not import tt-npe; NoC/DRAM metrics will be omitted. "
            "Build tt-npe (branch snadeem/blaze_layer_support_v2) and source its ENV_SETUP. "
            f"({e})"
        )
        return None, None


def run_npe_on_slice(
    npe,
    workload_file: Path,
    device_name: str,
    topology_json: str,
    cycles_per_timestep: int = 32,
) -> Optional[Any]:
    """Simulate one sliced trace. Returns a tt-npe Stats object, or None on failure.

    Deliberately does not emit timeline/viz files: they are large, unnecessary here, and hit a
    crash on empty timestep stats in some npe builds.
    """
    cfg = npe.Config()
    cfg.device_name = device_name
    cfg.workload_json_filepath = str(workload_file)
    cfg.cycles_per_timestep = cycles_per_timestep
    cfg.workload_is_noc_trace = True
    cfg.topology_json = topology_json
    cfg.set_verbosity_level(0)

    try:
        workload = npe.createWorkloadFromJSON(cfg.workload_json_filepath, cfg.device_name, is_noc_trace_format=True)
        if workload is None:
            logger.warning(f"tt-npe could not build a workload from {workload_file.name}; skipping")
            return None
        api = npe.InitAPI(cfg)
        if api is None:
            logger.warning("tt-npe InitAPI failed; skipping")
            return None
        return api.runNPE(workload)
    except Exception as e:
        logger.warning(f"tt-npe failed on {workload_file.name}: {e!r}")
        return None


def analyze_noc_per_iteration(
    merged_trace: Path,
    topology_json: Path,
    output_dir: Path,
    zone_name: str = ITERATION_ZONE,
    program_runtime_id: int = 0,
) -> Dict[Tuple[int, int], Dict[str, Optional[float]]]:
    """Slice a merged NoC trace by iteration and simulate each slice.

    Returns ``{(device_id, iteration): {metric: value|None}}``, ready to join onto the
    per-iteration perf-counter rows on the same key.
    """
    npe, TopologyGraph = import_npe()
    if npe is None:
        return {}

    # npe's device_name is the topology's cluster_type (e.g. "P150_X8") -- never hardcode it.
    try:
        device_name = TopologyGraph(str(topology_json)).cluster_type
    except Exception as e:
        logger.error(f"Could not read topology '{topology_json}': {e!r}")
        return {}

    with open(merged_trace) as f:
        records = json.load(f)

    slices, stats = slice_records_by_iteration(records, zone_name=zone_name)
    log_slice_summary(stats, zone_name)
    if not slices:
        logger.warning(f"No '{zone_name}' zones found in {merged_trace.name}; no NoC metrics produced")
        return {}

    slice_dir = output_dir / "iteration_slices"
    written = write_iteration_slices(
        slices, slice_dir, op_name=f"{zone_name.lower()}", program_runtime_id=program_runtime_id
    )

    results: Dict[Tuple[int, int], Dict[str, Optional[float]]] = {}
    for iteration, path in sorted(written.items()):
        if stats.per_iteration_events.get(iteration, 0) == 0:
            logger.info(f"iteration {iteration}: no NoC events, skipping simulation")
            continue
        logger.info(f"iteration {iteration}: simulating {stats.per_iteration_events[iteration]:,} NoC events")
        result = run_npe_on_slice(npe, path, device_name, str(topology_json))
        if result is None:
            continue
        # Only keep devices that actually contributed NoC events to this slice.
        devices_present = {r.get("src_device_id") for r in slices[iteration] if not is_zone_marker(r)} - {None}
        for device_id, metrics in summarize_npe_result(result, devices_present).items():
            results[(device_id, iteration)] = metrics

    return results


def log_slice_summary(stats: SliceStats, zone_name: str = ITERATION_ZONE) -> None:
    logger.info(
        f"Sliced on '{zone_name}': {stats.streams} streams, "
        f"{len(stats.per_iteration_events)} iterations, "
        f"{stats.assigned_records:,}/{stats.total_records:,} records assigned "
        f"({stats.unassigned_records:,} outside any iteration)"
    )
    if stats.unclosed_windows:
        logger.warning(
            f"{stats.unclosed_windows} iteration zone(s) never closed - capture truncated or "
            f"profiler buffer flushed mid-iteration. Those slices may be short."
        )
    empty = [i for i, n in sorted(stats.per_iteration_events.items()) if n == 0]
    if empty:
        logger.warning(f"Iterations with zero NoC events (metrics will be blank): {empty}")
