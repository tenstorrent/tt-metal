#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Generalized Fused-Kernel Micro-Op Profiler

Produces accurate per-micro-op execution time breakdowns for fused kernels,
using type-aware timing formulas and a YAML config that describes the kernel's
micro-ops, RISC roles, core roles, and inter-op dependencies.

The tool parses profile_log_device.csv (from the TT device profiler) and applies
formulas tailored to each micro-op type:
  - compute: max across cores of TRISC pipeline wall clock
  - mcast:   sender_duration + max(receiver_durations)
  - gather:  max(sender_durations) + receiver_duration
  - gather_reduce: max(senders) + receiver + reducer_compute
  - broadcast: sender + max(receivers)
  - dram_write: max(writer_durations)
  - span: max(end) - min(start) across all participants

Usage:
    python fused_kernel_profiler.py --kernel pre_sdpa --log profile_log_device.csv
    python fused_kernel_profiler.py --config my_kernel.yaml --log profile_log_device.csv --json
"""

from __future__ import annotations

import argparse
import csv
import enum
import itertools
import json
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ============================================================================
# Config Schema
# ============================================================================


class OpType(enum.Enum):
    COMPUTE = "compute"
    MCAST = "mcast"
    GATHER = "gather"
    GATHER_REDUCE = "gather_reduce"
    BROADCAST = "broadcast"
    DRAM_WRITE = "dram_write"
    DRAM_READ = "dram_read"
    SPAN = "span"


@dataclass
class Participant:
    role: str
    risc: str
    function: str


@dataclass
class CoreRoleConfig:
    infer_from_zone: str


@dataclass
class MicroOpConfig:
    zone: str
    type: OpType
    participants: list[Participant]


@dataclass
class KernelConfig:
    kernel_name: str
    core_roles: dict[str, CoreRoleConfig]
    micro_ops: list[MicroOpConfig]
    dependencies: list[tuple[str, str]]
    display_order: list[str]

    @property
    def allowed_zones(self) -> set[str]:
        return {op.zone for op in self.micro_ops}

    def get_micro_op(self, zone: str) -> MicroOpConfig | None:
        for op in self.micro_ops:
            if op.zone == zone:
                return op
        return None


# ============================================================================
# Config Loader
# ============================================================================

_CONFIG_DIR = Path(__file__).parent / "kernel_configs"


def load_kernel_config(path_or_name: str | Path) -> KernelConfig:
    """Load kernel config from a YAML file or a built-in kernel name."""
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")

    path = Path(path_or_name)
    if not path.exists():
        path = _CONFIG_DIR / f"{path_or_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path_or_name}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    core_roles = {
        name: CoreRoleConfig(infer_from_zone=cfg["infer_from_zone"]) for name, cfg in raw.get("core_roles", {}).items()
    }

    micro_ops = []
    for op_raw in raw.get("micro_ops", []):
        participants = [
            Participant(role=p["role"], risc=p["risc"], function=p["function"]) for p in op_raw.get("participants", [])
        ]
        micro_ops.append(
            MicroOpConfig(
                zone=op_raw["zone"],
                type=OpType(op_raw["type"]),
                participants=participants,
            )
        )

    dependencies = [(d[0], d[1]) for d in raw.get("dependencies", [])]
    display_order = raw.get("display_order", [op.zone for op in micro_ops])

    config = KernelConfig(
        kernel_name=raw["kernel_name"],
        core_roles=core_roles,
        micro_ops=micro_ops,
        dependencies=dependencies,
        display_order=display_order,
    )
    _validate_config(config)
    return config


def _validate_config(config: KernelConfig) -> None:
    zone_set = config.allowed_zones
    for child, parent in config.dependencies:
        if child not in zone_set:
            raise ValueError(f"Dependency child '{child}' not in micro_ops")
        if parent not in zone_set:
            raise ValueError(f"Dependency parent '{parent}' not in micro_ops")
    for op in config.micro_ops:
        for p in op.participants:
            if p.role not in config.core_roles:
                raise ValueError(f"Participant role '{p.role}' in op '{op.zone}' not in core_roles")


# ============================================================================
# Zone Interval
# ============================================================================


@dataclass
class ZoneInterval:
    """A single zone execution: start and end cycles for one (core, risc) pair."""

    chip_id: int
    core: tuple[int, int]
    risc: str
    zone_name: str
    start_cycle: int
    end_cycle: int
    timer_id: int

    @property
    def duration_cycles(self) -> int:
        return self.end_cycle - self.start_cycle


# ============================================================================
# Log Parser (generalized)
# ============================================================================


def extract_chip_freq_mhz(log_path: Path) -> float:
    """Extract chip frequency from first line of device log."""
    with log_path.open() as f:
        first = f.readline()
    m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+\.?\d*)", first or "")
    return float(m.group(1)) if m else 1000.0


def cycles_to_ns(cycles: int, freq_mhz: float) -> float:
    """Convert device cycles to nanoseconds."""
    return (cycles * 1000.0) / freq_mhz


def parse_device_log(
    log_path: Path,
    allowed_zones: set[str] | None = None,
    verbose: bool = False,
) -> tuple[list[ZoneInterval], float]:
    """
    Parse profile_log_device.csv and extract zone intervals.

    If allowed_zones is provided, only matching zone names are kept.
    """
    freq_mhz = extract_chip_freq_mhz(log_path)
    starts: dict[tuple, tuple[int, str]] = {}
    intervals: list[ZoneInterval] = []

    def _zone_ok(name: str) -> bool:
        if allowed_zones is None:
            return True
        return name in allowed_zones

    with log_path.open(newline="") as f:
        _ = f.readline()  # ARCH / CHIP_FREQ line
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return intervals, freq_mhz

        field_map: dict[str, str] = {}
        for col in header:
            n = (col or "").strip()
            if n:
                key = n.lower().replace(" ", "_").replace("[", "").replace("]", "")
                field_map[key] = n

        use_dict = "zone" in " ".join(header).lower() or "zone_name" in field_map

        if use_dict:
            f.seek(0)
            _ = f.readline()
            dict_reader = csv.DictReader(f, skipinitialspace=True)
            for row in dict_reader:
                _parse_row_dict(row, field_map, starts, intervals, _zone_ok)
        else:
            for row in itertools.chain([header], reader):
                if len(row) >= 13:
                    _parse_row_positional(tuple(row), starts, intervals, _zone_ok)

    if verbose:
        print(f"Parsed {len(intervals)} zone intervals, freq={freq_mhz} MHz", file=sys.stderr)

    return intervals, freq_mhz


def _parse_row_dict(
    row: dict,
    field_map: dict,
    starts: dict,
    intervals: list,
    zone_ok: Any,
) -> None:
    def get(*candidates: str) -> str | None:
        for c in candidates:
            key = c.lower().replace(" ", "_").replace("[", "").replace("]", "")
            if key in field_map:
                val = row.get(field_map[key], "")
                return str(val).strip() if val is not None else ""
        return None

    chip_s = get("PCIe slot", "chip_id", "chip")
    cx_s = get("core_x", "corex")
    cy_s = get("core_y", "corey")
    risc_s = get("RISC processor type", "risc")
    timer_s = get("timer_id")
    cycles_s = get("time[cycles since reset]", "time", "cycles")
    zone_s = get("zone name", "zone_name")
    phase_s = get("zone phase", "type", "phase")

    if not all([chip_s, cx_s, cy_s, risc_s, timer_s, cycles_s, zone_s, phase_s]):
        return
    try:
        chip_id = int(chip_s)
        core_x = int(cx_s)
        core_y = int(cy_s)
        cycles = int(float(cycles_s))
        timer_id = int(timer_s)
    except (ValueError, TypeError):
        return

    zone_name = (zone_s or "").strip()
    phase = (phase_s or "").strip().upper()
    is_start = phase in ("BEGIN", "ZONE_START", "START")
    is_end = phase in ("END", "ZONE_END", "END ")
    if not is_start and not is_end:
        return

    key = (chip_id, (core_x, core_y), risc_s.strip(), timer_id)
    if is_start:
        starts[key] = (cycles, zone_name)
    elif is_end and key in starts:
        start_cycle, start_zone = starts.pop(key)
        name = start_zone or zone_name
        if zone_ok(name):
            intervals.append(
                ZoneInterval(
                    chip_id=chip_id,
                    core=(core_x, core_y),
                    risc=risc_s.strip(),
                    zone_name=name,
                    start_cycle=start_cycle,
                    end_cycle=cycles,
                    timer_id=timer_id,
                )
            )


def _parse_row_positional(
    row: tuple,
    starts: dict,
    intervals: list,
    zone_ok: Any,
) -> None:
    if len(row) < 13:
        return
    try:
        chip_id = int(row[1])
        core_x = int(row[2])
        core_y = int(row[3])
        timer_id = int(float(row[5])) if str(row[5]).strip() else 0
        cycles = int(float(row[6]))
    except (ValueError, TypeError):
        return
    risc_s = str(row[4]).strip()
    zone_name = str(row[11]).strip()
    phase = str(row[12]).strip().upper()
    is_start = phase in ("BEGIN", "ZONE_START", "START")
    is_end = phase in ("END", "ZONE_END", "END ")
    if not is_start and not is_end:
        return

    key = (chip_id, (core_x, core_y), risc_s, timer_id)
    if is_start:
        starts[key] = (cycles, zone_name)
    elif is_end and key in starts:
        start_cycle, start_zone = starts.pop(key)
        name = start_zone or zone_name
        if zone_ok(name):
            intervals.append(
                ZoneInterval(
                    chip_id=chip_id,
                    core=(core_x, core_y),
                    risc=risc_s,
                    zone_name=name,
                    start_cycle=start_cycle,
                    end_cycle=cycles,
                    timer_id=timer_id,
                )
            )


# ============================================================================
# Interval Classifier
# ============================================================================


@dataclass
class ClassifiedInterval:
    interval: ZoneInterval
    micro_op_zone: str
    function: str
    core_role: str


def risc_matches(config_risc: str, csv_risc: str) -> bool:
    """Check if a config RISC spec matches a CSV RISC value."""
    if config_risc == csv_risc:
        return True
    if config_risc == "TRISC" and csv_risc.startswith("TRISC"):
        return True
    return False


NOOP_THRESHOLD_RATIO = 0.01


def infer_core_roles(
    intervals: list[ZoneInterval],
    config: KernelConfig,
) -> dict[str, set[tuple[int, int]]]:
    """
    Infer which cores belong to which role using config's infer_from_zone.

    Unified kernels fire DeviceZoneScopedN on ALL cores (including no-op cores),
    so we filter out cores whose max duration for the inference zone is below
    1% of the global max for that zone.
    """
    by_zone_core: dict[str, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    for iv in intervals:
        dur = iv.duration_cycles
        prev = by_zone_core[iv.zone_name][iv.core]
        if dur > prev:
            by_zone_core[iv.zone_name][iv.core] = dur

    roles: dict[str, set[tuple[int, int]]] = {}
    for role_name, role_cfg in config.core_roles.items():
        zone = role_cfg.infer_from_zone
        core_max = by_zone_core.get(zone, {})
        if not core_max:
            roles[role_name] = set()
            continue
        global_max = max(core_max.values())
        threshold = global_max * NOOP_THRESHOLD_RATIO
        roles[role_name] = {core for core, dur in core_max.items() if dur > threshold}
    return roles


def classify_intervals(
    intervals: list[ZoneInterval],
    config: KernelConfig,
    core_roles: dict[str, set[tuple[int, int]]],
) -> dict[str, list[ClassifiedInterval]]:
    """Classify each interval into (micro_op, participant_function)."""
    core_to_roles: dict[tuple[int, int], set[str]] = defaultdict(set)
    for role_name, cores in core_roles.items():
        for core in cores:
            core_to_roles[core].add(role_name)

    classified: dict[str, list[ClassifiedInterval]] = defaultdict(list)
    for iv in intervals:
        micro_op = config.get_micro_op(iv.zone_name)
        if micro_op is None:
            continue

        matched = False
        for participant in micro_op.participants:
            if participant.role in core_to_roles.get(iv.core, set()) and risc_matches(participant.risc, iv.risc):
                classified[iv.zone_name].append(
                    ClassifiedInterval(
                        interval=iv,
                        micro_op_zone=iv.zone_name,
                        function=participant.function,
                        core_role=participant.role,
                    )
                )
                matched = True
                break

        if not matched:
            classified[iv.zone_name].append(
                ClassifiedInterval(
                    interval=iv,
                    micro_op_zone=iv.zone_name,
                    function="unknown",
                    core_role="unknown",
                )
            )

    return classified


# ============================================================================
# Predecessor Gate Computation
# ============================================================================


def compute_predecessor_gates(
    zone: str,
    config: KernelConfig,
    all_classified: dict[str, list[ClassifiedInterval]],
) -> dict[tuple[int, int], int]:
    """
    For a given op zone, compute per-core gate times from its dependency parents.

    The gate time on a core is the latest end_cycle from any predecessor's
    participant that shares the same core role (and thus the same physical core).
    This represents when predecessor output becomes available on that core,
    allowing subtraction of blocking wait time from the current op's duration.

    For predecessors on a different core type (no shared role), use the op-level
    predecessor end as the gate for all child cores.
    """
    micro_op = config.get_micro_op(zone)
    if micro_op is None:
        return {}

    parent_zones = [parent for child, parent in config.dependencies if child == zone]
    if not parent_zones:
        return {}

    child_roles = {p.role for p in micro_op.participants}
    gates: dict[tuple[int, int], int] = {}

    for parent_zone in parent_zones:
        parent_op = config.get_micro_op(parent_zone)
        if parent_op is None:
            continue

        parent_roles = {p.role for p in parent_op.participants}
        shared_roles = child_roles & parent_roles

        parent_classified = all_classified.get(parent_zone, [])
        if not parent_classified:
            continue

        if shared_roles:
            for ci in parent_classified:
                if ci.core_role in shared_roles:
                    core = ci.interval.core
                    gates[core] = max(gates.get(core, 0), ci.interval.end_cycle)
        else:
            parent_end = max(ci.interval.end_cycle for ci in parent_classified)
            child_classified = all_classified.get(zone, [])
            for ci in child_classified:
                core = ci.interval.core
                gates[core] = max(gates.get(core, 0), parent_end)

    return gates


# ============================================================================
# Timing Engine
# ============================================================================


@dataclass
class OpTimingResult:
    zone: str
    duration_cycles: int
    duration_ns: float
    duration_us: float
    core_count: int
    interval_count: int
    breakdown: dict[str, int]
    raw_duration_cycles: int = 0
    raw_duration_us: float = 0.0
    blocking_cycles: int = 0
    blocking_us: float = 0.0
    slowest_core: tuple[int, int] | None = None
    slowest_risc: str | None = None


def _trisc_pipeline_wall_clock(
    intervals: list[ClassifiedInterval],
    gates: dict[tuple[int, int], int] | None = None,
) -> tuple[int, tuple[int, int] | None]:
    """
    Per-core: max(T_end) - min(T_start) across TRISC sub-processors.
    Then max across cores.  Returns (duration_cycles, slowest_core).

    If gates is provided, each core's start is clamped to at least the gate
    time, subtracting blocking wait on a predecessor.
    """
    if not intervals:
        return 0, None
    by_core: dict[tuple[int, int], list[ClassifiedInterval]] = defaultdict(list)
    for ci in intervals:
        by_core[ci.interval.core].append(ci)

    per_core: dict[tuple[int, int], int] = {}
    for core, cis in by_core.items():
        max_end = max(ci.interval.end_cycle for ci in cis)
        min_start = min(ci.interval.start_cycle for ci in cis)
        gate = gates.get(core, 0) if gates else 0
        per_core[core] = max(0, max_end - max(min_start, gate))

    slowest_core = max(per_core, key=lambda c: per_core[c])
    return per_core[slowest_core], slowest_core


def _max_duration(intervals: list[ClassifiedInterval]) -> int:
    if not intervals:
        return 0
    return max(ci.interval.duration_cycles for ci in intervals)


def _adjusted_max_duration(
    intervals: list[ClassifiedInterval],
    gates: dict[tuple[int, int], int] | None = None,
) -> int:
    """Max duration across intervals, with per-core gate adjustment."""
    if not intervals:
        return 0
    if not gates:
        return max(ci.interval.duration_cycles for ci in intervals)
    best = 0
    for ci in intervals:
        gate = gates.get(ci.interval.core, 0)
        adj = max(0, ci.interval.end_cycle - max(ci.interval.start_cycle, gate))
        best = max(best, adj)
    return best


def _span_cycles(intervals: list[ClassifiedInterval]) -> int:
    if not intervals:
        return 0
    return max(ci.interval.end_cycle for ci in intervals) - min(ci.interval.start_cycle for ci in intervals)


def _adjusted_span_cycles(
    intervals: list[ClassifiedInterval],
    gates: dict[tuple[int, int], int] | None = None,
) -> int:
    """Span from earliest start to latest end, with gate adjustment."""
    if not intervals:
        return 0
    max_end = max(ci.interval.end_cycle for ci in intervals)
    min_start = min(ci.interval.start_cycle for ci in intervals)
    if gates:
        cores = {ci.interval.core for ci in intervals}
        max_gate = max((gates.get(c, 0) for c in cores), default=0)
        return max(0, max_end - max(min_start, max_gate))
    return max_end - min_start


def _slowest_interval(
    intervals: list[ClassifiedInterval],
) -> tuple[tuple[int, int] | None, str | None]:
    if not intervals:
        return None, None
    s = max(intervals, key=lambda c: c.interval.duration_cycles)
    return s.interval.core, s.interval.risc


def _uniform_gate(
    intervals: list[ClassifiedInterval],
    gate_cycle: int,
) -> dict[tuple[int, int], int]:
    """Build a gate dict that applies the same gate_cycle to every core."""
    return {ci.interval.core: gate_cycle for ci in intervals}


def compute_op_duration(
    micro_op: MicroOpConfig,
    classified: list[ClassifiedInterval],
    freq_mhz: float,
    predecessor_gates: dict[tuple[int, int], int] | None = None,
) -> OpTimingResult:
    """
    Compute duration for a single micro-op using its type-specific formula.

    If predecessor_gates is provided, blocking time on predecessors is subtracted
    from the entry-point participants.  Within-op blocking (e.g. receiver on
    sender) is also subtracted internally.
    """
    if not classified:
        return OpTimingResult(
            zone=micro_op.zone,
            duration_cycles=0,
            duration_ns=0,
            duration_us=0,
            core_count=0,
            interval_count=0,
            breakdown={},
        )

    by_func: dict[str, list[ClassifiedInterval]] = defaultdict(list)
    for ci in classified:
        by_func[ci.function].append(ci)

    cores = {ci.interval.core for ci in classified}
    breakdown: dict[str, int] = {}
    slowest_core: tuple[int, int] | None = None
    slowest_risc: str | None = None
    raw_cycles = 0
    adj_cycles = 0

    if micro_op.type == OpType.COMPUTE:
        compute_ivs = by_func.get("compute", [])
        if compute_ivs:
            raw_cycles, _ = _trisc_pipeline_wall_clock(compute_ivs)
            adj_cycles, slowest_core = _trisc_pipeline_wall_clock(
                compute_ivs,
                predecessor_gates,
            )
            slowest_risc = "TRISC"
        else:
            raw_cycles = _span_cycles(classified)
            adj_cycles = _adjusted_span_cycles(classified, predecessor_gates)
            slowest_core, slowest_risc = _slowest_interval(classified)
        breakdown["compute"] = adj_cycles

    elif micro_op.type in (OpType.MCAST, OpType.BROADCAST):
        senders = by_func.get("sender", [])
        receivers = by_func.get("receiver", [])

        raw_s = _max_duration(senders)
        raw_r = _max_duration(receivers)
        raw_cycles = raw_s + raw_r

        adj_s = _adjusted_max_duration(senders, predecessor_gates)
        sender_end = max((ci.interval.end_cycle for ci in senders), default=0)
        recv_gate = _uniform_gate(receivers, sender_end) if receivers else None
        adj_r = _adjusted_max_duration(receivers, recv_gate)
        adj_cycles = adj_s + adj_r

        breakdown["sender"] = adj_s
        breakdown["receiver"] = adj_r
        all_ivs = senders + receivers
        slowest_core, slowest_risc = _slowest_interval(all_ivs)

    elif micro_op.type == OpType.GATHER:
        senders = by_func.get("sender", [])
        receivers = by_func.get("receiver", [])

        raw_s = _max_duration(senders)
        raw_r = _max_duration(receivers)
        raw_cycles = raw_s + raw_r

        adj_s = _adjusted_max_duration(senders, predecessor_gates)
        max_sender_end = max(
            (ci.interval.end_cycle for ci in senders),
            default=0,
        )
        recv_gate = _uniform_gate(receivers, max_sender_end) if receivers else None
        adj_r = _adjusted_max_duration(receivers, recv_gate)
        adj_cycles = adj_s + adj_r

        breakdown["sender"] = adj_s
        breakdown["receiver"] = adj_r
        all_ivs = senders + receivers
        slowest_core, slowest_risc = _slowest_interval(all_ivs)

    elif micro_op.type == OpType.GATHER_REDUCE:
        senders = by_func.get("sender", [])
        receivers = by_func.get("receiver", [])
        reducers = by_func.get("reducer", [])

        raw_s = _max_duration(senders)
        raw_r = _max_duration(receivers)
        raw_red, _ = _trisc_pipeline_wall_clock(reducers)
        raw_cycles = raw_s + raw_r + raw_red

        adj_s = _adjusted_max_duration(senders, predecessor_gates)
        max_sender_end = max(
            (ci.interval.end_cycle for ci in senders),
            default=0,
        )
        recv_gate = _uniform_gate(receivers, max_sender_end) if receivers else None
        adj_r = _adjusted_max_duration(receivers, recv_gate)
        max_recv_end = max(
            (ci.interval.end_cycle for ci in receivers),
            default=0,
        )
        red_gate = _uniform_gate(reducers, max_recv_end) if reducers else None
        adj_red, red_core = _trisc_pipeline_wall_clock(reducers, red_gate)
        adj_cycles = adj_s + adj_r + adj_red

        breakdown["sender"] = adj_s
        breakdown["receiver"] = adj_r
        breakdown["reducer"] = adj_red
        longest = max(
            ("sender", adj_s),
            ("receiver", adj_r),
            ("reducer", adj_red),
            key=lambda x: x[1],
        )
        if longest[0] == "reducer" and red_core:
            slowest_core, slowest_risc = red_core, "TRISC"
        else:
            pool = by_func.get(longest[0], [])
            slowest_core, slowest_risc = _slowest_interval(pool)

    elif micro_op.type == OpType.DRAM_WRITE:
        writers = by_func.get("writer", [])
        raw_cycles = _max_duration(writers)
        adj_cycles = _adjusted_max_duration(writers, predecessor_gates)
        breakdown["writer"] = adj_cycles
        slowest_core, slowest_risc = _slowest_interval(writers)

    elif micro_op.type == OpType.DRAM_READ:
        readers = by_func.get("reader", [])
        raw_cycles = _max_duration(readers)
        adj_cycles = _adjusted_max_duration(readers, predecessor_gates)
        breakdown["reader"] = adj_cycles
        slowest_core, slowest_risc = _slowest_interval(readers)

    elif micro_op.type == OpType.SPAN:
        raw_cycles = _span_cycles(classified)
        adj_cycles = _adjusted_span_cycles(classified, predecessor_gates)
        breakdown["span"] = adj_cycles
        slowest_core, slowest_risc = _slowest_interval(classified)

    else:
        raw_cycles = _span_cycles(classified)
        adj_cycles = _adjusted_span_cycles(classified, predecessor_gates)
        breakdown["span"] = adj_cycles

    blocking = max(0, raw_cycles - adj_cycles)
    dur_ns = cycles_to_ns(adj_cycles, freq_mhz)
    raw_ns = cycles_to_ns(raw_cycles, freq_mhz)
    return OpTimingResult(
        zone=micro_op.zone,
        duration_cycles=adj_cycles,
        duration_ns=dur_ns,
        duration_us=dur_ns / 1000.0,
        core_count=len(cores),
        interval_count=len(classified),
        breakdown=breakdown,
        raw_duration_cycles=raw_cycles,
        raw_duration_us=raw_ns / 1000.0,
        blocking_cycles=blocking,
        blocking_us=cycles_to_ns(blocking, freq_mhz) / 1000.0,
        slowest_core=slowest_core,
        slowest_risc=slowest_risc,
    )


# ============================================================================
# Critical-Path Solver
# ============================================================================


def _topological_sort(
    ops: set[str],
    dependencies: list[tuple[str, str]],
) -> list[str]:
    """Kahn's algorithm topological sort over *ops* respecting *dependencies*."""
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {op: 0 for op in ops}
    for child, parent in dependencies:
        if child in ops and parent in ops:
            adj[parent].append(child)
            in_degree[child] = in_degree.get(child, 0) + 1
    queue = deque(op for op in ops if in_degree[op] == 0)
    order: list[str] = []
    while queue:
        op = queue.popleft()
        order.append(op)
        for child in adj.get(op, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order


def compute_critical_path(
    op_durations: dict[str, int],
    dependencies: list[tuple[str, str]],
) -> tuple[dict[str, int], list[str], int]:
    """
    Compute critical path through the dependency DAG.

    Returns (earliest_finish, critical_path_ops, total_duration).
    """
    all_ops = set(op_durations.keys())
    parents: dict[str, set[str]] = defaultdict(set)
    for child, parent in dependencies:
        if child in all_ops and parent in all_ops:
            parents[child].add(parent)

    topo_order = _topological_sort(all_ops, dependencies)

    earliest_finish: dict[str, int] = {}
    predecessor: dict[str, str | None] = {}
    for op in topo_order:
        parent_deps = parents.get(op, set())
        best_parent: str | None = None
        earliest_start = 0
        for p in parent_deps:
            if p in all_ops:
                ef = earliest_finish.get(p, 0)
                if ef > earliest_start:
                    earliest_start = ef
                    best_parent = p
        earliest_finish[op] = earliest_start + op_durations.get(op, 0)
        predecessor[op] = best_parent

    if not earliest_finish:
        return {}, [], 0

    total = max(earliest_finish.values())
    end_op = max(earliest_finish, key=lambda o: earliest_finish[o])
    path: list[str] = []
    current: str | None = end_op
    while current is not None:
        path.append(current)
        current = predecessor.get(current)
    path.reverse()

    return earliest_finish, path, total


# ============================================================================
# Report Generator
# ============================================================================


def format_text_report(
    config: KernelConfig,
    op_results: dict[str, OpTimingResult],
    earliest_finish: dict[str, int],
    critical_path: list[str],
    total_critical_cycles: int,
    core_roles: dict[str, set[tuple[int, int]]],
    intervals: list[ZoneInterval],
    freq_mhz: float,
) -> str:
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(f"Fused-Kernel Micro-Op Timing: {config.kernel_name}")
    lines.append("=" * 90)
    lines.append(f"Chip frequency: {freq_mhz:.0f} MHz")
    lines.append(f"Total zone intervals: {len(intervals)}")
    lines.append("")

    if intervals:
        all_starts = [iv.start_cycle for iv in intervals]
        all_ends = [iv.end_cycle for iv in intervals]
        total_span_ns = cycles_to_ns(max(all_ends) - min(all_starts), freq_mhz)
        lines.append(f"Overall kernel span: {total_span_ns / 1000:.3f} us")

    total_cp_ns = cycles_to_ns(total_critical_cycles, freq_mhz)
    lines.append(f"Critical-path duration: {total_cp_ns / 1000:.3f} us")
    if critical_path:
        lines.append(f"Critical-path ops: {' -> '.join(critical_path)}")
    lines.append("")

    lines.append("--- Inferred Core Roles ---")
    for role, cores in sorted(core_roles.items()):
        if cores:
            cores_sorted = sorted(cores)
            shown = str(cores_sorted[:5])
            extra = f"...+{len(cores) - 5}" if len(cores) > 5 else ""
            lines.append(f"  {role}: {len(cores)} core(s) {shown}{extra}")
    lines.append("")

    hdr = f"{'Micro-Op':<22} {'Adjusted(us)':>12} {'Raw(us)':>10} " f"{'Blocked(us)':>11} {'Cores':>6} {'Breakdown'}"
    lines.append("--- Per Micro-Op Timing ---")
    lines.append(hdr)
    lines.append("-" * 100)

    for zone in config.display_order:
        result = op_results.get(zone)
        if result is None or result.duration_cycles == 0:
            continue
        parts = ", ".join(f"{k}={cycles_to_ns(v, freq_mhz)/1000:.1f}us" for k, v in result.breakdown.items() if v)
        on_cp = " *" if zone in critical_path else ""
        lines.append(
            f"{zone:<22} {result.duration_us:>12.3f} "
            f"{result.raw_duration_us:>10.3f} {result.blocking_us:>11.3f} "
            f"{result.core_count:>6} {parts}{on_cp}"
        )

    lines.append("")
    lines.append("  (* = on critical path)")
    lines.append("")

    lines.append("--- Dependency Chain ---")
    for child, parent in config.dependencies:
        p_res = op_results.get(parent)
        c_res = op_results.get(child)
        if p_res and c_res and p_res.duration_cycles > 0 and c_res.duration_cycles > 0:
            lines.append(f"  {parent} ({p_res.duration_us:.2f} us) -> " f"{child} ({c_res.duration_us:.2f} us)")

    if critical_path and op_results:
        bottleneck = max(
            (op_results[z] for z in critical_path if z in op_results),
            key=lambda r: r.duration_cycles,
            default=None,
        )
        if bottleneck and total_critical_cycles > 0:
            pct = 100.0 * bottleneck.duration_cycles / total_critical_cycles
            lines.append("")
            lines.append(
                f"Bottleneck: {bottleneck.zone} = {bottleneck.duration_us:.3f} us " f"({pct:.1f}% of critical path)"
            )

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


def format_json_report(
    config: KernelConfig,
    op_results: dict[str, OpTimingResult],
    earliest_finish: dict[str, int],
    critical_path: list[str],
    total_critical_cycles: int,
    core_roles: dict[str, set[tuple[int, int]]],
    freq_mhz: float,
) -> dict:
    ops = {}
    for zone, result in op_results.items():
        if result.duration_cycles == 0:
            continue
        ops[zone] = {
            "duration_us": result.duration_us,
            "duration_ns": result.duration_ns,
            "duration_cycles": result.duration_cycles,
            "raw_duration_us": result.raw_duration_us,
            "raw_duration_cycles": result.raw_duration_cycles,
            "blocking_us": result.blocking_us,
            "blocking_cycles": result.blocking_cycles,
            "core_count": result.core_count,
            "interval_count": result.interval_count,
            "slowest_core": result.slowest_core,
            "slowest_risc": result.slowest_risc,
            "breakdown": {
                k: {"cycles": v, "us": cycles_to_ns(v, freq_mhz) / 1000.0} for k, v in result.breakdown.items() if v
            },
            "on_critical_path": zone in critical_path,
        }
    total_cp_ns = cycles_to_ns(total_critical_cycles, freq_mhz)
    return {
        "kernel_name": config.kernel_name,
        "freq_mhz": freq_mhz,
        "critical_path": {
            "ops": critical_path,
            "total_duration_us": total_cp_ns / 1000.0,
            "total_duration_ns": total_cp_ns,
            "total_duration_cycles": total_critical_cycles,
        },
        "micro_ops": ops,
        "core_roles": {k: sorted(list(v)) for k, v in core_roles.items() if v},
    }


# ============================================================================
# Top-level analysis pipeline
# ============================================================================


def analyze(
    log_path: Path,
    config: KernelConfig,
    verbose: bool = False,
) -> tuple[
    dict[str, OpTimingResult],
    dict[str, int],
    list[str],
    int,
    dict[str, set[tuple[int, int]]],
    list[ZoneInterval],
    float,
]:
    """Run the full analysis pipeline."""
    intervals, freq_mhz = parse_device_log(log_path, config.allowed_zones, verbose)
    core_roles = infer_core_roles(intervals, config)
    classified = classify_intervals(intervals, config, core_roles)

    all_zones = {op.zone for op in config.micro_ops}
    topo_order = _topological_sort(all_zones, config.dependencies)

    op_results: dict[str, OpTimingResult] = {}
    op_dur_cycles: dict[str, int] = {}
    for zone in topo_order:
        micro_op = config.get_micro_op(zone)
        if micro_op is None:
            continue
        gates = compute_predecessor_gates(zone, config, classified)
        result = compute_op_duration(
            micro_op,
            classified.get(zone, []),
            freq_mhz,
            gates or None,
        )
        op_results[zone] = result
        op_dur_cycles[zone] = result.duration_cycles

    earliest_finish, critical_path, total_cp = compute_critical_path(
        op_dur_cycles,
        config.dependencies,
    )

    return (op_results, earliest_finish, critical_path, total_cp, core_roles, intervals, freq_mhz)


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generalized fused-kernel micro-op timing analysis",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kernel", type=str, help="Built-in kernel name (e.g. pre_sdpa)")
    group.add_argument("--config", type=Path, help="Path to custom YAML config")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("generated/profiler/.logs/profile_log_device.csv"),
        help="Path to profile_log_device.csv",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write report to file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    args = parser.parse_args()

    config_source = args.config if args.config else args.kernel
    try:
        config = load_kernel_config(config_source)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    log_path = args.log
    if not log_path.is_absolute():
        tt_home = Path(__file__).resolve().parents[4]
        log_path = tt_home / log_path
    if not log_path.exists():
        print(f"Error: device log not found: {log_path}", file=sys.stderr)
        return 1

    (op_results, earliest_finish, critical_path, total_cp, core_roles, intervals, freq_mhz) = analyze(
        log_path, config, args.verbose
    )

    if not intervals:
        print(f"No {config.kernel_name} zone intervals found in log.", file=sys.stderr)
        return 1

    if args.json:
        data = format_json_report(
            config,
            op_results,
            earliest_finish,
            critical_path,
            total_cp,
            core_roles,
            freq_mhz,
        )
        report = json.dumps(data, indent=2)
    else:
        report = format_text_report(
            config,
            op_results,
            earliest_finish,
            critical_path,
            total_cp,
            core_roles,
            intervals,
            freq_mhz,
        )

    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
