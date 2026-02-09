# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Simple host-side profiling utils with device synchronization.
Supports coarse-grained and fine-grained (instruction-level) profiling
with hierarchical output showing which fine ops belong to each coarse section.
"""

import time
from collections import defaultdict, OrderedDict

# ── Flags ──────────────────────────────────────────────────────────────
_coarse_enabled = False
_fine_enabled = False
_current_phase = "decode"
_current_section = None  # tracks which coarse section we're inside

# ── Coarse accumulators: phase -> section -> total_seconds ─────────────
_timers = {"prefill": defaultdict(float), "decode": defaultdict(float)}
_counts = {"prefill": defaultdict(int), "decode": defaultdict(int)}

# ── Fine accumulators: phase -> section -> subsection -> total_seconds ─
_fine_timers = {"prefill": defaultdict(lambda: defaultdict(float)), "decode": defaultdict(lambda: defaultdict(float))}
_fine_counts = {"prefill": defaultdict(lambda: defaultdict(int)), "decode": defaultdict(lambda: defaultdict(int))}


# ── Enable / disable ──────────────────────────────────────────────────
def enable_profiling(coarse=True, fine=False):
    global _coarse_enabled, _fine_enabled
    _coarse_enabled = coarse
    _fine_enabled = fine


def disable_profiling():
    global _coarse_enabled, _fine_enabled
    _coarse_enabled = False
    _fine_enabled = False


def is_profiling_enabled():
    """True if any profiling level is active (used to disable tracing)."""
    return _coarse_enabled or _fine_enabled


def is_coarse_enabled():
    return _coarse_enabled


def is_fine_enabled():
    return _fine_enabled


# ── Phase management ──────────────────────────────────────────────────
def set_phase(phase):
    global _current_phase
    assert phase in ("prefill", "decode"), f"Unknown phase: {phase}"
    _current_phase = phase


def get_phase():
    return _current_phase


# ── Section context (links fine records to their parent coarse section) ─
def begin_section(name):
    global _current_section
    _current_section = name


def get_current_section():
    return _current_section


# ── Device sync helper ────────────────────────────────────────────────
def sync_and_time(mesh_device):
    """Synchronize all devices and return current wall-clock time."""
    import ttnn

    ttnn.synchronize_device(mesh_device)
    return time.perf_counter()


# ── Recording ─────────────────────────────────────────────────────────
def record(section_name, elapsed):
    """Record a coarse-grained section time."""
    if _coarse_enabled:
        _timers[_current_phase][section_name] += elapsed
        _counts[_current_phase][section_name] += 1


def record_fine(subsection_name, elapsed):
    """Record a fine-grained subsection time under the current coarse section."""
    if _fine_enabled and _current_section is not None:
        _fine_timers[_current_phase][_current_section][subsection_name] += elapsed
        _fine_counts[_current_phase][_current_section][subsection_name] += 1


# ── Printing ──────────────────────────────────────────────────────────
def _print_phase(phase, label):
    """Print accumulated times for a single phase with optional hierarchy."""
    timers = _timers[phase]
    counts = _counts[phase]
    fine_t = _fine_timers[phase]
    fine_c = _fine_counts[phase]

    has_coarse = bool(timers)
    has_fine = bool(fine_t)

    if not has_coarse and not has_fine:
        print(f"\n[Profiling] No {label} timing data collected.")
        return

    print("\n" + "=" * 78)
    print(f"  {label} PROFILING SUMMARY (wall-clock with device sync)")
    print("=" * 78)
    print(f"  {'Section':<42} {'Total (ms)':>12} {'Calls':>8} {'Avg (ms)':>12}")
    print("-" * 78)

    # Collect all coarse sections (union of coarse + fine keys)
    all_sections = list(OrderedDict.fromkeys(list(timers.keys()) + list(fine_t.keys())))

    # Sort coarse sections by total time (longest first)
    def _coarse_time(name):
        if name in timers:
            return timers[name]
        # If only fine data exists, sum fine sub-times as the section total
        if name in fine_t:
            return sum(fine_t[name].values())
        return 0.0

    all_sections.sort(key=_coarse_time, reverse=True)

    total_all = 0.0
    for section in all_sections:
        # Print coarse line
        if section in timers:
            total_ms = timers[section] * 1000
            count = counts[section]
            avg_ms = total_ms / count if count > 0 else 0
            total_all += total_ms
            print(f"  {section:<42} {total_ms:>12.3f} {count:>8} {avg_ms:>12.3f}")
        elif section in fine_t:
            # No coarse record, but fine data exists — synthesise a total
            synth_ms = sum(fine_t[section].values()) * 1000
            synth_count = max(fine_c[section].values()) if fine_c[section] else 0
            avg_ms = synth_ms / synth_count if synth_count > 0 else 0
            total_all += synth_ms
            print(f"  {section:<42} {synth_ms:>12.3f} {synth_count:>8} {avg_ms:>12.3f}")

        # Print fine sub-sections (sorted by time, longest first)
        if section in fine_t and fine_t[section]:
            subs = fine_t[section]
            sub_counts = fine_c[section]
            sorted_subs = sorted(subs.keys(), key=lambda n: subs[n], reverse=True)
            for idx, sub in enumerate(sorted_subs):
                is_last = idx == len(sorted_subs) - 1
                prefix = "└─" if is_last else "├─"
                sub_ms = subs[sub] * 1000
                sub_cnt = sub_counts[sub]
                sub_avg = sub_ms / sub_cnt if sub_cnt > 0 else 0
                label_str = f"    {prefix} {sub}"
                print(f"  {label_str:<42} {sub_ms:>12.3f} {sub_cnt:>8} {sub_avg:>12.3f}")

    print("-" * 78)
    print(f"  {'TOTAL':<42} {total_all:>12.3f}")
    print("=" * 78)


def print_summary():
    """Print accumulated times for both prefill and decode phases."""
    _print_phase("prefill", "PREFILL")
    _print_phase("decode", "DECODE")


def export_json(filepath):
    """Export profiling data to a JSON file for external plotting."""
    import json

    data = {}
    for phase in ("prefill", "decode"):
        sections = {}
        for name in _timers[phase]:
            sections[name] = {
                "total_ms": _timers[phase][name] * 1000,
                "calls": _counts[phase][name],
                "avg_ms": (_timers[phase][name] * 1000) / _counts[phase][name] if _counts[phase][name] > 0 else 0,
            }
            # Include fine-grained sub-sections if present
            if name in _fine_timers[phase] and _fine_timers[phase][name]:
                subs = {}
                for sub_name, sub_time in _fine_timers[phase][name].items():
                    sub_cnt = _fine_counts[phase][name][sub_name]
                    subs[sub_name] = {
                        "total_ms": sub_time * 1000,
                        "calls": sub_cnt,
                        "avg_ms": (sub_time * 1000) / sub_cnt if sub_cnt > 0 else 0,
                    }
                sections[name]["fine"] = subs
        data[phase] = sections
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[Profiling] Data exported to {filepath}")


# ── Reset ─────────────────────────────────────────────────────────────
def reset():
    """Clear all accumulated data."""
    for phase in ("prefill", "decode"):
        _timers[phase].clear()
        _counts[phase].clear()
        _fine_timers[phase].clear()
        _fine_counts[phase].clear()


def reset_phase(phase):
    """Clear data for a single phase."""
    _timers[phase].clear()
    _counts[phase].clear()
    _fine_timers[phase].clear()
    _fine_counts[phase].clear()
