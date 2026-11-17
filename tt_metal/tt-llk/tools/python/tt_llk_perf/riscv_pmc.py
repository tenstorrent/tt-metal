# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decoder of the metal profiler's RISC-V performance monitor markers (ids 9092 to 9096).
Names come from tools/include/perf_counters/riscv_pmc_*.h through headers.py; the formulas are in metrics.py.
"""

from typing import Dict, NamedTuple, Optional

from . import headers, riscv_metrics

COUNTS_MARKER_ID = 9092  # (mhpmcounter4 delta << 32) | mhpmcounter3 delta
CYCLES_MARKER_ID = 9093  # (minstret delta << 32) | mcycle delta
EVENTS_MARKER_ID = 9094  # (mhpmevent4 << 32) | mhpmevent3
COUNTS_UPPER_MARKER_ID = (
    9095  # (mhpmcounter6 delta << 32) | mhpmcounter5, Quasar data-movement cores only
)
EVENTS_UPPER_MARKER_ID = 9096  # (mhpmevent6 << 32) | mhpmevent5
MARKER_IDS = (COUNTS_MARKER_ID, CYCLES_MARKER_ID, EVENTS_MARKER_ID)
UPPER_MARKER_IDS = (COUNTS_UPPER_MARKER_ID, EVENTS_UPPER_MARKER_ID)

_MASK32 = 0xFFFFFFFF


class Launch(NamedTuple):
    pmc3: int
    pmc4: int
    mcycle: int
    minstret: int
    event3: int
    event4: int
    pmc5: int = 0
    pmc6: int = 0
    event5: int = 0
    event6: int = 0


def core_kind(arch, risc_type) -> Optional[str]:
    """'baby', 'trisc' or 'dm' for a profiler RISC name on an arch, None when the core has no monitor capture."""
    arch = headers.normalize_arch(arch)
    risc = str(risc_type).upper()
    if arch == "quasar":
        if risc.startswith("QUASAR_DM"):
            return "dm"
        return "trisc" if "TRISC" in risc else None
    if arch == "blackhole" and risc in (
        "BRISC",
        "NCRISC",
        "TRISC_0",
        "TRISC_1",
        "TRISC_2",
        "ERISC",
    ):
        return "baby"
    return None


def unpack_launch(counts, cycles, events, counts_upper=0, events_upper=0) -> Launch:
    """Split the 64-bit marker payloads of one launch; the upper pair is only emitted for Quasar DMs."""
    return Launch(
        counts & _MASK32,
        counts >> 32,
        cycles & _MASK32,
        cycles >> 32,
        events & _MASK32,
        events >> 32,
        counts_upper & _MASK32,
        counts_upper >> 32,
        events_upper & _MASK32,
        events_upper >> 32,
    )


def event_label(arch, kind, select, include_dir=None) -> str:
    """Name of the events a select arms, joined with '|'; 'DISABLED' for a zero select."""
    names = headers.riscv_select_events(arch, kind, select, include_dir)
    return "|".join(names) if names else "DISABLED"


def decode_launch(arch, kind, launch: Launch, include_dir=None) -> Dict[str, object]:
    """Named view of one launch: labels of both counters plus every derived metric."""
    label3 = event_label(arch, kind, launch.event3, include_dir)
    label4 = event_label(arch, kind, launch.event4, include_dir)
    counts = {}
    if label3 != "DISABLED":
        counts[label3] = launch.pmc3
    if label4 != "DISABLED":
        counts[label4] = launch.pmc4
    out = {"event3": label3, "event4": label4}
    if kind == "dm":
        out["event5"] = event_label(arch, kind, launch.event5, include_dir)
        out["event6"] = event_label(arch, kind, launch.event6, include_dir)
        for label, count in (
            (out["event5"], launch.pmc5),
            (out["event6"], launch.pmc6),
        ):
            if label != "DISABLED":
                counts[label] = count
    out["mcycle"] = launch.mcycle
    out["minstret"] = launch.minstret
    out["metrics"] = riscv_metrics.compute_riscv_metrics(
        launch.mcycle, launch.minstret, counts
    )
    return out
