# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decoder of the metal profiler's RISC-V performance monitor markers (ids 9092, 9093, 9094).
Names come from tools/include/perf_counters/riscv_pmc_*.h through headers.py; the formulas are in metrics.py.
"""

from typing import Dict, NamedTuple, Optional

from . import headers, riscv_metrics

COUNTS_MARKER_ID = 9092  # (mhpmcounter4 delta << 32) | mhpmcounter3 delta
CYCLES_MARKER_ID = 9093  # (minstret delta << 32) | mcycle delta
EVENTS_MARKER_ID = 9094  # (mhpmevent4 << 32) | mhpmevent3
MARKER_IDS = (COUNTS_MARKER_ID, CYCLES_MARKER_ID, EVENTS_MARKER_ID)

_MASK32 = 0xFFFFFFFF


class Launch(NamedTuple):
    pmc3: int
    pmc4: int
    mcycle: int
    minstret: int
    event3: int
    event4: int


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


def unpack_launch(counts, cycles, events) -> Launch:
    """Split the three 64-bit marker payloads of one launch."""
    return Launch(
        counts & _MASK32,
        counts >> 32,
        cycles & _MASK32,
        cycles >> 32,
        events & _MASK32,
        events >> 32,
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
    return {
        "event3": label3,
        "event4": label4,
        "mcycle": launch.mcycle,
        "minstret": launch.minstret,
        "metrics": riscv_metrics.compute_riscv_metrics(
            launch.mcycle, launch.minstret, counts
        ),
    }
