# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Hardware database — Tenstorrent box specs and per-architecture overhead.

The numbers in this file are split into two categories:

  HARD CONSTANTS (from datasheets / `tt-smi`):
    - hbm_per_chip_gb, chips, mesh topology, arch

  CALIBRATED CONSTANTS (currently estimated; will be replaced by
  measured values from calibration runs — see calibration.py):
    - dispatch_overhead_gb_per_chip   : CQ buffers + kernel args + scratch
    - ccl_buffer_gb_per_chip          : all-gather / reduce-scatter staging
    - fragmentation_frac              : fraction of HBM lost to allocator
                                         fragmentation at steady state

Every estimated constant has a `source` field that says WHERE the number
came from (datasheet, measurement, educated guess).  Anything labelled
"estimate" should be considered provisional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Calibrated overhead by box name (GB/chip).  Populated by
# `_apply_calibration()` at module load when calibration data is available.
# When a box name appears here, the analytical overhead is ignored in
# favour of the measured value.
_CALIBRATED_OVERHEAD: Dict[str, float] = {}


@dataclass(frozen=True)
class Overhead:
    """Per-architecture memory overhead constants (per chip)."""

    dispatch_gb: float  # CQ + kernel args + dispatch scratch
    ccl_gb: float  # CCL bounce buffers (TP > 1 only)
    fragmentation_frac: float  # fraction of HBM lost to fragmentation
    source: str  # provenance for these numbers


# Per-architecture overhead.  These are the numbers that USED to be hidden
# inside the flat 0.80 safety factor.  Exposing them lets calibration
# adjust each independently as measurements arrive.
OVERHEAD_BY_ARCH = {
    "Wormhole": Overhead(
        dispatch_gb=0.5,  # ~512 MB CQ + dispatch scratch
        ccl_gb=0.4,  # all-gather staging on eth ring
        fragmentation_frac=0.05,  # ~5% steady-state fragmentation
        source="estimate; aligned with measurements from tt_transformers "
        "Llama-3.1-8B@N150 demos (PR# pending calibration)",
    ),
    "Blackhole": Overhead(
        dispatch_gb=1.2,  # larger CQ on BH, more dispatch scratch
        ccl_gb=0.6,  # PCIe-based CCL has larger buffers
        fragmentation_frac=0.05,
        source="estimate; BH dispatch path has more in-flight buffers " "than WH (per ttnn dispatch_kernel notes)",
    ),
}


@dataclass(frozen=True)
class Box:
    """A Tenstorrent box: a fixed multi-chip system with known mesh topology."""

    name: str
    arch: str  # "Wormhole" | "Blackhole"
    chips: int
    hbm_per_chip_gb: float
    mesh_shapes: List[Tuple[int, int]]  # canonical first, then alternatives
    notes: str = ""
    eth_link_gbps: float = 0.0  # peak inter-chip BW (for future BW checks)

    @property
    def total_hbm_gb(self) -> float:
        return self.chips * self.hbm_per_chip_gb

    @property
    def overhead(self) -> Overhead:
        return OVERHEAD_BY_ARCH[self.arch]

    def usable_per_chip_gb(self, tp_factor: int) -> float:
        """
        Per-chip HBM available for weights + KV + activations.

        If a calibrated overhead is registered for this box (from a
        measurement run), use it directly.  Otherwise fall back to the
        analytical decomposition: hbm - dispatch - ccl - frag×hbm.
        """
        cal = _CALIBRATED_OVERHEAD.get(self.name)
        if cal is not None:
            return self.hbm_per_chip_gb - cal
        o = self.overhead
        ccl = o.ccl_gb if tp_factor > 1 else 0.0
        return self.hbm_per_chip_gb - o.dispatch_gb - ccl - self.hbm_per_chip_gb * o.fragmentation_frac

    @property
    def calibrated(self) -> bool:
        return self.name in _CALIBRATED_OVERHEAD


# ---------------------------------------------------------------------------
# Box catalog
# ---------------------------------------------------------------------------
# Sources for HBM/chip and chips:
#   - N150/N300 : Tenstorrent Wormhole datasheet, 12 GB GDDR6 per chip
#   - T3K       : 8x Wormhole eth-mesh (LoudBox spec)
#   - QB2       : 4x Blackhole p150b OR p300c, 32 GB GDDR6 per chip (QuietBox 2 spec)
#   - Galaxy    : 32x Wormhole tray (Tenstorrent Galaxy datasheet)
#
# Mesh shapes: listed canonical-first per TP factor.  Probed via
# `ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))` on real hardware.

HARDWARE: List[Box] = [
    Box(
        name="N150",
        arch="Wormhole",
        chips=1,
        hbm_per_chip_gb=12.0,
        mesh_shapes=[(1, 1)],
        eth_link_gbps=0.0,
        notes="Single Wormhole. Sub-6B LLMs, small CNNs, embeddings.",
    ),
    Box(
        name="N300",
        arch="Wormhole",
        chips=2,
        hbm_per_chip_gb=12.0,
        mesh_shapes=[(1, 1), (1, 2), (2, 1)],
        eth_link_gbps=100.0,
        notes="Dual Wormhole. Sub-12B LLMs, most STT, most embeddings.",
    ),
    Box(
        name="T3K",
        arch="Wormhole",
        chips=8,
        hbm_per_chip_gb=12.0,
        mesh_shapes=[(1, 1), (1, 2), (2, 1), (1, 4), (2, 2), (4, 1), (1, 8), (2, 4), (4, 2), (8, 1)],
        eth_link_gbps=100.0,
        notes="8x Wormhole eth-mesh (LoudBox). Mature LLM box; canonical TP=[1,8].",
    ),
    Box(
        name="QB2",
        arch="Blackhole",
        chips=4,
        hbm_per_chip_gb=32.0,
        mesh_shapes=[(1, 1), (1, 2), (2, 1), (1, 4), (2, 2), (4, 1)],
        eth_link_gbps=0.0,
        notes="4x Blackhole (p150b or p300c), 128 GB. Canonical TP=[1,4]; " "[2,2] for 2D parallel.",
    ),
    Box(
        name="Galaxy",
        arch="Wormhole",
        chips=32,
        hbm_per_chip_gb=12.0,
        mesh_shapes=[
            (1, 1),
            (1, 8),
            (2, 4),
            (4, 2),
            (8, 1),
            (1, 16),
            (2, 8),
            (4, 4),
            (8, 2),
            (16, 1),
            (4, 8),
            (2, 16),
            (8, 4),
            (1, 32),
            (16, 2),
            (32, 1),
        ],
        eth_link_gbps=100.0,
        notes="32x Wormhole. Needed for 70B+ dense, large MoE, or video gen. " "Canonical large-scale shape is [4,8].",
    ),
]


def find_box(name: str) -> Box:
    for b in HARDWARE:
        if b.name == name:
            return b
    raise KeyError(f"unknown box: {name}; known: {[b.name for b in HARDWARE]}")


def _apply_calibration() -> None:
    """
    Load calibration.yaml and populate `_CALIBRATED_OVERHEAD` so that
    `Box.usable_per_chip_gb()` returns measured numbers instead of
    estimates.

    For each box that has at least one calibration run, we take the median
    of the implied overheads.  This is robust to one bad measurement and
    matches how vLLM aggregates per-GPU memory observations.
    """
    # Lazy import to avoid a circular import with calibration.py.
    try:
        from .calibration import load, DEFAULT_CALIBRATION_PATH
    except ImportError:
        return

    db = load(DEFAULT_CALIBRATION_PATH)
    if not db.runs:
        return

    overheads: Dict[str, List[float]] = {}
    for r in db.runs:
        overheads.setdefault(r.box, []).append(r.implied_overhead_gb)

    for box_name, vals in overheads.items():
        vals = sorted(vals)
        n = len(vals)
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        _CALIBRATED_OVERHEAD[box_name] = median


# Apply calibration on import.  Safe to call again (idempotent if the
# calibration file hasn't changed).
_apply_calibration()


def reload_calibration() -> Dict[str, float]:
    """Call from CLI after a `calibrate` run to pick up the new data."""
    _CALIBRATED_OVERHEAD.clear()
    _apply_calibration()
    return dict(_CALIBRATED_OVERHEAD)
