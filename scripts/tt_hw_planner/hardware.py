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


_CALIBRATED_OVERHEAD: Dict[str, float] = {}


@dataclass(frozen=True)
class Overhead:
    """Per-architecture memory overhead constants (per chip)."""

    dispatch_gb: float
    ccl_gb: float
    fragmentation_frac: float
    source: str


OVERHEAD_BY_ARCH = {
    "Wormhole": Overhead(
        dispatch_gb=0.5,
        ccl_gb=0.4,
        fragmentation_frac=0.05,
        source="estimate; aligned with measurements from tt_transformers "
        "Llama-3.1-8B@N150 demos (PR# pending calibration)",
    ),
    "Blackhole": Overhead(
        dispatch_gb=1.2,
        ccl_gb=0.6,
        fragmentation_frac=0.05,
        source="estimate; BH dispatch path has more in-flight buffers " "than WH (per ttnn dispatch_kernel notes)",
    ),
}


@dataclass(frozen=True)
class Box:
    """A Tenstorrent box: a fixed multi-chip system with known mesh topology."""

    name: str
    arch: str
    chips: int
    hbm_per_chip_gb: float
    mesh_shapes: List[Tuple[int, int]]
    notes: str = ""
    eth_link_gbps: float = 0.0
    default_mesh: Optional[Tuple[int, int]] = None
    board_types: Tuple[str, ...] = ()

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


HARDWARE: List[Box] = [
    Box(
        name="N150",
        arch="Wormhole",
        chips=1,
        hbm_per_chip_gb=12.0,
        mesh_shapes=[(1, 1)],
        eth_link_gbps=0.0,
        notes="Single Wormhole. Sub-6B LLMs, small CNNs, embeddings.",
        board_types=("n150",),
    ),
    Box(
        name="N300",
        arch="Wormhole",
        chips=2,
        hbm_per_chip_gb=12.0,
        mesh_shapes=[(1, 1), (1, 2), (2, 1)],
        eth_link_gbps=100.0,
        notes="Dual Wormhole. Sub-12B LLMs, most STT, most embeddings.",
        board_types=("n300",),
    ),
    Box(
        name="T3K",
        arch="Wormhole",
        chips=8,
        hbm_per_chip_gb=12.0,
        mesh_shapes=[(1, 1), (1, 2), (2, 1), (1, 4), (2, 2), (4, 1), (1, 8), (2, 4), (4, 2), (8, 1)],
        eth_link_gbps=100.0,
        notes="8x Wormhole eth-mesh (LoudBox). Mature LLM box; canonical TP=[1,8].",
        board_types=(),
    ),
    Box(
        name="QB2",
        arch="Blackhole",
        chips=4,
        hbm_per_chip_gb=32.0,
        mesh_shapes=[(1, 1), (2, 2)],
        eth_link_gbps=0.0,
        default_mesh=(2, 2),
        notes="4x Blackhole (p150c), 128 GB. Physical fabric is 2x2 (8 QSFP-DD, "
        "2 links/chip); (1,1) is available for single-chip bring-up on one of "
        "the 4 chips.",
        board_types=("p150c",),
    ),
    Box(
        name="GalaxyWH",
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
        default_mesh=(4, 8),
        eth_link_gbps=100.0,
        notes="32x Wormhole. Needed for 70B+ dense, large MoE, or video gen. " "Canonical large-scale shape is [4,8].",
        board_types=("tt-galaxy-wh", "ubb_wormhole"),
    ),
    Box(
        name="GalaxyBH",
        arch="Blackhole",
        chips=32,
        hbm_per_chip_gb=32.0,
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
        default_mesh=(4, 8),
        eth_link_gbps=100.0,
        notes="32x Blackhole (BHGLX). Blackhole counterpart of the Wormhole Galaxy; "
        "1024 GB total (32 GB/chip). Canonical large-scale shape is [4,8].",
        board_types=("tt-galaxy-bh", "ubb_blackhole"),
    ),
    Box(
        name="P100",
        arch="Blackhole",
        chips=1,
        hbm_per_chip_gb=32.0,
        mesh_shapes=[(1, 1)],
        eth_link_gbps=0.0,
        notes="Single Blackhole (p100).",
        board_types=("p100", "p100a"),
    ),
    Box(
        name="P150",
        arch="Blackhole",
        chips=1,
        hbm_per_chip_gb=32.0,
        mesh_shapes=[(1, 1)],
        eth_link_gbps=0.0,
        notes="Single Blackhole (p150).",
        board_types=("p150", "p150a", "p150b"),
    ),
    Box(
        name="P300",
        arch="Blackhole",
        chips=2,
        hbm_per_chip_gb=32.0,
        mesh_shapes=[(1, 1), (1, 2), (2, 1)],
        eth_link_gbps=0.0,
        notes="Dual Blackhole (p300).",
        board_types=("p300", "p300a"),
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


_apply_calibration()


def reload_calibration() -> Dict[str, float]:
    """Call from CLI after a `calibrate` run to pick up the new data."""
    _CALIBRATED_OVERHEAD.clear()
    _apply_calibration()
    return dict(_CALIBRATED_OVERHEAD)
