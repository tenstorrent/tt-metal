# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""BoxSpec + roofline ceilings for a Tenstorrent box.

A `BoxSpec` is the box-specific parameter bundle every optimizer block and
every chart reads to reason about "where does this op sit relative to its
hardware ceiling on the recommender-selected box." It is derived from the
existing `hardware.Box` (HBM/mesh) plus a small per-architecture table of
peak FPU/DRAM/NoC/ETH numbers (datasheet) so we have a single source of
truth for ceiling lines on the roofline chart.

The values for DRAM/NoC/ETH bandwidth and FPU peak per fidelity are
deliberately conservative. They live in this file as named constants
(rather than buried inside chart code) so they can be inspected and
updated as datasheets evolve. Every constant is annotated with its
provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..hardware import HARDWARE, Box, find_box


# Math fidelity labels match the ttnn enum names.
MATH_FIDELITY_LEVELS: Tuple[str, ...] = ("LoFi", "HiFi2", "HiFi3", "HiFi4")


@dataclass(frozen=True)
class ArchPeak:
    """Per-architecture peak hardware numbers (per-chip)."""

    fpu_peak_tflops_per_fidelity: Dict[str, float]
    dram_bw_gbps: float
    noc_bw_gbps: float
    eth_bw_gbps: float
    cores_per_chip: int
    l1_per_core_b: int
    sfpu_capabilities: Tuple[str, ...]
    source: str


ARCH_PEAKS: Dict[str, ArchPeak] = {
    "Wormhole": ArchPeak(
        fpu_peak_tflops_per_fidelity={
            "LoFi": 296.0,
            "HiFi2": 148.0,
            "HiFi3": 74.0,
            "HiFi4": 74.0,
        },
        dram_bw_gbps=288.0,
        noc_bw_gbps=900.0,
        eth_bw_gbps=100.0,
        cores_per_chip=64,
        l1_per_core_b=1_536 * 1024,
        sfpu_capabilities=("exp", "sqrt", "sigmoid", "gelu", "tanh", "softplus"),
        source="Wormhole datasheet (GDDR6, 1.5MB L1/core, 8x10 grid w/ harvesting)",
    ),
    "Blackhole": ArchPeak(
        fpu_peak_tflops_per_fidelity={
            "LoFi": 768.0,
            "HiFi2": 384.0,
            "HiFi3": 192.0,
            "HiFi4": 192.0,
        },
        dram_bw_gbps=1024.0,
        noc_bw_gbps=2000.0,
        eth_bw_gbps=0.0,
        cores_per_chip=140,
        l1_per_core_b=1_536 * 1024,
        sfpu_capabilities=("exp", "sqrt", "sigmoid", "gelu", "tanh", "softplus", "rsqrt"),
        source="Blackhole datasheet estimates (GDDR6, p150 spec); refine post-silicon",
    ),
}


@dataclass(frozen=True)
class BoxSpec:
    """The single source of truth for box-level numbers consumed by perf."""

    name: str
    arch: str
    mesh_shape: Tuple[int, int]
    chips: int
    hbm_per_chip_gb: float
    cores_per_chip: int
    l1_per_core_b: int
    dram_bw_gbps: float
    noc_bw_gbps: float
    eth_bw_gbps: float
    fpu_peak_tflops: Dict[str, float] = field(default_factory=dict)
    sfpu_capabilities: Tuple[str, ...] = ()

    @property
    def total_chips(self) -> int:
        return self.mesh_shape[0] * self.mesh_shape[1]

    @property
    def total_l1_b(self) -> int:
        return self.cores_per_chip * self.l1_per_core_b * self.total_chips

    def fpu_peak_flops_per_chip(self, fidelity: str) -> float:
        """FPU peak in FLOPS/s for a single chip at the requested fidelity."""
        tflops = self.fpu_peak_tflops.get(fidelity)
        if tflops is None:
            tflops = self.fpu_peak_tflops.get("HiFi2", 0.0)
        return tflops * 1e12

    def dram_bw_bytes_per_s(self) -> float:
        return self.dram_bw_gbps * 1e9

    def noc_bw_bytes_per_s(self) -> float:
        return self.noc_bw_gbps * 1e9

    def eth_bw_bytes_per_s(self) -> float:
        return self.eth_bw_gbps * 1e9


def load_box_spec(box_name: str, mesh_shape: Tuple[int, int]) -> BoxSpec:
    """Build a BoxSpec for (box_name, mesh_shape).

    Raises KeyError if the box name is unknown.
    """
    box: Box = find_box(box_name)
    peak = ARCH_PEAKS.get(box.arch)
    if peak is None:
        raise KeyError(f"no ARCH_PEAKS entry for arch={box.arch}")
    return BoxSpec(
        name=box.name,
        arch=box.arch,
        mesh_shape=mesh_shape,
        chips=box.chips,
        hbm_per_chip_gb=box.hbm_per_chip_gb,
        cores_per_chip=peak.cores_per_chip,
        l1_per_core_b=peak.l1_per_core_b,
        dram_bw_gbps=peak.dram_bw_gbps,
        noc_bw_gbps=peak.noc_bw_gbps,
        eth_bw_gbps=peak.eth_bw_gbps if box.eth_link_gbps == 0 else max(peak.eth_bw_gbps, box.eth_link_gbps),
        fpu_peak_tflops=dict(peak.fpu_peak_tflops_per_fidelity),
        sfpu_capabilities=peak.sfpu_capabilities,
    )


@dataclass(frozen=True)
class CeilingLine:
    """One ceiling on the roofline plot.

    For horizontal ceilings (FPU peak) `intensity_axis` is None and the
    line is a constant `flops_per_s`. For diagonal ceilings (DRAM, NoC,
    ETH) the FLOPS at intensity I is `intensity * bytes_per_s`.
    """

    label: str
    kind: str  # "fpu" | "dram" | "noc" | "eth"
    flops_per_s: Optional[float]  # set for horizontal (FPU) lines
    bytes_per_s: Optional[float]  # set for diagonal (BW) lines

    def y_at(self, intensity: float) -> float:
        """FLOPS at arithmetic intensity I (FLOPs / off-chip byte)."""
        if self.bytes_per_s is not None:
            return intensity * self.bytes_per_s
        if self.flops_per_s is not None:
            return self.flops_per_s
        return 0.0


def ceilings_for(box: BoxSpec) -> List[CeilingLine]:
    """All ceiling lines we want to draw on the roofline for one BoxSpec.

    Returns horizontal FPU ceilings (one per fidelity level present in the
    box's table) plus diagonal DRAM/NoC/ETH lines. The chart layer picks a
    subset to render based on what's interesting.
    """
    lines: List[CeilingLine] = []
    for fidelity in MATH_FIDELITY_LEVELS:
        peak = box.fpu_peak_tflops.get(fidelity)
        if peak is None or peak <= 0:
            continue
        lines.append(
            CeilingLine(
                label=f"FPU peak {fidelity} ({peak:.0f} TFLOPS)",
                kind="fpu",
                flops_per_s=peak * 1e12,
                bytes_per_s=None,
            )
        )
    lines.append(
        CeilingLine(
            label=f"DRAM BW ({box.dram_bw_gbps:.0f} GB/s)",
            kind="dram",
            flops_per_s=None,
            bytes_per_s=box.dram_bw_bytes_per_s(),
        )
    )
    lines.append(
        CeilingLine(
            label=f"NoC BW ({box.noc_bw_gbps:.0f} GB/s)",
            kind="noc",
            flops_per_s=None,
            bytes_per_s=box.noc_bw_bytes_per_s(),
        )
    )
    if box.eth_bw_gbps > 0:
        lines.append(
            CeilingLine(
                label=f"ETH BW ({box.eth_bw_gbps:.0f} GB/s)",
                kind="eth",
                flops_per_s=None,
                bytes_per_s=box.eth_bw_bytes_per_s(),
            )
        )
    return lines


def known_box_names() -> List[str]:
    return [b.name for b in HARDWARE]
