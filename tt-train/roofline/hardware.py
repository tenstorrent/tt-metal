# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware specifications for roofline modeling.

Adapted from tt_perf_calculator.py for standalone roofline estimation
without ttml/ttnn dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class MathFidelity(Enum):
    """Math fidelity levels affecting compute throughput."""

    LoFi = 1  # 1 cycle per tile operation
    HiFi2 = 2  # 2 cycles per tile operation
    HiFi3 = 3  # 3 cycles per tile operation
    HiFi4 = 4  # 4 cycles per tile operation


class DataType(Enum):
    """Supported data types with their byte sizes."""

    BFLOAT16 = 2
    FLOAT16 = 2
    FLOAT32 = 4
    BFLOAT8_B = 1
    BFLOAT4_B = 0.5
    INT8 = 1
    INT32 = 4


class BottleneckType(Enum):
    """Classification of operation bottleneck."""

    COMPUTE = "COMPUTE"
    DRAM = "DRAM"
    NOC = "NOC"
    L1_CAPACITY = "L1_CAPACITY"
    BOTH = "BOTH"  # Both compute and memory bound (>65% each)
    SLOW = "SLOW"  # Neither bound (<65% both) - needs optimization
    HOST = "HOST"


@dataclass
class HardwareSpec:
    """Hardware specifications for a TT device."""

    name: str

    # Compute
    tensix_cores_per_chip: int
    clock_ghz: float

    # Memory
    dram_gb_per_chip: float
    dram_bw_gb_s: float
    sram_mb_per_core: float

    # NoC
    noc_bw_tb_s: float
    noc_link_bw_gb_s: float

    # Fabric (for multi-chip)
    eth_bw_gb_s_per_chip: float

    # Defaults at end
    mul_adds_per_cycle: int = 4096  # 8x16 × 16x16 matrix engine
    chips_per_galaxy: int = 1

    @property
    def sram_mb_per_chip(self) -> float:
        return self.tensix_cores_per_chip * self.sram_mb_per_core

    @property
    def tflops_per_core_lofi(self) -> float:
        """Peak TFLOPS per core at LoFi."""
        return self.mul_adds_per_cycle * self.clock_ghz / 1000

    @property
    def tflops_per_chip_lofi(self) -> float:
        """Peak TFLOPS per chip at LoFi."""
        return self.tflops_per_core_lofi * self.tensix_cores_per_chip

    def tflops_per_chip(self, fidelity: MathFidelity) -> float:
        """Peak TFLOPS per chip at given math fidelity."""
        return self.tflops_per_chip_lofi / fidelity.value

    def tflops_per_galaxy(self, fidelity: MathFidelity) -> float:
        """Peak TFLOPS per Galaxy at given math fidelity."""
        return self.tflops_per_chip(fidelity) * self.chips_per_galaxy

    @property
    def dram_bw_tb_s_per_galaxy(self) -> float:
        """Aggregate DRAM bandwidth for Galaxy."""
        return self.dram_bw_gb_s * self.chips_per_galaxy / 1000

    @property
    def dram_gb_per_galaxy(self) -> float:
        """Total DRAM capacity for Galaxy."""
        return self.dram_gb_per_chip * self.chips_per_galaxy

    @property
    def sram_gb_per_galaxy(self) -> float:
        """Total SRAM capacity for Galaxy."""
        return self.sram_mb_per_chip * self.chips_per_galaxy / 1000

    def critical_intensity(self, fidelity: MathFidelity) -> float:
        """
        Critical arithmetic intensity (FLOPs/byte) where compute = memory time.
        Operations above this are compute-bound, below are memory-bound.
        """
        peak_tflops = self.tflops_per_chip(fidelity)
        dram_bw_tb_s = self.dram_bw_gb_s / 1000
        return peak_tflops / dram_bw_tb_s  # TFLOP/TB = FLOP/byte


# Pre-defined hardware configurations
WORMHOLE_N150 = HardwareSpec(
    name="Wormhole n150s",
    tensix_cores_per_chip=72,  # 80 total, 72 usable after harvesting
    clock_ghz=1.0,
    dram_gb_per_chip=12,
    dram_bw_gb_s=288,  # GDDR6 @ 12 GT/s
    sram_mb_per_core=1.5,
    noc_bw_tb_s=1.4,
    noc_link_bw_gb_s=83,
    eth_bw_gb_s_per_chip=200,  # 16 × 100GbE = 1.6 Tb/s = 200 GB/s
    chips_per_galaxy=1,
)

WORMHOLE_N300 = HardwareSpec(
    name="Wormhole n300s",
    tensix_cores_per_chip=128,  # 64 per ASIC × 2
    clock_ghz=1.0,
    dram_gb_per_chip=24,
    dram_bw_gb_s=576,  # 288 × 2
    sram_mb_per_core=1.5,
    noc_bw_tb_s=2.8,  # 1.4 × 2
    noc_link_bw_gb_s=83,
    eth_bw_gb_s_per_chip=400,
    chips_per_galaxy=1,
)

WORMHOLE_GALAXY = HardwareSpec(
    name="Wormhole Galaxy",
    tensix_cores_per_chip=80,  # Full 80 cores in Galaxy config
    clock_ghz=1.0,
    dram_gb_per_chip=12,
    dram_bw_gb_s=288,
    sram_mb_per_core=1.5,
    noc_bw_tb_s=1.4,
    noc_link_bw_gb_s=83,
    eth_bw_gb_s_per_chip=200,
    chips_per_galaxy=32,
)

BLACKHOLE_P100 = HardwareSpec(
    name="Blackhole",
    tensix_cores_per_chip=120,
    clock_ghz=1.35,
    dram_gb_per_chip=32,
    dram_bw_gb_s=480,
    sram_mb_per_core=1.5,  # 210 MB / 140 cores
    noc_bw_tb_s=1.4,
    noc_link_bw_gb_s=200,
    eth_bw_gb_s_per_chip=480,  # 10 × 400GbE × 48 GB/s effective
    chips_per_galaxy=1,
)

BLACKHOLE_P150 = HardwareSpec(
    name="Blackhole",
    tensix_cores_per_chip=120,
    clock_ghz=1.35,
    dram_gb_per_chip=32,
    dram_bw_gb_s=512,
    sram_mb_per_core=1.5,  # 210 MB / 140 cores
    noc_bw_tb_s=1.4,
    noc_link_bw_gb_s=200,
    eth_bw_gb_s_per_chip=480,  # 10 × 400GbE × 48 GB/s effective
    chips_per_galaxy=1,
)

BLACKHOLE_GALAXY = HardwareSpec(
    name="Blackhole Galaxy",
    tensix_cores_per_chip=140,
    clock_ghz=1.35,
    dram_gb_per_chip=32,
    dram_bw_gb_s=512,
    sram_mb_per_core=1.5,
    noc_bw_tb_s=1.4,
    noc_link_bw_gb_s=200,
    eth_bw_gb_s_per_chip=480,
    chips_per_galaxy=32,
)
