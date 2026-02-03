# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""RooflineEstimate and RooflineContext for tracking cumulative performance metrics.

This module provides classes for aggregating roofline estimates across
multiple operations in a forward/backward pass.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from ..hardware import HardwareSpec, BottleneckType, MathFidelity
from ..memory_tracker import MemoryTracker

if TYPE_CHECKING:
    from ..modules import MockModule
    from ..mock_tensor import MockTensor, TensorLabel


def fpu_eltwise_flops_per_core_per_cycle(
    fidelity: MathFidelity, is_fma: bool = False
) -> int:
    """FPU (Matrix Engine) throughput for elementwise operations (add, sub, fma)

    FPU works on 8x16 LoFi tiles, so that's 128 elements per cycle.
    """
    return 128 / fidelity.value if not is_fma else 256 / fidelity.value


def fpu_mm_flops_per_core_per_cycle(fidelity: MathFidelity) -> int:
    """FPU (Matrix Engine) throughput.

    Matrix engine multiplies tiles of size 8x16 and 16x16 in one cycle at LoFi.
    This gives 8*16*16 = 2048 multiply-accumulates = 4096 FLOPs per cycle.
    Higher fidelity requires more cycles (e.g., HiFi4 = 4 cycles for BF16/TF32).
    """
    return 4096 / fidelity.value


def sfpu_flops_per_core_per_cycle(fidelity: MathFidelity, is_fma: bool = False) -> int:
    """SFPU (Vector Engine) throughput.

    SFPU performs basic operations (add, sub, mul) on 32 fp32 numbers per cycle.
    For FMA (fused multiply-add), throughput doubles to 64 FLOPs/cycle.

    Note: Fidelity does not affect SFPU (it operates on fp32 natively).
    SFPU is preferred for elementwise ops due to FPU tile alignment constraints.

    Args:
        fidelity: Math fidelity (unused, kept for API consistency)
        is_fma: Whether operation uses FMA (doubles throughput)

    Returns:
        Operations per cycle per core
    """
    base_throughput = 32
    return base_throughput * 2 if is_fma else base_throughput


@dataclass
class RooflineEstimate:
    """Performance estimate for a single operation.

    Holds the essential metrics from a roofline analysis:
    - Operation name and phase (forward/backward)
    - FLOPs and bytes transferred
    - Ideal compute and memory times
    - Theoretical execution time and bottleneck classification

    This class combines the functionality of the former PerfModelResult
    (raw performance data) with derived metrics for roofline analysis.

    Essential fields:
        - operation, phase: metadata
        - total_flops, total_bytes: raw counts
        - ideal_compute_ns, ideal_memory_ns: roofline bounds
        - hw: hardware spec (for computing derived metrics)

    Derived (properties):
        - theoretical_time_ns: max(compute, memory)
        - bottleneck: which bound dominates
        - arithmetic_intensity, achieved_tflops, etc.
    """

    operation: str
    phase: str  # "forward" or "backward"
    total_flops: int
    total_bytes: int
    ideal_compute_ns: float
    ideal_memory_ns: float
    hw: HardwareSpec

    @property
    def theoretical_time_ns(self) -> float:
        """Roofline time = max(compute, memory)."""
        return max(self.ideal_compute_ns, self.ideal_memory_ns)

    @property
    def bottleneck(self) -> BottleneckType:
        """Classification based on which bound dominates."""
        if self.ideal_compute_ns > self.ideal_memory_ns * 1.5:
            return BottleneckType.COMPUTE
        elif self.ideal_memory_ns > self.ideal_compute_ns * 1.5:
            return BottleneckType.DRAM
        elif self.ideal_compute_ns > 0 and self.ideal_memory_ns > 0:
            return BottleneckType.BOTH
        else:
            return BottleneckType.SLOW

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte transferred."""
        return self.total_flops / self.total_bytes if self.total_bytes > 0 else 0

    @property
    def achieved_tflops(self) -> float:
        """Achieved TFLOP/s based on theoretical time."""
        t = self.theoretical_time_ns
        return (self.total_flops / t) / 1000 if t > 0 else 0

    @property
    def achieved_dram_bw_gb_s(self) -> float:
        """Achieved DRAM bandwidth in GB/s."""
        t = self.theoretical_time_ns
        return self.total_bytes / t if t > 0 else 0

    def summary(self) -> str:
        """Human-readable summary."""
        return f"""{self.operation} ({self.phase}) Performance
{'=' * 50}
FLOPs:      {self.total_flops:,} ({self.total_flops/1e12:.4f} TFLOPs)
Bytes:      {self.total_bytes:,} ({self.total_bytes/1e9:.4f} GB)
Time:       {self.theoretical_time_ns/1e6:.4f} ms
Bottleneck: {self.bottleneck.value}
Intensity:  {self.arithmetic_intensity:.1f} FLOPs/byte
"""

    def __repr__(self) -> str:
        return (
            f"RooflineEstimate({self.operation}, {self.phase}, "
            f"flops={self.total_flops:,}, bytes={self.total_bytes:,}, "
            f"time={self.theoretical_time_ns/1e6:.4f}ms, {self.bottleneck.value})"
        )


@dataclass
class RooflineContext:
    """Context for tracking cumulative roofline performance across operations.

    The context holds a hardware specification and accumulates estimates
    from each operation during forward and backward passes.

    Example:
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> # ... run forward/backward with operations adding estimates ...
        >>> print(ctx.summary())
    """

    hw: HardwareSpec
    estimates: List[RooflineEstimate] = field(default_factory=list)

    # Memory tracker for detailed allocation tracking
    memory_tracker: Optional[MemoryTracker] = field(default=None)

    def __post_init__(self):
        """Initialize and enable memory tracking."""
        if self.memory_tracker is None:
            self.memory_tracker = MemoryTracker()
        self.enable_memory_tracking()

    def enable_memory_tracking(self) -> None:
        """Enable global memory tracking for MockTensor allocations."""
        from ..mock_tensor import set_global_memory_tracker

        set_global_memory_tracker(self.memory_tracker)

    def disable_memory_tracking(self) -> None:
        """Disable global memory tracking."""
        from ..mock_tensor import set_global_memory_tracker

        set_global_memory_tracker(None)

    def add(self, estimate: RooflineEstimate) -> None:
        """Add an estimate to the context."""
        self.estimates.append(estimate)

    def add_perf_result(self, result: RooflineEstimate) -> None:
        """Add a RooflineEstimate to the context.

        This method exists for backwards compatibility and is equivalent to add().
        """
        self.add(result)

    # =========================================================================
    # Aggregate metrics
    # =========================================================================

    def total_time_ns(self) -> float:
        """Total theoretical time across all operations."""
        return sum(e.theoretical_time_ns for e in self.estimates)

    def total_time_ms(self) -> float:
        """Total theoretical time in milliseconds."""
        return self.total_time_ns() / 1e6

    def total_flops(self) -> int:
        """Total FLOPs across all operations."""
        return sum(e.total_flops for e in self.estimates)

    def total_bytes(self) -> int:
        """Total bytes transferred across all operations."""
        return sum(e.total_bytes for e in self.estimates)

    def forward_time_ns(self) -> float:
        """Total time for forward pass operations."""
        return sum(
            e.theoretical_time_ns for e in self.estimates if e.phase == "forward"
        )

    def backward_time_ns(self) -> float:
        """Total time for backward pass operations."""
        return sum(
            e.theoretical_time_ns for e in self.estimates if e.phase == "backward"
        )

    def forward_flops(self) -> int:
        """Total FLOPs for forward pass."""
        return sum(e.total_flops for e in self.estimates if e.phase == "forward")

    def backward_flops(self) -> int:
        """Total FLOPs for backward pass."""
        return sum(e.total_flops for e in self.estimates if e.phase == "backward")

    # =========================================================================
    # Memory estimation
    # =========================================================================

    def estimate_parameter_memory(self, module: "MockModule") -> int:
        """Estimate memory for model parameters.

        Args:
            module: MockModule to analyze

        Returns:
            Total parameter memory in bytes
        """
        total_bytes = 0
        for name, param in module.parameters().items():
            total_bytes += param.bytes()
        return total_bytes

    def estimate_gradient_memory(self, module: "MockModule") -> int:
        """Estimate memory for gradients (same as parameters).

        Args:
            module: MockModule to analyze

        Returns:
            Total gradient memory in bytes
        """
        # Gradients have same size as parameters
        return self.estimate_parameter_memory(module)

    # =========================================================================
    # Analysis
    # =========================================================================

    def bottleneck_breakdown(self) -> Dict[BottleneckType, int]:
        """Count operations by bottleneck type."""
        breakdown: Dict[BottleneckType, int] = {}
        for e in self.estimates:
            breakdown[e.bottleneck] = breakdown.get(e.bottleneck, 0) + 1
        return breakdown

    def operations_by_phase(self) -> Dict[str, List[RooflineEstimate]]:
        """Group operations by phase."""
        by_phase: Dict[str, List[RooflineEstimate]] = {"forward": [], "backward": []}
        for e in self.estimates:
            if e.phase in by_phase:
                by_phase[e.phase].append(e)
            else:
                by_phase[e.phase] = [e]
        return by_phase

    def achieved_tflops(self) -> float:
        """Calculate achieved TFLOPS across all operations."""
        total_time_s = self.total_time_ns() / 1e9
        if total_time_s == 0:
            return 0.0
        return self.total_flops() / total_time_s / 1e12

    def achieved_bandwidth_gb_s(self) -> float:
        """Calculate achieved memory bandwidth in GB/s."""
        total_time_s = self.total_time_ns() / 1e9
        if total_time_s == 0:
            return 0.0
        return self.total_bytes() / total_time_s / 1e9

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self, module: "MockModule" = None) -> str:
        """Generate a human-readable summary of the roofline analysis.

        Args:
            module: Optional MockModule for memory estimation

        Returns:
            Formatted string summary
        """
        lines = []
        lines.append("=" * 70)
        lines.append("ROOFLINE ANALYSIS SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Hardware: {self.hw.name}")
        lines.append(f"Peak TFLOPs: {self.hw.tflops_per_chip_lofi:.1f} (LoFi)")
        lines.append(f"DRAM BW: {self.hw.dram_bw_gb_s:.0f} GB/s")
        lines.append("")

        # Timing summary
        lines.append("TIMING:")
        lines.append(
            f"  Forward:  {self.forward_time_ns()/1e6:.4f} ms ({self.forward_flops()/1e12:.4f} TFLOPs)"
        )
        lines.append(
            f"  Backward: {self.backward_time_ns()/1e6:.4f} ms ({self.backward_flops()/1e12:.4f} TFLOPs)"
        )
        lines.append(
            f"  Total:    {self.total_time_ms():.4f} ms ({self.total_flops()/1e12:.4f} TFLOPs)"
        )
        lines.append("")

        # Throughput
        lines.append("THROUGHPUT:")
        lines.append(f"  Achieved TFLOPs: {self.achieved_tflops():.2f}")
        lines.append(f"  Achieved BW:     {self.achieved_bandwidth_gb_s():.1f} GB/s")
        lines.append("")

        # Bottleneck breakdown
        breakdown = self.bottleneck_breakdown()
        lines.append("BOTTLENECK BREAKDOWN:")
        for btype, count in sorted(breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"  {btype.value}: {count} ops")
        lines.append("")

        # Operation details
        lines.append("OPERATION DETAILS:")
        lines.append("-" * 70)
        lines.append(
            f"{'Operation':<35} {'Phase':<8} {'Time(ms)':<10} {'TFLOPs':<10} {'Bottleneck':<12}"
        )
        lines.append("-" * 70)
        for e in self.estimates:
            lines.append(
                f"{e.operation:<35} {e.phase:<8} {e.theoretical_time_ns/1e6:<10.4f} "
                f"{e.total_flops/1e12:<10.4f} {e.bottleneck.value:<12}"
            )
        lines.append("=" * 70)

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all accumulated estimates."""
        self.estimates.clear()
        if self.memory_tracker is not None:
            self.memory_tracker.clear()

    # =========================================================================
    # Memory tracking methods
    # =========================================================================

    def peak_memory_tracked(self) -> Tuple[int, Dict["TensorLabel", int]]:
        """Get peak memory usage from tracked allocations.

        Returns:
            Tuple of (peak_bytes, breakdown_by_label)
        """
        if self.memory_tracker is None:
            from ..mock_tensor import TensorLabel

            return 0, {label: 0 for label in TensorLabel}
        return self.memory_tracker.peak_memory()

    def print_peak_memory(self) -> None:
        """Print peak memory usage with breakdown by label."""
        if self.memory_tracker is not None:
            self.memory_tracker.print_peak_memory()

    def plot_memory_usage(
        self,
        filename: str = "memory_usage.png",
        title: str = "Memory Usage Over Time",
        stacked: bool = True,
    ) -> None:
        """Generate a memory usage plot.

        Args:
            filename: Output filename for the plot
            title: Plot title
            stacked: If True, stacked area plot. If False, separate line plots per category.
        """
        if self.memory_tracker is not None:
            self.memory_tracker.plot_memory_usage(filename, title, stacked=stacked)

    def get_memory_timeline(self):
        """Get memory timeline from tracker.

        Returns:
            List of MemorySnapshot objects
        """
        if self.memory_tracker is not None:
            return self.memory_tracker.get_memory_timeline()
        return []
