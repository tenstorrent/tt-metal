# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Memory tracking for roofline modeling.

This module provides MemoryTracker for tracking tensor allocations
and deallocations over time, enabling memory usage analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .mock_tensor import MockTensor, TensorLabel


@dataclass
class MemoryEvent:
    """A single memory allocation or deallocation event.

    Attributes:
        tick: The time tick (event number) when this occurred
        is_allocation: True for allocation, False for deallocation
        size_bytes: Size in bytes of the tensor
        label: Category label for the tensor
        name: Optional name for the tensor
        tensor_id: Unique ID of the tensor (id())
    """

    tick: int
    is_allocation: bool
    size_bytes: int
    label: Optional["TensorLabel"]
    name: Optional[str]
    tensor_id: int


@dataclass
class MemorySnapshot:
    """Memory state at a specific point in time.

    Attributes:
        tick: The time tick
        total_bytes: Total memory usage at this tick
        by_label: Memory usage breakdown by label
    """

    tick: int
    total_bytes: int
    by_label: Dict["TensorLabel", int]


class MemoryTracker:
    """Tracks memory allocations and deallocations over time.

    This class maintains a log of all tensor allocations/deallocations
    and can compute memory usage at any point, peak memory, and generate
    memory usage plots.

    Example:
        >>> tracker = MemoryTracker()
        >>> set_global_memory_tracker(tracker)
        >>> # ... run model ...
        >>> peak, breakdown = tracker.peak_memory()
        >>> tracker.plot_memory_usage("memory_plot.png")
    """

    def __init__(self):
        """Initialize the memory tracker."""
        self._events: List[MemoryEvent] = []
        self._current_tick: int = 0
        self._live_tensors: Dict[
            int, Tuple[int, Optional["TensorLabel"], Optional[str]]
        ] = {}  # id -> (bytes, label, name)

    def track_allocation(self, tensor: "MockTensor") -> None:
        """Track a tensor allocation.

        Args:
            tensor: The MockTensor being allocated
        """
        from .mock_tensor import TensorLabel

        size_bytes = tensor.bytes()
        label = tensor.label
        name = tensor.name
        tensor_id = id(tensor)

        # Record the event
        event = MemoryEvent(
            tick=self._current_tick,
            is_allocation=True,
            size_bytes=size_bytes,
            label=label,
            name=name,
            tensor_id=tensor_id,
        )
        self._events.append(event)
        self._current_tick += 1

        # Track live tensor
        self._live_tensors[tensor_id] = (size_bytes, label, name)

    def track_deallocation(self, tensor: "MockTensor") -> None:
        """Track a tensor deallocation.

        Args:
            tensor: The MockTensor being deallocated
        """
        tensor_id = id(tensor)

        # Only track if we tracked the allocation
        if tensor_id not in self._live_tensors:
            return

        size_bytes, label, name = self._live_tensors[tensor_id]

        # Record the event
        event = MemoryEvent(
            tick=self._current_tick,
            is_allocation=False,
            size_bytes=size_bytes,
            label=label,
            name=name,
            tensor_id=tensor_id,
        )
        self._events.append(event)
        self._current_tick += 1

        # Remove from live tensors
        del self._live_tensors[tensor_id]

    def get_memory_timeline(self) -> List[MemorySnapshot]:
        """Compute memory usage at each event tick.

        Returns:
            List of MemorySnapshot objects, one per tick
        """
        from .mock_tensor import TensorLabel

        snapshots = []
        total_bytes = 0
        by_label: Dict[TensorLabel, int] = {label: 0 for label in TensorLabel}

        for event in self._events:
            if event.is_allocation:
                total_bytes += event.size_bytes
                if event.label is not None:
                    by_label[event.label] = (
                        by_label.get(event.label, 0) + event.size_bytes
                    )
            else:
                total_bytes -= event.size_bytes
                if event.label is not None:
                    by_label[event.label] = (
                        by_label.get(event.label, 0) - event.size_bytes
                    )

            snapshots.append(
                MemorySnapshot(
                    tick=event.tick,
                    total_bytes=total_bytes,
                    by_label=dict(by_label),
                )
            )

        return snapshots

    def peak_memory(self) -> Tuple[int, Dict["TensorLabel", int]]:
        """Compute peak memory usage and breakdown by label.

        Returns:
            Tuple of (peak_bytes, breakdown_by_label_at_peak)
        """
        from .mock_tensor import TensorLabel

        snapshots = self.get_memory_timeline()

        if not snapshots:
            return 0, {label: 0 for label in TensorLabel}

        # Find peak
        peak_snapshot = max(snapshots, key=lambda s: s.total_bytes)
        return peak_snapshot.total_bytes, peak_snapshot.by_label

    def current_memory(self) -> Tuple[int, Dict["TensorLabel", int]]:
        """Get current memory usage and breakdown.

        Returns:
            Tuple of (current_bytes, breakdown_by_label)
        """
        from .mock_tensor import TensorLabel

        total = 0
        by_label: Dict[TensorLabel, int] = {label: 0 for label in TensorLabel}

        for tensor_id, (size_bytes, label, name) in self._live_tensors.items():
            total += size_bytes
            if label is not None:
                by_label[label] = by_label.get(label, 0) + size_bytes

        return total, by_label

    def print_peak_memory(self) -> None:
        """Print peak memory usage with breakdown."""
        from .mock_tensor import TensorLabel

        peak_bytes, breakdown = self.peak_memory()

        print(f"Peak Memory Usage: {peak_bytes / 1e9:.3f} GB ({peak_bytes:,} bytes)")
        print("Peak Memory Breakdown by category:")
        for label in TensorLabel:
            label_bytes = breakdown.get(label, 0)
            pct = (label_bytes / peak_bytes * 100) if peak_bytes > 0 else 0
            print(f"  {label.value:15s}: {label_bytes / 1e9:.3f} GB ({pct:.1f}%)")

    def plot_memory_usage(
        self,
        filename: str = "memory_usage.png",
        title: str = "Memory Usage Over Time",
        figsize: Tuple[int, int] = (14, 8),
        stacked: bool = True,
    ) -> None:
        """Generate a memory usage plot.

        Args:
            filename: Output filename for the plot
            title: Plot title
            figsize: Figure size (width, height) in inches
            stacked: If True, use stacked area plot. If False, use separate line plots.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib not available, skipping plot generation")
            return

        from .mock_tensor import TensorLabel

        snapshots = self.get_memory_timeline()

        if not snapshots:
            print("No memory events to plot")
            return

        # Prepare data
        ticks = [s.tick for s in snapshots]

        # Data by label
        label_data = {}
        for label in TensorLabel:
            label_data[label] = [
                s.by_label.get(label, 0) / 1e9 for s in snapshots
            ]  # Convert to GB

        # Color scheme
        colors = {
            TensorLabel.PARAMETER: "#2196F3",  # Blue
            TensorLabel.OPTIMIZER_STATE: "#4CAF50",  # Green
            TensorLabel.ACTIVATION: "#FF9800",  # Orange
            TensorLabel.GRADIENT: "#F44336",  # Red
        }

        if stacked:
            # Create single stacked area plot
            fig, ax = plt.subplots(figsize=figsize)

            y_stack = np.zeros(len(ticks))
            for label in TensorLabel:
                y_data = np.array(label_data[label])
                ax.fill_between(
                    ticks,
                    y_stack,
                    y_stack + y_data,
                    alpha=0.7,
                    label=label.value.replace("_", " ").title(),
                    color=colors.get(label, None),
                )
                y_stack += y_data

            # Plot total line
            total_data = [s.total_bytes / 1e9 for s in snapshots]
            ax.plot(
                ticks,
                total_data,
                color="black",
                linewidth=1,
                linestyle="--",
                alpha=0.5,
                label="Total",
            )

            # Mark peak
            peak_idx = np.argmax(total_data)
            ax.axvline(x=ticks[peak_idx], color="red", linestyle=":", alpha=0.7)
            ax.annotate(
                f"Peak: {total_data[peak_idx]:.2f} GB",
                xy=(ticks[peak_idx], total_data[peak_idx]),
                xytext=(ticks[peak_idx] + len(ticks) * 0.05, total_data[peak_idx]),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            )

            ax.set_xlabel("Time (tensor allocation tick)")
            ax.set_ylabel("Memory Usage (GB)")
            ax.set_title(title)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(ticks) if ticks else 1)
            ax.set_ylim(0, None)
        else:
            # Create subplots for each category to show individual trends clearly
            fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
            axes = axes.flatten()

            total_data = [s.total_bytes / 1e9 for s in snapshots]
            peak_idx = np.argmax(total_data)

            for i, label in enumerate(TensorLabel):
                ax = axes[i]
                y_data = np.array(label_data[label])

                ax.fill_between(ticks, 0, y_data, alpha=0.5, color=colors.get(label))
                ax.plot(ticks, y_data, color=colors.get(label), linewidth=1)

                # Mark peak tick
                ax.axvline(x=ticks[peak_idx], color="red", linestyle=":", alpha=0.5)

                label_name = label.value.replace("_", " ").title()
                max_val = max(y_data) if y_data.size > 0 else 0
                ax.set_title(f"{label_name} (max: {max_val:.3f} GB)")
                ax.set_ylabel("GB")
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, None)

            # Set common x-label
            for ax in axes[-2:]:
                ax.set_xlabel("Time (tensor allocation tick)")

            fig.suptitle(title, fontsize=12)

        # Save figure
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"Memory usage plot saved to: {filename}")

    def clear(self) -> None:
        """Clear all tracking data."""
        self._events.clear()
        self._current_tick = 0
        self._live_tensors.clear()

    def __repr__(self) -> str:
        return (
            f"MemoryTracker(events={len(self._events)}, "
            f"live_tensors={len(self._live_tensors)}, "
            f"current_tick={self._current_tick})"
        )
