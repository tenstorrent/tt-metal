# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared data classes and geometry utilities for the fusion infrastructure.

This module contains foundation types and utility functions used by multiple
fusion submodules (codegen, cb_allocator, composition, graph).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import ttnn


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BarrierConfig:
    """Configuration for the two-level barrier between phases.

    Holds GlobalSemaphore references (to prevent GC) and their L1 addresses,
    plus physical core coordinates for the global barrier.
    """

    # L1 addresses of per-core flags (from GlobalSemaphore.address())
    compute_done_addr: int = 0
    writer_done_addr: int = 0
    global_arrive_addr: int = 0
    global_release_addr: int = 0

    # Physical core coordinates for global barrier
    num_cores: int = 1
    core0_phys_x: int = 0
    core0_phys_y: int = 0
    mcast_start_x: int = 0
    mcast_start_y: int = 0
    mcast_end_x: int = 0
    mcast_end_y: int = 0

    # GlobalSemaphore references (prevent GC)
    _sem_refs: List[Any] = field(default_factory=list)


@dataclass
class BarrierSegment:
    """A barrier scope covering a range of phase transitions.

    Each segment has its own ``global_arrive`` / ``global_release``
    GlobalSemaphore pair and physical core coordinates for NOC multicast.
    """

    config: BarrierConfig  # Physical core coords + mcast params
    arrive_addr: int = 0  # GlobalSemaphore L1 address for arrive
    release_addr: int = 0  # GlobalSemaphore L1 address for release


@dataclass
class MultiBarrierSpec:
    """Multi-segment barrier for OpGraph paths.

    When a fused kernel transitions between phases, the barrier scope may
    change (e.g. stem barrier over 8 cores -> branch barrier over 4 cores).
    ``transition_map`` maps each phase-transition index to the barrier
    segment and per-segment call index to use.
    """

    segments: List[BarrierSegment] = field(default_factory=list)
    compute_done_addr: int = 0
    writer_done_addr: int = 0
    # Map: phase_transition_index -> (segment_index, call_index_within_segment)
    transition_map: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    _sem_refs: List[Any] = field(default_factory=list)


class _BuildResult:
    """Internal intermediate result from building a fused descriptor.

    NOT part of the public API.  Only converted to ``FusedOp`` at the
    outermost ``build()`` call.
    """

    __slots__ = ("descriptor", "input_tensors", "output_tensors", "semaphores", "kernel_labels", "kernel_phase_map")

    def __init__(self, descriptor, input_tensors, output_tensors, semaphores=(), kernel_labels=(), kernel_phase_map=()):
        self.descriptor = descriptor
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.semaphores = semaphores
        self.kernel_labels = kernel_labels
        self.kernel_phase_map = kernel_phase_map


# =============================================================================
# Geometry Utilities
# =============================================================================


def _core_range_set_to_coords(core_range_set: Any) -> Set[Tuple[int, int]]:
    """Convert a CoreRangeSet to a set of (x, y) coordinate tuples."""
    coords: Set[Tuple[int, int]] = set()
    for cr in core_range_set.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                coords.add((x, y))
    return coords


def _core_ranges_key(core_ranges: Any) -> frozenset:
    """Create a hashable key from a CoreRangeSet for grouping."""
    return frozenset((cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in core_ranges.ranges())


def _coords_to_core_range_set(coords: Set[Tuple[int, int]]) -> Any:
    """Convert a set of (x, y) tuples to a CoreRangeSet.

    Each coordinate becomes a single-core CoreRange.  CoreRangeSet
    merges adjacent ranges internally.
    """
    ranges = set()
    for x, y in coords:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)))
    return ttnn.CoreRangeSet(ranges)


def _get_node_core_range(node: Any) -> Any:
    """Extract the core range from a node's op descriptor.

    Returns the union of all kernel core_ranges in the node's
    ProgramDescriptor.
    """
    all_coords: Set[Tuple[int, int]] = set()
    for kernel in node.op.descriptor.kernels:
        all_coords |= _core_range_set_to_coords(kernel.core_ranges)
    return _coords_to_core_range_set(all_coords)


def _get_risc_type(kernel_desc: "ttnn.KernelDescriptor") -> str:
    """Return the RISC processor type: 'riscv_0', 'riscv_1', or 'compute'."""
    config = kernel_desc.config
    if isinstance(config, ttnn.ComputeConfigDescriptor):
        return "compute"
    elif isinstance(config, ttnn.ReaderConfigDescriptor):
        return "riscv_0"
    elif isinstance(config, ttnn.WriterConfigDescriptor):
        return "riscv_1"
    elif isinstance(config, ttnn.DataMovementConfigDescriptor):
        if config.processor == ttnn.DataMovementProcessor.RISCV_0:
            return "riscv_0"
        else:
            return "riscv_1"
    return "unknown"


def _kernel_overlaps_core_range(
    kernel_desc: "ttnn.KernelDescriptor",
    target_core_range: Optional[Any],
) -> bool:
    """Check whether *kernel_desc*'s core ranges overlap *target_core_range*.

    Returns ``True`` when *target_core_range* is ``None`` (no filtering).
    This is used to skip kernels that operate on disjoint core subsets
    during tree / branch builds.  For example, a block-sharded LayerNorm
    on a 2-row grid produces two riscv_0 kernels — a multicast sender on
    row 0 and a receiver on row 1.  When the tree builder targets only
    one branch (say row 0), the receiver kernel must be excluded so that
    it does not overwrite the sender in the role-key map.
    """
    if target_core_range is None:
        return True
    target_coords = _core_range_set_to_coords(target_core_range)
    kernel_coords = _core_range_set_to_coords(kernel_desc.core_ranges)
    return bool(target_coords & kernel_coords)


def _get_role_key(
    kernel_desc: "ttnn.KernelDescriptor",
    target_core_range: Optional[Any] = None,
) -> Tuple[str, frozenset]:
    """Return (risc_type, core_ranges_key) identifying this kernel's role.

    If target_core_range is set, all kernels are mapped to that range
    regardless of their native core_ranges.  This collapses kernels with
    different ranges (e.g. stem vs branch) into the same role when building
    a fused kernel for a specific core group.
    """
    cr = target_core_range if target_core_range is not None else kernel_desc.core_ranges
    return (_get_risc_type(kernel_desc), _core_ranges_key(cr))


__all__ = [
    "BarrierConfig",
    "BarrierSegment",
    "MultiBarrierSpec",
    "_BuildResult",
    "_core_range_set_to_coords",
    "_core_ranges_key",
    "_coords_to_core_range_set",
    "_get_node_core_range",
    "_get_risc_type",
    "_get_role_key",
    "_kernel_overlaps_core_range",
]
