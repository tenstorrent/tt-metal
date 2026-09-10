# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BarrierConfig:
    """Configuration for the two-level barrier between phases.

    Holds L1 bank-word addresses and physical core coordinates for the
    cross-core barrier.
    """

    # L1 addresses of per-core bank words
    compute_done_addr: int = 0
    writer_done_addr: int = 0
    global_arrive_addr: int = 0
    global_release_addr: int = 0

    # Physical core coordinates for global barrier
    num_release_cores: int = 1  # Total cores that receive release (for unicast)
    num_arrive_cores: int = 1  # Cores that arrive (for threshold)
    core0_phys_x: int = 0
    core0_phys_y: int = 0
    # Physical (x, y) of every core EXCEPT core 0, for unicast release
    other_core_phys_coords: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class BarrierSegment:
    """A barrier scope covering a range of phase transitions.

    Each segment has its own ``global_arrive`` / ``global_release`` bank words
    and physical core coordinates for NOC unicast.
    """

    config: BarrierConfig  # Physical core coords + mcast params
    arrive_addr: int = 0  # L1 bank-word address for arrive
    release_addr: int = 0  # L1 bank-word address for release


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
    reset_done_addr: int = 0
    pack_drained_addr: int = 0
    math_drained_addr: int = 0
    # Map: phase_transition_index -> (segment_index, call_index_within_segment)
    transition_map: Dict[int, Tuple[int, int]] = field(default_factory=dict)


@dataclass
class _SemaphoreSpec:
    """Blueprint for one logical word in a fusion semaphore bank.

    Stored in ``_CacheEntry`` so no L1 is pinned between dispatches. At each
    dispatch, one bank is allocated from all specs, its word addresses are
    patched into the cached ``ProgramDescriptor`` runtime args, and the bank
    is released after dispatch submission via command-queue-safe lifetime.
    """

    core_ranges: Any  # CoreRangeSet
    initial_value: int = 0


def _allocate_fusion_semaphore_bank(device, sem_specs):
    """Allocate one command-lifetime L1 bank for all fusion barrier words."""
    if not sem_specs:
        return None
    return ttnn._ttnn.operations.experimental.FusionSemaphoreBank(
        device,
        [spec.core_ranges for spec in sem_specs],
        [spec.initial_value for spec in sem_specs],
    )


def _cb_has_backing(cb_descriptor) -> bool:
    """True if a CBDescriptor has Buffer* or tensor backing."""
    try:
        if isinstance(cb_descriptor, ttnn.CBDescriptor):
            return ttnn._ttnn.operations.experimental.cb_has_backing(cb_descriptor)
    except TypeError:
        pass
    has_buffer = getattr(cb_descriptor, "has_buffer", False)
    return has_buffer() if callable(has_buffer) else bool(has_buffer)


def _cb_backing_address(cb_descriptor):
    """L1 address of a CBDescriptor's Buffer* or tensor backing, or None."""
    try:
        if isinstance(cb_descriptor, ttnn.CBDescriptor):
            return ttnn._ttnn.operations.experimental.cb_backing_address(cb_descriptor)
    except TypeError:
        pass
    buffer_address = getattr(cb_descriptor, "buffer_address", None)
    return buffer_address() if callable(buffer_address) else None


def _copy_cb_backing(dst, src) -> None:
    """Copy Buffer* and tensor backing from one CBDescriptor to another."""
    try:
        if isinstance(dst, ttnn.CBDescriptor) and isinstance(src, ttnn.CBDescriptor):
            ttnn._ttnn.operations.experimental.copy_cb_backing(dst, src)
            return
    except TypeError:
        pass
    set_buffer = getattr(dst, "set_buffer_from_cb", None)
    if callable(set_buffer):
        set_buffer(src)


class _BuildResult:
    """Internal intermediate result from building a fused descriptor.

    NOT part of the public API.  Only converted to ``FusedOp`` at the
    outermost ``build()`` call.
    """

    __slots__ = (
        "descriptor",
        "input_tensors",
        "output_tensors",
        "kernel_labels",
        "kernel_phase_map",
        "cb_source_map",
        "rebind_source_map",
        "global_cb_source_map",
        "output_source_map",
        "sem_specs",
        "sem_addrs",
    )

    def __init__(
        self,
        descriptor,
        input_tensors,
        output_tensors,
        kernel_labels=(),
        kernel_phase_map=(),
        cb_source_map=(),
        rebind_source_map=(),
        global_cb_source_map=(),
        output_source_map=(),
        sem_specs=(),
        sem_addrs=(),
    ):
        self.descriptor = descriptor
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.kernel_labels = kernel_labels
        self.kernel_phase_map = kernel_phase_map
        self.cb_source_map = cb_source_map
        self.rebind_source_map = rebind_source_map
        self.global_cb_source_map = global_cb_source_map
        self.output_source_map = output_source_map
        self.sem_specs = sem_specs
        self.sem_addrs = sem_addrs


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
    """Return the RISC processor type: 'riscv_0', 'riscv_1', or 'compute'.

    ReaderConfigDescriptor maps to RISCV_1 (NCRISC) and
    WriterConfigDescriptor maps to RISCV_0 (BRISC) — matching the
    hardware mapping in ``kernel_types.cpp`` where
    ``ReaderDataMovementConfig`` sets ``processor = RISCV_1`` and
    ``WriterDataMovementConfig`` sets ``processor = RISCV_0``.
    """
    config = kernel_desc.config
    if isinstance(config, ttnn.ComputeConfigDescriptor):
        return "compute"
    elif isinstance(config, ttnn.ReaderConfigDescriptor):
        return "riscv_1"
    elif isinstance(config, ttnn.WriterConfigDescriptor):
        return "riscv_0"
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
    on a 2-row grid produces two riscv_1 kernels — a multicast sender on
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


# =============================================================================
# No-Op Sentinel
# =============================================================================


class _NoOpProgramDescriptor:
    """A ProgramDescriptor with no kernels, CBs, or semaphores.

    Used by the narrow→wide topology support: cores active in a wide
    child but not in the narrow parent get a ``_NOOP_OP`` entry so
    their phase count matches cores that participate in every phase.
    """

    kernels = []
    cbs = []
    semaphores = []


_NOOP_OP = OpDescriptor(
    descriptor=_NoOpProgramDescriptor(),
    input_tensors=[],
    output_tensors=[],
    name="noop",
    program_cache_key=0,
)


__all__ = [
    "BarrierConfig",
    "BarrierSegment",
    "MultiBarrierSpec",
    "_BuildResult",
    "_NOOP_OP",
    "_SemaphoreSpec",
    "_allocate_fusion_semaphore_bank",
    "_core_range_set_to_coords",
    "_core_ranges_key",
    "_coords_to_core_range_set",
    "_get_node_core_range",
    "_get_risc_type",
    "_get_role_key",
    "_kernel_overlaps_core_range",
]
