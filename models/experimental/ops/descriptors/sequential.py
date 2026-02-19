# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sequential Kernel Chaining Infrastructure

Fuses multiple operations to run sequentially on the SAME cores within a
single program.  All readers/writers run for every phase.  Data flows through
DRAM between phases (Writer_N -> DRAM -> Reader_{N+1}).

The fusion tree is a standard tree of OpNode objects.  Each node holds one
operation and optional children.  Parent→child edges encode sequential
ordering; sibling nodes run in parallel on disjoint core subsets.  A linear
chain is a tree with branching factor 1.

CB pool allocation: CBs from all phases are assigned to hardware slots based
on a compatibility key (data_format, page_size, unpack_to_dest_mode).  Phases
with matching configs share a slot; mismatches get separate slots.  Errors if
the total exceeds the device's CB slot limit.

Two-level barrier synchronization between phases:
  - Local barrier (per core): L1 flags allocated via GlobalSemaphore.
    Compute/writer signal done, reader waits then resets CBs.
  - Global barrier (across cores): Reader uses noc_semaphore_inc/wait
    on GlobalSemaphore L1 words, then sets global_release which also
    serves as the phase release signal for compute/writer.

Usage (linear chain):
    >>> fused = Sequential(op0, op1, op2).build()
    >>> composite.launch([fused])

Usage (branching tree):
    >>> fused = Sequential(stem, Parallel(branch_a, branch_b)).build()
    >>> composite.launch([fused])
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set, Any
import logging
import re
import os

import ttnn
from ttnn._ttnn.program_descriptor import UnpackToDestMode

from models.experimental.ops.descriptors.op_descriptor import FusedOp, OpDescriptor
from models.experimental.ops.descriptors import cpp_parser

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _core_range_set_to_coords(core_range_set: Any) -> Set[Tuple[int, int]]:
    """Convert a CoreRangeSet to a set of (x, y) coordinate tuples."""
    coords: Set[Tuple[int, int]] = set()
    for cr in core_range_set.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                coords.add((x, y))
    return coords


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CBInfo:
    """Information about a circular buffer extracted from a CBDescriptor."""

    original_index: int
    total_size: int
    data_format: Any  # tt::DataFormat
    page_size: int
    core_ranges: Any  # CoreRangeSet
    has_buffer: bool = False  # True if backed by an L1 Buffer allocation
    unpack_to_dest_mode: Any = None  # UnpackToDestMode enum (Default or UnpackToDestFp32)


@dataclass
class PhaseInfo:
    """Information about a phase (op) in the sequential chain."""

    phase_idx: int
    op_descriptor: OpDescriptor
    cb_info: Dict[int, CBInfo] = field(default_factory=dict)


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
class OpNode:
    """A node in the fusion tree.

    Each node holds one operation.  Parent→child edges encode sequential
    ordering (parent runs before child).  Sibling nodes run in parallel on
    disjoint core subsets.  A leaf node has no children.

    The node's core range is derived from its op's ProgramDescriptor
    kernels via ``_get_node_core_range()`` — there is no separate
    core_range field.
    """

    op: OpDescriptor
    children: List["OpNode"] = field(default_factory=list)


# =============================================================================
# Internal intermediate result (not part of public API)
# =============================================================================


class _BuildResult:
    """Internal intermediate result from building a fused descriptor.

    NOT part of the public API.  Only converted to ``FusedOp`` at the
    outermost ``build()`` call.
    """

    __slots__ = ("descriptor", "input_tensors", "output_tensors", "semaphores")

    def __init__(self, descriptor, input_tensors, output_tensors, semaphores=()):
        self.descriptor = descriptor
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.semaphores = semaphores


# =============================================================================
# High-Level API: Sequential / Parallel
# =============================================================================


class Sequential:
    """A sequence of ops to fuse into a single dispatch.

    Items can be ``OpDescriptor``, ``Sequential``, or ``Parallel`` objects.
    Nested ``Sequential`` items are automatically flattened.

    Usage::

        # Inline
        fused = Sequential(op0, op1, op2).build()

        # Incremental
        s = Sequential(op0)
        s.add(op1).add(op2)
        fused = s.build()

        # Composition
        stem = Sequential(op0, op1)
        full = Sequential(stem, op2).build()  # flattened
    """

    def __init__(self, *items):
        if not items:
            raise ValueError("Sequential() requires at least 1 item")
        self._items = list(items)

    def add(self, item):
        """Append an item.  Returns self for chaining."""
        self._items.append(item)
        return self

    def build(self, device=None) -> FusedOp:
        """Build the fused op.  Device is auto-extracted from tensors if not provided."""
        r = self._build_internal(device)
        return FusedOp(
            op=OpDescriptor(r.descriptor, r.input_tensors, r.output_tensors),
            semaphores=r.semaphores,
        )

    def _build_internal(self, device=None) -> _BuildResult:
        """Internal build returning intermediate _BuildResult."""
        if device is None:
            device = _extract_device(self._items)
        nodes = _resolve(self)
        if len(nodes) != 1:
            raise ValueError("Sequential must resolve to a single root node")
        return OpGraphBuilder(nodes[0])._build_internal(device)


class Parallel:
    """Items that run in parallel on disjoint core subsets.

    Each item runs independently on its own cores.  Items can be
    ``OpDescriptor``, ``Sequential``, or ``Parallel`` objects.

    Usage::

        # Inline
        fused = Parallel(op_a, op_b).build()
        composite.launch([fused])

        # As part of a Sequential
        fused = Sequential(stem, Parallel(branch_a, branch_b)).build()
    """

    def __init__(self, *items):
        if len(items) < 2:
            raise ValueError("Parallel() requires at least 2 items")
        self._items = list(items)

    def add(self, item):
        """Add a branch.  Returns self for chaining."""
        self._items.append(item)
        return self

    def build(self, device=None) -> FusedOp:
        """Build each item independently and merge into one FusedOp."""
        r = self._build_internal(device)
        return FusedOp(
            op=OpDescriptor(r.descriptor, r.input_tensors, r.output_tensors),
            semaphores=r.semaphores,
        )

    def _build_internal(self, device=None) -> _BuildResult:
        """Internal build returning intermediate _BuildResult."""
        if device is None:
            device = _extract_device(self._items)
        built = [_build_item(item, device) for item in self._items]
        return _merge_build_results(built)


def _resolve(item) -> List[OpNode]:
    """Convert a user-facing item into a list of OpNode trees.

    Handles all three types uniformly:
    - ``OpDescriptor`` → ``[OpNode(op)]``
    - ``Parallel`` → flat list of children (one OpNode per branch)
    - ``Sequential`` → single-element list containing a chain of OpNodes

    When a ``Sequential`` item in the middle resolves to multiple nodes
    (i.e. it's a ``Parallel``), a ``ValueError`` is raised because the
    tree diverges and can't rejoin.
    """
    if isinstance(item, OpDescriptor):
        return [OpNode(item)]

    if isinstance(item, Parallel):
        return [node for child in item._items for node in _resolve(child)]

    if isinstance(item, Sequential):
        # Flatten nested Sequential items
        flat: List = []
        for sub in item._items:
            if isinstance(sub, Sequential):
                flat.extend(sub._items)
            else:
                flat.append(sub)

        if not flat:
            raise ValueError("Sequential() has no items after flattening")

        # Process right-to-left: resolve the last item to get tail nodes,
        # then walk remaining items in reverse, each becoming parent of tail.
        tail = _resolve(flat[-1])

        for sub_item in reversed(flat[:-1]):
            parents = _resolve(sub_item)
            if len(parents) != 1:
                raise ValueError(
                    "Items before a Parallel in a Sequential are not allowed — "
                    "the tree would diverge and can't rejoin.  Place trailing "
                    "items inside each branch instead."
                )
            _set_leaf_children(parents[0], tail)
            tail = parents

        return tail

    if isinstance(item, FusedOp):
        raise TypeError(
            "FusedOp cannot be nested in Sequential/Parallel — "
            "it is the result of build() and has already been fused."
        )

    raise TypeError(f"Unsupported item type: {type(item).__name__}")


def _extract_device(items):
    """Walk item tree, return device from first tensor found."""
    for item in items:
        if isinstance(item, OpDescriptor):
            for t in item.input_tensors:
                return t.device()
            for t in item.output_tensors:
                return t.device()
        elif isinstance(item, (Sequential, Parallel)):
            dev = _extract_device(item._items)
            if dev is not None:
                return dev
    raise ValueError(
        "Cannot auto-extract device: no items contain device-backed tensors. "
        "Pass device explicitly to build(device=...)."
    )


def _extract_device_from_tree(node: OpNode):
    """Walk an OpNode tree, return device from first tensor found."""
    for t in node.op.input_tensors:
        return t.device()
    for t in node.op.output_tensors:
        return t.device()
    for child in node.children:
        dev = _extract_device_from_tree(child)
        if dev is not None:
            return dev
    return None


def _set_leaf_children(node: OpNode, children: List[OpNode]):
    """Attach children to the deepest single-child leaf of a node subtree."""
    current = node
    while current.children:
        if len(current.children) != 1:
            raise ValueError("Cannot attach children to a node that already branches")
        current = current.children[0]
    current.children = list(children)


def _build_item(item, device) -> _BuildResult:
    """Build a single item into a _BuildResult."""
    if isinstance(item, OpDescriptor):
        return _BuildResult(
            descriptor=item.descriptor,
            input_tensors=item.input_tensors,
            output_tensors=item.output_tensors,
        )
    if isinstance(item, (Sequential, Parallel)):
        return item._build_internal(device)
    raise TypeError(f"Unsupported item type: {type(item).__name__}")


def _merge_build_results(results: List[_BuildResult]) -> _BuildResult:
    """Merge multiple _BuildResults into one.

    Combines ProgramDescriptors, deduplicates input tensors (by identity),
    concatenates output tensors, and unions semaphore refs.
    """
    if len(results) == 1:
        return results[0]

    merged_desc = ttnn.merge_program_descriptors([r.descriptor for r in results])

    # Deduplicate input tensors (shared inputs appear in multiple ops)
    all_inputs: List = []
    seen_ids: Set[int] = set()
    for r in results:
        for t in r.input_tensors:
            tid = id(t)
            if tid not in seen_ids:
                all_inputs.append(t)
                seen_ids.add(tid)

    # Output tensors: one per result, in order
    all_outputs = [t for r in results for t in r.output_tensors]

    # Union semaphore refs
    all_semaphores = tuple(ref for r in results for ref in r.semaphores)

    return _BuildResult(
        descriptor=merged_desc,
        input_tensors=all_inputs,
        output_tensors=all_outputs,
        semaphores=all_semaphores,
    )


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
    change (e.g. stem barrier over 8 cores → branch barrier over 4 cores).
    ``transition_map`` maps each phase-transition index to the barrier
    segment and per-segment call index to use.
    """

    segments: List[BarrierSegment] = field(default_factory=list)
    compute_done_addr: int = 0
    writer_done_addr: int = 0
    # Map: phase_transition_index -> (segment_index, call_index_within_segment)
    transition_map: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    _sem_refs: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class CBPoolKey:
    """Compatibility key for a CB slot. CBs with the same key can share a slot."""

    data_format: Any  # tt::DataFormat
    page_size: int
    has_buffer: bool  # True if backed by an L1 Buffer (prevents sharing with non-buffer CBs)
    unpack_to_dest_mode: Any  # UnpackToDestMode enum


@dataclass
class CBSlot:
    """A slot in the CB pool."""

    slot_index: int
    config: CBPoolKey
    cb_descriptor: Any  # Best CBDescriptor (largest total_size, phase 0 buffer preferred)
    total_size: int  # Max total_size across all phases sharing this slot
    source_phase: int  # Phase that provided the current cb_descriptor


class CBPoolAllocator:
    """Pool-allocates CB hardware slots based on compatibility keys.

    CBs from different phases that share the same (data_format, page_size,
    unpack_to_dest_mode) configuration are assigned to the same slot.
    Different configs get separate slots.  Raises ValueError if the total
    number of slots exceeds the device limit.
    """

    MAX_SLOTS = 32

    def __init__(self, max_slots: int = MAX_SLOTS):
        self.max_slots = max_slots
        self._slots: Dict[int, CBSlot] = {}  # slot_index -> CBSlot
        # Maps CBPoolKey -> list of slot indices with that config.
        # Within a phase, each CB gets its own slot even if configs match;
        # across phases, CBs with the same config can share a slot.
        self._config_to_slots: Dict[CBPoolKey, List[int]] = {}
        self._allocated_indices: Set[int] = set()
        self.phase_remaps: List[Dict[int, int]] = []  # phase_idx -> {orig_cb -> slot_idx}
        self._next_index = 0  # Next candidate index for allocation
        # Maps slot_index -> original CB index that created it (for identity-preference)
        self._slot_to_orig_index: Dict[int, int] = {}

    def _alloc_index(self) -> int:
        """Find the next free slot index."""
        while self._next_index in self._allocated_indices:
            self._next_index += 1
        idx = self._next_index
        self._next_index += 1
        return idx

    def allocate_phase(
        self,
        phase_idx: int,
        cb_info: Dict[int, CBInfo],
        phantom_cb_indices: Set[int],
    ) -> None:
        """Allocate slots for a phase's CBs.

        Within a single phase, each CB gets its own slot even if multiple CBs
        share the same config (they hold different data concurrently).  Across
        phases, CBs with matching configs can share a slot because only one
        phase runs at a time.

        Args:
            phase_idx: Phase index.
            cb_info: Dict mapping original CB index -> CBInfo for this phase.
            phantom_cb_indices: CB indices referenced in named compile-time args
                but without a corresponding CBDescriptor.  These get identity-mapped
                reservations to prevent collisions.
        """
        remap: Dict[int, int] = {}
        slots_used_this_phase: Set[int] = set()

        # Reserve phantom CB indices first (identity mapping)
        for phantom_idx in phantom_cb_indices:
            if phantom_idx not in self._allocated_indices:
                self._allocated_indices.add(phantom_idx)
            remap[phantom_idx] = phantom_idx

        # Two-pass allocation: identity-matching CBs first (preserves cross-phase
        # slot assignments), then remaining CBs from the pool.
        identity_cbs, remaining_cbs = self._partition_by_identity(cb_info)

        for orig_idx, info, key in identity_cbs + remaining_cbs:
            slot_idx = self._find_reusable_slot(key, orig_idx, slots_used_this_phase)
            if slot_idx is not None:
                self._reuse_slot(slot_idx, info, phase_idx)
            else:
                slot_idx = self._allocate_new_slot(key, info, orig_idx, phase_idx)
            slots_used_this_phase.add(slot_idx)
            remap[orig_idx] = slot_idx

        self.phase_remaps.append(remap)

    def _partition_by_identity(
        self, cb_info: Dict[int, CBInfo]
    ) -> Tuple[List[Tuple[int, CBInfo, CBPoolKey]], List[Tuple[int, CBInfo, CBPoolKey]]]:
        """Split CBs into those with an existing identity-matching slot and the rest."""
        identity_cbs = []
        remaining_cbs = []
        for orig_idx, info in sorted(cb_info.items()):
            key = CBPoolKey(
                data_format=info.data_format,
                page_size=info.page_size,
                has_buffer=info.has_buffer,
                unpack_to_dest_mode=info.unpack_to_dest_mode,
            )
            has_identity = False
            if key in self._config_to_slots:
                for candidate_idx in self._config_to_slots[key]:
                    if self._slot_to_orig_index.get(candidate_idx) == orig_idx:
                        has_identity = True
                        break
            if has_identity:
                identity_cbs.append((orig_idx, info, key))
            else:
                remaining_cbs.append((orig_idx, info, key))
        return identity_cbs, remaining_cbs

    def _find_reusable_slot(self, key: CBPoolKey, orig_idx: int, slots_used_this_phase: Set[int]) -> Optional[int]:
        """Find an existing slot compatible with *key* not used by this phase.

        Prefers a slot created from the same original CB index (identity match).
        """
        if key not in self._config_to_slots:
            return None
        # First: identity match
        for candidate_idx in self._config_to_slots[key]:
            if candidate_idx not in slots_used_this_phase:
                if self._slot_to_orig_index.get(candidate_idx) == orig_idx:
                    return candidate_idx
        # Second: any compatible slot
        for candidate_idx in self._config_to_slots[key]:
            if candidate_idx not in slots_used_this_phase:
                return candidate_idx
        return None

    def _reuse_slot(self, slot_idx: int, info: CBInfo, phase_idx: int) -> None:
        """Reuse an existing slot, updating total_size if needed."""
        slot = self._slots[slot_idx]
        if info.total_size > slot.total_size:
            slot.total_size = info.total_size
            if slot.source_phase != 0:
                slot.cb_descriptor = self._get_cb_descriptor(info, phase_idx)
                slot.source_phase = phase_idx

    def _allocate_new_slot(self, key: CBPoolKey, info: CBInfo, orig_idx: int, phase_idx: int) -> int:
        """Allocate a fresh slot, raising ValueError on overflow.

        Prefers identity mapping (orig_idx -> orig_idx) when the slot is free,
        so that CBs keep their original hardware indices.  This avoids collisions
        when different phases have non-overlapping CB index sets (e.g., matmul
        uses {0,1,4,5} and RMS uses {0,2,3,5,...}).
        """
        if orig_idx not in self._allocated_indices and orig_idx < self.max_slots:
            slot_idx = orig_idx
        else:
            slot_idx = self._alloc_index()
        if len(self._slots) + 1 > self.max_slots:
            breakdown = [
                f"  slot {si}: fmt={sl.config.data_format}, "
                f"page_size={sl.config.page_size}, "
                f"unpack={sl.config.unpack_to_dest_mode}"
                for si, sl in sorted(self._slots.items())
            ]
            raise ValueError(
                f"CB pool overflow: need {len(self._slots) + 1} slots but "
                f"device limit is {self.max_slots}.\n"
                f"Allocated slots:\n" + "\n".join(breakdown)
            )
        self._allocated_indices.add(slot_idx)
        self._slots[slot_idx] = CBSlot(
            slot_index=slot_idx,
            config=key,
            cb_descriptor=self._get_cb_descriptor(info, phase_idx),
            total_size=info.total_size,
            source_phase=phase_idx,
        )
        self._slot_to_orig_index[slot_idx] = orig_idx
        if key not in self._config_to_slots:
            self._config_to_slots[key] = []
        self._config_to_slots[key].append(slot_idx)
        return slot_idx

    @staticmethod
    def _get_cb_descriptor(info: CBInfo, phase_idx: int) -> Dict[str, int]:
        """Create a lookup key for finding the CBDescriptor later.

        We can't deepcopy CBDescriptors (C++ bindings), so we store the
        phase index and original CB index.  build_merged_cb_descriptors()
        uses these to look up the actual CBDescriptor from the phase's
        ProgramDescriptor.
        """
        return {"phase_idx": phase_idx, "cb_idx": info.original_index}

    def get_remap(self, phase_idx: int) -> Dict[int, int]:
        """Return {orig_cb_idx: slot_idx} for a phase."""
        return self.phase_remaps[phase_idx]

    def build_merged_cb_descriptors(
        self,
        phases: List["PhaseInfo"],
    ) -> list:
        """Build merged CB descriptors from the pool.

        Returns one CBDescriptor per allocated slot, sorted by slot index.
        Uses the largest total_size and prefers phase 0's buffer-backed descriptor.

        Two-phase approach: first collect all (slot, descriptor, format_descriptor)
        tuples while buffer_indices are still at their original values, then apply
        all in-place modifications.  This prevents earlier modifications from
        making later searches find the wrong CBDescriptor (e.g., when slot N's
        assigned index coincidentally equals another CB's original index).
        """
        # Phase 1: Collect all results while buffer_indices are unmodified
        results = []
        for slot_idx in sorted(self._slots.keys()):
            slot = self._slots[slot_idx]
            ref = slot.cb_descriptor
            phase = phases[ref["phase_idx"]]
            orig_idx = ref["cb_idx"]

            # Find the actual CBDescriptor from the phase's ProgramDescriptor
            best_desc = None
            best_fmt = None
            for cb_desc in phase.op_descriptor.descriptor.cbs:
                for fmt_desc in cb_desc.format_descriptors:
                    if fmt_desc.buffer_index == orig_idx:
                        best_desc = cb_desc
                        best_fmt = fmt_desc
                        break
                if best_desc is not None:
                    break

            if best_desc is not None:
                results.append((slot_idx, slot.total_size, best_desc, best_fmt))

        # Phase 2: Apply all in-place modifications.
        # A single CBDescriptor can have multiple format_descriptors (e.g.,
        # matmul's output and intermediate share one CBDescriptor at indices
        # {4, 5}).  Each format_descriptor's buffer_index must be updated to
        # its slot index, but the CBDescriptor must appear exactly once in the
        # merged list.  Use id() tracking to avoid duplicates.
        merged = []
        seen_ids: Set[int] = set()
        # Compute max total_size per CBDescriptor across all its slots
        max_size_by_id: Dict[int, int] = {}
        for slot_idx, total_size, cb_desc, fmt_desc in results:
            fmt_desc.buffer_index = slot_idx
            cid = id(cb_desc)
            max_size_by_id[cid] = max(max_size_by_id.get(cid, 0), total_size)

        for _, _, cb_desc, _ in results:
            cid = id(cb_desc)
            if cid not in seen_ids:
                seen_ids.add(cid)
                cb_desc.total_size = max_size_by_id[cid]
                merged.append(cb_desc)

        return merged

    def build_unpack_to_dest_mode(self) -> list:
        """Build merged unpack_to_dest_mode vector indexed by slot index.

        Returns a list of exactly max_slots entries (matching the device's
        NUM_CIRCULAR_BUFFERS) with the correct mode at each slot's index,
        Default elsewhere.  The C++ JIT compiler requires this size.
        """
        result = [UnpackToDestMode.Default] * self.max_slots
        for slot_idx, slot in self._slots.items():
            if slot.config.unpack_to_dest_mode is not None:
                result[slot_idx] = slot.config.unpack_to_dest_mode
        return result

    def get_all_slot_indices(self) -> Set[int]:
        """All allocated slot indices (for sweep/clear)."""
        return set(self._slots.keys())


# =============================================================================
# Analysis Functions
# =============================================================================


def extract_cb_info(
    descriptor: "ttnn.ProgramDescriptor",
    unpack_to_dest_modes: Optional[list] = None,
) -> Dict[int, CBInfo]:
    """Extract CB information from a ProgramDescriptor.

    Args:
        descriptor: The ProgramDescriptor to extract CB info from.
        unpack_to_dest_modes: Optional vector of UnpackToDestMode indexed by CB index,
            typically from ComputeConfigDescriptor.unpack_to_dest_mode.

    Returns a dict mapping CB index -> CBInfo.
    """
    cb_info = {}
    for cb_desc in descriptor.cbs:
        if cb_desc.has_global_circular_buffer():
            raise ValueError(
                "Sequential fusion does not support GlobalCircularBuffer CBs. "
                "CB with global_circular_buffer detected in ProgramDescriptor."
            )
        for fmt_desc in cb_desc.format_descriptors:
            cb_idx = fmt_desc.buffer_index
            try:
                data_format = fmt_desc.data_format
            except (TypeError, AttributeError):
                data_format = None
            # Look up unpack_to_dest_mode for this CB
            utd_mode = None
            if unpack_to_dest_modes is not None:
                try:
                    if cb_idx < len(unpack_to_dest_modes):
                        utd_mode = unpack_to_dest_modes[cb_idx]
                except (TypeError, IndexError):
                    pass
            if utd_mode is None:
                utd_mode = UnpackToDestMode.Default
            cb_info[cb_idx] = CBInfo(
                original_index=cb_idx,
                total_size=cb_desc.total_size,
                data_format=data_format,
                page_size=fmt_desc.page_size,
                core_ranges=cb_desc.core_ranges,
                has_buffer=cb_desc.has_buffer(),
                unpack_to_dest_mode=utd_mode,
            )
    return cb_info


# Convention: CB-reference named compile-time args MUST start with this prefix
# and have a value in range [0, 31]. Non-CB args MUST NOT use this prefix.
CB_ARG_PREFIX = "cb_"


def _is_cb_named_arg(name: str, value: Any) -> bool:
    """Check if a named compile-time arg refers to a CB index.

    Returns True if the name starts with CB_ARG_PREFIX and the value
    is an integer in [0, 31] (valid CB slot range).
    """
    if not name.startswith(CB_ARG_PREFIX):
        return False
    if not isinstance(value, int) or value < 0 or value >= CBPoolAllocator.MAX_SLOTS:
        logger.warning(
            "Named arg '%s' starts with '%s' but has value %s outside CB range [0,31]. "
            "If this is not a CB arg, rename it to not start with '%s'.",
            name,
            CB_ARG_PREFIX,
            value,
            CB_ARG_PREFIX,
        )
        return False
    return True


def extract_cb_names_from_kernel(kernel_desc: "ttnn.KernelDescriptor") -> Dict[str, int]:
    """Extract CB name -> index mapping from kernel's named compile-time args."""
    cb_names = {}
    if hasattr(kernel_desc, "named_compile_time_args"):
        for name, value in kernel_desc.named_compile_time_args:
            if _is_cb_named_arg(name, value):
                cb_names[name] = value
    return cb_names


# =============================================================================
# CB State Save/Restore
# =============================================================================


def _save_cb_state(program_descriptors: List[Any]) -> List[dict]:
    """Save mutable CB descriptor state before a fused build.

    _build_fused_descriptor mutates buffer_index, total_size, and
    core_ranges on the original CBDescriptors (can't deepcopy C++
    bindings).  Save these fields so they can be restored after build.

    Args:
        program_descriptors: List of ProgramDescriptor objects whose
            CB state should be saved.  Deduplicates by object id.
    """
    saved = []
    seen_cb_ids: set = set()
    for prog_desc in program_descriptors:
        for cb_desc in prog_desc.cbs:
            cb_id = id(cb_desc)
            if cb_id in seen_cb_ids:
                continue
            seen_cb_ids.add(cb_id)
            saved.append(
                {
                    "cb": cb_desc,
                    "total_size": cb_desc.total_size,
                    "core_ranges": cb_desc.core_ranges,
                    "fmt": [(fmt, fmt.buffer_index) for fmt in cb_desc.format_descriptors],
                }
            )
    return saved


def _restore_cb_state(saved: List[dict]) -> None:
    """Restore CB descriptor state saved by _save_cb_state."""
    for entry in saved:
        entry["cb"].total_size = entry["total_size"]
        entry["cb"].core_ranges = entry["core_ranges"]
        for fmt, orig_idx in entry["fmt"]:
            fmt.buffer_index = orig_idx


def _verify_cb_restore(saved: List[dict]) -> None:
    """Verify that CB descriptor state was correctly restored.

    Called after _restore_cb_state to catch any future code changes
    that add mutation paths not covered by save/restore.

    Raises:
        RuntimeError: If any CB field doesn't match its saved value.
    """
    for entry in saved:
        cb = entry["cb"]
        if cb.total_size != entry["total_size"]:
            raise RuntimeError(f"CB restore failed: total_size is {cb.total_size}, " f"expected {entry['total_size']}")
        for fmt, expected_idx in entry["fmt"]:
            if fmt.buffer_index != expected_idx:
                raise RuntimeError(
                    f"CB restore failed: buffer_index is {fmt.buffer_index}, " f"expected {expected_idx}"
                )


def _create_phase_info(op_descriptor: OpDescriptor, phase_idx: int) -> PhaseInfo:
    """Create a PhaseInfo from an OpDescriptor.

    Extracts CB info and unpack_to_dest_mode from the op's kernels.
    """
    utd_modes = None
    for kd in op_descriptor.descriptor.kernels:
        config = kd.config
        if hasattr(config, "unpack_to_dest_mode"):
            modes = config.unpack_to_dest_mode
            if modes is not None and len(modes) > 0:
                utd_modes = modes
                break
    cb_info = extract_cb_info(op_descriptor.descriptor, utd_modes)
    return PhaseInfo(phase_idx=phase_idx, op_descriptor=op_descriptor, cb_info=cb_info)


# =============================================================================
# Kernel Classification
# =============================================================================


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


def _core_ranges_key(core_ranges: Any) -> frozenset:
    """Create a hashable key from a CoreRangeSet for grouping."""
    return frozenset((cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in core_ranges.ranges())


def _same_core_range(a: Any, b: Any) -> bool:
    """Compare two CoreRangeSets by their keys."""
    return _core_ranges_key(a) == _core_ranges_key(b)


def _coords_to_core_range_set(coords: Set[Tuple[int, int]]) -> Any:
    """Convert a set of (x, y) tuples to a CoreRangeSet.

    Each coordinate becomes a single-core CoreRange.  CoreRangeSet
    merges adjacent ranges internally.
    """
    ranges = set()
    for x, y in coords:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)))
    return ttnn.CoreRangeSet(ranges)


def _get_node_core_range(node: "OpNode") -> Any:
    """Extract the core range from a node's op descriptor.

    Returns the union of all kernel core_ranges in the node's
    ProgramDescriptor.
    """
    all_coords: Set[Tuple[int, int]] = set()
    for kernel in node.op.descriptor.kernels:
        all_coords |= _core_range_set_to_coords(kernel.core_ranges)
    return _coords_to_core_range_set(all_coords)


def _get_role_key(
    kernel_desc: "ttnn.KernelDescriptor",
    core_range_override: Optional[Any] = None,
) -> Tuple[str, frozenset]:
    """Return (risc_type, core_ranges_key) identifying this kernel's role.

    If core_range_override is set, all kernels are mapped to that range
    regardless of their native core_ranges.  This collapses kernels with
    different ranges (e.g. stem vs branch) into the same role.
    """
    cr = core_range_override if core_range_override is not None else kernel_desc.core_ranges
    return (_get_risc_type(kernel_desc), _core_ranges_key(cr))


# =============================================================================
# Source Code Utilities
# =============================================================================


def _read_kernel_source(kernel_desc: "ttnn.KernelDescriptor") -> Tuple[str, Optional[str]]:
    """Read kernel source code from a kernel descriptor.

    Returns (source_code, kernel_file_dir) where kernel_file_dir is the
    directory of the source file (for resolving local includes), or None
    if the kernel is inline SOURCE_CODE.
    """
    if kernel_desc.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE:
        return kernel_desc.kernel_source, None

    base_paths = [
        os.environ.get("TT_METAL_HOME", ""),
        "",
    ]
    for base in base_paths:
        full_path = os.path.join(base, kernel_desc.kernel_source) if base else kernel_desc.kernel_source
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read(), os.path.dirname(full_path)
    return "", None


def _prefix_phase_names_in_source(source: str, phase_idx: int, names: List[str]) -> str:
    """Prefix phase-specific names (functions + globals) in source code.

    Applied to each phase's kernel body to ensure references to
    phase-specific functions and variables use the prefixed names
    that match the prefixed declarations in the merged pre-main.

    Every phase (including phase 0) gets prefixed.  Replacements are
    scoped to code only — identifiers inside string literals and
    comments are left unchanged.
    """
    if not names:
        return source
    for name in names:
        source = cpp_parser.replace_in_code_only(source, name, f"phase{phase_idx}_{name}")
    return source


def _collect_all_pre_main_code(
    sources_with_indices: List[Tuple[int, str]],
) -> Tuple[str, Dict[int, str], Dict[int, List[str]]]:
    """Merge pre-main code from all phases with per-phase name isolation.

    Uses tree-sitter (via cpp_parser) to categorize top-level blocks
    before kernel_main into *shared* or *phase-specific*:

    **Shared** (deduped across phases, no define wrapper needed):
      - Namespace blocks: dedup by signature (first occurrence wins).
      - Single-line declarations (namespace aliases, ``using``,
        ``typedef``): dedup by exact normalized content.

    **Phase-specific** (prefixed with ``phaseN_``, wrapped by caller
    with per-phase ``#define``/``#undef`` blocks):
      - Free function definitions (``ALWI``, ``FORCE_INLINE``, etc.):
        each phase gets its own copy with the function name prefixed.
      - Global/static variables: each phase gets its own copy with
        the variable name prefixed.
      - Preprocessor conditional blocks (``#ifdef``/``#if``/``#ifndef``):
        preserved as-is but with inner function/variable names prefixed.
        Each phase gets its own copy.

    ALL phases (including phase 0) get prefixed.  This eliminates:
      - Silent first-wins drops when two phases define the same function
        with different bodies.
      - Redefinition errors when an included header also defines a
        function that appears inline in another phase's pre-main.

    Returns:
        A tuple of (shared_pre_main, per_phase_pre_main, phase_names):
        - shared_pre_main: namespace blocks, using declarations, etc.
        - per_phase_pre_main: dict mapping phase_idx -> prefixed code.
          Callers wrap each with ``#define``/``#undef`` for that phase.
        - phase_names: dict mapping phase_idx -> list of original names
          that were prefixed.  Callers apply the same prefixing to each
          phase's kernel body.
    """
    if not sources_with_indices:
        return "", {}, {}

    shared_blocks: List[str] = []
    per_phase_blocks: Dict[int, List[str]] = {}
    seen_signatures: Set[str] = set()
    seen_shared_content: Set[str] = set()
    seen_phase_content: Dict[int, Set[str]] = {}
    phase_names: Dict[int, List[str]] = {}

    for phase_idx, source in sources_with_indices:
        blocks = cpp_parser.categorize_pre_main(source)
        seen_phase_content.setdefault(phase_idx, set())

        for block in blocks:
            normalized = cpp_parser.normalize_block(block.text)
            if not normalized:
                continue

            # --- Phase-specific: preprocessor conditional blocks ---
            if block.kind == "preproc_block":
                text = block.text
                names = block.inner_names or []
                for name in names:
                    text = re.sub(
                        rf"\b{re.escape(name)}\b",
                        f"phase{phase_idx}_{name}",
                        text,
                    )
                    phase_names.setdefault(phase_idx, []).append(name)
                prefixed_norm = cpp_parser.normalize_block(text)
                if prefixed_norm not in seen_phase_content[phase_idx]:
                    seen_phase_content[phase_idx].add(prefixed_norm)
                    per_phase_blocks.setdefault(phase_idx, []).append(text)
                continue

            # --- Phase-specific: global/static variables ---
            if block.kind == "variable" and block.name:
                prefixed = re.sub(
                    rf"\b{re.escape(block.name)}\b",
                    f"phase{phase_idx}_{block.name}",
                    block.text,
                )
                phase_names.setdefault(phase_idx, []).append(block.name)
                prefixed_norm = cpp_parser.normalize_block(prefixed)
                if prefixed_norm not in seen_phase_content[phase_idx]:
                    seen_phase_content[phase_idx].add(prefixed_norm)
                    per_phase_blocks.setdefault(phase_idx, []).append(prefixed)
                continue

            # --- Phase-specific: free function definitions ---
            if block.kind == "function" and block.name:
                prefixed = re.sub(
                    rf"\b{re.escape(block.name)}\b",
                    f"phase{phase_idx}_{block.name}",
                    block.text,
                )
                phase_names.setdefault(phase_idx, []).append(block.name)
                prefixed_norm = cpp_parser.normalize_block(prefixed)
                if prefixed_norm not in seen_phase_content[phase_idx]:
                    seen_phase_content[phase_idx].add(prefixed_norm)
                    per_phase_blocks.setdefault(phase_idx, []).append(prefixed)
                continue

            # --- Shared: namespace blocks ---
            if block.kind == "namespace":
                sig = block.text.split("{")[0].strip() if "{" in block.text else normalized
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    shared_blocks.append(block.text)
            else:
                # --- Shared: namespace_alias, using, struct, template, other ---
                if normalized not in seen_shared_content:
                    seen_shared_content.add(normalized)
                    shared_blocks.append(block.text)

    shared_pre_main = "\n\n".join(shared_blocks)
    per_phase_pre_main = {idx: "\n\n".join(blocks) for idx, blocks in per_phase_blocks.items()}

    return shared_pre_main, per_phase_pre_main, phase_names


# =============================================================================
# Source Transformations for Phase N>0
# =============================================================================


def _prefix_named_args_in_source(source: str, phase_idx: int) -> str:
    """Replace get_named_compile_time_arg_val("X") with phase-prefixed version."""
    if phase_idx == 0:
        return source

    def replace_named_arg(match):
        name = match.group(1)
        return f'get_named_compile_time_arg_val("phase{phase_idx}_{name}")'

    return re.sub(
        r'get_named_compile_time_arg_val\("([^"]+)"\)',
        replace_named_arg,
        source,
    )


def _offset_compile_time_args_in_source(source: str, phase_idx: int, ct_arg_offset: int) -> str:
    """Offset get_compile_time_arg_val(N) and TensorAccessorArgs<N> for phase N>0.

    Instead of substituting literal values, we offset the indices so that
    each phase reads from its own slice of the concatenated compile-time arg
    array.  This also handles TensorAccessorArgs<N> which internally calls
    get_compile_time_arg_val(N).
    """
    if phase_idx == 0 or ct_arg_offset == 0:
        return source

    def offset_ct_arg(match):
        arg_idx = int(match.group(1))
        return f"get_compile_time_arg_val({arg_idx + ct_arg_offset})"

    source = re.sub(
        r"get_compile_time_arg_val\((\d+)\)",
        offset_ct_arg,
        source,
    )

    def offset_tensor_accessor(match):
        arg_idx = int(match.group(1))
        return f"TensorAccessorArgs<{arg_idx + ct_arg_offset}>"

    source = re.sub(
        r"TensorAccessorArgs<(\d+)>",
        offset_tensor_accessor,
        source,
    )

    return source


def _emit_rt_arg_wrapper(phase_idx: int, rt_offset: int) -> List[str]:
    """Emit the wrapper function definition for phase N>0.

    Emitted once at file scope (before any #define) so the wrapper body
    references the real get_arg_val.
    """
    wrapper_name = f"__phase{phase_idx}_get_arg_val"
    return [
        f"template <typename T>",
        f"FORCE_INLINE T {wrapper_name}(int arg_idx) {{",
        f"    return get_arg_val<T>(arg_idx + {rt_offset});",
        f"}}",
    ]


def _emit_rt_arg_define(phase_idx: int) -> str:
    """Emit #define to redirect get_arg_val to the phase wrapper."""
    return f"#define get_arg_val __phase{phase_idx}_get_arg_val"


def _emit_rt_arg_undef() -> str:
    """Emit #undef to restore get_arg_val after a phase."""
    return "#undef get_arg_val"


def _transform_phase_source(
    source: str,
    phase_idx: int,
    ct_arg_offset: int = 0,
    phase_names: Optional[List[str]] = None,
) -> str:
    """Apply all transformations for a phase's kernel body.

    Args:
        source: Kernel body source code.
        phase_idx: Phase index.
        ct_arg_offset: Compile-time arg offset for this phase.
        phase_names: Names (functions + globals) that were prefixed in
            pre-main and must also be prefixed in the kernel body.
    """
    source = _prefix_named_args_in_source(source, phase_idx)
    source = _offset_compile_time_args_in_source(source, phase_idx, ct_arg_offset)
    # Runtime arg offsetting is now handled by #define/#undef redirect of
    # get_arg_val (see _emit_rt_arg_define/_emit_rt_arg_undef), not by
    # source-level regex rewriting.
    source = _prefix_phase_names_in_source(source, phase_idx, phase_names or [])
    return source


# =============================================================================
# CB Descriptor Merging
# =============================================================================


def _get_compute_unpack_to_dest_modes(phase: PhaseInfo) -> Optional[list]:
    """Get the unpack_to_dest_mode vector from a phase's compute kernel config."""
    for kernel_desc in phase.op_descriptor.descriptor.kernels:
        config = kernel_desc.config
        if hasattr(config, "unpack_to_dest_mode"):
            modes = config.unpack_to_dest_mode
            if modes is not None and len(modes) > 0:
                return modes
    return None


def _get_phantom_cb_indices(phase: PhaseInfo) -> Set[int]:
    """Get CB indices referenced in named compile-time args but without CBDescriptors.

    These "phantom" CBs need identity-mapped reservations in the pool to prevent
    real CBs from being allocated at conflicting indices.
    """
    # Collect all CB indices that have actual descriptors
    real_cb_indices = set(phase.cb_info.keys())

    # Collect all CB indices referenced in named compile-time args
    phantom = set()
    for kernel_desc in phase.op_descriptor.descriptor.kernels:
        for name, value in kernel_desc.named_compile_time_args:
            if _is_cb_named_arg(name, value) and value not in real_cb_indices:
                phantom.add(value)

    return phantom


# =============================================================================
# CB Address Rebinding
# =============================================================================


def _compute_rebind_info(
    phases: List[PhaseInfo],
    phase_remaps: List[Dict[int, int]],
) -> Dict[int, List[Tuple[int, int, int]]]:
    """Compute which CB slots need address rebinding at each phase transition.

    For each phase 1+, identifies remapped slot indices where the buffer address
    differs from what was set in the previous phase.  Phase 0 never needs
    rebinding because build_merged_cb_descriptors prefers phase 0's buffer.

    Args:
        phases: All PhaseInfo objects.
        phase_remaps: Per-phase {orig_cb_idx: slot_idx} from the pool allocator.

    Returns:
        Dict mapping phase_idx -> list of (slot_idx, new_addr, new_size) tuples.
    """
    # Collect per-phase buffer addresses, mapped to slot indices
    phase_slot_addrs: List[Dict[int, Tuple[int, int]]] = []
    for phase_idx, phase in enumerate(phases):
        remap = phase_remaps[phase_idx] if phase_idx < len(phase_remaps) else {}
        addrs: Dict[int, Tuple[int, int]] = {}
        for cb_desc in phase.op_descriptor.descriptor.cbs:
            for fmt_desc in cb_desc.format_descriptors:
                orig_idx = fmt_desc.buffer_index
                slot_idx = remap.get(orig_idx, orig_idx)
                if cb_desc.has_buffer():
                    addr = cb_desc.buffer_address()
                    if addr is not None:
                        addrs[slot_idx] = (addr, cb_desc.total_size)
        phase_slot_addrs.append(addrs)

    if not phase_slot_addrs:
        return {}

    # Start with phase 0's addresses as baseline
    rebind_info: Dict[int, List[Tuple[int, int, int]]] = {}
    current_addrs = dict(phase_slot_addrs[0])

    for phase_idx in range(1, len(phases)):
        rebinds: List[Tuple[int, int, int]] = []
        for slot_idx, (phase_addr, phase_size) in phase_slot_addrs[phase_idx].items():
            current = current_addrs.get(slot_idx)
            if current is None or current[0] != phase_addr:
                rebinds.append((slot_idx, phase_addr, phase_size))
                current_addrs[slot_idx] = (phase_addr, phase_size)
        rebind_info[phase_idx] = rebinds

    return rebind_info


def _generate_rebind_code(
    rebinds: List[Tuple[int, int, int]],
    phase_idx: int,
    for_compute: bool = False,
) -> List[str]:
    """Generate C++ code to rebind CB addresses for a phase.

    Args:
        rebinds: List of (slot_idx, addr, size) tuples for this phase.
        phase_idx: Which phase these rebinds are for.
        for_compute: If True, shift addresses by >> 4 for TRISC and guard
            with #ifndef TRISC_MATH (TRISC1 has no cb_interface).

    Returns:
        List of C++ source lines (indented with 4 spaces).
    """
    if not rebinds:
        return []
    shift = " >> 4" if for_compute else ""
    lines = [f"    // Rebind CB addresses for phase {phase_idx}"]
    if for_compute:
        # TRISC1 (math) doesn't have cb_interface linked in — skip it
        lines.append("#ifndef TRISC_MATH")
    for slot_idx, _, _ in rebinds:
        prefix = f"phase{phase_idx}_cb{slot_idx}"
        lines.append(f"    {{")
        lines.append(
            f'        constexpr uint32_t new_addr = get_named_compile_time_arg_val("{prefix}_rebind_addr"){shift};'
        )
        lines.append(
            f'        constexpr uint32_t new_size = get_named_compile_time_arg_val("{prefix}_rebind_size"){shift};'
        )
        lines.append(f"        get_local_cb_interface({slot_idx}).fifo_rd_ptr = new_addr;")
        lines.append(f"        get_local_cb_interface({slot_idx}).fifo_wr_ptr = new_addr;")
        lines.append(f"        get_local_cb_interface({slot_idx}).fifo_size = new_size;")
        lines.append(f"        get_local_cb_interface({slot_idx}).fifo_limit = new_addr + new_size;")
        lines.append(f"    }}")
    if for_compute:
        lines.append("#endif")
    return lines


# =============================================================================
# Fused Kernel Source Generation
# =============================================================================


def _generate_fused_riscv0_source(
    phase_kernels: List[Dict[str, Any]],
    role_key: Any,
    phases: List[PhaseInfo],
    ct_arg_offsets: Dict[int, int],
    sweep_cb_indices: List[int],
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    op_semaphore_info: Optional[List[Tuple[int, int]]] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
    rt_arg_offsets: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """Generate fused RISCV_0 (reader/BRISC) kernel source with two-level barrier sync.

    Between phases, the RISCV_0 processor acts as the barrier coordinator:
      1. Wait for local compute + writer to signal done (L1 flag spin)
      2. Reset residual tiles from ALL CBs on BRISC
      3. Global barrier across cores (sets global_release which also serves
         as the phase release signal for compute/writer)

    The BRISC reset updates stream register tiles_acked but NOT TRISC0's
    local copy.  Compute must resync after being released (see compute source).
    """
    reader_sources = []

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(role_key)
        if kernel is None:
            continue
        source, kernel_dir = _read_kernel_source(kernel)
        if not source:
            continue
        source = cpp_parser.inline_local_includes(source, kernel_dir)
        reader_sources.append((i, source))

    if not reader_sources:
        return None

    all_sources = [s for _, s in reader_sources]
    includes = cpp_parser.collect_includes(all_sources)
    source_defines = cpp_parser.collect_defines(all_sources)
    uniform_defines, per_phase_defines = _categorize_phase_defines(phase_kernels, role_key)
    shared_pre_main, per_phase_pre_main, phase_names = _collect_all_pre_main_code(reader_sources)

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused reader kernel - {len(reader_sources)} phases",
        "",
    ]
    lines.extend(_emit_define_lines(uniform_defines))
    lines.extend(source_defines)
    lines.append("")
    lines.extend(includes)
    lines.append("")
    if shared_pre_main.strip():
        lines.append(shared_pre_main)
        lines.append("")

    # Emit RT arg wrapper function definitions at file scope (before any #define)
    if rt_arg_offsets:
        for phase_idx, _ in reader_sources:
            if phase_idx > 0 and phase_idx in rt_arg_offsets:
                lines.extend(_emit_rt_arg_wrapper(phase_idx, rt_arg_offsets[phase_idx]))
        lines.append("")

    # Per-phase pre-main code, each wrapped with that phase's varying defines
    # and RT arg redirect so helper functions get the offset too
    for phase_idx, _ in reader_sources:
        phase_code = per_phase_pre_main.get(phase_idx, "")
        if not phase_code.strip():
            continue
        varying = per_phase_defines.get(phase_idx, [])
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_define(phase_idx))
        if varying:
            lines.extend(_emit_define_lines(varying))
        lines.append(phase_code)
        if varying:
            lines.extend(_emit_undef_lines(varying))
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_undef())
        lines.append("")

    is_multi_phase = len(reader_sources) > 1
    needs_barrier = multi_barrier is not None and len(multi_barrier.transition_map) > 0

    if needs_barrier:
        # Barrier named compile-time args
        lines.append('constexpr uint32_t __barrier_rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')

        # Per-segment compile-time constants
        for seg_idx in range(len(multi_barrier.segments)):
            s = f"seg{seg_idx}"
            lines.append(f'constexpr uint32_t __{s}_num_cores = get_named_compile_time_arg_val("{s}_num_cores");')
            lines.append(f'constexpr uint32_t __{s}_core0_phys_x = get_named_compile_time_arg_val("{s}_core0_phys_x");')
            lines.append(f'constexpr uint32_t __{s}_core0_phys_y = get_named_compile_time_arg_val("{s}_core0_phys_y");')
            lines.append(
                f'constexpr uint32_t __{s}_mcast_start_x = get_named_compile_time_arg_val("{s}_mcast_start_x");'
            )
            lines.append(
                f'constexpr uint32_t __{s}_mcast_start_y = get_named_compile_time_arg_val("{s}_mcast_start_y");'
            )
            lines.append(f'constexpr uint32_t __{s}_mcast_end_x = get_named_compile_time_arg_val("{s}_mcast_end_x");')
            lines.append(f'constexpr uint32_t __{s}_mcast_end_y = get_named_compile_time_arg_val("{s}_mcast_end_y");')
        lines.append("")

        # BRISC-side CB reset: equalize stream registers + reset pointers to CB start.
        # Uses direct tt_reg_ptr stream register increment (no cb_pop_front dependency).
        # The stream controller requires per-tile increments — bulk acked += N hangs.
        lines.append("// BRISC-side CB reset: equalize stream registers + reset pointers to CB start.")
        lines.append("FORCE_INLINE void __cb_reset_to_empty() {")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(f"        uint16_t remaining = (uint16_t)(*get_cb_tiles_received_ptr({cb_idx}))")
            lines.append(f"                          - (uint16_t)(*get_cb_tiles_acked_ptr({cb_idx}));")
            lines.append(f"        volatile tt_reg_ptr uint32_t* acked_ptr = (volatile tt_reg_ptr uint32_t*)")
            lines.append(f"            ((uint32_t)(uintptr_t)get_cb_tiles_acked_ptr({cb_idx}));")
            lines.append(f"        for (uint16_t i = 0; i < remaining; i++) {{")
            lines.append(f"            acked_ptr[0] += 1;")
            lines.append(f"        }}")
            lines.append(f"        // Reset BRISC local pointers to CB start")
            lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
            lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = fifo_start;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = fifo_start;")
            lines.append(f"    }}")
        lines.append("}")
        lines.append("")

        # Per-segment global barrier functions
        for seg_idx in range(len(multi_barrier.segments)):
            s = f"seg{seg_idx}"
            lines.append(f"// Barrier segment {seg_idx}: global barrier across segment cores.")
            lines.append(
                f"FORCE_INLINE void __barrier_{s}(uint32_t call_idx, "
                f"volatile tt_l1_ptr uint32_t* arrive, volatile tt_l1_ptr uint32_t* release) {{"
            )
            lines.append(f"    if constexpr (__{s}_num_cores > 1) {{")
            lines.append(
                f"        uint64_t core0_arrive_noc_addr = get_noc_addr(__{s}_core0_phys_x, __{s}_core0_phys_y, (uint32_t)arrive);"
            )
            lines.append(f"        noc_semaphore_inc(core0_arrive_noc_addr, 1);")
            lines.append(f"")
            lines.append(f"        bool is_core_0 = (my_x[0] == __{s}_core0_phys_x && my_y[0] == __{s}_core0_phys_y);")
            lines.append(f"        if (is_core_0) {{")
            lines.append(f"            noc_semaphore_wait_min(arrive, __{s}_num_cores * (call_idx + 1));")
            lines.append(f"            *release = call_idx + 1;")
            lines.append(
                f"            uint64_t mcast_addr = get_noc_multicast_addr("
                f"__{s}_mcast_start_x, __{s}_mcast_start_y, __{s}_mcast_end_x, __{s}_mcast_end_y, (uint32_t)release);"
            )
            lines.append(
                f"            noc_semaphore_set_multicast_loopback_src((uint32_t)release, mcast_addr, __{s}_num_cores);"
            )
            lines.append(f"            noc_async_write_barrier();")
            lines.append(f"        }} else {{")
            lines.append(f"            noc_semaphore_wait_min(release, call_idx + 1);")
            lines.append(f"        }}")
            lines.append(f"    }} else {{")
            lines.append(f"        *release = call_idx + 1;")
            lines.append(f"    }}")
            lines.append(f"}}")
            lines.append("")

    # Generate phase functions
    for phase_idx, raw_source in reader_sources:
        body = cpp_parser.extract_kernel_body(raw_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        pnames = phase_names.get(phase_idx, [])
        transformed = _transform_phase_source(body, phase_idx, offset, phase_names=pnames)

        varying = per_phase_defines.get(phase_idx, [])
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_define(phase_idx))
        lines.append(f"// Phase {phase_idx} reader")
        lines.append(f"FORCE_INLINE void phase{phase_idx}_reader() {{")
        if varying:
            for dl in _emit_define_lines(varying):
                lines.append(f"    {dl}")
        for line in transformed.split("\n"):
            lines.append(f"    {line}")
        if varying:
            for ul in _emit_undef_lines(varying):
                lines.append(f"    {ul}")
        lines.append("}")
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_undef())
        lines.append("")

    # Generate kernel_main
    lines.append("void kernel_main() {")

    if is_multi_phase:
        lines.append("    // Read barrier L1 flag addresses from runtime args")
        lines.append("    const uint32_t __compute_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset);")
        lines.append("    const uint32_t __writer_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 1);")
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__compute_done_addr);"
        )
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__writer_done_addr);"
        )
        # Per-segment arrive/release pointers
        for seg_idx in range(len(multi_barrier.segments)):
            s = f"seg{seg_idx}"
            off = 2 + seg_idx * 2
            lines.append(f"    const uint32_t __{s}_arrive_addr = get_arg_val<uint32_t>(__barrier_rt_offset + {off});")
            lines.append(
                f"    const uint32_t __{s}_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + {off + 1});"
            )
            lines.append(
                f"    volatile tt_l1_ptr uint32_t* __{s}_arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__{s}_arrive_addr);"
            )
            lines.append(
                f"    volatile tt_l1_ptr uint32_t* __{s}_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__{s}_release_addr);"
            )
        lines.append("")

    rebind_info = rebind_info or {}

    first = True
    for phase_idx, _ in reader_sources:
        if not first and needs_barrier:
            transition_idx = phase_idx - 1
            if transition_idx in multi_barrier.transition_map:
                lines.append("")
                lines.append(f"    // === Barrier: Phase {phase_idx - 1} -> Phase {phase_idx} ===")
                lines.append("    // Invariant: BRISC (reader) coordinates all inter-phase cleanup.")
                lines.append("    // Order: noc_barrier -> wait compute/writer done -> reset CBs ->")
                lines.append("    //         reset semaphores -> rebind CB addrs -> global barrier.")
                lines.append("    // Compute/writer must NOT touch CBs until global_release is set.")
                lines.append("    noc_async_full_barrier();")
                lines.append("")
                lines.append(f"    // Wait for local compute + writer to finish Phase {phase_idx - 1}")
                lines.append(f"    noc_semaphore_wait_min(__compute_done, {phase_idx});")
                lines.append(f"    noc_semaphore_wait_min(__writer_done, {phase_idx});")
                lines.append("")
                lines.append("    // Reset residual tiles from ALL CBs")
                lines.append("    __cb_reset_to_empty();")
                lines.append("")
                # Reset op semaphores to their initial values so next phase starts clean
                if op_semaphore_info:
                    lines.append("    // Reset op semaphores to initial values (as if each phase runs standalone)")
                    for sem_id, initial_val in op_semaphore_info:
                        lines.append(
                            f"    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore({sem_id})) = {initial_val};"
                        )
                    lines.append("")
                # Rebind CB addresses before global barrier (so BRISC has correct state)
                rebind_lines = _generate_rebind_code(rebind_info.get(phase_idx, []), phase_idx, for_compute=False)
                if rebind_lines:
                    lines.extend(rebind_lines)
                    lines.append("")
                seg_idx, call_idx = multi_barrier.transition_map[transition_idx]
                s = f"seg{seg_idx}"
                lines.append(f"    // Global barrier segment {seg_idx}, call {call_idx}")
                lines.append(f"    __barrier_{s}({call_idx}, __{s}_arrive, __{s}_release);")
                lines.append("")
        lines.append(f"    phase{phase_idx}_reader();")
        first = False

    # Trailing barrier: after the last phase, if there's a pending transition
    # (e.g. empty leaf branch participating in an ancestor's barrier)
    if needs_barrier:
        last_phase_idx = reader_sources[-1][0]
        trailing_transition = last_phase_idx
        if trailing_transition in multi_barrier.transition_map:
            seg_idx, call_idx = multi_barrier.transition_map[trailing_transition]
            s = f"seg{seg_idx}"
            lines.append("")
            lines.append(f"    // === Trailing barrier after Phase {last_phase_idx} ===")
            lines.append("    noc_async_full_barrier();")
            lines.append(f"    noc_semaphore_wait_min(__compute_done, {last_phase_idx + 1});")
            lines.append(f"    noc_semaphore_wait_min(__writer_done, {last_phase_idx + 1});")
            lines.append("    __cb_reset_to_empty();")
            lines.append(f"    __barrier_{s}({call_idx}, __{s}_arrive, __{s}_release);")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _generate_fused_riscv1_source(
    phase_kernels: List[Dict[str, Any]],
    role_key: Any,
    phases: List[PhaseInfo],
    ct_arg_offsets: Dict[int, int],
    sweep_cb_indices: List[int],
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
    rt_arg_offsets: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """Generate fused RISCV_1 (writer/NCRISC) kernel source with L1 flag barrier sync.

    Between phases, the writer:
      1. Signals done by writing phase+1 to writer_done L1 flag
      2. Spins on global_release L1 flag (plain volatile read, no NOC APIs)
      3. Resyncs NCRISC local CB pointers to CB start
    """
    writer_sources = []

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(role_key)
        if kernel is None:
            continue
        source, kernel_dir = _read_kernel_source(kernel)
        if not source:
            continue
        source = cpp_parser.inline_local_includes(source, kernel_dir)
        writer_sources.append((i, source))

    if not writer_sources:
        return None

    all_sources = [s for _, s in writer_sources]
    includes = cpp_parser.collect_includes(all_sources)
    source_defines = cpp_parser.collect_defines(all_sources)
    uniform_defines, per_phase_defines = _categorize_phase_defines(phase_kernels, role_key)
    shared_pre_main, per_phase_pre_main, phase_names = _collect_all_pre_main_code(writer_sources)

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused writer kernel - {len(writer_sources)} phases",
        "",
    ]
    lines.extend(_emit_define_lines(uniform_defines))
    lines.extend(source_defines)
    lines.append("")
    lines.extend(includes)
    lines.append("")
    if shared_pre_main.strip():
        lines.append(shared_pre_main)
        lines.append("")

    # Emit RT arg wrapper function definitions at file scope (before any #define)
    if rt_arg_offsets:
        for phase_idx, _ in writer_sources:
            if phase_idx > 0 and phase_idx in rt_arg_offsets:
                lines.extend(_emit_rt_arg_wrapper(phase_idx, rt_arg_offsets[phase_idx]))
        lines.append("")

    # Per-phase pre-main code, each wrapped with that phase's varying defines
    # and RT arg redirect so helper functions get the offset too
    for phase_idx, _ in writer_sources:
        phase_code = per_phase_pre_main.get(phase_idx, "")
        if not phase_code.strip():
            continue
        varying = per_phase_defines.get(phase_idx, [])
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_define(phase_idx))
        if varying:
            lines.extend(_emit_define_lines(varying))
        lines.append(phase_code)
        if varying:
            lines.extend(_emit_undef_lines(varying))
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_undef())
        lines.append("")

    is_multi_phase = len(writer_sources) > 1
    needs_barrier = multi_barrier is not None and len(multi_barrier.transition_map) > 0

    if needs_barrier:
        lines.append('constexpr uint32_t __barrier_rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
        lines.append("")

    # Generate NCRISC CB state resync function (resets local pointers to CB start).
    if sweep_cb_indices and needs_barrier:
        lines.append("// Resync NCRISC local CB pointers to CB start between phases.")
        lines.append("FORCE_INLINE void __resync_ncrisc_cb_state() {")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
            lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = fifo_start;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = fifo_start;")
            lines.append(f"    }}")
        lines.append("}")
        lines.append("")

    # Generate phase functions
    for phase_idx, raw_source in writer_sources:
        body = cpp_parser.extract_kernel_body(raw_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        pnames = phase_names.get(phase_idx, [])
        transformed = _transform_phase_source(body, phase_idx, offset, phase_names=pnames)

        varying = per_phase_defines.get(phase_idx, [])
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_define(phase_idx))
        lines.append(f"// Phase {phase_idx} writer")
        lines.append(f"FORCE_INLINE void phase{phase_idx}_writer() {{")
        if varying:
            for dl in _emit_define_lines(varying):
                lines.append(f"    {dl}")
        for line in transformed.split("\n"):
            lines.append(f"    {line}")
        if varying:
            for ul in _emit_undef_lines(varying):
                lines.append(f"    {ul}")
        lines.append("}")
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_undef())
        lines.append("")

    # Generate kernel_main
    lines.append("void kernel_main() {")

    if needs_barrier:
        lines.append("    // Read barrier L1 flag addresses from runtime args")
        lines.append("    const uint32_t __writer_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset);")
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__writer_done_addr);"
        )
        # Per-segment release pointers
        for seg_idx in range(len(multi_barrier.segments)):
            s = f"seg{seg_idx}"
            off = 1 + seg_idx
            lines.append(f"    const uint32_t __{s}_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + {off});")
            lines.append(
                f"    volatile tt_l1_ptr uint32_t* __{s}_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__{s}_release_addr);"
            )
        lines.append("")

    rebind_info = rebind_info or {}

    num_writers = len(writer_sources)
    for count, (phase_idx, _) in enumerate(writer_sources):
        lines.append(f"    phase{phase_idx}_writer();")
        if count < num_writers - 1 and needs_barrier:
            transition_idx = phase_idx
            if transition_idx in multi_barrier.transition_map:
                next_phase_idx = writer_sources[count + 1][0]
                lines.append("")
                lines.append(f"    // Ensure all async NOC writes from Phase {phase_idx} are complete")
                lines.append("    noc_async_write_barrier();")
                lines.append(f"    // Signal done for Phase {phase_idx}")
                lines.append(f"    *__writer_done = {phase_idx + 1};")
                lines.append("")
                seg_idx, call_idx = multi_barrier.transition_map[transition_idx]
                s = f"seg{seg_idx}"
                lines.append(f"    // Wait for segment {seg_idx} release (call {call_idx})")
                lines.append(f"    while (*__{s}_release < {call_idx + 1}) {{ }}")
                lines.append("")
                # Resync NCRISC local CB pointers to CB start
                if sweep_cb_indices:
                    lines.append("    // Resync NCRISC CB pointers to start")
                    lines.append("    __resync_ncrisc_cb_state();")
                    lines.append("")
                # Rebind CB addresses after barrier wait
                rebind_lines = _generate_rebind_code(
                    rebind_info.get(next_phase_idx, []), next_phase_idx, for_compute=False
                )
                if rebind_lines:
                    lines.extend(rebind_lines)
                    lines.append("")

    # Trailing barrier: after the last phase, if there's a pending transition
    if needs_barrier:
        last_phase_idx = writer_sources[-1][0]
        trailing_transition = last_phase_idx
        if trailing_transition in multi_barrier.transition_map:
            seg_idx, call_idx = multi_barrier.transition_map[trailing_transition]
            s = f"seg{seg_idx}"
            lines.append("")
            lines.append(f"    // === Trailing barrier after Phase {last_phase_idx} ===")
            lines.append("    noc_async_write_barrier();")
            lines.append(f"    *__writer_done = {last_phase_idx + 1};")
            lines.append(f"    while (*__{s}_release < {call_idx + 1}) {{ }}")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _generate_fused_compute_source(
    phase_kernels: List[Dict[str, Any]],
    role_key: Any,
    phases: List[PhaseInfo],
    ct_arg_offsets: Optional[Dict[int, int]] = None,
    sweep_cb_indices: Optional[List[int]] = None,
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
    rt_arg_offsets: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """Generate fused compute kernel with L1 flag barrier sync.

    Between phases, compute:
      1. Signals done by writing phase+1 to compute_done L1 flag
      2. Spins on global_release L1 flag (plain volatile read, no NOC APIs)
      3. Resyncs TRISC0 local CB state with stream registers

    Step 3 is critical: BRISC reset (in reader) updates the hardware stream
    register tiles_acked but NOT TRISC0's local copy.  Without resync,
    compute sees stale tiles_acked and reads garbage.
    """
    if ct_arg_offsets is None:
        ct_arg_offsets = {}

    compute_sources = []

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(role_key)
        if kernel is None:
            continue
        source, kernel_dir = _read_kernel_source(kernel)
        if not source:
            continue
        source = cpp_parser.inline_local_includes(source, kernel_dir)
        compute_sources.append((i, source))

    if not compute_sources:
        return None

    all_sources = [s for _, s in compute_sources]
    includes = cpp_parser.collect_includes(all_sources)
    source_defines = cpp_parser.collect_defines(all_sources)
    uniform_defines, per_phase_defines = _categorize_phase_defines(phase_kernels, role_key)
    shared_pre_main, per_phase_pre_main, phase_names = _collect_all_pre_main_code(compute_sources)

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused compute kernel - {len(compute_sources)} phases",
        "",
    ]
    lines.extend(_emit_define_lines(uniform_defines))
    lines.extend(source_defines)
    lines.append("")
    lines.extend(includes)
    lines.append("")
    if shared_pre_main.strip():
        lines.append(shared_pre_main)
        lines.append("")

    # Emit RT arg wrapper function definitions at file scope (before any #define)
    if rt_arg_offsets:
        for phase_idx, _ in compute_sources:
            if phase_idx > 0 and phase_idx in rt_arg_offsets:
                lines.extend(_emit_rt_arg_wrapper(phase_idx, rt_arg_offsets[phase_idx]))
        lines.append("")

    # Per-phase pre-main code, each wrapped with that phase's varying defines
    # and RT arg redirect so helper functions get the offset too
    for phase_idx, _ in compute_sources:
        phase_code = per_phase_pre_main.get(phase_idx, "")
        if not phase_code.strip():
            continue
        varying = per_phase_defines.get(phase_idx, [])
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_define(phase_idx))
        if varying:
            lines.extend(_emit_define_lines(varying))
        lines.append(phase_code)
        if varying:
            lines.extend(_emit_undef_lines(varying))
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_undef())
        lines.append("")

    is_multi_phase = len(compute_sources) > 1
    needs_barrier = multi_barrier is not None and len(multi_barrier.transition_map) > 0

    if needs_barrier:
        lines.append('constexpr uint32_t __barrier_rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
        lines.append("")

    # Generate compute-side CB state resync function.
    # After BRISC's __cb_reset_to_empty(), stream registers are equalized and
    # BRISC pointers are at CB start. TRISC0 and TRISC2 need to sync their
    # local state and reset pointers to CB start as well.
    if sweep_cb_indices and needs_barrier:
        lines.append("// Resync compute-side local CB state after BRISC reset.")
        lines.append("// TRISC0: sync tiles_acked + reset fifo_rd_ptr to CB start.")
        lines.append("// TRISC2: sync tiles_received + reset fifo_wr_ptr to CB start.")
        lines.append("FORCE_INLINE void __resync_cb_state_after_sweep() {")
        lines.append("#ifdef TRISC_UNPACK")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(
                f"        uint16_t stream_acked = (uint16_t)reg_read((uint32_t)get_cb_tiles_acked_ptr({cb_idx}));"
            )
            lines.append(f"        get_local_cb_interface({cb_idx}).tiles_acked = stream_acked;")
            lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
            lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = fifo_start;")
            lines.append(f"    }}")
        lines.append("#endif")
        lines.append("#ifdef TRISC_PACK")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(
                f"        uint16_t stream_received = (uint16_t)reg_read((uint32_t)get_cb_tiles_received_ptr({cb_idx}));"
            )
            lines.append(f"        get_local_cb_interface({cb_idx}).tiles_received = stream_received;")
            lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
            lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = fifo_start;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_tile_ptr = 0;")
            lines.append(f"    }}")
        lines.append("#endif")
        lines.append("}")
        lines.append("")

    # Generate phase functions
    for phase_idx, raw_source in compute_sources:
        body = cpp_parser.extract_kernel_body(raw_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        pnames = phase_names.get(phase_idx, [])
        transformed = _transform_phase_source(body, phase_idx, offset, phase_names=pnames)

        varying = per_phase_defines.get(phase_idx, [])
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_define(phase_idx))
        lines.append(f"// Phase {phase_idx} compute")
        lines.append(f"FORCE_INLINE void phase{phase_idx}_compute() {{")
        if varying:
            for dl in _emit_define_lines(varying):
                lines.append(f"    {dl}")
        for line in transformed.split("\n"):
            lines.append(f"    {line}")
        if varying:
            for ul in _emit_undef_lines(varying):
                lines.append(f"    {ul}")
        lines.append("}")
        if phase_idx > 0 and rt_arg_offsets and phase_idx in rt_arg_offsets:
            lines.append(_emit_rt_arg_undef())
        lines.append("")

    # Generate kernel_main
    lines.append("void kernel_main() {")

    if needs_barrier:
        lines.append("    // Read barrier L1 flag addresses from runtime args")
        lines.append("    const uint32_t __compute_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset);")
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__compute_done_addr);"
        )
        # Per-segment release pointers
        for seg_idx in range(len(multi_barrier.segments)):
            s = f"seg{seg_idx}"
            off = 1 + seg_idx
            lines.append(f"    const uint32_t __{s}_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + {off});")
            lines.append(
                f"    volatile tt_l1_ptr uint32_t* __{s}_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__{s}_release_addr);"
            )
        lines.append("")

    rebind_info = rebind_info or {}

    first = True
    for count, (phase_idx, _) in enumerate(compute_sources):
        if not first and needs_barrier:
            transition_idx = phase_idx - 1
            if transition_idx in multi_barrier.transition_map:
                lines.append("")
                lines.append(f"    // Signal done for Phase {phase_idx - 1}")
                lines.append(f"    *__compute_done = {phase_idx};")
                lines.append("")
                seg_idx, call_idx = multi_barrier.transition_map[transition_idx]
                s = f"seg{seg_idx}"
                lines.append(f"    // Wait for segment {seg_idx} release (call {call_idx})")
                lines.append(f"    while (*__{s}_release < {call_idx + 1}) {{ }}")
                lines.append("")
                if sweep_cb_indices:
                    lines.append("    // Resync TRISC0 local CB state (tiles_acked, fifo_rd_ptr)")
                    lines.append("    __resync_cb_state_after_sweep();")
                    lines.append("")
                # Rebind CB addresses after resync (all TRISC instances, with >> 4 shift)
                rebind_lines = _generate_rebind_code(rebind_info.get(phase_idx, []), phase_idx, for_compute=True)
                if rebind_lines:
                    lines.extend(rebind_lines)
                    lines.append("")
        lines.append(f"    phase{phase_idx}_compute();")
        first = False

    # Trailing barrier: after the last phase, if there's a pending transition
    if needs_barrier:
        last_phase_idx = compute_sources[-1][0]
        trailing_transition = last_phase_idx
        if trailing_transition in multi_barrier.transition_map:
            seg_idx, call_idx = multi_barrier.transition_map[trailing_transition]
            s = f"seg{seg_idx}"
            lines.append("")
            lines.append(f"    // === Trailing barrier after Phase {last_phase_idx} ===")
            lines.append(f"    *__compute_done = {last_phase_idx + 1};")
            lines.append(f"    while (*__{s}_release < {call_idx + 1}) {{ }}")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Runtime Arg Handling
# =============================================================================


def _compute_runtime_arg_offsets(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
    core_range_override: Optional[Any] = None,
) -> Dict[int, int]:
    """Compute per-phase runtime arg offsets.

    Returns {phase_idx: offset} where offset is the cumulative count of
    runtime args from all prior phases (max across cores).

    RuntimeArgsView API: runtime_args[col_idx] -> RuntimeArgsColProxy,
    runtime_args[col_idx][0] -> VectorUInt32 of args for that core.

    If core_range_override is set, use those core ranges to determine which
    cores to count args for (needed for OpGraph paths where stem ops cover
    all cores but only a subset runs this path's kernel).
    """
    offsets: Dict[int, int] = {}
    cumulative = 0

    for i, pk in enumerate(phase_kernels):
        offsets[i] = cumulative
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        # Count runtime args for this phase (max across cores).
        # RuntimeArgsView uses coordinate-based 2D indexing: [x][y] -> CoreCoord(x,y).
        max_args = 0
        if core_range_override is not None:
            core_coords = _get_core_coords_from_ranges(core_range_override)
        else:
            core_coords = _get_core_coords_from_ranges(kernel.core_ranges)
        for core in core_coords:
            try:
                args = kernel.runtime_args[core.x][core.y]
                max_args = max(max_args, len(args))
            except (IndexError, KeyError):
                if core_range_override is not None:
                    logger.warning(
                        "Phase %d %s: no runtime args for core (%d,%d) "
                        "with core_range_override (stem op may not cover this core)",
                        i,
                        kernel_type,
                        core.x,
                        core.y,
                    )

        cumulative += max_args

    return offsets


def _get_core_coords_from_ranges(core_ranges: Any) -> List[Any]:
    """Extract ordered list of CoreCoords from a CoreRangeSet."""
    coords = []
    for cr in core_ranges.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                coords.append(ttnn.CoreCoord(x, y))
    return coords


def _concatenate_runtime_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
    core_range_override: Optional[Any] = None,
) -> List[Tuple[Any, List[int]]]:
    """Concatenate per-core runtime args from all phases.

    Returns list of (CoreCoord, concatenated_args) pairs.

    RuntimeArgsView uses coordinate-based 2D indexing: runtime_args[x][y]
    maps to CoreCoord(x, y). We must use actual core coordinates, not
    sequential indices.

    If core_range_override is set, use those core ranges instead of the
    kernel's native ranges.  This filters stem ops' args to only the cores
    in this path's branch range (OpGraph support).
    """
    if core_range_override is not None:
        core_coords = _get_core_coords_from_ranges(core_range_override)
    else:
        # Find core_ranges from first available kernel
        core_coords = None
        for pk in phase_kernels:
            kernel = pk.get(kernel_type)
            if kernel is not None:
                core_coords = _get_core_coords_from_ranges(kernel.core_ranges)
                break
    if not core_coords:
        return []

    num_cols = len(core_coords)
    col_args: List[List[int]] = [[] for _ in range(num_cols)]

    for phase_idx, pk in enumerate(phase_kernels):
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        # First pass: compute max_args for this phase (must match _compute_runtime_arg_offsets)
        phase_max_args = 0
        for core in core_coords:
            try:
                args = kernel.runtime_args[core.x][core.y]
                phase_max_args = max(phase_max_args, len(args))
            except (IndexError, KeyError):
                pass

        # Second pass: append args + pad to phase_max_args so offsets align
        for col_idx, core in enumerate(core_coords):
            try:
                args = kernel.runtime_args[core.x][core.y]
                arg_list = list(args)
                col_args[col_idx].extend(arg_list)
                pad_count = phase_max_args - len(arg_list)
                if pad_count > 0:
                    col_args[col_idx].extend([0] * pad_count)
            except (IndexError, KeyError):
                # No args for this core — pad entire phase width
                if phase_max_args > 0:
                    col_args[col_idx].extend([0] * phase_max_args)
                if core_range_override is not None:
                    logger.warning(
                        "Phase %d %s: no runtime args for core (%d,%d) "
                        "with core_range_override (stem op may not cover this core)",
                        phase_idx,
                        kernel_type,
                        core.x,
                        core.y,
                    )

    return [(core_coords[i], col_args[i]) for i in range(num_cols) if col_args[i]]


def _append_barrier_runtime_args(
    rt_args: List[Tuple[Any, List[int]]],
    barrier_addrs: List[int],
) -> Tuple[List[Tuple[Any, List[int]]], int]:
    """Append barrier L1 flag addresses to each core's runtime args.

    Returns (updated_rt_args, barrier_rt_offset) where barrier_rt_offset
    is the index in each core's args where the barrier addresses start.
    """
    if not rt_args:
        return rt_args, 0

    # Offset = length of first core's existing args (all cores should have same count)
    barrier_offset = len(rt_args[0][1])

    updated = []
    for core_coord, args in rt_args:
        updated.append((core_coord, args + barrier_addrs))

    return updated, barrier_offset


def _concatenate_common_runtime_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> List[int]:
    """Concatenate common runtime args from all phases."""
    common_args: List[int] = []
    for pk in phase_kernels:
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue
        try:
            common_args.extend(list(kernel.common_runtime_args))
        except (AttributeError, TypeError):
            pass
    return common_args


# =============================================================================
# Named Compile-Time Arg Merging
# =============================================================================


def _merge_named_compile_time_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
    barrier_rt_offset: Optional[int] = None,
    phase_remaps: Optional[List[Dict[int, int]]] = None,
) -> List[Tuple[str, int]]:
    """Merge named compile-time args from all phases with phase prefixes.

    Phase 0 keeps original names. Phase N>0 gets "phaseN_" prefix.
    CB-reference args (names starting with "cb_") are remapped to pool slot indices.
    Per-segment barrier constants are added externally by the caller.
    """
    merged = []

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        remap = phase_remaps[i] if phase_remaps else None

        for name, value in kernel.named_compile_time_args:
            actual_value = value
            # Remap CB-reference named args to pool slot indices
            if remap is not None and _is_cb_named_arg(name, value):
                actual_value = remap.get(value, value)

            if i == 0:
                merged.append((name, actual_value))
            else:
                merged.append((f"phase{i}_{name}", actual_value))

    # Add barrier runtime arg offset
    if barrier_rt_offset is not None:
        merged.append(("barrier_rt_offset", barrier_rt_offset))

    return merged


def _merge_compile_time_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> Tuple[List[int], Dict[int, int]]:
    """Concatenate all phases' compile-time args and return (merged_args, offsets).

    Phase 0's args go first, then phase 1's, etc.  The offsets dict maps
    phase_idx -> starting index in the merged array so that phase N's source
    can reference get_compile_time_arg_val(original_idx + offset).
    """
    merged: List[int] = []
    offsets: Dict[int, int] = {}

    for i, pk in enumerate(phase_kernels):
        offsets[i] = len(merged)
        kernel = pk.get(kernel_type)
        if kernel is not None:
            merged.extend(list(kernel.compile_time_args))

    return merged, offsets


# These defines are referenced by LLK headers at include time and cannot
# vary per-phase.  They MUST have identical values across all fused phases.
_MUST_MATCH_DEFINES = frozenset({"REDUCE_OP", "REDUCE_DIM", "BCAST_LLKOP", "BCAST_DIM"})


def _categorize_phase_defines(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    """Categorize defines from all phases into uniform and per-phase.

    Uniform defines (same name+value in ALL phases) are emitted once as
    ``#define`` lines at the top of the generated source.  Varying defines
    (different value or present in only some phases) are emitted per-phase
    with ``#define``/``#undef`` pairs around each phase's code.

    ``_MUST_MATCH_DEFINES`` are validated for consistency and treated as
    uniform.

    Returns:
        (uniform, per_phase) where:
        - uniform: list of (name, value) for defines identical across all phases
        - per_phase: dict mapping phase_index -> list of (name, value) for that phase

    Raises:
        ValueError: If a MUST_MATCH define has inconsistent values across phases.
    """
    # Collect per-phase define dicts: name -> value
    per_phase_defs: Dict[int, Dict[str, str]] = {}
    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(kernel_type)
        if kernel is None:
            per_phase_defs[i] = {}
            continue
        defs = {}
        if hasattr(kernel, "defines"):
            for name, value in kernel.defines:
                defs[name] = value
        per_phase_defs[i] = defs

    # Collect all define names across all phases
    all_names: Set[str] = set()
    for defs in per_phase_defs.values():
        all_names.update(defs.keys())

    # Classify each define name
    uniform: List[Tuple[str, str]] = []
    seen_uniform: Set[str] = set()
    varying_names: Set[str] = set()
    must_match_values: Dict[str, Tuple[str, int]] = {}  # name -> (value, first_phase)

    for name in sorted(all_names):
        # Collect (value, phase_idx) for phases that have this define
        occurrences = [(defs[name], idx) for idx, defs in per_phase_defs.items() if name in defs]

        if name in _MUST_MATCH_DEFINES:
            # Validate consistency
            for value, phase_idx in occurrences:
                if name in must_match_values:
                    prev_val, prev_phase = must_match_values[name]
                    if value != prev_val:
                        raise ValueError(
                            f"Define '{name}' has inconsistent values across phases: "
                            f"phase {prev_phase} has '{prev_val}', phase {phase_idx} has '{value}'. "
                            f"These defines must have identical values in all fused phases "
                            f"because they are referenced by LLK headers at include time."
                        )
                else:
                    must_match_values[name] = (value, phase_idx)
            # Treat as uniform
            if occurrences and name not in seen_uniform:
                uniform.append((name, occurrences[0][0]))
                seen_uniform.add(name)
            continue

        # Check if uniform: present in ALL phases with same value
        if len(occurrences) == len(per_phase_defs):
            values = {v for v, _ in occurrences}
            if len(values) == 1:
                uniform.append((name, occurrences[0][0]))
                seen_uniform.add(name)
                continue

        # Varying: present in only some phases, or different values
        varying_names.add(name)

    # Build per-phase varying define lists
    per_phase: Dict[int, List[Tuple[str, str]]] = {}
    for idx, defs in per_phase_defs.items():
        phase_varying = [(name, defs[name]) for name in sorted(varying_names) if name in defs]
        if phase_varying:
            per_phase[idx] = phase_varying

    return uniform, per_phase


def _emit_define_lines(defines: List[Tuple[str, str]]) -> List[str]:
    """Generate ``#define NAME VALUE`` lines from a list of (name, value) pairs."""
    lines = []
    for name, value in defines:
        if value:
            lines.append(f"#define {name} {value}")
        else:
            lines.append(f"#define {name}")
    return lines


def _emit_undef_lines(defines: List[Tuple[str, str]]) -> List[str]:
    """Generate ``#undef NAME`` lines from a list of (name, value) pairs."""
    return [f"#undef {name}" for name, _ in defines]


# =============================================================================
# Compute Config Validation
# =============================================================================


def _validate_and_get_compute_config_for_role(
    phase_kernels: List[Dict[Any, Any]],
    role_key: Any,
) -> "ttnn.ComputeConfigDescriptor":
    """Validate compute config consistency for a specific role across phases."""
    base = None
    base_phase = -1

    for phase_idx, pk in enumerate(phase_kernels):
        kernel = pk.get(role_key)
        if kernel is None:
            continue

        config = kernel.config
        if base is None:
            base = config
            base_phase = phase_idx
            continue

        mismatches = []
        for field in ("fp32_dest_acc_en", "math_approx_mode", "math_fidelity", "dst_full_sync_en", "bfp8_pack_precise"):
            base_val = getattr(base, field, None)
            this_val = getattr(config, field, None)
            if base_val != this_val:
                mismatches.append(f"  {field}: phase {base_phase}={base_val}, phase {phase_idx}={this_val}")

        if mismatches:
            raise ValueError(f"Compute config mismatch for role {role_key}.\n" + "\n".join(mismatches))

    if base is None:
        return ttnn.ComputeConfigDescriptor()

    return base


# =============================================================================
# Validation
# =============================================================================


def _validate_fp32_consistency(op_descriptors: List[OpDescriptor]) -> None:
    """Validate fp32_dest_acc_en consistency across all phases.

    DST_ACCUM_MODE is a compile-time constant that cannot change mid-kernel.
    All fused phases must use the same fp32_dest_acc_en setting.
    """
    fp32_settings = []
    for i, desc in enumerate(op_descriptors):
        for kernel_desc in desc.descriptor.kernels:
            config = kernel_desc.config
            if hasattr(config, "fp32_dest_acc_en"):
                fp32_settings.append((i, config.fp32_dest_acc_en))
                break

    if not fp32_settings:
        return

    fp32_values = {v for _, v in fp32_settings}
    if len(fp32_values) <= 1:
        return

    phases_with = [i for i, v in fp32_settings if v]
    phases_without = [i for i, v in fp32_settings if not v]

    raise ValueError(
        f"fp32_dest_acc_en mismatch: phases {phases_with} use fp32=True, "
        f"phases {phases_without} use fp32=False. "
        f"DST_ACCUM_MODE is a kernel-level hardware setting that cannot be "
        f"changed mid-kernel. All phases must use the same fp32_dest_acc_en "
        f"setting. To fix: create all op descriptors with a consistent "
        f"compute_kernel_config."
    )


# =============================================================================
# Barrier Configuration
# =============================================================================


def _create_barrier_segment_config(device: Any, core_ranges: Any) -> BarrierConfig:
    """Create a lightweight barrier config for OpGraph segments.

    Only allocates ``global_arrive`` and ``global_release`` GlobalSemaphores
    (2 instead of 4).  The per-core ``compute_done`` / ``writer_done`` flags
    are shared across all segments and allocated separately in
    ``OpGraphBuilder.build()``, so per-segment copies would waste L1.
    """
    config = BarrierConfig()

    sem_global_arrive = ttnn.create_global_semaphore(device, core_ranges, 0)
    sem_global_release = ttnn.create_global_semaphore(device, core_ranges, 0)

    config._sem_refs = [sem_global_arrive, sem_global_release]
    config.global_arrive_addr = ttnn.get_global_semaphore_address(sem_global_arrive)
    config.global_release_addr = ttnn.get_global_semaphore_address(sem_global_release)

    logical_coords = _get_core_coords_from_ranges(core_ranges)
    config.num_cores = len(logical_coords)

    if config.num_cores > 0:
        phys_coords = [device.worker_core_from_logical_core(c) for c in logical_coords]
        config.core0_phys_x = phys_coords[0].x
        config.core0_phys_y = phys_coords[0].y
        config.mcast_start_x = min(c.x for c in phys_coords)
        config.mcast_start_y = min(c.y for c in phys_coords)
        config.mcast_end_x = max(c.x for c in phys_coords)
        config.mcast_end_y = max(c.y for c in phys_coords)

        if config.num_cores > 1:
            _validate_rectangular_grid(phys_coords, config)

    return config


def _validate_rectangular_grid(phys_coords: List[Any], config: BarrierConfig) -> None:
    """Validate that physical cores form a rectangle for safe NOC multicast.

    NOC multicast sends to ALL cores in the bounding box. If the actual core
    set is non-rectangular (e.g., L-shaped), the multicast would write to
    unintended cores, corrupting their L1 memory.
    """
    phys_set = set((c.x, c.y) for c in phys_coords)
    bbox_w = config.mcast_end_x - config.mcast_start_x + 1
    bbox_h = config.mcast_end_y - config.mcast_start_y + 1
    bbox_area = bbox_w * bbox_h
    if len(phys_set) != bbox_area:
        raise ValueError(
            f"Fused kernel global barrier requires rectangular core grid for "
            f"safe NOC multicast. Got {len(phys_set)} physical cores in "
            f"bounding box {bbox_w}x{bbox_h} ({bbox_area} cores). "
            f"Physical coords: {sorted(phys_set)}"
        )


# =============================================================================
# Fused Descriptor Builder
# =============================================================================


def _build_fused_descriptor(
    phases: List[PhaseInfo],
    device: Any,
    core_range_override: Optional[Any] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
) -> _BuildResult:
    """Build a fused ProgramDescriptor with multi-segment barrier sync.

    Dynamically discovers kernel roles from the ProgramDescriptor using
    (risc_type, core_ranges) as a unique key. This supports any op type
    (interleaved with 3 kernels, sharded with up to 7 kernels, etc.).

    Args:
        phases: List of PhaseInfo objects for each phase.
        device: The device for GlobalSemaphore allocation.
        core_range_override: If set, overrides role keys and fused kernel
            core_ranges. Used by OpGraphBuilder for paths where stem ops
            have wider core ranges than the path's leaf branch.
        multi_barrier: Multi-segment barrier spec. Required for multi-phase
            chains. Provides barrier segment configs and transition map.
    """
    # Validate fp32 consistency
    _validate_fp32_consistency([p.op_descriptor for p in phases])

    # Discover all kernel roles from phase 0
    role_keys: List[Tuple[str, frozenset]] = []
    role_keys_set: Set[Tuple[str, frozenset]] = set()
    for kernel_desc in phases[0].op_descriptor.descriptor.kernels:
        rk = _get_role_key(kernel_desc, core_range_override)
        if rk not in role_keys_set:
            role_keys.append(rk)
            role_keys_set.add(rk)

    # Build phase_kernels as List[Dict[role_key, KernelDescriptor]]
    phase_kernels: List[Dict[Any, Any]] = []
    for phase_idx, phase in enumerate(phases):
        role_map: Dict[Any, Any] = {}
        for kernel_desc in phase.op_descriptor.descriptor.kernels:
            rk = _get_role_key(kernel_desc, core_range_override)
            role_map[rk] = kernel_desc
        phase_kernels.append(role_map)

    # Pool-allocate CB slots based on compatibility keys
    pool = CBPoolAllocator(max_slots=32)
    for phase_idx, phase in enumerate(phases):
        phantom_indices = _get_phantom_cb_indices(phase)
        pool.allocate_phase(phase_idx, phase.cb_info, phantom_indices)

    # Compute CB address rebinding info using remapped slot indices.
    rebind_info = _compute_rebind_info(phases, pool.phase_remaps)

    # Build merged CB descriptors from pool (modifies buffer_index in-place)
    merged_cbs = pool.build_merged_cb_descriptors(phases)

    # Override CB core_ranges when building a path through an OpGraph.
    if core_range_override is not None:
        for cb_desc in merged_cbs:
            cb_desc.core_ranges = core_range_override

    # Sweep indices = all allocated CB pool slots
    sweep_cb_indices = sorted(pool.get_all_slot_indices())

    # Collect all unique op semaphore (id, initial_value) pairs used by any phase.
    op_semaphore_info: List[Tuple[int, int]] = []
    seen_sem_ids_for_reset: Set[int] = set()
    for phase in phases:
        for sem in phase.op_descriptor.descriptor.semaphores:
            if sem.id not in seen_sem_ids_for_reset:
                op_semaphore_info.append((sem.id, sem.initial_value))
                seen_sem_ids_for_reset.add(sem.id)
    op_semaphore_info.sort(key=lambda x: x[0])

    fused_kernels = []

    for role_key in role_keys:
        risc_type, core_key = role_key

        # Get role-specific core_ranges
        if core_range_override is not None:
            role_core_ranges = core_range_override
        else:
            role_core_ranges = None
            for pk in phase_kernels:
                kernel = pk.get(role_key)
                if kernel is not None:
                    role_core_ranges = kernel.core_ranges
                    break

        if role_core_ranges is None:
            continue

        # Merge compile-time args and compute offsets
        ct_args, ct_offsets = _merge_compile_time_args(phase_kernels, role_key)
        rt_offsets = _compute_runtime_arg_offsets(phase_kernels, role_key, core_range_override=core_range_override)

        # Generate fused source and determine barrier addresses per RISC type
        if risc_type == "riscv_0":
            fused_source = _generate_fused_riscv0_source(
                phase_kernels,
                role_key,
                phases,
                ct_offsets,
                sweep_cb_indices,
                rebind_info=rebind_info,
                op_semaphore_info=op_semaphore_info,
                multi_barrier=multi_barrier,
                rt_arg_offsets=rt_offsets,
            )
            barrier_addrs = []
            if multi_barrier is not None:
                barrier_addrs = [multi_barrier.compute_done_addr, multi_barrier.writer_done_addr]
                for seg in multi_barrier.segments:
                    barrier_addrs.extend([seg.arrive_addr, seg.release_addr])
        elif risc_type == "riscv_1":
            fused_source = _generate_fused_riscv1_source(
                phase_kernels,
                role_key,
                phases,
                ct_offsets,
                sweep_cb_indices,
                rebind_info=rebind_info,
                multi_barrier=multi_barrier,
                rt_arg_offsets=rt_offsets,
            )
            barrier_addrs = []
            if multi_barrier is not None:
                barrier_addrs = [multi_barrier.writer_done_addr]
                for seg in multi_barrier.segments:
                    barrier_addrs.append(seg.release_addr)
        elif risc_type == "compute":
            fused_source = _generate_fused_compute_source(
                phase_kernels,
                role_key,
                phases,
                ct_offsets,
                sweep_cb_indices,
                rebind_info=rebind_info,
                multi_barrier=multi_barrier,
                rt_arg_offsets=rt_offsets,
            )
            barrier_addrs = []
            if multi_barrier is not None:
                barrier_addrs = [multi_barrier.compute_done_addr]
                for seg in multi_barrier.segments:
                    barrier_addrs.append(seg.release_addr)
        else:
            continue

        if fused_source is None:
            continue

        # Concatenate runtime args and append barrier addresses
        rt_args = _concatenate_runtime_args(phase_kernels, role_key, core_range_override=core_range_override)
        rt_args, barrier_offset = _append_barrier_runtime_args(rt_args, barrier_addrs)

        # Merge named compile-time args
        named_ct_args = _merge_named_compile_time_args(
            phase_kernels,
            role_key,
            barrier_rt_offset=barrier_offset if barrier_addrs else None,
            phase_remaps=pool.phase_remaps,
        )
        # Add per-segment named compile-time args (only riscv_0 needs them)
        if multi_barrier is not None and risc_type == "riscv_0":
            for seg_idx, seg in enumerate(multi_barrier.segments):
                s = f"seg{seg_idx}"
                named_ct_args.append((f"{s}_num_cores", seg.config.num_cores))
                named_ct_args.append((f"{s}_core0_phys_x", seg.config.core0_phys_x))
                named_ct_args.append((f"{s}_core0_phys_y", seg.config.core0_phys_y))
                named_ct_args.append((f"{s}_mcast_start_x", seg.config.mcast_start_x))
                named_ct_args.append((f"{s}_mcast_start_y", seg.config.mcast_start_y))
                named_ct_args.append((f"{s}_mcast_end_x", seg.config.mcast_end_x))
                named_ct_args.append((f"{s}_mcast_end_y", seg.config.mcast_end_y))

        # Add rebind named compile-time args (addr + size for each CB that changes)
        for phase_idx, rebinds in rebind_info.items():
            for slot_idx, addr, size in rebinds:
                prefix = f"phase{phase_idx}_cb{slot_idx}"
                named_ct_args.append((f"{prefix}_rebind_addr", addr))
                named_ct_args.append((f"{prefix}_rebind_size", size))

        # Get config from first available kernel for this role
        role_config = None
        for pk in phase_kernels:
            kernel = pk.get(role_key)
            if kernel is not None:
                role_config = kernel.config
                break

        # For compute roles, validate configs match across phases and
        # rebuild unpack_to_dest_mode from pool-allocated slot indices
        if risc_type == "compute":
            role_config = _validate_and_get_compute_config_for_role(phase_kernels, role_key)
            role_config.unpack_to_dest_mode = pool.build_unpack_to_dest_mode()

        # Build fused kernel descriptor
        desc = ttnn.KernelDescriptor()
        desc.kernel_source = fused_source
        desc.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        desc.core_ranges = role_core_ranges
        desc.compile_time_args = ct_args
        desc.named_compile_time_args = named_ct_args
        # Only uniform defines go to the compiler as -D flags.
        # Varying defines are handled by #define/#undef in the generated source.
        uniform_defs, _ = _categorize_phase_defines(phase_kernels, role_key)
        desc.defines = uniform_defs
        desc.runtime_args = rt_args
        desc.common_runtime_args = _concatenate_common_runtime_args(phase_kernels, role_key)
        desc.config = role_config
        fused_kernels.append(desc)

    # Merge semaphores (dedup by ID)
    all_semaphores = []
    seen_sem_ids: Set[int] = set()
    for phase in phases:
        for sem in phase.op_descriptor.descriptor.semaphores:
            if sem.id not in seen_sem_ids:
                all_semaphores.append(sem)
                seen_sem_ids.add(sem.id)

    # Collect input/output tensors (use id() for dedup because ttnn Tensor's
    # __eq__ returns an element-wise Tensor, making `in` unreliable)
    all_input_tensors = []
    seen_tensor_ids: Set[int] = set()
    for phase in phases:
        for tensor in phase.op_descriptor.input_tensors:
            tid = id(tensor)
            if tid not in seen_tensor_ids:
                all_input_tensors.append(tensor)
                seen_tensor_ids.add(tid)

    output_tensor = None
    if phases[-1].op_descriptor.output_tensors:
        output_tensor = phases[-1].op_descriptor.output_tensors[0]

    # Create the merged ProgramDescriptor
    merged_descriptor = ttnn.ProgramDescriptor()
    merged_descriptor.kernels = fused_kernels
    merged_descriptor.cbs = merged_cbs
    merged_descriptor.semaphores = all_semaphores

    # Collect semaphore references to prevent GC of GlobalSemaphores
    sem_refs = tuple(multi_barrier._sem_refs) if multi_barrier is not None else ()

    return _BuildResult(
        descriptor=merged_descriptor,
        input_tensors=all_input_tensors,
        output_tensors=[output_tensor] if output_tensor else [],
        semaphores=sem_refs,
    )


# =============================================================================
# OpGraph Builder
# =============================================================================


class OpGraphBuilder:
    """Builds fused descriptors from a tree of OpNode objects.

    The fusion tree is a standard tree where each node holds one operation.
    Parent→child edges encode sequential ordering; sibling nodes run in
    parallel on disjoint core subsets.  For each root-to-leaf path, a
    separate fused kernel binary is generated.  Nodes sharing a path
    segment synchronize via shared GlobalSemaphore addresses.

    Usage::

        root = OpNode(ln_op, children=[OpNode(rms_op_A), OpNode(rms_op_B)])
        fused = OpGraphBuilder(root).build()
        # Returns a single FusedOp, suitable for composite.launch()
    """

    def __init__(self, root: OpNode):
        self._root = root
        self._built = False

    def build(self, device: Any = None) -> FusedOp:
        """Build a fused descriptor from the tree.

        Returns a single self-contained FusedOp.  For branching trees,
        path ProgramDescriptors are merged internally so the result can be
        dispatched as one unit via ``composite.launch([result])``.

        Output tensors are ordered by leaf in left-to-right DFS order.

        Args:
            device: Optional device for GlobalSemaphore allocation.
                If *None*, auto-extracted from the first tensor found
                in the tree's OpDescriptors.
        """
        r = self._build_internal(device)
        return FusedOp(
            op=OpDescriptor(r.descriptor, r.input_tensors, r.output_tensors),
            semaphores=r.semaphores,
        )

    def _build_internal(self, device: Any = None) -> _BuildResult:
        """Internal build returning intermediate _BuildResult."""
        if self._built:
            raise ValueError("Already built")
        self._built = True

        # Single node with no children = nothing to fuse
        if not self._root.children:
            op = self._root.op
            return _BuildResult(
                descriptor=op.descriptor,
                input_tensors=op.input_tensors,
                output_tensors=op.output_tensors,
            )

        # Validate tree topology before doing any device allocation
        self._validate_topology()

        # Auto-extract device from tensors if not provided
        if device is None:
            device = _extract_device_from_tree(self._root)
            if device is None:
                raise ValueError("Cannot auto-extract device: no tensors found in tree. " "Pass device explicitly.")

        # Trace all root-to-leaf paths
        paths = self._trace_paths()

        # Compute union of all leaf core ranges
        union_range = self._compute_union_ranges()

        # Allocate shared per-core monotonic semaphores on union range.
        sem_compute_done = ttnn.create_global_semaphore(device, union_range, 0)
        sem_writer_done = ttnn.create_global_semaphore(device, union_range, 0)
        compute_done_addr = ttnn.get_global_semaphore_address(sem_compute_done)
        writer_done_addr = ttnn.get_global_semaphore_address(sem_writer_done)
        all_sem_refs = [sem_compute_done, sem_writer_done]

        # Pre-allocate barrier configs for each unique core range across all
        # path segments.  Paths that share a segment MUST use the same
        # arrive/release GlobalSemaphore L1 addresses so that cores running
        # different kernel binaries synchronize at the same barrier.
        segment_cache: Dict[frozenset, BarrierConfig] = {}
        for path in paths:
            for core_range, _ in path:
                key = _core_ranges_key(core_range)
                if key not in segment_cache:
                    segment_cache[key] = _create_barrier_segment_config(device, core_range)
                    all_sem_refs.extend(segment_cache[key]._sem_refs)

        results = []
        for path in paths:
            # Save CB descriptor state before building each path.
            # _build_fused_descriptor mutates buffer_index, total_size, and
            # core_ranges IN-PLACE on the original CBDescriptors (can't
            # deepcopy C++ bindings).  When paths share ops (e.g. root),
            # the first path's mutations corrupt subsequent paths' cb_info.
            path_prog_descs = [op.descriptor for _, phases in path for op in phases]
            saved_cb_state = _save_cb_state(path_prog_descs)

            result = self._build_path(
                device,
                path,
                compute_done_addr,
                writer_done_addr,
                all_sem_refs,
                segment_cache,
            )
            results.append(result)

            # Restore original state so the next path sees uncorrupted indices
            _restore_cb_state(saved_cb_state)
            _verify_cb_restore(saved_cb_state)

        return _merge_build_results(results)

    def _trace_paths(self) -> List[List[Tuple[Any, List[OpDescriptor]]]]:
        """Trace all root-to-leaf paths through the tree.

        Returns a list of paths.  Each path is a list of
        ``(core_range, [phases])`` segments.  Consecutive nodes with the
        same core range are grouped into one segment.

        For intermediate nodes (those with children), the segment's
        core_range is the *effective* range — the union of descendant leaf
        ranges — rather than the node's own core_range.  This ensures
        barrier scopes match the cores that actually run through each
        segment.
        """
        raw_paths: List[List[Tuple[Any, OpDescriptor]]] = []

        def _collect(node: OpNode, prefix: List[Tuple[Any, OpDescriptor]]):
            if not node.children:
                # Leaf: use the node's own core range
                core_range = _get_node_core_range(node)
                raw_paths.append(prefix + [(core_range, node.op)])
            else:
                # Internal node: use effective range (union of descendant leaves)
                eff_coords = self._effective_leaf_range(node)
                eff_range = _coords_to_core_range_set(eff_coords)
                current = prefix + [(eff_range, node.op)]
                for child in node.children:
                    _collect(child, current)

        _collect(self._root, [])
        return [self._group_into_segments(raw) for raw in raw_paths]

    @staticmethod
    def _group_into_segments(
        raw_path: List[Tuple[Any, OpDescriptor]],
    ) -> List[Tuple[Any, List[OpDescriptor]]]:
        """Group consecutive same-range nodes into segments."""
        segments: List[Tuple[Any, List[OpDescriptor]]] = []
        for core_range, op in raw_path:
            if segments and _same_core_range(segments[-1][0], core_range):
                segments[-1][1].append(op)
            else:
                segments.append((core_range, [op]))
        return segments

    def _compute_union_ranges(self) -> Any:
        """Compute the union CoreRangeSet of all leaf core ranges.

        Only leaf nodes (those without children) contribute core ranges.
        """
        all_coords: Set[Tuple[int, int]] = set()

        def _collect_leaves(node: OpNode):
            if not node.children:
                all_coords.update(_core_range_set_to_coords(_get_node_core_range(node)))
            else:
                for child in node.children:
                    _collect_leaves(child)

        _collect_leaves(self._root)
        return _coords_to_core_range_set(all_coords)

    def _validate_topology(self) -> None:
        """Validate the tree topology.

        Checks:
        - Each child's core_range is a subset of its parent's range.
        - Sibling nodes have disjoint core ranges.

        Children are NOT required to fully cover their parent's range.
        Unused cores simply don't participate in child phases.

        Raises:
            ValueError: On any topology violation.
        """

        def _validate_children(node: OpNode, parent_coords: Set[Tuple[int, int]], depth: int):
            if not node.children:
                return

            # Check sibling disjointness
            seen_coords: Set[Tuple[int, int]] = set()
            for child in node.children:
                child_coords = _core_range_set_to_coords(_get_node_core_range(child))
                overlap = seen_coords & child_coords
                if overlap:
                    raise ValueError(
                        f"OpGraph topology error: sibling nodes at depth {depth + 1} "
                        f"have overlapping cores {sorted(overlap)}"
                    )
                seen_coords |= child_coords

            # Check each child is subset of parent, then recurse
            for child in node.children:
                child_coords = _core_range_set_to_coords(_get_node_core_range(child))
                if not child_coords.issubset(parent_coords):
                    extra = sorted(child_coords - parent_coords)
                    raise ValueError(
                        f"OpGraph topology error: node at depth {depth + 1} "
                        f"(cores {sorted(child_coords)}) has cores {extra} "
                        f"outside parent range {sorted(parent_coords)}"
                    )
                _validate_children(child, child_coords, depth + 1)

        root_coords = _core_range_set_to_coords(_get_node_core_range(self._root))
        _validate_children(self._root, root_coords, depth=0)

    @staticmethod
    def _effective_leaf_range(node: OpNode) -> Set[Tuple[int, int]]:
        """Compute the union of all descendant leaf core coordinates.

        For leaf nodes, returns the node's own core coords.
        For internal nodes, returns the union of all leaf descendants.
        """
        if not node.children:
            return _core_range_set_to_coords(_get_node_core_range(node))
        coords: Set[Tuple[int, int]] = set()
        for child in node.children:
            coords |= OpGraphBuilder._effective_leaf_range(child)
        return coords

    def _build_path(
        self,
        device: Any,
        path: List[Tuple[Any, List[OpDescriptor]]],
        compute_done_addr: int,
        writer_done_addr: int,
        shared_sem_refs: List[Any],
        segment_cache: Dict[frozenset, BarrierConfig],
    ) -> _BuildResult:
        """Build a fused _BuildResult for one root-to-leaf path."""
        # Flatten phases and determine leaf core range
        all_phases: List[OpDescriptor] = []
        for _, phases in path:
            all_phases.extend(phases)

        # Leaf core range = last segment's core range.
        # For linear chains (single segment), leaf == union so no override needed.
        leaf_core_range = path[-1][0] if len(path) > 1 else None

        # Build barrier segments and transition map
        segments, transition_map = self._build_barrier_segments(
            path,
            all_phases,
            segment_cache,
        )

        # Build MultiBarrierSpec
        multi_barrier = MultiBarrierSpec(
            segments=segments,
            compute_done_addr=compute_done_addr,
            writer_done_addr=writer_done_addr,
            transition_map=transition_map,
            _sem_refs=list(shared_sem_refs),
        )

        # Build PhaseInfo list and call module-level _build_fused_descriptor
        phase_infos = [_create_phase_info(op, i) for i, op in enumerate(all_phases)]

        return _build_fused_descriptor(
            phase_infos,
            device,
            core_range_override=leaf_core_range,
            multi_barrier=multi_barrier,
        )

    def _build_barrier_segments(
        self,
        path: List[Tuple[Any, List[OpDescriptor]]],
        all_phases: List[OpDescriptor],
        segment_cache: Dict[frozenset, BarrierConfig],
    ) -> Tuple[List[BarrierSegment], Dict[int, Tuple[int, int]]]:
        """Build barrier segments and transition map for a path.

        Uses ``segment_cache`` (keyed by core_ranges_key) so that paths
        sharing a segment (e.g. the stem) reuse the same GlobalSemaphore
        arrive/release addresses for cross-kernel synchronization.

        Returns (segments, transition_map).
        """
        segments: List[BarrierSegment] = []

        # Build one barrier segment per path segment that has at least one
        # transition (i.e. the segment contains phases followed by more phases).
        # Determine which global-phase transitions belong to which segment.
        #
        # Phase layout example for path [(union, [op0, op1]), (branchA, [op2]), (leafA1, [op3])]:
        #   Global phases: 0=op0, 1=op1, 2=op2, 3=op3
        #   Transitions:   0 (after op0), 1 (after op1), 2 (after op2)
        #   Segment 0 (union):  transitions 0, 1  (between stem phases, and stem->branch)
        #   Segment 1 (branchA): transition 2     (between branch and leaf)

        transition_map: Dict[int, Tuple[int, int]] = {}
        global_phase_offset = 0

        for seg_path_idx, (core_range, phases) in enumerate(path):
            num_phases_in_seg = len(phases)
            if num_phases_in_seg == 0:
                global_phase_offset += num_phases_in_seg
                continue

            # Determine how many transitions this segment owns:
            # - All transitions between consecutive phases within this segment
            # - Plus the transition from last phase of this segment to first
            #   phase of the NEXT segment (if there is a next segment)
            is_last_segment = seg_path_idx == len(path) - 1
            num_transitions_within = num_phases_in_seg - 1
            num_transitions = num_transitions_within
            if not is_last_segment:
                num_transitions += 1  # transition to next segment

            if num_transitions > 0:
                # Look up pre-allocated barrier config from cache
                key = _core_ranges_key(core_range)
                barrier_cfg = segment_cache[key]
                seg = BarrierSegment(
                    config=barrier_cfg,
                    arrive_addr=barrier_cfg.global_arrive_addr,
                    release_addr=barrier_cfg.global_release_addr,
                )
                segments.append(seg)

                seg_idx = len(segments) - 1
                call_idx = 0

                # Map transitions within this segment
                for t in range(num_transitions_within):
                    global_transition = global_phase_offset + t
                    transition_map[global_transition] = (seg_idx, call_idx)
                    call_idx += 1

                # Map transition to next segment (if any)
                if not is_last_segment:
                    global_transition = global_phase_offset + num_phases_in_seg - 1
                    transition_map[global_transition] = (seg_idx, call_idx)

            global_phase_offset += num_phases_in_seg

        return segments, transition_map


# =============================================================================
# Convenience Functions
# =============================================================================


def build_op_graph(
    root_phases: List[OpDescriptor],
    children: List[OpNode],
    device: Any = None,
) -> FusedOp:
    """Build a fused descriptor for a tree topology.

    Convenience wrapper around :class:`OpGraphBuilder`.  Converts
    ``root_phases`` into a chain of nodes, attaches ``children`` to
    the last root-phase node, then builds.

    Args:
        root_phases: Phases that run before the tree splits.  Converted
            into a chain of OpNode objects.
        children: Subtrees to attach to the last root-phase node.
        device: Optional device for GlobalSemaphore allocation.
            If *None*, auto-extracted from the first tensor found
            in the tree's OpDescriptors.

    Returns:
        A single self-contained FusedOp suitable for
        ``composite.launch([result])``.
    """
    if not root_phases:
        raise ValueError("root_phases cannot be empty")
    # Last root phase gets children attached
    last = OpNode(root_phases[-1], children=list(children))
    node = last
    for desc in reversed(root_phases[:-1]):
        node = OpNode(desc, children=[node])
    return OpGraphBuilder(node).build(device)


__all__ = [
    # High-level API
    "Sequential",
    "Parallel",
    # Core classes
    "OpGraphBuilder",
    "OpNode",
    "CBPoolAllocator",
    "PhaseInfo",
    "CBInfo",
    "BarrierConfig",
    # Functions
    "build_op_graph",
    "extract_cb_info",
    "extract_cb_names_from_kernel",
]
