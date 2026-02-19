# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CB Pool Allocation and Analysis for Kernel Fusion.

Manages circular buffer (CB) hardware slot allocation across fused phases.
CBs from different phases with matching configurations (data_format, page_size,
unpack_to_dest_mode) share hardware slots; mismatches get separate slots.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

import ttnn
from ttnn._ttnn.program_descriptor import UnpackToDestMode

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor

logger = logging.getLogger(__name__)


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
        so that CBs keep their original hardware indices.
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
        """Create a lookup key for finding the CBDescriptor later."""
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
        all in-place modifications.
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
        merged = []
        seen_ids: Set[int] = set()
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
        Default elsewhere.
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


__all__ = [
    "CBInfo",
    "PhaseInfo",
    "CBPoolKey",
    "CBSlot",
    "CBPoolAllocator",
    "extract_cb_info",
    "CB_ARG_PREFIX",
    "_is_cb_named_arg",
    "extract_cb_names_from_kernel",
    "_save_cb_state",
    "_restore_cb_state",
    "_verify_cb_restore",
]
