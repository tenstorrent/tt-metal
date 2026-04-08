# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CB Pool Allocation and Analysis for Kernel Fusion.

Manages circular buffer (CB) hardware slot allocation across fused phases.
CBs from different phases with matching configurations (data_format, page_size,
unpack_to_dest_mode) share hardware slots; mismatches get separate slots.
"""

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

import ttnn
from ttnn import UnpackToDestMode

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
    data_format: Any  # int (raw tt::DataFormat uint8) or mock value in tests
    page_size: int
    core_ranges: Any  # CoreRangeSet
    has_buffer: bool = False  # True if backed by an L1 Buffer allocation
    unpack_to_dest_mode: Any = None  # UnpackToDestMode enum (Default or UnpackToDestFp32)
    tile: Any = None  # Optional TileDescriptor from CBFormatDescriptor
    alias_group: int = -1  # Sequential ID per CBDescriptor in cbs list
    address_offset: int = 0  # From CBDescriptor.address_offset
    source_fmt: Any = None  # Reference to original CBFormatDescriptor
    source_cb: Any = None  # Reference to original CBDescriptor

    @property
    def pool_key(self) -> "CBPoolKey":
        """Compatibility key for CB pool allocation."""
        return CBPoolKey(
            data_format=self.data_format,
            page_size=self.page_size,
            has_buffer=self.has_buffer,
            unpack_to_dest_mode=self.unpack_to_dest_mode,
        )


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
    total_size: int  # Max total_size across all phases sharing this slot
    source_cb: Any = None  # Reference to the original CBDescriptor (set on first alloc, not updated on reuse)
    source_fmt: Any = None  # Reference to the original CBFormatDescriptor


def num_cbs_for_device(device) -> int:
    """Return the number of circular buffer slots for the given device.

    Wormhole has 32 CB slots; Blackhole has 64.
    """
    if device is not None and hasattr(device, "arch") and callable(device.arch):
        if device.arch() == ttnn.Arch.WORMHOLE_B0:
            return 32
        return 64
    # Default to 32 (Wormhole) when device is unknown
    return 32


class CBPoolAllocator:
    """Pool-allocates CB hardware slots based on compatibility keys.

    CBs from different phases that share the same (data_format, page_size,
    unpack_to_dest_mode) configuration are assigned to the same slot.
    Different configs get separate slots.

    The allocator itself imposes no slot limit.  The hardware limit
    (32 for Wormhole, 64 for Blackhole) is enforced externally when
    projecting to a per-group pool via ``project_to_group()``, or by
    the caller when using the pool directly for single-group builds.
    """

    def __init__(self, max_slots: int = 32):
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
        # Alias groups: tracks which pool slots form an alias group (shared L1).
        # Aliased CBs (multiple format_descriptors on one CBDescriptor) must
        # either reuse an existing complete alias group or get fresh slots —
        # reusing non-aliased slots would corrupt phases that use them independently.
        self._slot_alias_groups: Dict[int, frozenset] = {}  # slot_index -> group members
        self._unique_alias_groups: Set[frozenset] = set()  # deduped set of alias groups

    def reserve_index(self, index: int) -> None:
        """Reserve a slot index without creating a pool slot or remap entry.

        Used for GlobalCB remote indices which occupy a hardware slot but must
        not be pool-allocated, remapped, or included in per-phase CB reset.
        """
        self._allocated_indices.add(index)

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

        Aliased CBs (multiple format_descriptors sharing one CBDescriptor)
        either reuse an existing pool alias group wholesale or get fresh
        slots.  Reusing non-aliased slots would corrupt phases that use
        them independently (the merged CBDescriptor would force aliasing).

        Args:
            phase_idx: Phase index.
            cb_info: Dict mapping original CB index -> CBInfo for this phase.
            phantom_cb_indices: CB indices referenced in named compile-time args
                but without a corresponding CBDescriptor.  These get identity-mapped
                reservations to prevent collisions.
        """
        remap: Dict[int, int] = {}
        slots_used_this_phase: Set[int] = set()

        # Reserve phantom CB indices first (identity mapping).
        # Do NOT add to slots_used_this_phase — phantom CBs (e.g., cb_bias
        # when bias is absent) are never accessed at runtime, so real CBs
        # can safely share their slot index.
        for phantom_idx in phantom_cb_indices:
            if phantom_idx not in self._allocated_indices:
                self._allocated_indices.add(phantom_idx)
            remap[phantom_idx] = phantom_idx

        # Identify alias groups in this phase (alias_group_id -> list of orig indices).
        phase_alias_groups: Dict[int, List[int]] = defaultdict(list)
        for orig_idx, info in cb_info.items():
            phase_alias_groups[info.alias_group].append(orig_idx)

        # Split CBs into aliased (multi-member group) and non-aliased (single).
        aliased_indices: Set[int] = set()
        for members in phase_alias_groups.values():
            if len(members) > 1:
                aliased_indices.update(members)

        # --- Non-aliased CBs: normal two-pass allocation (identity first) ---
        non_aliased_info = {k: v for k, v in cb_info.items() if k not in aliased_indices}
        identity_cbs, remaining_cbs = self._partition_by_identity(non_aliased_info)

        for orig_idx, info, key in identity_cbs + remaining_cbs:
            slot_idx = self._find_reusable_slot(key, orig_idx, slots_used_this_phase)
            if slot_idx is not None:
                self._reuse_slot(slot_idx, info, phase_idx)
            else:
                slot_idx = self._allocate_new_slot(key, info, orig_idx, phase_idx)
            slots_used_this_phase.add(slot_idx)
            remap[orig_idx] = slot_idx

        # --- Aliased CBs: reuse existing alias group or allocate fresh ---
        for alias_id, members in phase_alias_groups.items():
            if len(members) <= 1:
                continue
            members_sorted = sorted(members)
            reused = self._try_reuse_alias_group(members_sorted, cb_info, slots_used_this_phase, phase_idx)
            if reused is not None:
                for orig_idx, slot_idx in reused:
                    slots_used_this_phase.add(slot_idx)
                    remap[orig_idx] = slot_idx
            else:
                # Allocate fresh slots for all members
                for orig_idx in members_sorted:
                    info = cb_info[orig_idx]
                    slot_idx = self._allocate_new_slot(info.pool_key, info, orig_idx, phase_idx)
                    slots_used_this_phase.add(slot_idx)
                    remap[orig_idx] = slot_idx
                # Record the new alias group
                new_group = frozenset(remap[m] for m in members_sorted)
                for slot_idx in new_group:
                    self._slot_alias_groups[slot_idx] = new_group
                self._unique_alias_groups.add(new_group)

        self.phase_remaps.append(remap)

    def _try_reuse_alias_group(
        self,
        members: List[int],
        cb_info: Dict[int, CBInfo],
        slots_used_this_phase: Set[int],
        phase_idx: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """Try to reuse an existing pool alias group for aliased CBs.

        Returns list of (orig_idx, slot_idx) pairs if successful, None otherwise.
        An existing alias group can be reused when:
        - It has the same number of members
        - Each member slot is compatible (same CBPoolKey) with a current CB
        - No member slot is already used this phase
        """
        for group in self._unique_alias_groups:
            if len(group) != len(members):
                continue
            # Check no slot in this group is used this phase
            if group & slots_used_this_phase:
                continue
            # Try to match each member to a group slot
            group_slots = sorted(group)
            matched = self._match_alias_members(members, group_slots, cb_info)
            if matched is not None:
                # Reuse: update total_size for each slot
                for orig_idx, slot_idx in matched:
                    self._reuse_slot(slot_idx, cb_info[orig_idx], phase_idx)
                return matched

        return None

    def _match_alias_members(
        self,
        members: List[int],
        group_slots: List[int],
        cb_info: Dict[int, CBInfo],
    ) -> Optional[List[Tuple[int, int]]]:
        """Match aliased CB members to existing group slots by compatible CBPoolKey.

        Tries all permutations of group_slots so that aliased CBs whose keys
        appear in a different order than the existing group still match
        (N=2 → 2 perms, N=3 → 6 perms, trivial cost).

        Returns list of (orig_idx, slot_idx) pairs or None if no valid matching.
        """
        member_keys = [(orig_idx, cb_info[orig_idx].pool_key) for orig_idx in members]

        for perm in itertools.permutations(group_slots):
            result = []
            valid = True
            for (orig_idx, key), slot_idx in zip(member_keys, perm):
                slot = self._slots.get(slot_idx)
                if slot is None or slot.config != key:
                    valid = False
                    break
                result.append((orig_idx, slot_idx))
            if valid:
                return result
        return None

    def _partition_by_identity(
        self, cb_info: Dict[int, CBInfo]
    ) -> Tuple[List[Tuple[int, CBInfo, CBPoolKey]], List[Tuple[int, CBInfo, CBPoolKey]]]:
        """Split CBs into those with an existing identity-matching slot and the rest."""
        identity_cbs = []
        remaining_cbs = []
        for orig_idx, info in sorted(cb_info.items()):
            key = info.pool_key
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
        """Reuse an existing slot, updating total_size if needed.

        source_cb/source_fmt are NOT updated — the first allocating phase's
        references are kept stable for build_merged_cb_descriptors.
        """
        slot = self._slots[slot_idx]
        slot.total_size = max(slot.total_size, info.total_size)

    def _allocate_new_slot(self, key: CBPoolKey, info: CBInfo, orig_idx: int, phase_idx: int) -> int:
        """Allocate a fresh slot.

        Prefers identity mapping (orig_idx -> orig_idx) when the slot is free,
        so that CBs keep their original hardware indices.
        """
        if orig_idx not in self._allocated_indices:
            slot_idx = orig_idx
        else:
            slot_idx = self._alloc_index()
        self._allocated_indices.add(slot_idx)
        self._slots[slot_idx] = CBSlot(
            slot_index=slot_idx,
            config=key,
            total_size=info.total_size,
            source_cb=info.source_cb,
            source_fmt=info.source_fmt,
        )
        self._slot_to_orig_index[slot_idx] = orig_idx
        if key not in self._config_to_slots:
            self._config_to_slots[key] = []
        self._config_to_slots[key].append(slot_idx)
        return slot_idx

    def get_remap(self, phase_idx: int) -> Dict[int, int]:
        """Return {orig_cb_idx: slot_idx} for a phase."""
        return self.phase_remaps[phase_idx]

    def _compute_slot_alias_groups(
        self,
        phases: List["PhaseInfo"],
    ) -> Dict[int, int]:
        """Compute which pool slots should share an L1 allocation.

        Returns Dict[slot_index, alias_group_id].  Slots with the same
        group_id came from the same multi-format CBDescriptor in at least
        one phase and will be emitted as a single merged CBDescriptor.
        """
        slot_groups: Dict[int, int] = {}
        next_group = 0

        for phase_idx, phase in enumerate(phases):
            remap = self.phase_remaps[phase_idx]
            by_alias: Dict[int, List[int]] = defaultdict(list)
            for orig_idx, info in phase.cb_info.items():
                if orig_idx in remap:
                    by_alias[info.alias_group].append(remap[orig_idx])

            for alias_id, slot_indices in by_alias.items():
                if len(slot_indices) <= 1:
                    continue
                existing = {slot_groups[s] for s in slot_indices if s in slot_groups}
                if existing:
                    group_id = min(existing)
                else:
                    group_id = next_group
                    next_group += 1
                for s in slot_indices:
                    slot_groups[s] = group_id

        for slot_idx in self._slots:
            if slot_idx not in slot_groups:
                slot_groups[slot_idx] = next_group
                next_group += 1

        return slot_groups

    def build_merged_cb_descriptors(
        self,
        phases: List["PhaseInfo"],
    ) -> list:
        """Build merged CB descriptors from the pool.

        Uses alias groups to determine which pool slots share an L1 allocation
        (e.g. matmul's c_4/c_5 aliased CBDescriptor).  Constructs NEW
        CBDescriptor objects per alias group — never emits originals — so that
        multi-phase slot sharing cannot produce duplicate buffer_index conflicts.

        MUTATION CONTRACT:
        This method mutates ``source_fmt.buffer_index`` on the original
        CBFormatDescriptor objects (line: ``slot.source_fmt.buffer_index =
        slot_idx``).  Callers MUST bracket the build with save/restore:

            graph.py _build_groups() → _save_cb_state()
                → ... → build_merged_cb_descriptors()
                → ... → _restore_cb_state()

        The C++ CBFormatDescriptor bindings do not support deepcopy, so
        in-place mutation + restore is the only viable pattern.
        """
        slot_alias = self._compute_slot_alias_groups(phases)

        # Group slots by alias group
        groups: Dict[int, List[int]] = defaultdict(list)
        for slot_idx in sorted(self._slots.keys()):
            groups[slot_alias[slot_idx]].append(slot_idx)

        merged = []
        seen_ids: Set[int] = set()  # Track GlobalCB pass-through dedup

        for group_id in sorted(groups.keys()):
            slot_indices = groups[group_id]
            slot_set = set(slot_indices)
            max_total = 0
            fmts = []

            for slot_idx in slot_indices:
                slot = self._slots[slot_idx]
                max_total = max(max_total, slot.total_size)
                if slot.source_fmt is not None:
                    # MUTATION: overwrites original buffer_index; caller's
                    # _save_cb_state/_restore_cb_state bracket reverts this.
                    slot.source_fmt.buffer_index = slot_idx
                    fmts.append(slot.source_fmt)

            if not fmts:
                raise ValueError(
                    f"Alias group {group_id} has slots {slot_indices} but no "
                    f"format descriptors — every allocated CBSlot must have a "
                    f"source_fmt (set by _allocate_new_slot from extract_cb_info)"
                )

            # Find buffer from the EARLIEST phase that has a buffer-backed CB
            # mapping to any slot in this group.  This matches the rebind logic
            # which computes address diffs relative to phase 0's baseline.
            buffer_source_cb = None
            for phase_idx, phase in enumerate(phases):
                remap = self.phase_remaps[phase_idx]
                for orig_idx, info in sorted(phase.cb_info.items()):
                    if info.has_buffer and remap.get(orig_idx) in slot_set:
                        buffer_source_cb = info.source_cb
                        break
                if buffer_source_cb is not None:
                    break

            # Use first slot's source_cb for core_ranges / address_offset
            rep_cb = self._slots[slot_indices[0]].source_cb

            new_cb = ttnn.CBDescriptor()
            new_cb.total_size = max_total
            new_cb.core_ranges = rep_cb.core_ranges if rep_cb else fmts[0].core_ranges
            new_cb.format_descriptors = fmts
            if rep_cb is not None:
                new_cb.address_offset = rep_cb.address_offset
            if buffer_source_cb is not None:
                new_cb.set_buffer_from_cb(buffer_source_cb)

            merged.append(new_cb)

        # Pass through GlobalCB-backed CBDescriptors not covered by pool slots
        for phase in phases:
            for cb_desc in phase.op_descriptor.descriptor.cbs:
                if cb_desc.has_global_circular_buffer():
                    cid = id(cb_desc)
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        merged.append(cb_desc)

        return merged

    def build_unpack_to_dest_mode(self) -> list:
        """Build merged unpack_to_dest_mode vector indexed by slot index.

        Returns a list of exactly MAX_SLOTS entries (matching the device's
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

    def project_to_group(
        self,
        group_global_indices: List[int],
        padding_slots: Set[int],
    ) -> "CBPoolAllocator":
        """Project this global pool to a per-group pool for a core group.

        Takes a list of global phase indices (one per group-local phase) and
        a set of padding slot indices (for L1 alignment), and returns a new
        ``CBPoolAllocator`` with:

        - ``phase_remaps`` re-indexed from the global pool's remaps
        - ``_slots`` containing referenced slots PLUS padding slots
        - ``_config_to_slots``, ``_slot_to_orig_index``, ``_allocated_indices``
          copied for all included slots
        - ``_slot_alias_groups`` / ``_unique_alias_groups`` filtered to included

        Validates that the projected pool has <= 32 slots.

        Args:
            group_global_indices: Index into this pool's ``phase_remaps`` for
                each phase the group executes.  E.g., if the group runs
                global ops [0, 2, 3], pass ``[0, 2, 3]``.
            padding_slots: Extra slot indices to include for L1 alignment
                (ensures identical CB L1 layout across groups sharing slots).

        Returns:
            A new ``CBPoolAllocator`` whose ``phase_remaps`` are local (0..K)
            and whose ``_slots`` are the union of referenced + padding slots.

        Raises:
            ValueError: If the projected pool has > 32 slots.
        """
        projected = CBPoolAllocator(max_slots=self.max_slots)

        # Collect phase_remaps for this group (re-indexed to local 0..K)
        for global_idx in group_global_indices:
            projected.phase_remaps.append(dict(self.phase_remaps[global_idx]))

        # Determine which slots are referenced by this group
        referenced_slots: Set[int] = set()
        for remap in projected.phase_remaps:
            referenced_slots.update(remap.values())

        # Include padding slots
        included_slots = referenced_slots | padding_slots

        # Copy slot data for all included slots
        for slot_idx in included_slots:
            if slot_idx in self._slots:
                projected._slots[slot_idx] = self._slots[slot_idx]
                projected._allocated_indices.add(slot_idx)
                if slot_idx in self._slot_to_orig_index:
                    projected._slot_to_orig_index[slot_idx] = self._slot_to_orig_index[slot_idx]

        # Rebuild _config_to_slots for included slots
        for slot_idx, slot in projected._slots.items():
            key = slot.config
            if key not in projected._config_to_slots:
                projected._config_to_slots[key] = []
            projected._config_to_slots[key].append(slot_idx)

        # Copy reserved indices (e.g. remote CB indices)
        for idx in self._allocated_indices:
            if idx not in projected._allocated_indices:
                # Only copy reserved-but-no-slot indices if they would
                # conflict with included slots
                projected._allocated_indices.add(idx)

        # Filter alias groups to included slots
        for slot_idx, group in self._slot_alias_groups.items():
            if slot_idx in included_slots:
                projected._slot_alias_groups[slot_idx] = group
        for group in self._unique_alias_groups:
            if group & included_slots:
                projected._unique_alias_groups.add(group)

        # Validate hardware slot limit
        if len(projected._slots) > self.max_slots:
            breakdown = [
                f"  slot {si}: fmt={sl.config.data_format}, "
                f"page_size={sl.config.page_size}, "
                f"unpack={sl.config.unpack_to_dest_mode}"
                for si, sl in sorted(projected._slots.items())
            ]
            raise ValueError(
                f"CB pool overflow: group projection needs {len(projected._slots)} "
                f"slots but device limit is {self.max_slots}.\n"
                f"Allocated slots:\n" + "\n".join(breakdown)
            )

        return projected


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
    for cb_group_id, cb_desc in enumerate(descriptor.cbs):
        for fmt_desc in cb_desc.format_descriptors:
            cb_idx = fmt_desc.buffer_index
            data_format = fmt_desc.data_format_as_uint8  # int (raw tt::DataFormat uint8)
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
            # Extract optional tile descriptor
            tile = None
            try:
                tile = fmt_desc.tile
            except (TypeError, AttributeError):
                pass
            cb_info[cb_idx] = CBInfo(
                original_index=cb_idx,
                total_size=cb_desc.total_size,
                data_format=data_format,
                page_size=fmt_desc.page_size,
                core_ranges=cb_desc.core_ranges,
                has_buffer=cb_desc.has_buffer(),
                unpack_to_dest_mode=utd_mode,
                tile=tile,
                alias_group=cb_group_id,
                address_offset=cb_desc.address_offset,
                source_fmt=fmt_desc,
                source_cb=cb_desc,
            )
    return cb_info


def _extract_remote_cb_indices(descriptor: "ttnn.ProgramDescriptor") -> Set[int]:
    """Get buffer indices from remote_format_descriptors of GlobalCB-backed CBs.

    These indices occupy hardware CB slots but are managed by the
    GlobalCircularBuffer (L1-based tracking, no stream registers).
    They must be reserved in the pool to prevent collisions but must NOT
    be pool-allocated, remapped, or included in inter-phase CB reset.
    """
    indices: Set[int] = set()
    for cb_desc in descriptor.cbs:
        if cb_desc.has_global_circular_buffer():
            for fmt_desc in cb_desc.remote_format_descriptors:
                indices.add(fmt_desc.buffer_index)
    return indices


# Convention: CB-reference named compile-time args MUST start with this prefix
# and have a value in range [0, 31]. Non-CB args MUST NOT use this prefix.
CB_ARG_PREFIX = "cb_"


def _is_cb_named_arg(name: str, value: Any) -> bool:
    """Check if a named compile-time arg refers to a CB index.

    Returns True if the name starts with CB_ARG_PREFIX and the value
    is a non-negative integer.  The actual slot bound is validated by
    the pool allocator, not here.
    """
    if not name.startswith(CB_ARG_PREFIX):
        return False
    if not isinstance(value, int) or value < 0:
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


def _get_phantom_cb_indices(phase: PhaseInfo) -> Set[int]:
    """Get CB indices referenced in named compile-time args but without CBDescriptors.

    These "phantom" CBs need identity-mapped reservations in the pool to prevent
    real CBs from being allocated at conflicting indices.

    CONTRACT:
    - Phantom CBs are identity-mapped (orig_idx == slot_idx) in the remap.
    - They are NOT added to slots_used_this_phase, so real CBs CAN share
      their slot index in a later phase (the phantom's code path is dead
      at runtime — e.g., cb_bias when bias is absent).
    - They are NOT in self._slots, so they are excluded from:
      - per-phase CB reset arrays (emitted by _emit_phase_cb_arrays)
      - build_merged_cb_descriptors (no CBSlot → no merged CB)
      - build_unpack_to_dest_mode (no CBSlot → Default mode)
    - Safe because the kernel code path referencing the phantom CB index
      is unreachable at runtime (guarded by a compile-time or runtime flag).
    """
    real_cb_indices = set(phase.cb_info.keys())

    phantom = set()
    for kernel_desc in phase.op_descriptor.descriptor.kernels:
        for name, value in kernel_desc.named_compile_time_args:
            if _is_cb_named_arg(name, value) and value not in real_cb_indices:
                phantom.add(value)

    if phantom:
        logger.debug(
            "Phase %d: phantom CB indices %s (referenced in named args but no CBDescriptor)",
            phase.phase_idx,
            sorted(phantom),
        )

    return phantom


def _compute_rebind_info(
    phases: List[PhaseInfo],
    phase_remaps: List[Dict[int, int]],
) -> Dict[int, List[Tuple[int, int, int]]]:
    """Compute which CB slots need address rebinding at each phase transition.

    For each phase 1+, identifies remapped slot indices where the buffer address
    differs from what was set in the previous phase.
    """
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
    "_get_phantom_cb_indices",
    "_compute_rebind_info",
    "_extract_remote_cb_indices",
]
