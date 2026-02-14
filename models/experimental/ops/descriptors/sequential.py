# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sequential Kernel Chaining Infrastructure

Fuses multiple operations to run sequentially on the SAME cores within a
single program.  All readers/writers run for every phase.  Data flows through
DRAM between phases (Writer_N -> DRAM -> Reader_{N+1}).

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

Usage:
    >>> builder = SequentialChainBuilder()
    >>> builder.add_phase(op0_desc)
    >>> builder.add_phase(op1_desc)
    >>> fused = builder.build(device)
    >>> outputs = composite.launch([fused])
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set, Any
import re
import os

import ttnn
from ttnn._ttnn.program_descriptor import UnpackToDestMode

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors import cpp_parser


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
class BranchSpec:
    """A branch in an OpGraph tree.

    Represents a subset of cores that diverge from the stem to run their
    own phases.  Branches can nest via the ``children`` field.
    """

    core_range: Any  # CoreRangeSet for this branch's cores
    phases: List[OpDescriptor] = field(default_factory=list)
    children: List["BranchSpec"] = field(default_factory=list)


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
        # Track which slots are used by THIS phase (to avoid sharing within a phase)
        slots_used_this_phase: Set[int] = set()

        # Reserve phantom CB indices first (identity mapping)
        for phantom_idx in phantom_cb_indices:
            if phantom_idx not in self._allocated_indices:
                self._allocated_indices.add(phantom_idx)
            remap[phantom_idx] = phantom_idx

        # Two-pass allocation: first claim identity-matching slots (CBs that
        # existed in previous phases keep their slot), then allocate remaining
        # CBs from the pool.  This prevents new CBs from stealing slots that
        # belong to cross-phase shared CBs.
        identity_cbs = []  # (orig_idx, info, key) with identity match
        remaining_cbs = []  # (orig_idx, info, key) without identity match
        for orig_idx, info in sorted(cb_info.items()):
            key = CBPoolKey(
                data_format=info.data_format,
                page_size=info.page_size,
                has_buffer=info.has_buffer,
                unpack_to_dest_mode=info.unpack_to_dest_mode,
            )
            # Check if there's an existing slot from the same original index
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

        # Process identity-matching CBs first, then remaining
        for orig_idx, info, key in identity_cbs + remaining_cbs:
            # Look for an existing slot with the same config that's NOT
            # already used by this phase.  Prefer the slot created from
            # the same original CB index (preserves identity mapping).
            reused_slot = None
            if key in self._config_to_slots:
                # First: look for a slot from the same original index
                for candidate_idx in self._config_to_slots[key]:
                    if candidate_idx not in slots_used_this_phase:
                        if self._slot_to_orig_index.get(candidate_idx) == orig_idx:
                            reused_slot = candidate_idx
                            break
                # Second: any compatible slot
                if reused_slot is None:
                    for candidate_idx in self._config_to_slots[key]:
                        if candidate_idx not in slots_used_this_phase:
                            reused_slot = candidate_idx
                            break

            if reused_slot is not None:
                # Reuse existing slot from a different phase
                slot_idx = reused_slot
                slot = self._slots[slot_idx]
                # Update total_size to max
                if info.total_size > slot.total_size:
                    slot.total_size = info.total_size
                    # Keep phase 0's descriptor for correct initial buffer setup
                    if slot.source_phase != 0:
                        slot.cb_descriptor = self._get_cb_descriptor(info, phase_idx)
                        slot.source_phase = phase_idx
            else:
                # Allocate new slot
                slot_idx = self._alloc_index()
                if len(self._slots) + 1 > self.max_slots:
                    breakdown = []
                    for si, sl in sorted(self._slots.items()):
                        breakdown.append(
                            f"  slot {si}: fmt={sl.config.data_format}, "
                            f"page_size={sl.config.page_size}, "
                            f"unpack={sl.config.unpack_to_dest_mode}"
                        )
                    raise ValueError(
                        f"CB pool overflow: need {len(self._slots) + 1} slots but "
                        f"device limit is {self.max_slots}.\n"
                        f"Allocated slots:\n" + "\n".join(breakdown)
                    )
                self._allocated_indices.add(slot_idx)
                desc = self._get_cb_descriptor(info, phase_idx)
                self._slots[slot_idx] = CBSlot(
                    slot_index=slot_idx,
                    config=key,
                    cb_descriptor=desc,
                    total_size=info.total_size,
                    source_phase=phase_idx,
                )
                self._slot_to_orig_index[slot_idx] = orig_idx
                if key not in self._config_to_slots:
                    self._config_to_slots[key] = []
                self._config_to_slots[key].append(slot_idx)

            slots_used_this_phase.add(slot_idx)
            remap[orig_idx] = slot_idx

        self.phase_remaps.append(remap)

    @staticmethod
    def _get_cb_descriptor(info: CBInfo, phase_idx: int) -> Any:
        """Get the CBDescriptor from a CBInfo's source descriptor.

        We can't deepcopy CBDescriptors, so we store a reference.
        The CBDescriptor is looked up later from the phase's ProgramDescriptor.
        """
        # Return a lightweight reference to reconstruct later
        return {"phase_idx": phase_idx, "cb_idx": info.original_index, "info": info}

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
        """
        merged = []
        for slot_idx in sorted(self._slots.keys()):
            slot = self._slots[slot_idx]
            ref = slot.cb_descriptor
            phase = phases[ref["phase_idx"]]
            orig_idx = ref["cb_idx"]

            # Find the actual CBDescriptor from the phase's ProgramDescriptor
            best_desc = None
            for cb_desc in phase.op_descriptor.descriptor.cbs:
                for fmt_desc in cb_desc.format_descriptors:
                    if fmt_desc.buffer_index == orig_idx:
                        best_desc = cb_desc
                        break
                if best_desc is not None:
                    break

            if best_desc is not None:
                # Update the format descriptor's buffer_index to the slot index
                # We need to modify in-place since we can't deepcopy
                for fmt_desc in best_desc.format_descriptors:
                    if fmt_desc.buffer_index == orig_idx:
                        fmt_desc.buffer_index = slot_idx
                        break
                # Update total_size to the max across phases
                best_desc.total_size = slot.total_size
                merged.append(best_desc)

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


def extract_cb_names_from_kernel(kernel_desc: "ttnn.KernelDescriptor") -> Dict[str, int]:
    """Extract CB name -> index mapping from kernel's named compile-time args."""
    cb_names = {}
    if hasattr(kernel_desc, "named_compile_time_args"):
        for name, value in kernel_desc.named_compile_time_args:
            if name.startswith("cb_"):
                cb_names[name] = value
    return cb_names


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


def _collect_all_pre_main_code(sources_with_indices: List[Tuple[int, str]]) -> Tuple[str, Dict[int, List[str]]]:
    """Merge pre-main code from all phases with per-phase name isolation.

    Uses tree-sitter (via cpp_parser) to categorize top-level blocks
    before kernel_main into *shared* or *phase-specific*:

    **Shared** (deduped across phases):
      - Namespace blocks: dedup by signature (first occurrence wins).
      - Single-line declarations (namespace aliases, ``using``,
        ``typedef``): dedup by exact normalized content.

    **Phase-specific** (prefixed with ``phaseN_``):
      - Free function definitions (``ALWI``, ``FORCE_INLINE``, etc.):
        each phase gets its own copy with the function name prefixed.
      - Global/static variables: each phase gets its own copy with
        the variable name prefixed.

    ALL phases (including phase 0) get prefixed.  This eliminates:
      - Silent first-wins drops when two phases define the same function
        with different bodies.
      - Redefinition errors when an included header also defines a
        function that appears inline in another phase's pre-main.

    Returns:
        A tuple of (pre_main_code, phase_names) where phase_names is a
        dict mapping phase_idx -> list of original names that were prefixed.
        Callers must apply the same prefixing to each phase's kernel body.
    """
    if not sources_with_indices:
        return "", {}

    all_blocks: List[str] = []
    seen_signatures: Set[str] = set()
    seen_content: Set[str] = set()
    phase_names: Dict[int, List[str]] = {}

    for phase_idx, source in sources_with_indices:
        blocks = cpp_parser.categorize_pre_main(source)

        for block in blocks:
            normalized = cpp_parser.normalize_block(block.text)
            if not normalized:
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
                if prefixed_norm not in seen_content:
                    seen_content.add(prefixed_norm)
                    all_blocks.append(prefixed)
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
                if prefixed_norm not in seen_content:
                    seen_content.add(prefixed_norm)
                    all_blocks.append(prefixed)
                continue

            # --- Shared: namespace blocks ---
            if block.kind == "namespace":
                sig = block.text.split("{")[0].strip() if "{" in block.text else normalized
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    all_blocks.append(block.text)
            else:
                # --- Shared: namespace_alias, using, struct, template, other ---
                if normalized not in seen_content:
                    seen_content.add(normalized)
                    all_blocks.append(block.text)

    return "\n\n".join(all_blocks), phase_names


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


def _offset_runtime_args_in_source(source: str, phase_idx: int) -> str:
    """Replace get_arg_val<T>(N) with offset version for phase N>0."""
    if phase_idx == 0:
        return source

    offset_name = f"__phase{phase_idx}_rt_offset"
    offset_decl = (
        f'    constexpr uint32_t {offset_name} = get_named_compile_time_arg_val("phase{phase_idx}_rt_arg_offset");\n'
    )

    def replace_rt_arg(match):
        type_name = match.group(1)
        arg_idx = match.group(2)
        return f"get_arg_val<{type_name}>({offset_name} + {arg_idx})"

    source = re.sub(
        r"get_arg_val<(\w+)>\((\d+)\)",
        replace_rt_arg,
        source,
    )

    # Handle incrementing variable pattern: uint32_t rt_args_idx = 0;
    # Requires "arg" or "rt" in the variable name to avoid false positives
    # on unrelated uint32_t initializations.
    source = re.sub(
        r"(uint32_t\s+\w*(?:arg|rt)\w*\s*=\s*)0(\s*;)",
        rf"\g<1>{offset_name}\2",
        source,
    )

    return offset_decl + source


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
    source = _offset_runtime_args_in_source(source, phase_idx)
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
            if name.startswith("cb_") and isinstance(value, int) and value not in real_cb_indices:
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
    barrier_config: Optional[BarrierConfig] = None,
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    op_semaphore_ids: Optional[List[int]] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
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
        phase_defs = {name for name, _ in kernel.defines} if hasattr(kernel, "defines") else set()
        resolved = cpp_parser.resolve_ifdef_directives(source, phase_defs)
        reader_sources.append((i, resolved))

    if not reader_sources:
        return None

    all_sources = [s for _, s in reader_sources]
    includes = cpp_parser.collect_includes(all_sources)
    defines = cpp_parser.collect_defines(all_sources)
    # Merge pre-main code from all phases with per-phase name isolation.
    pre_main, phase_names = _collect_all_pre_main_code(reader_sources)

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused reader kernel - {len(reader_sources)} phases",
        "",
    ]
    lines.extend(defines)
    lines.append("")
    lines.extend(includes)
    lines.append("")
    if pre_main.strip():
        lines.append(pre_main)
        lines.append("")

    is_multi_phase = len(reader_sources) > 1

    use_multi_barrier = multi_barrier is not None

    if is_multi_phase:
        # Barrier named compile-time args
        lines.append('constexpr uint32_t __barrier_rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')

        if use_multi_barrier:
            # Per-segment compile-time constants
            for seg_idx in range(len(multi_barrier.segments)):
                s = f"seg{seg_idx}"
                lines.append(f'constexpr uint32_t __{s}_num_cores = get_named_compile_time_arg_val("{s}_num_cores");')
                lines.append(
                    f'constexpr uint32_t __{s}_core0_phys_x = get_named_compile_time_arg_val("{s}_core0_phys_x");'
                )
                lines.append(
                    f'constexpr uint32_t __{s}_core0_phys_y = get_named_compile_time_arg_val("{s}_core0_phys_y");'
                )
                lines.append(
                    f'constexpr uint32_t __{s}_mcast_start_x = get_named_compile_time_arg_val("{s}_mcast_start_x");'
                )
                lines.append(
                    f'constexpr uint32_t __{s}_mcast_start_y = get_named_compile_time_arg_val("{s}_mcast_start_y");'
                )
                lines.append(
                    f'constexpr uint32_t __{s}_mcast_end_x = get_named_compile_time_arg_val("{s}_mcast_end_x");'
                )
                lines.append(
                    f'constexpr uint32_t __{s}_mcast_end_y = get_named_compile_time_arg_val("{s}_mcast_end_y");'
                )
            lines.append("")
        else:
            lines.append(
                'constexpr uint32_t __num_barrier_cores = get_named_compile_time_arg_val("num_barrier_cores");'
            )
            lines.append("")

            # Global barrier compile-time args (only used if num_cores > 1)
            lines.append('constexpr uint32_t __core0_phys_x = get_named_compile_time_arg_val("core0_phys_x");')
            lines.append('constexpr uint32_t __core0_phys_y = get_named_compile_time_arg_val("core0_phys_y");')
            lines.append('constexpr uint32_t __mcast_start_x = get_named_compile_time_arg_val("mcast_start_x");')
            lines.append('constexpr uint32_t __mcast_start_y = get_named_compile_time_arg_val("mcast_start_y");')
            lines.append('constexpr uint32_t __mcast_end_x = get_named_compile_time_arg_val("mcast_end_x");')
            lines.append('constexpr uint32_t __mcast_end_y = get_named_compile_time_arg_val("mcast_end_y");')
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

        if use_multi_barrier:
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
                lines.append(
                    f"        bool is_core_0 = (my_x[0] == __{s}_core0_phys_x && my_y[0] == __{s}_core0_phys_y);"
                )
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
        else:
            # Global barrier helper (also serves as phase release for compute/writer)
            lines.append("// Global barrier across cores. Sets global_release which compute/writer spin on.")
            lines.append(
                "FORCE_INLINE void __global_barrier(uint32_t phase, volatile tt_l1_ptr uint32_t* global_arrive, volatile tt_l1_ptr uint32_t* global_release) {"
            )
            lines.append("    if constexpr (__num_barrier_cores > 1) {")
            lines.append("        // Arrive: all cores send atomic inc to core 0's arrive semaphore")
            lines.append(
                "        uint64_t core0_arrive_noc_addr = get_noc_addr(__core0_phys_x, __core0_phys_y, (uint32_t)global_arrive);"
            )
            lines.append("        noc_semaphore_inc(core0_arrive_noc_addr, 1);")
            lines.append("")
            lines.append("        bool is_core_0 = (my_x[0] == __core0_phys_x && my_y[0] == __core0_phys_y);")
            lines.append("        if (is_core_0) {")
            lines.append("            // Core 0: wait for all cores to arrive")
            lines.append("            noc_semaphore_wait_min(global_arrive, __num_barrier_cores * (phase + 1));")
            lines.append("            // Multicast release to all cores (including self via loopback)")
            lines.append("            *global_release = phase + 1;")
            lines.append(
                "            uint64_t mcast_addr = get_noc_multicast_addr(__mcast_start_x, __mcast_start_y, __mcast_end_x, __mcast_end_y, (uint32_t)global_release);"
            )
            lines.append(
                "            noc_semaphore_set_multicast_loopback_src((uint32_t)global_release, mcast_addr, __num_barrier_cores);"
            )
            lines.append("            noc_async_write_barrier();")
            lines.append("        } else {")
            lines.append("            // Other cores: wait for release from core 0")
            lines.append("            noc_semaphore_wait_min(global_release, phase + 1);")
            lines.append("        }")
            lines.append("    } else {")
            lines.append("        // Single core: set release directly (no NOC ops needed)")
            lines.append("        *global_release = phase + 1;")
            lines.append("    }")
            lines.append("}")
            lines.append("")

    # Generate phase functions
    for phase_idx, resolved_source in reader_sources:
        body = cpp_parser.extract_kernel_body(resolved_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        pnames = phase_names.get(phase_idx, [])
        transformed = _transform_phase_source(body, phase_idx, offset, phase_names=pnames)

        lines.append(f"// Phase {phase_idx} reader")
        lines.append(f"FORCE_INLINE void phase{phase_idx}_reader() {{")
        for line in transformed.split("\n"):
            lines.append(f"    {line}")
        lines.append("}")
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
        if use_multi_barrier:
            # Per-segment arrive/release pointers
            for seg_idx in range(len(multi_barrier.segments)):
                s = f"seg{seg_idx}"
                off = 2 + seg_idx * 2
                lines.append(
                    f"    const uint32_t __{s}_arrive_addr = get_arg_val<uint32_t>(__barrier_rt_offset + {off});"
                )
                lines.append(
                    f"    const uint32_t __{s}_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + {off + 1});"
                )
                lines.append(
                    f"    volatile tt_l1_ptr uint32_t* __{s}_arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__{s}_arrive_addr);"
                )
                lines.append(
                    f"    volatile tt_l1_ptr uint32_t* __{s}_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__{s}_release_addr);"
                )
        else:
            lines.append("    const uint32_t __global_arrive_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 2);")
            lines.append("    const uint32_t __global_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 3);")
            lines.append(
                "    volatile tt_l1_ptr uint32_t* __global_arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__global_arrive_addr);"
            )
            lines.append(
                "    volatile tt_l1_ptr uint32_t* __global_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__global_release_addr);"
            )
        lines.append("")

    if rebind_info is None:
        rebind_info = {}

    first = True
    for phase_idx, _ in reader_sources:
        if not first and is_multi_phase:
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
            # Reset op semaphores to initial value (0) so next phase starts clean
            if op_semaphore_ids:
                lines.append("    // Reset op semaphores to 0 (as if each phase runs standalone)")
                for sem_id in op_semaphore_ids:
                    lines.append(f"    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore({sem_id})) = 0;")
                lines.append("")
            # Rebind CB addresses before global barrier (so BRISC has correct state)
            rebind_lines = _generate_rebind_code(rebind_info.get(phase_idx, []), phase_idx, for_compute=False)
            if rebind_lines:
                lines.extend(rebind_lines)
                lines.append("")
            if use_multi_barrier:
                # Transition index = phase_idx - 1 (transition after the preceding phase)
                transition_idx = phase_idx - 1
                seg_idx, call_idx = multi_barrier.transition_map[transition_idx]
                s = f"seg{seg_idx}"
                lines.append(f"    // Global barrier segment {seg_idx}, call {call_idx}")
                lines.append(f"    __barrier_{s}({call_idx}, __{s}_arrive, __{s}_release);")
            else:
                lines.append("    // Global barrier (sets global_release, releasing compute/writer)")
                lines.append(f"    __global_barrier({phase_idx - 1}, __global_arrive, __global_release);")
            lines.append("")
        lines.append(f"    phase{phase_idx}_reader();")
        first = False
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
    barrier_config: Optional[BarrierConfig] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
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
        phase_defs = {name for name, _ in kernel.defines} if hasattr(kernel, "defines") else set()
        resolved = cpp_parser.resolve_ifdef_directives(source, phase_defs)
        writer_sources.append((i, resolved))

    if not writer_sources:
        return None

    all_sources = [s for _, s in writer_sources]
    includes = cpp_parser.collect_includes(all_sources)
    defines = cpp_parser.collect_defines(all_sources)
    pre_main, phase_names = _collect_all_pre_main_code(writer_sources)

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused writer kernel - {len(writer_sources)} phases",
        "",
    ]
    lines.extend(defines)
    lines.append("")
    lines.extend(includes)
    lines.append("")
    if pre_main.strip():
        lines.append(pre_main)
        lines.append("")

    is_multi_phase = len(writer_sources) > 1

    if is_multi_phase:
        lines.append('constexpr uint32_t __barrier_rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
        lines.append("")

    # Generate NCRISC CB state resync function (resets local pointers to CB start).
    if sweep_cb_indices and is_multi_phase:
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
    for phase_idx, resolved_source in writer_sources:
        body = cpp_parser.extract_kernel_body(resolved_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        pnames = phase_names.get(phase_idx, [])
        transformed = _transform_phase_source(body, phase_idx, offset, phase_names=pnames)

        lines.append(f"// Phase {phase_idx} writer")
        lines.append(f"FORCE_INLINE void phase{phase_idx}_writer() {{")
        for line in transformed.split("\n"):
            lines.append(f"    {line}")
        lines.append("}")
        lines.append("")

    # Generate kernel_main
    lines.append("void kernel_main() {")

    use_multi_barrier = multi_barrier is not None

    if is_multi_phase:
        lines.append("    // Read barrier L1 flag addresses from runtime args")
        lines.append("    const uint32_t __writer_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset);")
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__writer_done_addr);"
        )
        if use_multi_barrier:
            # Per-segment release pointers
            for seg_idx in range(len(multi_barrier.segments)):
                s = f"seg{seg_idx}"
                off = 1 + seg_idx
                lines.append(
                    f"    const uint32_t __{s}_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + {off});"
                )
                lines.append(
                    f"    volatile tt_l1_ptr uint32_t* __{s}_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__{s}_release_addr);"
                )
        else:
            lines.append("    const uint32_t __global_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 1);")
            lines.append(
                "    volatile tt_l1_ptr uint32_t* __global_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__global_release_addr);"
            )
        lines.append("")

    if rebind_info is None:
        rebind_info = {}

    num_writers = len(writer_sources)
    for count, (phase_idx, _) in enumerate(writer_sources):
        lines.append(f"    phase{phase_idx}_writer();")
        if count < num_writers - 1 and is_multi_phase:
            next_phase_idx = writer_sources[count + 1][0]
            lines.append("")
            lines.append(f"    // Ensure all async NOC writes from Phase {phase_idx} are complete")
            lines.append("    noc_async_write_barrier();")
            lines.append(f"    // Signal done for Phase {phase_idx}")
            lines.append(f"    *__writer_done = {phase_idx + 1};")
            lines.append("")
            if use_multi_barrier:
                transition_idx = phase_idx
                seg_idx, call_idx = multi_barrier.transition_map[transition_idx]
                s = f"seg{seg_idx}"
                lines.append(f"    // Wait for segment {seg_idx} release (call {call_idx})")
                lines.append(f"    while (*__{s}_release < {call_idx + 1}) {{ }}")
            else:
                lines.append(f"    // Wait for global release (Phase {phase_idx + 1})")
                lines.append(f"    while (*__global_release < {phase_idx + 1}) {{ }}")
            lines.append("")
            # Resync NCRISC local CB pointers to CB start
            if sweep_cb_indices:
                lines.append("    // Resync NCRISC CB pointers to start")
                lines.append("    __resync_ncrisc_cb_state();")
                lines.append("")
            # Rebind CB addresses after barrier wait
            rebind_lines = _generate_rebind_code(rebind_info.get(next_phase_idx, []), next_phase_idx, for_compute=False)
            if rebind_lines:
                lines.extend(rebind_lines)
                lines.append("")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _generate_fused_compute_source(
    phase_kernels: List[Dict[str, Any]],
    role_key: Any,
    phases: List[PhaseInfo],
    ct_arg_offsets: Optional[Dict[int, int]] = None,
    sweep_cb_indices: Optional[List[int]] = None,
    barrier_config: Optional[BarrierConfig] = None,
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
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
        phase_defs = {name for name, _ in kernel.defines}
        resolved = cpp_parser.resolve_ifdef_directives(source, phase_defs)
        compute_sources.append((i, resolved))

    if not compute_sources:
        return None

    all_sources = [s for _, s in compute_sources]
    includes = cpp_parser.collect_includes(all_sources)
    defines = cpp_parser.collect_defines(all_sources)
    pre_main, phase_names = _collect_all_pre_main_code(compute_sources)

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused compute kernel - {len(compute_sources)} phases",
        "",
    ]
    lines.extend(defines)
    lines.append("")
    lines.extend(includes)
    lines.append("")
    if pre_main.strip():
        lines.append(pre_main)
        lines.append("")

    is_multi_phase = len(compute_sources) > 1

    if is_multi_phase:
        lines.append('constexpr uint32_t __barrier_rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
        lines.append("")

    # Generate compute-side CB state resync function.
    # After BRISC's __cb_reset_to_empty(), stream registers are equalized and
    # BRISC pointers are at CB start. TRISC0 and TRISC2 need to sync their
    # local state and reset pointers to CB start as well.
    if sweep_cb_indices and is_multi_phase:
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
    for phase_idx, resolved_source in compute_sources:
        body = cpp_parser.extract_kernel_body(resolved_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        pnames = phase_names.get(phase_idx, [])
        transformed = _transform_phase_source(body, phase_idx, offset, phase_names=pnames)

        lines.append(f"// Phase {phase_idx} compute")
        lines.append(f"FORCE_INLINE void phase{phase_idx}_compute() {{")
        for line in transformed.split("\n"):
            lines.append(f"    {line}")
        lines.append("}")
        lines.append("")

    # Generate kernel_main
    lines.append("void kernel_main() {")

    use_multi_barrier = multi_barrier is not None

    if is_multi_phase:
        lines.append("    // Read barrier L1 flag addresses from runtime args")
        lines.append("    const uint32_t __compute_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset);")
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__compute_done_addr);"
        )
        if use_multi_barrier:
            # Per-segment release pointers
            for seg_idx in range(len(multi_barrier.segments)):
                s = f"seg{seg_idx}"
                off = 1 + seg_idx
                lines.append(
                    f"    const uint32_t __{s}_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + {off});"
                )
                lines.append(
                    f"    volatile tt_l1_ptr uint32_t* __{s}_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__{s}_release_addr);"
                )
        else:
            lines.append("    const uint32_t __global_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 1);")
            lines.append(
                "    volatile tt_l1_ptr uint32_t* __global_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__global_release_addr);"
            )
        lines.append("")

    if rebind_info is None:
        rebind_info = {}

    first = True
    for count, (phase_idx, _) in enumerate(compute_sources):
        if not first and is_multi_phase:
            lines.append("")
            lines.append(f"    // Signal done for Phase {phase_idx - 1}")
            lines.append(f"    *__compute_done = {phase_idx};")
            lines.append("")
            if use_multi_barrier:
                transition_idx = phase_idx - 1
                seg_idx, call_idx = multi_barrier.transition_map[transition_idx]
                s = f"seg{seg_idx}"
                lines.append(f"    // Wait for segment {seg_idx} release (call {call_idx})")
                lines.append(f"    while (*__{s}_release < {call_idx + 1}) {{ }}")
            else:
                lines.append(f"    // Wait for global release (Phase {phase_idx})")
                lines.append(f"    while (*__global_release < {phase_idx}) {{ }}")
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
                pass

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

    for pk in phase_kernels:
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue
        for col_idx, core in enumerate(core_coords):
            try:
                args = kernel.runtime_args[core.x][core.y]
                col_args[col_idx].extend(list(args))
            except (IndexError, KeyError):
                pass

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
        except Exception:
            pass
    return common_args


# =============================================================================
# Named Compile-Time Arg Merging
# =============================================================================


def _merge_named_compile_time_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
    rt_arg_offsets: Optional[Dict[int, int]] = None,
    barrier_rt_offset: Optional[int] = None,
    barrier_config: Optional[BarrierConfig] = None,
    phase_remaps: Optional[List[Dict[int, int]]] = None,
) -> List[Tuple[str, int]]:
    """Merge named compile-time args from all phases with phase prefixes.

    Phase 0 keeps original names. Phase N>0 gets "phaseN_" prefix.
    Runtime arg offsets and barrier config are added as named args.
    CB-reference args (names starting with "cb_") are remapped to pool slot indices.
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
            if remap is not None and name.startswith("cb_") and isinstance(value, int):
                actual_value = remap.get(value, value)

            if i == 0:
                merged.append((name, actual_value))
            else:
                merged.append((f"phase{i}_{name}", actual_value))

        # Add runtime arg offset for phase 1+
        if i > 0 and rt_arg_offsets is not None and i in rt_arg_offsets:
            merged.append((f"phase{i}_rt_arg_offset", rt_arg_offsets[i]))

    # Add barrier named args
    if barrier_rt_offset is not None:
        merged.append(("barrier_rt_offset", barrier_rt_offset))
    if barrier_config is not None:
        merged.append(("num_barrier_cores", barrier_config.num_cores))
        merged.append(("core0_phys_x", barrier_config.core0_phys_x))
        merged.append(("core0_phys_y", barrier_config.core0_phys_y))
        merged.append(("mcast_start_x", barrier_config.mcast_start_x))
        merged.append(("mcast_start_y", barrier_config.mcast_start_y))
        merged.append(("mcast_end_x", barrier_config.mcast_end_x))
        merged.append(("mcast_end_y", barrier_config.mcast_end_y))

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


def _merge_defines(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> List[Tuple[str, str]]:
    """Merge defines from all phases' kernels.

    Source-level defines (RMSNORM etc.) are resolved per-phase into source.
    Common defines (REDUCE_OP etc.) are kept as-is. Others get phase-prefixed.
    """
    merged = []
    seen_common = set()
    common_defines = {"REDUCE_OP", "REDUCE_DIM", "BCAST_LLKOP", "BCAST_DIM"}

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        for name, value in kernel.defines:
            if name in common_defines:
                if name not in seen_common:
                    merged.append((name, value))
                    seen_common.add(name)
            elif name in cpp_parser.SOURCE_LEVEL_DEFINES:
                continue
            else:
                if i == 0:
                    merged.append((name, value))
                else:
                    merged.append((f"PHASE{i}_{name}", value))

    return merged


# =============================================================================
# Compute Config Validation
# =============================================================================


def _validate_and_get_compute_config(
    phase_kernels: List[Dict[str, Any]],
) -> "ttnn.ComputeConfigDescriptor":
    """Validate that all phases have identical compute configs and return it.

    Compute kernel configs (fp32_dest_acc_en, math_fidelity, math_approx_mode,
    etc.) are hardware settings that cannot be reconfigured mid-kernel.  All
    phases must use exactly the same config.
    """
    base = None
    base_phase = -1

    for phase_idx, pk in enumerate(phase_kernels):
        compute = pk.get("compute")
        if compute is None:
            continue

        config = compute.config
        if base is None:
            base = config
            base_phase = phase_idx
            continue

        # Validate all fields match
        mismatches = []
        for field in (
            "fp32_dest_acc_en",
            "math_approx_mode",
            "math_fidelity",
            "dst_full_sync_en",
            "bfp8_pack_precise",
        ):
            base_val = getattr(base, field, None)
            this_val = getattr(config, field, None)
            if base_val != this_val:
                mismatches.append(f"  {field}: phase {base_phase}={base_val}, phase {phase_idx}={this_val}")

        if mismatches:
            raise ValueError(
                f"Compute config mismatch between phases. These are hardware "
                f"settings that cannot change mid-kernel — all phases must use "
                f"identical compute configs.\n" + "\n".join(mismatches)
            )

    if base is None:
        return ttnn.ComputeConfigDescriptor()

    return base


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
        for fld in ("fp32_dest_acc_en", "math_approx_mode", "math_fidelity", "dst_full_sync_en", "bfp8_pack_precise"):
            base_val = getattr(base, fld, None)
            this_val = getattr(config, fld, None)
            if base_val != this_val:
                mismatches.append(f"  {fld}: phase {base_phase}={base_val}, phase {phase_idx}={this_val}")

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
        f"setting. To fix: create all descriptors with the same "
        f"compute_kernel_config. For example:\n"
        f"  config = ttnn.layernorm_default_compute_config(device.arch())\n"
        f"  rms = rms_norm.rms_norm(input, ..., compute_kernel_config=config)\n"
        f"  ln  = layer_norm.layer_norm(input, ..., compute_kernel_config=config)"
    )


# =============================================================================
# Barrier Configuration
# =============================================================================


def _create_barrier_config(device: Any, core_ranges: Any) -> BarrierConfig:
    """Create barrier configuration with GlobalSemaphore L1 flags.

    Allocates 4 GlobalSemaphores (one 4-byte L1 word per core each):
      - compute_done: compute signals phase completion
      - writer_done: writer signals phase completion
      - global_arrive: cross-core barrier arrive counter
      - global_release: cross-core barrier release flag (also serves as
        phase release — compute/writer spin on this directly)

    Also computes physical core coordinates for NOC addressing.
    """
    config = BarrierConfig()

    # Create GlobalSemaphores for per-core L1 flags
    sem_compute_done = ttnn.create_global_semaphore(device, core_ranges, 0)
    sem_writer_done = ttnn.create_global_semaphore(device, core_ranges, 0)
    sem_global_arrive = ttnn.create_global_semaphore(device, core_ranges, 0)
    sem_global_release = ttnn.create_global_semaphore(device, core_ranges, 0)

    # Store references to prevent GC
    config._sem_refs = [sem_compute_done, sem_writer_done, sem_global_arrive, sem_global_release]

    # Get L1 addresses
    config.compute_done_addr = ttnn.get_global_semaphore_address(sem_compute_done)
    config.writer_done_addr = ttnn.get_global_semaphore_address(sem_writer_done)
    config.global_arrive_addr = ttnn.get_global_semaphore_address(sem_global_arrive)
    config.global_release_addr = ttnn.get_global_semaphore_address(sem_global_release)

    # Compute physical core coordinates for global barrier
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

        # Validate rectangular grid for safe NOC multicast
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


def _create_role_barrier_config(
    device: Any,
    role_core_ranges: Any,
    shared_config: BarrierConfig,
) -> BarrierConfig:
    """Create a role-specific barrier config sharing semaphore addresses.

    Uses the shared GlobalSemaphore L1 addresses but computes role-specific
    core counts and physical coordinates for the multicast barrier.
    """
    cfg = BarrierConfig()
    cfg.compute_done_addr = shared_config.compute_done_addr
    cfg.writer_done_addr = shared_config.writer_done_addr
    cfg.global_arrive_addr = shared_config.global_arrive_addr
    cfg.global_release_addr = shared_config.global_release_addr

    logical_coords = _get_core_coords_from_ranges(role_core_ranges)
    cfg.num_cores = len(logical_coords)

    if cfg.num_cores > 0:
        phys_coords = [device.worker_core_from_logical_core(c) for c in logical_coords]
        cfg.core0_phys_x = phys_coords[0].x
        cfg.core0_phys_y = phys_coords[0].y
        cfg.mcast_start_x = min(c.x for c in phys_coords)
        cfg.mcast_start_y = min(c.y for c in phys_coords)
        cfg.mcast_end_x = max(c.x for c in phys_coords)
        cfg.mcast_end_y = max(c.y for c in phys_coords)

        # Validate rectangular grid for safe NOC multicast
        if cfg.num_cores > 1:
            _validate_rectangular_grid(phys_coords, cfg)

    return cfg


def _compute_union_core_ranges(phases: List[PhaseInfo]) -> Any:
    """Compute the union of all core ranges across all kernels in phase 0.

    Returns a CoreRangeSet covering all cores used by any kernel.
    Handles overlapping core ranges by collecting individual cores and
    creating a bounding box CoreRange that covers all of them.
    """
    # Collect all unique logical core coordinates
    all_coords = set()
    for kernel_desc in phases[0].op_descriptor.descriptor.kernels:
        for cr in kernel_desc.core_ranges.ranges():
            for y in range(cr.start.y, cr.end.y + 1):
                for x in range(cr.start.x, cr.end.x + 1):
                    all_coords.add((x, y))

    if not all_coords:
        return phases[0].op_descriptor.descriptor.kernels[0].core_ranges

    # Create a bounding box CoreRange covering all cores
    min_x = min(x for x, y in all_coords)
    max_x = max(x for x, y in all_coords)
    min_y = min(y for x, y in all_coords)
    max_y = max(y for x, y in all_coords)

    bounding_range = ttnn.CoreRange(
        ttnn.CoreCoord(min_x, min_y),
        ttnn.CoreCoord(max_x, max_y),
    )
    return ttnn.CoreRangeSet([bounding_range])


# =============================================================================
# Sequential Chain Builder
# =============================================================================


class SequentialChainBuilder:
    """Builds a fused ProgramDescriptor from a sequence of OpDescriptors.

    All readers/writers run for every phase.  Data flows through DRAM between
    phases.  No CB remapping — each phase uses native CB indices (0-31).

    Uses two-level barrier synchronization:
      - Local: L1 flags (via GlobalSemaphore) for per-core RISC coordination
      - Global: NOC semaphore ops for cross-core barrier (dataflow RISC only)
    """

    def __init__(self):
        self.phases: List[PhaseInfo] = []
        self._built = False
        self._barrier_config: Optional[BarrierConfig] = None

    def add_phase(self, op_descriptor: OpDescriptor) -> "SequentialChainBuilder":
        """Add a phase to the sequential chain.

        Args:
            op_descriptor: The OpDescriptor for this phase.  For phases 1+,
                the input tensor should be the previous phase's output tensor
                so the reader reads from the correct DRAM address.

        Returns:
            self for method chaining
        """
        phase_idx = len(self.phases)
        # Get unpack_to_dest_mode vector from compute kernel config
        utd_modes = None
        for kd in op_descriptor.descriptor.kernels:
            config = kd.config
            if hasattr(config, "unpack_to_dest_mode"):
                modes = config.unpack_to_dest_mode
                if modes is not None and len(modes) > 0:
                    utd_modes = modes
                    break
        cb_info = extract_cb_info(op_descriptor.descriptor, utd_modes)
        phase = PhaseInfo(
            phase_idx=phase_idx,
            op_descriptor=op_descriptor,
            cb_info=cb_info,
        )
        self.phases.append(phase)
        return self

    def build(self, device: Any) -> OpDescriptor:
        """Build the fused OpDescriptor from the chain.

        Args:
            device: The device (MeshDevice or IDevice) for GlobalSemaphore
                allocation and coordinate conversion.

        Returns:
            Fused OpDescriptor that executes all phases sequentially.
        """
        if self._built:
            raise ValueError("Chain has already been built")
        if not self.phases:
            raise ValueError("Chain has no phases")

        self._built = True

        if len(self.phases) == 1:
            return self.phases[0].op_descriptor

        return self._build_fused_descriptor(device)

    def _build_fused_descriptor(
        self,
        device: Any,
        core_range_override: Optional[Any] = None,
        multi_barrier: Optional[MultiBarrierSpec] = None,
    ) -> OpDescriptor:
        """Build the fused descriptor with two-level barrier sync.

        Dynamically discovers kernel roles from the ProgramDescriptor using
        (risc_type, core_ranges) as a unique key. This supports any op type
        (interleaved with 3 kernels, sharded with up to 7 kernels, etc.).

        Args:
            device: The device for GlobalSemaphore allocation.
            core_range_override: If set, overrides role keys and fused kernel
                core_ranges. Used by OpGraphBuilder for paths where stem ops
                have wider core ranges than the path's leaf branch.
            multi_barrier: If set, uses multi-segment barrier instead of
                single barrier. Used by OpGraphBuilder for barrier scope
                transitions between stem and branch phases.
        """
        # Validate fp32 consistency
        _validate_fp32_consistency([p.op_descriptor for p in self.phases])

        # Discover all kernel roles from phase 0
        # Role key = (risc_type, frozenset of core range tuples)
        # When core_range_override is set, all kernels are mapped to the
        # override range (needed for OpGraph paths where stem ops have wider
        # core ranges than the path's leaf branch).
        role_keys: List[Tuple[str, frozenset]] = []
        role_keys_set: Set[Tuple[str, frozenset]] = set()
        for kernel_desc in self.phases[0].op_descriptor.descriptor.kernels:
            rk = _get_role_key(kernel_desc, core_range_override)
            if rk not in role_keys_set:
                role_keys.append(rk)
                role_keys_set.add(rk)

        # Build phase_kernels as List[Dict[role_key, KernelDescriptor]]
        phase_kernels: List[Dict[Any, Any]] = []
        for phase_idx, phase in enumerate(self.phases):
            role_map: Dict[Any, Any] = {}
            for kernel_desc in phase.op_descriptor.descriptor.kernels:
                rk = _get_role_key(kernel_desc, core_range_override)
                role_map[rk] = kernel_desc
            phase_kernels.append(role_map)

        # Pool-allocate CB slots based on compatibility keys
        pool = CBPoolAllocator(max_slots=32)
        for phase_idx, phase in enumerate(self.phases):
            phantom_indices = _get_phantom_cb_indices(phase)
            pool.allocate_phase(phase_idx, phase.cb_info, phantom_indices)

        # Compute CB address rebinding info using remapped slot indices.
        # Must be computed BEFORE build_merged_cb_descriptors, which
        # modifies buffer_index in-place on the original CBDescriptors.
        rebind_info = _compute_rebind_info(self.phases, pool.phase_remaps)

        # Build merged CB descriptors from pool (modifies buffer_index in-place)
        merged_cbs = pool.build_merged_cb_descriptors(self.phases)

        # Override CB core_ranges when building a path through an OpGraph.
        # Stem ops have CBs with union_range, but each path's fused descriptor
        # must only claim its leaf core range so paths don't overlap when
        # merged by composite.launch().
        if core_range_override is not None:
            for cb_desc in merged_cbs:
                cb_desc.core_ranges = core_range_override

        # Create barrier config with GlobalSemaphores on union of all core ranges
        # (skip if multi_barrier already provides barrier configuration)
        if multi_barrier is None:
            union_core_ranges = _compute_union_core_ranges(self.phases)
            self._barrier_config = _create_barrier_config(device, union_core_ranges)
            bc = self._barrier_config
        else:
            bc = None  # multi_barrier provides all barrier info

        # Sweep indices = all allocated CB pool slots
        sweep_cb_indices = sorted(pool.get_all_slot_indices())

        # Collect all unique op semaphore IDs (0-15) used by any phase.
        # These need to be reset to 0 between phases so each phase starts clean.
        op_semaphore_ids: List[int] = []
        seen_sem_ids_for_reset: Set[int] = set()
        for phase in self.phases:
            for sem in phase.op_descriptor.descriptor.semaphores:
                if sem.id not in seen_sem_ids_for_reset:
                    op_semaphore_ids.append(sem.id)
                    seen_sem_ids_for_reset.add(sem.id)
        op_semaphore_ids.sort()

        fused_kernels = []

        # For each discovered role: generate fused source, merge args, build descriptor
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
            if multi_barrier is not None:
                # --- Multi-barrier path (OpGraph) ---
                if risc_type == "riscv_0":
                    fused_source = _generate_fused_riscv0_source(
                        phase_kernels,
                        role_key,
                        self.phases,
                        ct_offsets,
                        sweep_cb_indices,
                        barrier_config=None,
                        rebind_info=rebind_info,
                        op_semaphore_ids=op_semaphore_ids,
                        multi_barrier=multi_barrier,
                    )
                    # riscv0: compute_done, writer_done, then per-segment arrive+release
                    barrier_addrs = [multi_barrier.compute_done_addr, multi_barrier.writer_done_addr]
                    for seg in multi_barrier.segments:
                        barrier_addrs.extend([seg.arrive_addr, seg.release_addr])
                elif risc_type == "riscv_1":
                    fused_source = _generate_fused_riscv1_source(
                        phase_kernels,
                        role_key,
                        self.phases,
                        ct_offsets,
                        sweep_cb_indices,
                        rebind_info=rebind_info,
                        barrier_config=None,
                        multi_barrier=multi_barrier,
                    )
                    # riscv1: writer_done, then per-segment release
                    barrier_addrs = [multi_barrier.writer_done_addr]
                    for seg in multi_barrier.segments:
                        barrier_addrs.append(seg.release_addr)
                elif risc_type == "compute":
                    fused_source = _generate_fused_compute_source(
                        phase_kernels,
                        role_key,
                        self.phases,
                        ct_offsets,
                        sweep_cb_indices,
                        barrier_config=None,
                        rebind_info=rebind_info,
                        multi_barrier=multi_barrier,
                    )
                    # compute: compute_done, then per-segment release
                    barrier_addrs = [multi_barrier.compute_done_addr]
                    for seg in multi_barrier.segments:
                        barrier_addrs.append(seg.release_addr)
                else:
                    continue
            else:
                # --- Single-barrier path (standard linear chain) ---
                # IMPORTANT: riscv_0 must use the GLOBAL barrier config (bc) so that
                # ALL riscv_0 cores across ALL roles synchronize via a single barrier.
                if risc_type == "riscv_0":
                    fused_source = _generate_fused_riscv0_source(
                        phase_kernels,
                        role_key,
                        self.phases,
                        ct_offsets,
                        sweep_cb_indices,
                        bc,
                        rebind_info,
                        op_semaphore_ids=op_semaphore_ids,
                    )
                    barrier_addrs = [
                        bc.compute_done_addr,
                        bc.writer_done_addr,
                        bc.global_arrive_addr,
                        bc.global_release_addr,
                    ]
                elif risc_type == "riscv_1":
                    fused_source = _generate_fused_riscv1_source(
                        phase_kernels,
                        role_key,
                        self.phases,
                        ct_offsets,
                        sweep_cb_indices,
                        rebind_info,
                        bc,
                    )
                    barrier_addrs = [bc.writer_done_addr, bc.global_release_addr]
                elif risc_type == "compute":
                    fused_source = _generate_fused_compute_source(
                        phase_kernels,
                        role_key,
                        self.phases,
                        ct_offsets,
                        sweep_cb_indices,
                        bc,
                        rebind_info,
                    )
                    barrier_addrs = [bc.compute_done_addr, bc.global_release_addr]
                else:
                    continue

            if fused_source is None:
                continue

            # Concatenate runtime args and append barrier addresses
            rt_args = _concatenate_runtime_args(phase_kernels, role_key, core_range_override=core_range_override)
            rt_args, barrier_offset = _append_barrier_runtime_args(rt_args, barrier_addrs)

            # Merge named compile-time args
            if multi_barrier is not None:
                # Multi-barrier: no single BarrierConfig for named args;
                # per-segment constants are added below.
                named_ct_args = _merge_named_compile_time_args(
                    phase_kernels,
                    role_key,
                    rt_offsets,
                    barrier_rt_offset=barrier_offset,
                    barrier_config=None,
                    phase_remaps=pool.phase_remaps,
                )
                # Add per-segment named compile-time args (only riscv_0 needs them)
                if risc_type == "riscv_0":
                    for seg_idx, seg in enumerate(multi_barrier.segments):
                        s = f"seg{seg_idx}"
                        named_ct_args.append((f"{s}_num_cores", seg.config.num_cores))
                        named_ct_args.append((f"{s}_core0_phys_x", seg.config.core0_phys_x))
                        named_ct_args.append((f"{s}_core0_phys_y", seg.config.core0_phys_y))
                        named_ct_args.append((f"{s}_mcast_start_x", seg.config.mcast_start_x))
                        named_ct_args.append((f"{s}_mcast_start_y", seg.config.mcast_start_y))
                        named_ct_args.append((f"{s}_mcast_end_x", seg.config.mcast_end_x))
                        named_ct_args.append((f"{s}_mcast_end_y", seg.config.mcast_end_y))
            else:
                # Single barrier: riscv_0 gets full barrier config for global barrier
                barrier_cfg_for_named = bc if risc_type == "riscv_0" else None
                named_ct_args = _merge_named_compile_time_args(
                    phase_kernels,
                    role_key,
                    rt_offsets,
                    barrier_rt_offset=barrier_offset,
                    barrier_config=barrier_cfg_for_named,
                    phase_remaps=pool.phase_remaps,
                )

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
            desc.defines = _merge_defines(phase_kernels, role_key)
            desc.runtime_args = rt_args
            desc.common_runtime_args = _concatenate_common_runtime_args(phase_kernels, role_key)
            desc.config = role_config
            fused_kernels.append(desc)

        # Merge semaphores (dedup by ID)
        all_semaphores = []
        seen_sem_ids: Set[int] = set()
        for phase in self.phases:
            for sem in phase.op_descriptor.descriptor.semaphores:
                if sem.id not in seen_sem_ids:
                    all_semaphores.append(sem)
                    seen_sem_ids.add(sem.id)

        # Collect input/output tensors (use id() for dedup because ttnn Tensor's
        # __eq__ returns an element-wise Tensor, making `in` unreliable)
        all_input_tensors = []
        seen_tensor_ids: Set[int] = set()
        for phase in self.phases:
            for tensor in phase.op_descriptor.input_tensors:
                tid = id(tensor)
                if tid not in seen_tensor_ids:
                    all_input_tensors.append(tensor)
                    seen_tensor_ids.add(tid)

        output_tensor = None
        if self.phases[-1].op_descriptor.output_tensors:
            output_tensor = self.phases[-1].op_descriptor.output_tensors[0]

        # Create the merged ProgramDescriptor
        merged_descriptor = ttnn.ProgramDescriptor()
        merged_descriptor.kernels = fused_kernels
        merged_descriptor.cbs = merged_cbs
        merged_descriptor.semaphores = all_semaphores

        # Collect keepalive references to prevent GC of GlobalSemaphores
        if multi_barrier is not None:
            keepalive_refs = tuple(multi_barrier._sem_refs)
        else:
            keepalive_refs = tuple(self._barrier_config._sem_refs)

        return OpDescriptor(
            descriptor=merged_descriptor,
            input_tensors=all_input_tensors,
            output_tensors=[output_tensor] if output_tensor else [],
            keepalive=keepalive_refs,
        )


class OpGraphBuilder:
    """Builds fused descriptors for OpGraph (branching tree) topologies.

    A shared stem of phases runs on ALL branch cores, then cores split
    into disjoint branches each running their own subsequent phases.
    Branches can nest via ``BranchSpec.children``.

    For each root-to-leaf path through the tree, a separate fused kernel
    binary is generated.  During stem phases, different kernel binaries
    on different cores synchronize via shared GlobalSemaphore addresses.
    After the split, each branch barriers independently within its core
    subset.

    Usage::

        builder = OpGraphBuilder()
        builder.add_stem_phase(ln_op)
        builder.add_branch(cores_A, [rms_op_A])
        builder.add_branch(cores_B, [rms_op_B])
        fused_ops = builder.build(device)
        # Returns 2 OpDescriptors, suitable for composite.launch()
    """

    def __init__(self):
        self.stem_phases: List[OpDescriptor] = []
        self.branches: List[BranchSpec] = []
        self._built = False

    def add_stem_phase(self, op: OpDescriptor) -> "OpGraphBuilder":
        """Add a phase that runs on ALL branch cores before the split."""
        self.stem_phases.append(op)
        return self

    def add_branch(
        self,
        core_range: Any,
        phases: Optional[List[OpDescriptor]] = None,
        children: Optional[List[BranchSpec]] = None,
    ) -> "OpGraphBuilder":
        """Add a top-level branch after the stem phases.

        Args:
            core_range: CoreRangeSet for this branch's cores.
            phases: Sequential phases within this branch.
            children: Sub-branches for nested splitting.
        """
        self.branches.append(BranchSpec(core_range, phases or [], children or []))
        return self

    def build(self, device: Any) -> List[OpDescriptor]:
        """Build fused descriptors for all root-to-leaf paths.

        Returns a list of OpDescriptors, one per path, suitable for
        ``composite.launch()``.
        """
        if self._built:
            raise ValueError("Already built")
        self._built = True

        if not self.branches:
            raise ValueError("OpGraph has no branches")
        if not self.stem_phases:
            raise ValueError("OpGraph has no stem phases")

        # Trace all root-to-leaf paths
        paths = self._trace_paths()

        # Compute union of all branch core ranges
        union_range = self._compute_union_ranges()

        # Allocate shared per-core monotonic semaphores on union range.
        # compute_done and writer_done are per-core and don't involve cross-core
        # communication, so they can be shared across all barrier segments.
        sem_compute_done = ttnn.create_global_semaphore(device, union_range, 0)
        sem_writer_done = ttnn.create_global_semaphore(device, union_range, 0)
        compute_done_addr = ttnn.get_global_semaphore_address(sem_compute_done)
        writer_done_addr = ttnn.get_global_semaphore_address(sem_writer_done)
        all_sem_refs = [sem_compute_done, sem_writer_done]

        # Pre-allocate barrier configs for each unique core range across all
        # path segments.  Paths that share a segment (e.g. the stem) MUST use
        # the same arrive/release GlobalSemaphore L1 addresses so that cores
        # running different kernel binaries synchronize at the same barrier.
        segment_cache: Dict[frozenset, BarrierConfig] = {}
        for path in paths:
            for core_range, _ in path:
                key = _core_ranges_key(core_range)
                if key not in segment_cache:
                    segment_cache[key] = _create_barrier_config(device, core_range)
                    all_sem_refs.extend(segment_cache[key]._sem_refs)

        fused_ops = []
        for path in paths:
            # Save CB descriptor state before building each path.
            # _build_fused_descriptor mutates buffer_index, total_size, and
            # core_ranges IN-PLACE on the original CBDescriptors (can't
            # deepcopy C++ bindings).  When paths share ops (e.g. stem),
            # the first path's mutations corrupt subsequent paths' cb_info.
            saved_cb_state = self._save_cb_state(path)

            fused = self._build_path(
                device,
                path,
                compute_done_addr,
                writer_done_addr,
                all_sem_refs,
                segment_cache,
            )
            fused_ops.append(fused)

            # Restore original state so the next path sees uncorrupted indices
            self._restore_cb_state(saved_cb_state)

        return fused_ops

    def _trace_paths(self) -> List[List[Tuple[Any, List[OpDescriptor]]]]:
        """Trace all root-to-leaf paths through the tree.

        Returns a list of paths.  Each path is a list of
        ``(core_range, [phases])`` segments from root (stem) to leaf.
        """
        union_range = self._compute_union_ranges()
        paths: List[List[Tuple[Any, List[OpDescriptor]]]] = []

        def _collect(branch: BranchSpec, prefix: List[Tuple[Any, List[OpDescriptor]]]):
            current = prefix + [(branch.core_range, list(branch.phases))]
            if not branch.children:
                paths.append(current)
            else:
                for child in branch.children:
                    _collect(child, current)

        stem_segment = (union_range, list(self.stem_phases))
        for branch in self.branches:
            _collect(branch, [stem_segment])

        return paths

    def _compute_union_ranges(self) -> Any:
        """Compute the union CoreRangeSet of all leaf branch core ranges.

        Only leaf branches (those without children) contribute core ranges.
        Intermediate branches' ranges overlap with their children, so
        including both would trigger CoreRangeSet's overlap validation.
        """
        all_ranges = set()

        def _collect_leaf_ranges(branches: List[BranchSpec]):
            for b in branches:
                if b.children:
                    _collect_leaf_ranges(b.children)
                else:
                    for cr in b.core_range.ranges():
                        all_ranges.add(
                            ttnn.CoreRange(
                                ttnn.CoreCoord(cr.start.x, cr.start.y),
                                ttnn.CoreCoord(cr.end.x, cr.end.y),
                            )
                        )

        _collect_leaf_ranges(self.branches)
        return ttnn.CoreRangeSet(all_ranges)

    @staticmethod
    def _save_cb_state(
        path: List[Tuple[Any, List[OpDescriptor]]],
    ) -> List[dict]:
        """Save mutable CB descriptor state for all ops in a path.

        _build_fused_descriptor mutates buffer_index, total_size, and
        core_ranges on the original CBDescriptors (can't deepcopy C++
        bindings).  Save these fields so they can be restored after
        each path build to prevent cross-path corruption.
        """
        saved = []
        seen_cb_ids: set = set()
        for _, phases in path:
            for phase in phases:
                for cb_desc in phase.descriptor.cbs:
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

    @staticmethod
    def _restore_cb_state(saved: List[dict]) -> None:
        """Restore CB descriptor state saved by _save_cb_state."""
        for entry in saved:
            entry["cb"].total_size = entry["total_size"]
            entry["cb"].core_ranges = entry["core_ranges"]
            for fmt, orig_idx in entry["fmt"]:
                fmt.buffer_index = orig_idx

    def _build_path(
        self,
        device: Any,
        path: List[Tuple[Any, List[OpDescriptor]]],
        compute_done_addr: int,
        writer_done_addr: int,
        shared_sem_refs: List[Any],
        segment_cache: Dict[frozenset, BarrierConfig],
    ) -> OpDescriptor:
        """Build a fused OpDescriptor for one root-to-leaf path."""
        # Flatten phases and determine leaf core range
        all_phases: List[OpDescriptor] = []
        for _, phases in path:
            all_phases.extend(phases)

        # Leaf core range = last segment's core range
        leaf_core_range = path[-1][0]

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

        # Use a SequentialChainBuilder with the flattened phase list
        builder = SequentialChainBuilder()
        for op in all_phases:
            builder.add_phase(op)
        builder._built = True  # Skip build() validation, call _build directly

        return builder._build_fused_descriptor(
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


def chain_descriptors(descriptors: List[OpDescriptor], device: Any) -> OpDescriptor:
    """Chain multiple OpDescriptors sequentially.

    For phases 1+, the input tensor should be the previous phase's output
    tensor.  This ensures each reader reads from the correct DRAM address.

    Args:
        descriptors: List of OpDescriptors to chain sequentially.
        device: The device for GlobalSemaphore allocation.

    Returns:
        Fused OpDescriptor.
    """
    builder = SequentialChainBuilder()
    for desc in descriptors:
        builder.add_phase(desc)
    return builder.build(device)


def create_parallel_chain_descriptors(
    chains: List[List[OpDescriptor]],
    device: Any,
) -> List[OpDescriptor]:
    """Create fused descriptors for multiple parallel chains.

    Each chain is fused sequentially, and the resulting fused ops can be
    run in parallel using composite.launch().

    Args:
        chains: List of chains, where each chain is a list of OpDescriptors.
        device: The device for GlobalSemaphore allocation.

    Returns:
        List of fused OpDescriptors, one per chain.
    """
    fused_descriptors = []
    for chain in chains:
        if not chain:
            continue
        if len(chain) == 1:
            fused_descriptors.append(chain[0])
        else:
            fused_descriptors.append(chain_descriptors(chain, device))
    return fused_descriptors


def build_op_graph(
    stem_phases: List[OpDescriptor],
    branches: List[BranchSpec],
    device: Any,
) -> List[OpDescriptor]:
    """Build fused descriptors for an OpGraph (branching tree) topology.

    Convenience wrapper around :class:`OpGraphBuilder`.

    Args:
        stem_phases: Shared phases that run on all branch cores.
        branches: List of BranchSpec defining the tree structure.
        device: The device for GlobalSemaphore allocation.

    Returns:
        One OpDescriptor per root-to-leaf path, suitable for
        ``composite.launch()``.
    """
    builder = OpGraphBuilder()
    for op in stem_phases:
        builder.add_stem_phase(op)
    for branch in branches:
        builder.add_branch(branch.core_range, branch.phases, branch.children)
    return builder.build(device)


__all__ = [
    # Core classes
    "SequentialChainBuilder",
    "OpGraphBuilder",
    "BranchSpec",
    "PhaseInfo",
    "CBInfo",
    "BarrierConfig",
    # Functions
    "chain_descriptors",
    "create_parallel_chain_descriptors",
    "build_op_graph",
    "extract_cb_info",
    "extract_cb_names_from_kernel",
]
