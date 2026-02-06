# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sequential Kernel Chaining Infrastructure

This module provides infrastructure for chaining multiple operations sequentially
on the SAME cores within a single program. Unlike composite.py (which runs ops on
DIFFERENT cores in parallel), sequential.py fuses ops on the same cores with the
output of op N becoming the input of op N+1.

The infrastructure is GENERAL - it works with any ProgramDescriptor objects,
not just specific op types. The key requirement is that kernels must accept
CB indices as compile-time arguments or defines for remapping to work.

Key Concepts:
-------------
1. CB Remapping: Each op's CBs are remapped to non-conflicting indices
2. Output→Input Chaining: Output CB of phase N becomes input CB of phase N+1
3. Kernel Modification: Compile-time args/defines are updated with new CB indices

Architecture:
-------------
                     ┌──────────────────────────────────────┐
                     │         SequentialChainBuilder        │
                     │  - Takes list of OpDescriptor         │
                     │  - Analyzes CB usage                  │
                     │  - Remaps CBs via CBRemapper          │
                     │  - Generates merged ProgramDescriptor │
                     └──────────────────────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
           ┌─────────────────┐               ┌─────────────────┐
           │   CBRemapper    │               │  KernelChainer  │
           │ - Tracks 32 CBs │               │ - Updates args  │
           │ - Assigns new   │               │ - Updates defs  │
           │   indices       │               │ - Chains phases │
           │ - Chains data   │               │                 │
           └─────────────────┘               └─────────────────┘

Usage Example:
--------------
    >>> # Get OpDescriptors for ops to chain
    >>> ln1_desc = descriptors.normalization.layer_norm(input, weight=w1, ...)
    >>> ln2_desc = descriptors.normalization.rms_norm(None, weight=w2, ...)  # Input from prev
    >>>
    >>> # Build sequential chain
    >>> builder = SequentialChainBuilder()
    >>> builder.add_phase(ln1_desc)
    >>> builder.add_phase(ln2_desc, input_from_previous=True)
    >>> fused_desc = builder.build()
    >>>
    >>> # Execute via composite.launch (or directly)
    >>> outputs = composite.launch([fused_desc])
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set, Any
from copy import deepcopy

import ttnn
from ttnn._ttnn.program_descriptor import UnpackToDestMode

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


# =============================================================================
# CB Analysis and Remapping
# =============================================================================


@dataclass
class CBInfo:
    """Information about a circular buffer extracted from a CBDescriptor."""

    original_index: int
    total_size: int
    data_format: Any  # tt::DataFormat
    page_size: int
    core_ranges: Any  # CoreRangeSet
    is_input: bool = False  # Determined by kernel analysis
    is_output: bool = False  # Determined by kernel analysis


@dataclass
class PhaseInfo:
    """Information about a phase (op) in the sequential chain."""

    phase_idx: int
    op_descriptor: OpDescriptor
    cb_info: Dict[int, CBInfo] = field(default_factory=dict)  # original_index -> CBInfo
    input_cb_indices: Set[int] = field(default_factory=set)  # CB indices used as inputs
    output_cb_indices: Set[int] = field(default_factory=set)  # CB indices used as outputs
    # Remapped CB indices: original -> new
    cb_remapping: Dict[int, int] = field(default_factory=dict)


class CBRemapper:
    """
    Manages CB index remapping across sequential phases.

    Ensures:
    1. No CB index conflicts between phases
    2. Output CB of phase N can be reused as input CB of phase N+1
    3. Unused CBs are freed for reuse by later phases

    The 32 available CBs (0-31) are allocated to phases, with careful
    tracking of which CBs hold "live" data needed by the next phase.
    """

    NUM_CBS = 32

    def __init__(self):
        self.allocated: Set[int] = set()  # Currently allocated CB indices
        self.phase_allocations: List[Dict[int, int]] = []  # Per-phase: original -> remapped
        self.live_data_cbs: Set[int] = set()  # CBs holding data for next phase

    def allocate_for_phase(
        self,
        phase: PhaseInfo,
        chain_from_previous: bool = False,
        previous_output_cb: Optional[int] = None,
        target_input_cb: Optional[int] = None,
        shared_cb_mapping: Optional[Dict[int, int]] = None,
        phantom_cb_indices: Optional[Set[int]] = None,
    ) -> Dict[int, int]:
        """
        Allocate remapped CB indices for a phase.

        For phase 0 (first phase), keep original CB indices - no remapping needed.
        For phases 1+, remap to avoid conflicts with previous phases.

        Args:
            phase: Phase information with CB requirements
            chain_from_previous: If True, connect previous output to this phase's input
            previous_output_cb: The remapped CB index holding previous phase's output
            target_input_cb: The original CB index that should receive the chained input
            shared_cb_mapping: Optional mapping of original CB index -> physical CB index
                              to reuse from a previous phase (e.g., for shared gamma/beta/scratch)
            phantom_cb_indices: CB indices referenced in named args but without descriptors.
                               Identity-mapped and marked as allocated.

        Returns:
            Mapping of original CB index -> remapped CB index
        """
        remapping: Dict[int, int] = {}

        # Phase 0: keep original indices (identity mapping)
        if not chain_from_previous:
            for original_idx in phase.cb_info.keys():
                remapping[original_idx] = original_idx
                self.allocated.add(original_idx)
            # Mark phantom CBs as allocated and add identity mapping
            if phantom_cb_indices:
                for phantom_idx in phantom_cb_indices:
                    self.allocated.add(phantom_idx)
                    remapping[phantom_idx] = phantom_idx
            self.phase_allocations.append(remapping)
            return remapping

        # Phases 1+: remap to avoid conflicts
        # If chaining from previous phase, reuse the output CB as this phase's input
        if previous_output_cb is not None and target_input_cb is not None:
            remapping[target_input_cb] = previous_output_cb
            self.allocated.add(previous_output_cb)

        # Apply shared CB mappings (reuse phase 0's physical CBs)
        if shared_cb_mapping:
            for original_idx, physical_idx in shared_cb_mapping.items():
                if original_idx not in remapping and original_idx in phase.cb_info:
                    remapping[original_idx] = physical_idx

        # Allocate new CBs for remaining requirements
        for original_idx, cb_info in phase.cb_info.items():
            if original_idx in remapping:
                continue  # Already mapped

            # Find a free CB index
            new_idx = self._find_free_cb()
            remapping[original_idx] = new_idx
            self.allocated.add(new_idx)

        self.phase_allocations.append(remapping)
        return remapping

    def finish_phase(self, remapping: Dict[int, int], output_cb_original: Optional[int] = None):
        """
        Mark a phase as complete.

        For fused compute kernels where all phases run in the same kernel,
        we do NOT free any CBs - they're all in use simultaneously.

        Args:
            remapping: The CB remapping for this phase
            output_cb_original: The original index of the output CB (to preserve for next phase)
        """
        output_cb_remapped = remapping.get(output_cb_original) if output_cb_original else None

        # For fused compute kernels, keep ALL CBs allocated since all phases
        # run in the same kernel and all CBs are in scope.
        # Only mark the output CB as live data for chaining.
        if output_cb_remapped is not None:
            self.live_data_cbs.add(output_cb_remapped)

    def _find_free_cb(self) -> int:
        """Find the next available CB index."""
        for i in range(self.NUM_CBS):
            if i not in self.allocated:
                return i
        raise RuntimeError(f"No free CBs available! All {self.NUM_CBS} are in use.")

    def get_total_cbs_used(self) -> int:
        """Get the total number of unique CB indices used across all phases."""
        all_used = set()
        for mapping in self.phase_allocations:
            all_used.update(mapping.values())
        return len(all_used)


# =============================================================================
# Program Descriptor Analysis
# =============================================================================


def extract_cb_names_from_kernel(kernel_desc: "ttnn.KernelDescriptor") -> Dict[str, int]:
    """
    Extract CB name -> index mapping from kernel's named compile-time args.

    This allows the infrastructure to understand what each CB is used for,
    enabling intelligent remapping of input/output CBs when chaining.

    Returns:
        Dict mapping CB names (e.g., "cb_in", "cb_out") to their indices
    """
    cb_names = {}
    if hasattr(kernel_desc, "named_compile_time_args"):
        for name, value in kernel_desc.named_compile_time_args:
            if name.startswith("cb_"):
                cb_names[name] = value
    return cb_names


def extract_cb_info(descriptor: "ttnn.ProgramDescriptor") -> Dict[int, CBInfo]:
    """
    Extract CB information from a ProgramDescriptor.

    Returns a dict mapping CB index -> CBInfo.

    Note: data_format may be None if the Python binding doesn't properly
    expose the tt::DataFormat enum.
    """
    cb_info = {}

    for cb_desc in descriptor.cbs:
        for fmt_desc in cb_desc.format_descriptors:
            cb_idx = fmt_desc.buffer_index
            # Try to get data_format, but it may fail due to binding issues
            try:
                data_format = fmt_desc.data_format
            except (TypeError, AttributeError):
                data_format = None

            cb_info[cb_idx] = CBInfo(
                original_index=cb_idx,
                total_size=cb_desc.total_size,
                data_format=data_format,
                page_size=fmt_desc.page_size,
                core_ranges=cb_desc.core_ranges,
            )

    return cb_info


def analyze_kernel_cb_usage(
    kernel_desc: "ttnn.KernelDescriptor",
) -> Tuple[Set[int], Set[int]]:
    """
    Analyze a kernel descriptor to determine which CBs are inputs vs outputs.

    This is a heuristic based on common patterns:
    - Reader kernels typically write to low-numbered CBs (inputs)
    - Writer kernels typically read from CB 16 (output)
    - Compute kernels read from inputs and write to outputs

    Returns:
        (input_cb_indices, output_cb_indices)
    """
    # TODO: More sophisticated analysis based on kernel source or annotations
    # For now, use convention: CB 16 is typically output, others are inputs
    input_cbs = set()
    output_cbs = set()

    # Check kernel config type
    config = kernel_desc.config
    if isinstance(config, ttnn.ComputeConfigDescriptor):
        # Compute kernel - need to analyze compile-time args or source
        # Default assumption: reads from 0-15, writes to 16+
        pass
    elif isinstance(config, ttnn.DataMovementConfigDescriptor):
        processor = config.processor
        if processor == ttnn.DataMovementProcessor.RISCV_1:
            # Reader - writes to CBs (makes them available)
            pass
        else:
            # Writer - reads from CBs
            output_cbs.add(16)  # Convention: output CB

    return input_cbs, output_cbs


# =============================================================================
# Kernel Modification for CB Remapping
# =============================================================================


def remap_kernel_cb_indices_inplace(
    kernel_desc: "ttnn.KernelDescriptor",
    cb_remapping: Dict[int, int],
) -> None:
    """
    Remap CB indices in a kernel descriptor IN PLACE.

    This modifies the original descriptor rather than creating a copy.
    Use this when the original descriptor won't be reused (e.g., after fusion).

    Args:
        kernel_desc: Kernel descriptor to modify
        cb_remapping: Mapping of original CB index -> new CB index
    """
    # Remap via named compile-time args (preferred method)
    if hasattr(kernel_desc, "named_compile_time_args"):
        named_args = list(kernel_desc.named_compile_time_args)
        for i, (name, value) in enumerate(named_args):
            if value in cb_remapping:
                named_args[i] = (name, cb_remapping[value])
        kernel_desc.named_compile_time_args = named_args


def remap_kernel_cb_indices(
    kernel_desc: "ttnn.KernelDescriptor",
    cb_remapping: Dict[int, int],
    cb_arg_positions: Optional[Dict[int, int]] = None,
    cb_defines: Optional[Dict[int, str]] = None,
) -> "ttnn.KernelDescriptor":
    """
    Create a modified kernel descriptor with remapped CB indices.

    There are three ways kernels can receive CB indices:
    1. Named compile-time args: CB index is passed as a named arg (e.g., {"cb_in": 0})
       This is the PREFERRED method as it's explicit and self-documenting.
    2. Compile-time args: CB index is at a specific position in compile_time_args
    3. Defines: CB index is passed via a #define (e.g., #define CB_IN 0)

    Args:
        kernel_desc: Original kernel descriptor
        cb_remapping: Mapping of original CB index -> new CB index
        cb_arg_positions: Maps original CB index -> position in compile_time_args
        cb_defines: Maps original CB index -> define name (e.g., {0: "CB_IN"})

    Returns:
        The kernel descriptor with remapped CB indices (may be same object if in-place)
    """
    # Try deepcopy first for mock objects in tests
    try:
        new_desc = deepcopy(kernel_desc)
    except TypeError:
        # C++ binding objects can't be deep copied - modify in place
        # This is safe because fused descriptors replace the originals
        new_desc = kernel_desc

    # Remap via named compile-time args (preferred method)
    # Named args like cb_in, cb_out, etc. are automatically remapped
    if hasattr(new_desc, "named_compile_time_args"):
        named_args = list(new_desc.named_compile_time_args)
        for i, (name, value) in enumerate(named_args):
            # Check if this named arg's value is a CB index that needs remapping
            if value in cb_remapping:
                named_args[i] = (name, cb_remapping[value])
        new_desc.named_compile_time_args = named_args

    # Remap via compile-time args (legacy method)
    if cb_arg_positions:
        new_args = list(new_desc.compile_time_args)
        for original_cb, arg_pos in cb_arg_positions.items():
            if original_cb in cb_remapping and arg_pos < len(new_args):
                new_args[arg_pos] = cb_remapping[original_cb]
        new_desc.compile_time_args = new_args

    # Remap via defines (legacy method)
    if cb_defines:
        new_defines = list(new_desc.defines)
        for original_cb, define_name in cb_defines.items():
            if original_cb in cb_remapping:
                # Update or add the define
                new_value = str(cb_remapping[original_cb])
                found = False
                for i, (name, value) in enumerate(new_defines):
                    if name == define_name:
                        new_defines[i] = (name, new_value)
                        found = True
                        break
                if not found:
                    new_defines.append((define_name, new_value))
        new_desc.defines = new_defines

    return new_desc


def _copy_cb_descriptor(cb_desc: "ttnn.CBDescriptor") -> "ttnn.CBDescriptor":
    """
    Create a copy of a CBDescriptor.

    Since C++ binding objects can't be deep copied, we create a new descriptor
    and copy each field manually.
    """
    new_desc = ttnn.CBDescriptor()
    new_desc.total_size = cb_desc.total_size
    new_desc.core_ranges = cb_desc.core_ranges
    # Format descriptors need to be copied separately
    new_desc.format_descriptors = []
    return new_desc


def _copy_format_descriptor(fmt_desc: "ttnn.CBFormatDescriptor") -> "ttnn.CBFormatDescriptor":
    """
    Create a copy of a CBFormatDescriptor.
    """
    new_fmt = ttnn.CBFormatDescriptor()
    new_fmt.buffer_index = fmt_desc.buffer_index
    new_fmt.page_size = fmt_desc.page_size
    try:
        new_fmt.data_format = fmt_desc.data_format
    except (TypeError, AttributeError):
        pass  # data_format may not be copyable
    return new_fmt


def remap_cb_descriptors(
    cb_descs: List["ttnn.CBDescriptor"],
    cb_remapping: Dict[int, int],
) -> List["ttnn.CBDescriptor"]:
    """
    Create remapped CB descriptors with new indices.

    Args:
        cb_descs: Original CB descriptors
        cb_remapping: Mapping of original CB index -> new CB index

    Returns:
        New list of CB descriptors with remapped indices
    """
    new_descs = []

    for cb_desc in cb_descs:
        # Try deepcopy first for mock objects, fall back to in-place modification
        try:
            new_cb_desc = deepcopy(cb_desc)
            is_copy = True
        except TypeError:
            # C++ binding objects can't be deep copied - use original
            new_cb_desc = cb_desc
            is_copy = False

        # Remap format descriptors - modify buffer_index in place for C++ objects
        if is_copy:
            # For copies, we can freely modify
            new_formats = []
            for fmt_desc in new_cb_desc.format_descriptors:
                original_idx = fmt_desc.buffer_index
                if original_idx in cb_remapping:
                    fmt_desc.buffer_index = cb_remapping[original_idx]
                new_formats.append(fmt_desc)
            new_cb_desc.format_descriptors = new_formats
        else:
            # For C++ objects, modify buffer_index directly on format descriptors
            for fmt_desc in new_cb_desc.format_descriptors:
                original_idx = fmt_desc.buffer_index
                if original_idx in cb_remapping:
                    fmt_desc.buffer_index = cb_remapping[original_idx]

        new_descs.append(new_cb_desc)

    return new_descs


# =============================================================================
# Sequential Chain Builder
# =============================================================================


@dataclass
class PhaseConnection:
    """Describes how phases are connected."""

    source_phase_idx: int
    source_output_cb: int  # Original CB index of output in source phase
    target_phase_idx: int
    target_input_cb: int  # Original CB index of input in target phase


class SequentialChainBuilder:
    """
    Builds a fused ProgramDescriptor from a sequence of OpDescriptors.

    The builder:
    1. Collects OpDescriptors for each phase
    2. Analyzes their CB usage
    3. Remaps CBs to avoid conflicts
    4. Chains output→input between phases
    5. Generates a merged ProgramDescriptor

    Requirements for kernels to be chainable:
    - CB indices should be configurable via compile-time args or defines
    - Input/output CB conventions should be documented
    - Kernels should properly synchronize via CB push/pop
    """

    def __init__(self):
        self.phases: List[PhaseInfo] = []
        self.connections: List[PhaseConnection] = []
        self._built = False

    def add_phase(
        self,
        op_descriptor: OpDescriptor,
        input_from_previous: bool = False,
        input_cb: Optional[int] = None,
        output_cb: Optional[int] = None,
    ) -> "SequentialChainBuilder":
        """
        Add a phase to the sequential chain.

        Args:
            op_descriptor: The OpDescriptor for this phase
            input_from_previous: If True, this phase receives input from previous phase's output
            input_cb: The CB index that receives input (default: 0 for first input)
            output_cb: The CB index that produces output (default: 16)

        Returns:
            self for method chaining
        """
        phase_idx = len(self.phases)

        # Extract CB info from the descriptor
        cb_info = extract_cb_info(op_descriptor.descriptor)

        # Create phase info
        phase = PhaseInfo(
            phase_idx=phase_idx,
            op_descriptor=op_descriptor,
            cb_info=cb_info,
        )

        # Determine input/output CBs
        # Default conventions: input CB 0, output CB 16
        phase.input_cb_indices.add(input_cb if input_cb is not None else 0)
        phase.output_cb_indices.add(output_cb if output_cb is not None else 16)

        self.phases.append(phase)

        # Set up connection from previous phase
        if input_from_previous and phase_idx > 0:
            prev_phase = self.phases[phase_idx - 1]
            # Get the output CB of previous phase (default: 16)
            prev_output = list(prev_phase.output_cb_indices)[0] if prev_phase.output_cb_indices else 16
            target_input = input_cb if input_cb is not None else 0

            self.connections.append(
                PhaseConnection(
                    source_phase_idx=phase_idx - 1,
                    source_output_cb=prev_output,
                    target_phase_idx=phase_idx,
                    target_input_cb=target_input,
                )
            )

        return self

    def set_phase_io(
        self,
        phase_idx: int,
        input_cbs: Optional[List[int]] = None,
        output_cbs: Optional[List[int]] = None,
    ) -> "SequentialChainBuilder":
        """
        Explicitly set input/output CBs for a phase.

        Use this when the default conventions (input=0, output=16) don't apply.

        Args:
            phase_idx: Index of the phase to configure
            input_cbs: List of CB indices used as inputs
            output_cbs: List of CB indices used as outputs

        Returns:
            self for method chaining
        """
        if phase_idx >= len(self.phases):
            raise ValueError(f"Phase index {phase_idx} out of range")

        phase = self.phases[phase_idx]
        if input_cbs is not None:
            phase.input_cb_indices = set(input_cbs)
        if output_cbs is not None:
            phase.output_cb_indices = set(output_cbs)

        return self

    def connect_phases(
        self,
        source_phase: int,
        source_output_cb: int,
        target_phase: int,
        target_input_cb: int,
    ) -> "SequentialChainBuilder":
        """
        Explicitly connect two phases.

        The output CB of source_phase will be connected to the input CB of target_phase.
        The data written to source_output_cb will be readable from target_input_cb.

        Args:
            source_phase: Phase index that produces output
            source_output_cb: CB index (original) that holds output
            target_phase: Phase index that consumes input
            target_input_cb: CB index (original) that receives input

        Returns:
            self for method chaining
        """
        self.connections.append(
            PhaseConnection(
                source_phase_idx=source_phase,
                source_output_cb=source_output_cb,
                target_phase_idx=target_phase,
                target_input_cb=target_input_cb,
            )
        )
        return self

    def build(
        self,
        cb_arg_positions: Optional[Dict[int, Dict[int, int]]] = None,
        cb_defines: Optional[Dict[int, Dict[int, str]]] = None,
    ) -> OpDescriptor:
        """
        Build the fused OpDescriptor from the chain.

        Args:
            cb_arg_positions: Per-phase mapping of CB index -> compile-time arg position
                              e.g., {0: {0: 5, 16: 6}} means phase 0's CB 0 is at arg[5]
            cb_defines: Per-phase mapping of CB index -> define name
                        e.g., {0: {0: "CB_IN", 16: "CB_OUT"}}

        Returns:
            Fused OpDescriptor that executes all phases sequentially

        Raises:
            ValueError: If chain is empty or already built
        """
        if self._built:
            raise ValueError("Chain has already been built")
        if not self.phases:
            raise ValueError("Chain has no phases")

        self._built = True

        # For single-phase chains, return as-is (no remapping needed)
        if len(self.phases) == 1:
            return self.phases[0].op_descriptor

        # Multi-phase: perform CB remapping and merge descriptors
        return self._build_fused_descriptor(cb_arg_positions, cb_defines)

    def _build_fused_descriptor(
        self,
        cb_arg_positions: Optional[Dict[int, Dict[int, int]]],
        cb_defines: Optional[Dict[int, Dict[int, str]]],
    ) -> OpDescriptor:
        """
        Build the fused descriptor with true kernel fusion.

        Instead of merging all kernels (which causes NOC conflicts), this creates:
        1. A single reader kernel (from phase 0, with remapped CBs)
        2. A single fused compute kernel (chains all phase computations)
        3. A single writer kernel (from last phase, with remapped CBs)
        """
        # Validate fp32_dest_acc_en consistency across phases before fusion.
        _validate_fp32_consistency([phase.op_descriptor for phase in self.phases])

        # Validate that CB requirements won't overflow the 32 available CBs
        _validate_cb_capacity(self.phases)

        remapper = CBRemapper()

        # Only reader-filled constant CBs are shared across phases.  All other
        # CBs (input, output, scratch/intermediate) get unique per-phase
        # allocations because they may differ in page_size (e.g. Float32 vs
        # Float16_b when fp32_dest_acc_en differs) or blk-dependent tile counts.
        READER_FILLED_CB_INDICES = {2, 3, 5, 6}  # cb_scaler, cb_eps, cb_gamma, cb_beta
        per_phase_original_indices: Set[int] = set()
        for phase in self.phases:
            for cb_idx in phase.cb_info.keys():
                if cb_idx not in READER_FILLED_CB_INDICES:
                    per_phase_original_indices.add(cb_idx)

        # Detect phantom CB indices: those referenced in named args but without descriptors.
        # These must be identity-mapped to prevent collisions.
        phantom_cb_indices: Set[int] = set()
        for phase in self.phases:
            for kernel_desc in phase.op_descriptor.descriptor.kernels:
                if hasattr(kernel_desc, "named_compile_time_args"):
                    for name, value in kernel_desc.named_compile_time_args:
                        if name.startswith("cb_") and isinstance(value, int):
                            if value not in phase.cb_info:
                                phantom_cb_indices.add(value)

        # First pass: allocate CB indices for all phases
        phase0_remapping: Dict[int, int] = {}
        for i, phase in enumerate(self.phases):
            # Find if this phase receives input from a previous phase
            chain_from_prev = False
            prev_output_cb = None
            target_input_cb = None

            for conn in self.connections:
                if conn.target_phase_idx == i:
                    chain_from_prev = True
                    source_phase = self.phases[conn.source_phase_idx]
                    prev_output_cb = source_phase.cb_remapping.get(conn.source_output_cb)
                    target_input_cb = conn.target_input_cb
                    break

            # For phases 1+, only reader-filled constant CBs are shared.
            # All others get unique per-phase allocations.
            shared_cb_mapping: Optional[Dict[int, int]] = None
            if i > 0 and phase0_remapping:
                shared_cb_mapping = {}
                for orig_idx, phys_idx in phase0_remapping.items():
                    if orig_idx not in per_phase_original_indices:
                        shared_cb_mapping[orig_idx] = phys_idx

            # Allocate CBs for this phase
            # For phase 0, also pass phantom CB indices to mark them as allocated
            phantoms = phantom_cb_indices if i == 0 else None
            remapping = remapper.allocate_for_phase(
                phase,
                chain_from_previous=chain_from_prev,
                previous_output_cb=prev_output_cb,
                target_input_cb=target_input_cb,
                shared_cb_mapping=shared_cb_mapping,
                phantom_cb_indices=phantoms,
            )
            phase.cb_remapping = remapping

            if i == 0:
                phase0_remapping = dict(remapping)

            # Mark phase complete
            output_cb = list(phase.output_cb_indices)[0] if phase.output_cb_indices else None
            remapper.finish_phase(remapping, output_cb)

        # Second pass: collect CB descriptors and classify kernels by type
        all_cbs = []
        all_semaphores = []
        all_input_tensors = []
        output_tensor = None

        # Track which CB indices have been added to avoid duplicates
        added_cb_indices: Set[int] = set()

        # Classify kernels by type for each phase
        phase_kernels: List[Dict[str, Any]] = []  # List of {reader, writer, compute} per phase

        for i, phase in enumerate(self.phases):
            orig_desc = phase.op_descriptor.descriptor

            # Remap CB descriptors.  For shared CBs (e.g. gamma/beta) that appear
            # in multiple phases, keep the descriptor with the larger total_size so
            # the CB is big enough for the phase with the largest blk.
            remapped_cbs = remap_cb_descriptors(list(orig_desc.cbs), phase.cb_remapping)
            for cb_desc in remapped_cbs:
                if cb_desc.format_descriptors:
                    cb_idx = cb_desc.format_descriptors[0].buffer_index
                    if cb_idx not in added_cb_indices:
                        all_cbs.append(cb_desc)
                        added_cb_indices.add(cb_idx)
                    else:
                        # Shared CB: use the larger total_size across phases
                        for j, existing_cb in enumerate(all_cbs):
                            if (
                                existing_cb.format_descriptors
                                and existing_cb.format_descriptors[0].buffer_index == cb_idx
                            ):
                                if cb_desc.total_size > existing_cb.total_size:
                                    all_cbs[j] = cb_desc
                                break
                else:
                    all_cbs.append(cb_desc)

            # Classify and remap kernel descriptors
            phase_cb_args = cb_arg_positions.get(i) if cb_arg_positions else None
            phase_cb_defs = cb_defines.get(i) if cb_defines else None

            kernels_by_type: Dict[str, Any] = {"reader": None, "writer": None, "compute": None}
            for kernel_desc in orig_desc.kernels:
                remapped_kernel = remap_kernel_cb_indices(
                    kernel_desc,
                    phase.cb_remapping,
                    phase_cb_args,
                    phase_cb_defs,
                )
                kernel_type = _classify_kernel(remapped_kernel)
                kernels_by_type[kernel_type] = remapped_kernel

            phase_kernels.append(kernels_by_type)

            # Collect semaphores
            all_semaphores.extend(list(orig_desc.semaphores))

            # Collect input tensors (but not intermediate outputs)
            if i == 0:
                all_input_tensors.extend(phase.op_descriptor.input_tensors)
            else:
                # For subsequent phases, only add non-chained inputs
                for tensor in phase.op_descriptor.input_tensors:
                    if tensor not in all_input_tensors:
                        all_input_tensors.append(tensor)

            # The final output is from the last phase
            if i == len(self.phases) - 1:
                output_tensor = phase.op_descriptor.output_tensors[0] if phase.op_descriptor.output_tensors else None

        # Build fused kernels - for multi-phase chains, this generates
        # combined kernel source code that runs all phases sequentially.
        fused_kernels = _build_fused_kernel_list(phase_kernels, self.phases)

        # Create the merged ProgramDescriptor
        merged_descriptor = ttnn.ProgramDescriptor()
        merged_descriptor.kernels = fused_kernels
        merged_descriptor.cbs = all_cbs
        merged_descriptor.semaphores = all_semaphores

        return OpDescriptor(
            descriptor=merged_descriptor,
            input_tensors=all_input_tensors,
            output_tensors=[output_tensor] if output_tensor else [],
        )


def _build_fused_kernel_list(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
) -> List["ttnn.KernelDescriptor"]:
    """
    Build the list of kernels for the fused descriptor.

    For single-phase chains, returns all kernels from that phase.
    For multi-phase chains, generates truly fused kernels by combining
    source code from all phases.
    """
    if not phase_kernels:
        return []

    if len(phase_kernels) == 1:
        # Single phase - just return its kernels
        kernels = []
        for kernel_type in ["reader", "writer", "compute"]:
            if phase_kernels[0][kernel_type] is not None:
                kernels.append(phase_kernels[0][kernel_type])
        return kernels

    # Multi-phase: generate truly fused kernels
    return _generate_fused_kernels(phase_kernels, phases)


def _classify_kernel(kernel_desc: "ttnn.KernelDescriptor") -> str:
    """
    Classify a kernel as reader, writer, or compute based on its config type.
    """
    config = kernel_desc.config
    if isinstance(config, ttnn.ComputeConfigDescriptor):
        return "compute"
    elif isinstance(config, ttnn.ReaderConfigDescriptor):
        return "reader"
    elif isinstance(config, ttnn.WriterConfigDescriptor):
        return "writer"
    elif isinstance(config, ttnn.DataMovementConfigDescriptor):
        # Check processor to determine reader vs writer
        if config.processor == ttnn.DataMovementProcessor.RISCV_1:
            return "reader"
        else:
            return "writer"
    return "unknown"


def _read_kernel_source_from_descriptor(kernel_desc: "ttnn.KernelDescriptor") -> str:
    """Read kernel source code from a kernel descriptor."""
    if kernel_desc.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE:
        return kernel_desc.kernel_source
    else:
        # It's a file path - read the file
        # The path is relative to the tt-metal root
        import os

        # Try common base paths
        base_paths = [
            "",  # Current directory
            os.environ.get("TT_METAL_HOME", ""),
            "/localdev/rmiller/tt-metal",  # Development path
        ]

        for base in base_paths:
            full_path = os.path.join(base, kernel_desc.kernel_source) if base else kernel_desc.kernel_source
            if os.path.exists(full_path):
                with open(full_path, "r") as f:
                    return f.read()

        # If we can't find it, return empty string
        return ""


def _generate_fused_kernels(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
) -> List["ttnn.KernelDescriptor"]:
    """
    Generate truly fused kernels by combining source code from all phases.

    Returns a list of 3 kernel descriptors: [fused_reader, fused_writer, fused_compute]
    """
    if not phase_kernels:
        return []

    fused_kernels = []

    # Generate fused reader kernel
    fused_reader = _generate_fused_reader(phase_kernels, phases)
    if fused_reader is not None:
        fused_kernels.append(fused_reader)

    # Generate fused writer kernel
    fused_writer = _generate_fused_writer(phase_kernels, phases)
    if fused_writer is not None:
        fused_kernels.append(fused_writer)

    # Generate fused compute kernel
    fused_compute = _generate_fused_compute(phase_kernels, phases)
    if fused_compute is not None:
        fused_kernels.append(fused_compute)

    return fused_kernels


def _generate_fused_reader(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
) -> Optional["ttnn.KernelDescriptor"]:
    """
    Generate a fused reader kernel.

    The fused reader:
    - Phase 0: Reads input tensor from DRAM, generates scaler/eps tiles, reads gamma/beta
    - Phase 1+: Reads gamma/beta from DRAM (input comes from previous phase's output CB)

    For single phase, use phase 0's reader directly.
    For multi-phase, use phase 0's reader with FILE_PATH source type (preserves include paths).

    Note: We use phase 0's file-based reader instead of generating source code because
    the LayerNorm reader includes relative headers (layernorm_dataflow_utils.h) that
    require specific include paths set up by the build system. Generating SOURCE_CODE
    would lose these include paths.

    Future enhancement: To fully support multi-phase fusion with all gamma/beta reads,
    either:
    1. Add the layernorm kernel directory to the compiler's include paths
    2. Use absolute includes in the generated source
    3. Extend the runtime args mechanism to pass gamma/beta for phases 1+
    """
    if not phase_kernels or phase_kernels[0]["reader"] is None:
        return None

    # Single phase - use as-is
    if len(phase_kernels) == 1:
        return phase_kernels[0]["reader"]

    # Multi-phase: Use phase 0's reader directly (preserves FILE_PATH and include paths)
    # This means phases 1+ won't have their gamma/beta read from DRAM - they must either:
    # 1. Use the same gamma/beta as phase 0 (common in many models)
    # 2. Be extended later to read additional gamma/beta via extended runtime args
    #
    # For now, this enables the basic fusion case where all phases use the same weights.
    base_reader = phase_kernels[0]["reader"]

    # Update the named compile-time args to include phase-prefixed CB names
    # This allows the reader to reference remapped CBs for all phases
    merged_named_args = _merge_named_compile_time_args_for_readers(phase_kernels)

    # Modify the reader's named args in place (C++ binding limitation prevents deepcopy)
    base_reader.named_compile_time_args = merged_named_args

    return base_reader


def _generate_fused_reader_source(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
) -> Optional[str]:
    """
    Generate fused reader source that reads inputs for all phases.

    Phase 0's reader reads the main input tensor + scaler/eps + gamma/beta.
    For phases 1+, we add code to read their gamma/beta from additional runtime args.
    """
    if not phase_kernels or phase_kernels[0]["reader"] is None:
        return None

    # Read phase 0's reader source as the base
    base_source = _read_kernel_source_from_descriptor(phase_kernels[0]["reader"])
    if not base_source:
        return None

    # For phases 1+, we need to add gamma/beta reads
    # The runtime args for phases 1+ start after phase 0's args (index 10+)
    additional_reads = []
    runtime_arg_offset = 10  # Phase 0 uses args 0-9

    for i in range(1, len(phase_kernels)):
        phase = phases[i]
        pk = phase_kernels[i]
        if pk["reader"] is None:
            continue

        # Check if this phase has gamma/beta by looking at its reader's defines
        has_gamma = False
        has_beta = False
        for name, value in pk["reader"].defines:
            if name == "FUSE_GAMMA":
                has_gamma = True
            if name == "FUSE_BETA":
                has_beta = True

        if has_gamma or has_beta:
            # Get remapped CB indices for this phase
            gamma_cb = phase.cb_remapping.get(5, 5)  # cb_gamma is typically index 5
            beta_cb = phase.cb_remapping.get(6, 6)  # cb_beta is typically index 6

            if has_gamma:
                additional_reads.append(
                    f"""
    // Read gamma for phase {i}
    {{
        uint32_t phase{i}_gamma_addr = get_arg_val<uint32_t>({runtime_arg_offset});
        constexpr uint32_t phase{i}_cb_gamma = get_named_compile_time_arg_val("phase{i}_cb_gamma");
        const uint32_t phase{i}_gamma_tile_bytes = get_tile_size(phase{i}_cb_gamma);
        const auto phase{i}_addrg = TensorAccessor(gamma_args, phase{i}_gamma_addr, phase{i}_gamma_tile_bytes);
        for (auto block : generic::blocks(Wt, blk)) {{
            layernorm_dataflow_utils::read_block_to_cb(
                phase{i}_cb_gamma, phase{i}_addrg, phase{i}_gamma_tile_bytes, block.start(), block);
        }}
    }}"""
                )
                runtime_arg_offset += 1

            if has_beta:
                additional_reads.append(
                    f"""
    // Read beta for phase {i}
    {{
        uint32_t phase{i}_beta_addr = get_arg_val<uint32_t>({runtime_arg_offset});
        constexpr uint32_t phase{i}_cb_beta = get_named_compile_time_arg_val("phase{i}_cb_beta");
        const uint32_t phase{i}_beta_tile_bytes = get_tile_size(phase{i}_cb_beta);
        const auto phase{i}_addrb = TensorAccessor(beta_args, phase{i}_beta_addr, phase{i}_beta_tile_bytes);
        for (auto block : generic::blocks(Wt, blk)) {{
            layernorm_dataflow_utils::read_block_to_cb(
                phase{i}_cb_beta, phase{i}_addrb, phase{i}_beta_tile_bytes, block.start(), block);
        }}
    }}"""
                )
                runtime_arg_offset += 1

    if not additional_reads:
        # No additional reads needed, use base source with CB remapping
        return _apply_cb_remapping_to_source(base_source, phases[0].cb_remapping, phase_idx=0)

    # Insert additional reads before the end of kernel_main
    # Find the closing brace of kernel_main and insert before it
    lines = base_source.split("\n")
    result_lines = []
    inserted = False

    # Apply phase 0 CB remapping to the base source first
    remapped_base = _apply_cb_remapping_to_source(base_source, phases[0].cb_remapping, phase_idx=0)
    lines = remapped_base.split("\n")

    brace_depth = 0
    in_kernel_main = False

    for line in lines:
        stripped = line.strip()

        if "void kernel_main()" in line:
            in_kernel_main = True

        if in_kernel_main:
            brace_depth += stripped.count("{") - stripped.count("}")

            # Insert additional reads just before the final closing brace
            if brace_depth == 1 and stripped == "}" and not inserted:
                # Insert the additional reads
                for read_code in additional_reads:
                    result_lines.append(read_code)
                inserted = True

        result_lines.append(line)

    return "\n".join(result_lines)


def _merge_named_compile_time_args_for_readers(
    phase_kernels: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """
    Merge named compile-time args from all phases' readers.

    Phase 0's args are used AS-IS (no prefix) because we use phase 0's
    file-based reader which expects the original names.
    Phases 1+ args get phase-prefixed (for future fused reader source).

    All reader-referenced CBs are shared with Phase 0's physical indices since
    Phase 0's reader fills them once and the data persists (never popped).
    """
    merged = []
    phase0_values = {}

    for i, pk in enumerate(phase_kernels):
        if pk["reader"] is None:
            continue

        for name, value in pk["reader"].named_compile_time_args:
            if i == 0:
                merged.append((name, value))
                phase0_values[name] = value
            else:
                # Phase 1+: prefix, but reuse Phase 0's physical CB indices
                # for all CBs that the reader fills (gamma, beta, eps, scaler, etc.)
                prefixed_name = f"phase{i}_{name}"
                if name in phase0_values:
                    merged.append((prefixed_name, phase0_values[name]))
                else:
                    merged.append((prefixed_name, value))

    return merged


def _merge_reader_runtime_args(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
) -> Any:
    """
    Merge runtime args from all phases' readers.

    Phase 0's args are used as the base (indices 0-9).
    Phases 1+ append their gamma/beta addresses (indices 10+).

    Returns the merged runtime args in the same format as the original.
    """
    if not phase_kernels or phase_kernels[0]["reader"] is None:
        return []

    base_args = phase_kernels[0]["reader"].runtime_args

    # For single phase, just return base args
    if len(phase_kernels) == 1:
        return base_args

    # For multi-phase, we need to extend each core's args
    # Runtime args are a list of (CoreCoord, vector<uint32_t>) pairs
    # We can't easily iterate them (C++ binding limitation), so we return base for now
    # The actual extension would need to happen in C++ or via a different mechanism

    # TODO: Properly merge runtime args - for now, return base
    # This means phases 1+ won't have their gamma/beta read, but the infrastructure is in place
    return base_args


def _generate_fused_writer(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
) -> Optional["ttnn.KernelDescriptor"]:
    """
    Generate a fused writer kernel.

    Only the last phase's output needs to be written to DRAM.
    The writer's blk naturally matches the last phase's compute blk since
    both come from the same factory. No normalization needed — each phase
    uses its own independent blk, and the writer matches the last phase.
    """
    if not phase_kernels:
        return None

    last_phase_idx = len(phase_kernels) - 1
    if phase_kernels[last_phase_idx]["writer"] is None:
        return None

    return phase_kernels[last_phase_idx]["writer"]


def _generate_fused_compute(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
) -> Optional["ttnn.KernelDescriptor"]:
    """
    Generate a fused compute kernel that chains all phase computations.

    For LayerNorm/RMSNorm kernels, the fusion approach is:
    1. Use phase 0's compute kernel as the base (it reads from remapped input CB)
    2. The output CB is remapped to chain to the next phase's input
    3. Each phase's compute runs to completion before the next starts

    Since the layernorm kernel structure is complex and self-contained, we use
    a simpler approach: keep the first phase's kernel but modify its named
    compile-time args to use the final output CB.

    For true multi-phase fusion, we would need to generate kernel source that
    loops over all phases, but this requires deeper kernel modification.
    """
    if not phase_kernels or phase_kernels[0]["compute"] is None:
        return None

    # For now, use a simpler approach: chain by running phases sequentially
    # Each phase reads from its input CB (which is previous output) and writes to its output CB

    # The issue is that multiple compute kernels on the same core with the same name conflict.
    # We need to either:
    # 1. Generate a single fused kernel (complex)
    # 2. Use different kernel names (not straightforward)
    # 3. Run phases as separate programs (defeats fusion purpose)

    # For initial implementation, return just the first phase's compute kernel
    # This demonstrates the CB remapping works, even if full fusion isn't complete
    base_compute = phase_kernels[0]["compute"]

    # If there's only one phase, just use it directly
    if len(phase_kernels) == 1:
        return base_compute

    # For multiple phases, we need true kernel fusion
    # For now, let's generate a kernel that calls each phase sequentially
    # by including all the phase source code as separate functions

    fused_source = _generate_multi_phase_compute_source(phase_kernels, phases)

    if fused_source is None:
        # Fall back to just the first phase's compute
        return base_compute

    # Create a new kernel descriptor with the fused source
    fused_compute = ttnn.KernelDescriptor()
    fused_compute.kernel_source = fused_source
    fused_compute.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    fused_compute.core_ranges = base_compute.core_ranges
    # Merge compile-time args from all phases
    fused_compute.compile_time_args = _merge_compile_time_args(phase_kernels)
    # Combine named compile-time args from all phases with phase prefixes
    fused_compute.named_compile_time_args = _merge_named_compile_time_args(phase_kernels)
    # Merge defines from all phases
    fused_compute.defines = _merge_defines(phase_kernels)
    # Note: Cannot use list(runtime_args) - causes infinite loop with C++ bindings
    # For compute kernels, all phases typically have the same runtime args (NCHt)
    # since they operate on the same data. Use phase 0's args.
    fused_compute.runtime_args = base_compute.runtime_args

    # Merge compute config from all phases (including unpack_to_dest_mode).
    fused_compute.config = _merge_compute_configs(phase_kernels, phases)

    return fused_compute


def _merge_compute_configs(
    phase_kernels: List[Dict[str, Any]],
    phases: Optional[List[PhaseInfo]] = None,
) -> "ttnn.ComputeConfigDescriptor":
    """
    Merge compute configs from all phases.

    Key constraint: fp32_dest_acc_en is a kernel-level hardware setting that
    controls whether dest registers use 32-bit (fp32) or 16-bit (fp16_b)
    format.  On Wormhole, dest can hold 8 fp32 tiles or 16 fp16_b tiles.

    If ANY phase requires fp32_dest_acc_en=True (for FLOAT32_REDUCTION), the
    fused kernel must use True.  Phases with blk>4 still work because the 8
    fp32 dest registers accommodate up to blk=8.

    For math_approx_mode: use True if ANY phase uses it (safe superset).

    For unpack_to_dest_mode: builds a merged vector (indexed by physical CB
    index) from each phase's per-CB modes, remapped through cb_remapping.
    Validates that shared CBs (same physical index across phases) have
    consistent unpack modes.
    """
    base = phase_kernels[0]["compute"].config

    # fp32_dest_acc_en: True if ANY phase needs it
    need_fp32 = False
    need_approx = False
    for pk in phase_kernels:
        if pk["compute"] is not None:
            if pk["compute"].config.fp32_dest_acc_en:
                need_fp32 = True
            if pk["compute"].config.math_approx_mode:
                need_approx = True

    # Build merged unpack_to_dest_mode vector across all phases.
    # Each phase's original CB indices are remapped to physical indices;
    # the merged vector must have the correct mode at each physical index.
    merged_unpack = _merge_unpack_to_dest_modes(phase_kernels, phases)

    needs_new_config = (
        need_fp32 != base.fp32_dest_acc_en or need_approx != base.math_approx_mode or merged_unpack is not None
    )
    if not needs_new_config:
        return base

    # Need a modified config
    merged = ttnn.ComputeConfigDescriptor()
    merged.math_fidelity = base.math_fidelity
    merged.fp32_dest_acc_en = need_fp32
    merged.math_approx_mode = need_approx
    merged.dst_full_sync_en = base.dst_full_sync_en
    merged.bfp8_pack_precise = base.bfp8_pack_precise
    if merged_unpack is not None:
        merged.unpack_to_dest_mode = merged_unpack
    elif base.unpack_to_dest_mode:
        merged.unpack_to_dest_mode = base.unpack_to_dest_mode
    return merged


NUM_CIRCULAR_BUFFERS = 32


def _merge_unpack_to_dest_modes(
    phase_kernels: List[Dict[str, Any]],
    phases: Optional[List[PhaseInfo]] = None,
) -> Optional[List]:
    """
    Build a merged unpack_to_dest_mode vector for the fused kernel.

    Each phase may have its own unpack_to_dest_mode vector indexed by
    *original* CB index.  After CB remapping, these must be translated
    to *physical* CB indices.  If any physical CB gets UnpackToDestFp32
    from any phase, the merged vector must reflect that.

    Validates that shared CBs (multiple phases mapping different original
    indices to the same physical index) have consistent unpack modes.

    Returns None if all phases use Default for all CBs (no vector needed).
    """
    if not phases:
        return None

    # Collect per-physical-CB unpack modes from all phases.
    # Track which phase set each mode for error reporting.
    physical_modes: Dict[int, "ttnn.UnpackToDestMode"] = {}
    # For conflict detection: physical_cb -> list of (phase_idx, original_cb, mode)
    physical_sources: Dict[int, List[tuple]] = {}

    for phase_idx, pk in enumerate(phase_kernels):
        compute = pk.get("compute")
        if compute is None:
            continue

        config = compute.config
        unpack_modes = config.unpack_to_dest_mode
        if not unpack_modes:
            continue

        # Get this phase's CB remapping
        if phase_idx < len(phases):
            remapping = phases[phase_idx].cb_remapping
        else:
            continue

        for original_cb, physical_cb in remapping.items():
            if original_cb >= len(unpack_modes):
                continue

            mode = unpack_modes[original_cb]

            if physical_cb not in physical_sources:
                physical_sources[physical_cb] = []
            physical_sources[physical_cb].append((phase_idx, original_cb, mode))

            if physical_cb in physical_modes:
                existing = physical_modes[physical_cb]
                if mode != existing:
                    # Conflict: same physical CB, different unpack modes
                    prev_entries = [e for e in physical_sources[physical_cb] if e[2] != mode]
                    raise ValueError(
                        f"UnpackToDestMode conflict on physical CB {physical_cb}: "
                        f"phase {phase_idx} (original CB {original_cb}) requires "
                        f"{mode}, but previous phase(s) set it to {existing}. "
                        f"Previous: {prev_entries}. "
                        f"Shared CBs must have consistent UnpackToDestMode settings."
                    )
            else:
                physical_modes[physical_cb] = mode

    # Check if any CB needs non-Default mode
    has_non_default = any(mode != UnpackToDestMode.Default for mode in physical_modes.values())
    if not has_non_default:
        return None

    # Build the full vector
    merged = [UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
    for physical_cb, mode in physical_modes.items():
        if physical_cb < NUM_CIRCULAR_BUFFERS:
            merged[physical_cb] = mode

    return merged


def _merge_compile_time_args(
    phase_kernels: List[Dict[str, Any]],
) -> List[int]:
    """
    Merge compile-time args from all phases' compute kernels.

    For LayerNorm, compile-time args are: Wt, blk, do_gamma, do_beta, etc.
    These should be the same across phases operating on the same tensor shape.
    """
    if not phase_kernels:
        return []

    # Use phase 0's compile-time args as the base
    # All phases should have compatible args since they operate on same tensor
    base_compute = phase_kernels[0].get("compute")
    if base_compute is None:
        return []

    return list(base_compute.compile_time_args)


def _merge_defines(
    phase_kernels: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """
    Merge defines from all phases' compute kernels.

    Source-level defines (RMSNORM, FUSE_PRE_ADD, etc.) are NOT included here
    because they are resolved per-phase directly into the generated source code
    by _resolve_ifdef_directives. Only compiler-level defines that affect all
    phases uniformly (REDUCE_OP, REDUCE_DIM, etc.) are passed through.

    Other phase-specific defines are prefixed with PHASE{i}_ to avoid conflicts.
    """
    merged = []
    seen_common = set()  # Track common defines to avoid duplicates

    # Common defines that shouldn't be prefixed
    common_defines = {"REDUCE_OP", "REDUCE_DIM", "BCAST_LLKOP", "BCAST_DIM"}

    for i, pk in enumerate(phase_kernels):
        if pk["compute"] is None:
            continue

        for name, value in pk["compute"].defines:
            if name in common_defines:
                if name not in seen_common:
                    merged.append((name, value))
                    seen_common.add(name)
            elif name in _SOURCE_LEVEL_DEFINES:
                # Already resolved into source per-phase; don't pass to compiler
                continue
            else:
                # Phase-specific defines get prefixed
                prefixed_name = f"PHASE{i}_{name}"
                merged.append((prefixed_name, value))

    return merged


# Defines that are resolved per-phase in source code (not passed to compiler)
_SOURCE_LEVEL_DEFINES = {"RMSNORM", "FUSE_PRE_ADD", "FUSED_PRE_ADD"}

# Positional compile-time arg names for LayerNorm/RMSNorm compute kernels.
# Index 0=Wt, 1=blk, 2=do_gamma, 3=do_beta, 4=FLOAT32_DTYPE,
# 5=FLOAT32_REDUCTION, 6=LEGACY_RSQRT, 7=W
_COMPUTE_ARG_NAMES = [
    "Wt",
    "blk",
    "do_gamma",
    "do_beta",
    "FLOAT32_DTYPE",
    "FLOAT32_REDUCTION",
    "LEGACY_RSQRT",
    "W",
]


def _total_with_remainder(Wt: int, blk: int) -> int:
    """Python mirror of C++ blocks(Wt, blk).total_with_remainder().

    Returns the total number of tiles including padding for the last block.
    """
    if Wt == 0:
        return 0
    if blk >= Wt:
        return blk  # Single partial block, padded to full block size
    remainder = Wt % blk
    if remainder == 0:
        return Wt  # All full blocks, no padding
    return Wt + (blk - remainder)  # Pad last block to full block size


def _resolve_ifdef_directives(source: str, active_defines: set) -> str:
    """
    Resolve preprocessor #ifdef/#ifndef/#if defined directives in source code.

    Only resolves directives involving known source-level defines (RMSNORM,
    FUSE_PRE_ADD, FUSED_PRE_ADD). Other #ifdef directives are left untouched.

    This is needed for fused kernels where each phase may have different defines
    (e.g., Phase 0 is LayerNorm, Phase 1 is RMSNorm). Since all phases share
    one compilation unit, preprocessor defines can't vary per-phase. Instead,
    we resolve them into the source before extraction.

    Args:
        source: The kernel source code
        active_defines: Set of define names that are active for this phase

    Returns:
        Source with relevant #ifdef blocks resolved (included or excluded)
    """
    import re

    lines = source.split("\n")
    result = []
    # Stack entries: (include_current_branch, is_known_define)
    stack: List[Tuple[bool, bool]] = []

    for line in lines:
        stripped = line.strip()

        directive_handled = False

        if stripped.startswith("#if defined"):
            # Handle: #if defined NAME and not defined NAME2
            # Handle: #if defined NAME
            names_found = re.findall(r"\bdefined\s+(\w+)", stripped)
            involved = [n for n in names_found if n in _SOURCE_LEVEL_DEFINES]
            if involved:
                # Evaluate the expression
                result_val = _eval_ifdef_expression(stripped, active_defines)
                stack.append((result_val, True))
                directive_handled = True
            else:
                stack.append((True, False))

        elif stripped.startswith("#ifdef "):
            name = stripped[7:].strip()
            if name in _SOURCE_LEVEL_DEFINES:
                stack.append((name in active_defines, True))
                directive_handled = True
            else:
                stack.append((True, False))

        elif stripped.startswith("#ifndef "):
            name = stripped[8:].strip()
            if name in _SOURCE_LEVEL_DEFINES:
                stack.append((name not in active_defines, True))
                directive_handled = True
            else:
                stack.append((True, False))

        elif stripped == "#else":
            if stack and stack[-1][1]:
                incl, known = stack[-1]
                stack[-1] = (not incl, known)
                directive_handled = True

        elif stripped == "#endif":
            if stack:
                _, known = stack[-1]
                stack.pop()
                if known:
                    directive_handled = True

        if directive_handled:
            continue

        # Include line only if all known-define branches on the stack are active
        include = True
        for incl, known in stack:
            if known and not incl:
                include = False
                break

        if include:
            result.append(line)

    return "\n".join(result)


def _eval_ifdef_expression(directive: str, active_defines: set) -> bool:
    """
    Evaluate a #if defined expression like:
      #if defined RMSNORM and not defined FUSE_PRE_ADD

    Returns True if the condition is satisfied.
    """
    import re

    # Find all "defined NAME" and "not defined NAME" clauses
    # Pattern: optional "not" followed by "defined NAME"
    clauses = re.findall(r"(not\s+)?defined\s+(\w+)", directive)

    result = True
    for negation, name in clauses:
        is_defined = name in active_defines
        if negation:
            result = result and (not is_defined)
        else:
            result = result and is_defined

    return result


def _generate_multi_phase_compute_source(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
) -> Optional[str]:
    """
    Generate a fused compute kernel that runs multiple phases sequentially.

    Each phase is wrapped as a separate function (phase0_compute, phase1_compute, etc.)
    and kernel_main() calls them in order.
    """
    if not phase_kernels:
        return None

    # Read source from all compute kernels, resolving per-phase #ifdef directives.
    # Each phase may have different defines (e.g., RMSNORM for RMSNorm phases).
    # Since all phases share one compilation unit, we resolve these into the source.
    phase_sources = []
    phase_define_sets = []
    for pk in phase_kernels:
        if pk["compute"] is not None:
            source = _read_kernel_source_from_descriptor(pk["compute"])
            if source:
                phase_defs = {name for name, _ in pk["compute"].defines}
                resolved = _resolve_ifdef_directives(source, phase_defs)
                phase_sources.append(resolved)
                phase_define_sets.append(phase_defs)

    if not phase_sources:
        return None

    # Extract defines, includes, and other pre-main code separately.
    # Use Phase 0's resolved source for defines/includes (since all phases
    # share the same includes and global defines like REDUCE_OP).
    defines = []
    includes = set()
    other_pre_main = []

    for source in phase_sources:
        for line in source.split("\n"):
            stripped = line.strip()
            if "void kernel_main()" in line:
                break
            if stripped.startswith("#define"):
                if stripped not in [d.strip() for d in defines]:
                    defines.append(line)
            elif stripped.startswith("#include"):
                includes.add(stripped)

    # Extract other pre-main code (namespaces, macros like ALWI, etc.) from first source
    for line in phase_sources[0].split("\n"):
        stripped = line.strip()
        if "void kernel_main()" in line:
            break
        if (
            not stripped.startswith("#define")
            and not stripped.startswith("#include")
            and not stripped.startswith("//")
            and stripped
        ):
            other_pre_main.append(line)

    # Build the fused source - ORDER IS CRITICAL:
    # 1. Header comments
    # 2. Defines (REDUCE_OP, REDUCE_DIM must come before includes!)
    # 3. Includes
    # 4. Other pre-main code (namespaces, helper macros)
    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused compute kernel - {len(phase_sources)} phases",
        "",
    ]

    # Add defines FIRST (before includes)
    lines.extend(defines)
    lines.append("")

    # Add includes
    lines.extend(sorted(includes))
    lines.append("")

    # Add other pre-main code (namespaces, helper macros)
    lines.extend(other_pre_main)
    lines.append("")

    # Generate a phase function for each phase.
    # Each phase resolves its own compile-time args independently via
    # phase-prefixed named args (e.g., "phase1_blk" instead of normalizing to
    # Phase 0's blk). The intermediate CB tile count mismatch between phases
    # with different blk values is handled by transition padding in kernel_main().
    source_idx = 0
    for i, (pk, phase) in enumerate(zip(phase_kernels, phases)):
        if pk["compute"] is None:
            continue

        # Use the already-resolved source (ifdefs flattened per-phase)
        resolved_source = phase_sources[source_idx]
        source_idx += 1
        body = _extract_kernel_body_for_fusion(resolved_source)

        # Substitute all get_compile_time_arg_val(N) with literal values.
        # Every phase needs this because the phase bodies are FORCE_INLINE
        # functions and positional arg resolution must happen at source level.
        ct_args = list(pk["compute"].compile_time_args)
        remapped_body = _apply_cb_remapping_to_source(body, phase.cb_remapping, phase_idx=i, compile_time_args=ct_args)

        lines.append(f"// Phase {i} compute function")
        lines.append(f"FORCE_INLINE void phase{i}_compute() {{")

        # Add the remapped body with proper indentation
        for line in remapped_body.split("\n"):
            lines.append(f"    {line}")

        lines.append("}")
        lines.append("")

    # Generate kernel_main that calls all phase functions, with transition
    # padding between phases when the consumer expects more tiles than the
    # producer pushed (due to different blk values).
    lines.append("void kernel_main() {")
    for i in range(len(phase_sources)):
        lines.append(f"    phase{i}_compute();")

        # Add transition padding between consecutive phases if needed
        if i < len(phase_sources) - 1:
            pk_i = phase_kernels[i]
            pk_next = phase_kernels[i + 1]

            if pk_i["compute"] is not None and pk_next["compute"] is not None:
                args_i = list(pk_i["compute"].compile_time_args)
                args_next = list(pk_next["compute"].compile_time_args)

                Wt_i = args_i[0] if args_i else 0
                blk_i = args_i[1] if len(args_i) > 1 else Wt_i
                blk_next = args_next[1] if len(args_next) > 1 else (args_next[0] if args_next else 0)

                tiles_produced = _total_with_remainder(Wt_i, blk_i)
                tiles_needed = _total_with_remainder(Wt_i, blk_next)

                if tiles_needed > tiles_produced:
                    pad_count = tiles_needed - tiles_produced
                    # Get intermediate CB: Phase i's remapped output CB
                    intermediate_cb = phases[i].cb_remapping.get(16, 16)
                    lines.append(f"    // Transition: phase {i} (blk={blk_i}) pushed {tiles_produced} tiles,")
                    lines.append(f"    // phase {i + 1} (blk={blk_next}) needs {tiles_needed} tiles")
                    lines.append(f"    cb_reserve_back({intermediate_cb}, {pad_count});")
                    lines.append(f"    cb_push_back({intermediate_cb}, {pad_count});")

                    # Gamma/beta are loaded by the reader with Phase 0's blk but
                    # never popped.  If Phase i+1 has a larger blk, its
                    # cb_wait_front expects more tiles than the reader loaded.
                    # Pad these CBs too (reader loaded with blk_0, we need blk_next).
                    do_gamma_next = args_next[2] if len(args_next) > 2 else 0
                    do_beta_next = args_next[3] if len(args_next) > 3 else 0
                    # Reader loaded gamma/beta with Phase 0's blk
                    args_0 = list(phase_kernels[0]["compute"].compile_time_args)
                    blk_0 = args_0[1] if len(args_0) > 1 else (args_0[0] if args_0 else 0)
                    gamma_loaded = _total_with_remainder(Wt_i, blk_0)
                    gamma_needed = _total_with_remainder(Wt_i, blk_next)
                    gamma_pad = gamma_needed - gamma_loaded
                    if gamma_pad > 0 and do_gamma_next:
                        gamma_cb = phases[i + 1].cb_remapping.get(5, 5)
                        lines.append(f"    // Pad gamma CB for phase {i + 1} ({gamma_loaded} -> {gamma_needed} tiles)")
                        lines.append(f"    cb_reserve_back({gamma_cb}, {gamma_pad});")
                        lines.append(f"    cb_push_back({gamma_cb}, {gamma_pad});")
                    if gamma_pad > 0 and do_beta_next:
                        beta_cb = phases[i + 1].cb_remapping.get(6, 6)
                        lines.append(f"    // Pad beta CB for phase {i + 1} ({gamma_loaded} -> {gamma_needed} tiles)")
                        lines.append(f"    cb_reserve_back({beta_cb}, {gamma_pad});")
                        lines.append(f"    cb_push_back({beta_cb}, {gamma_pad});")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _merge_named_compile_time_args(
    phase_kernels: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """
    Merge named compile-time args from all phases.

    Phase 0's args use ORIGINAL names (cb_in, cb_out, etc.) because the
    original kernel body expects them.
    Phases 1+ use phase-prefixed names to avoid conflicts.

    CB sharing strategy for fused phases:
    - READER-FILLED CBs (cb_eps, cb_scaler, cb_gamma, cb_beta): These are filled
      by the reader kernel once and never popped. Phase 1+ reuses Phase 0's
      physical CBs since the data persists.
    - ALL OTHER CBs (input, output, scratch/intermediate): Get unique per-phase
      physical CB indices because they may differ in page_size (e.g. Float32
      vs Float16_b when fp32_dest_acc_en differs) or blk-dependent tile counts.
    """
    # Only reader-filled constant CBs are shared across phases.  Scratch CBs
    # get per-phase physical indices because they may have different page_sizes
    # (e.g. Float32 vs Float16_b) or blk-dependent tile counts.
    SHARED_CBS = {"cb_eps", "cb_scaler", "cb_gamma", "cb_beta"}

    merged = []
    phase0_values = {}  # Cache phase 0's CB values

    for i, pk in enumerate(phase_kernels):
        if pk["compute"] is not None:
            for name, value in pk["compute"].named_compile_time_args:
                if i == 0:
                    # Phase 0: use original names and cache all values
                    merged.append((name, value))
                    phase0_values[name] = value
                else:
                    # Phase 1+: add phase prefix
                    prefixed_name = f"phase{i}_{name}"
                    if name in SHARED_CBS and name in phase0_values:
                        # Reuse Phase 0's physical CB index
                        merged.append((prefixed_name, phase0_values[name]))
                    else:
                        merged.append((prefixed_name, value))

    return merged


def _extract_pre_kernel_main_code(source: str) -> str:
    """
    Extract code that appears before kernel_main() in the source.

    This includes namespace aliases, macro definitions, helper functions, etc.
    """
    lines = source.split("\n")
    pre_main_lines = []

    for line in lines:
        if "void kernel_main()" in line:
            break
        stripped = line.strip()
        # Skip includes (handled separately) and empty lines
        if stripped and not stripped.startswith("#include"):
            pre_main_lines.append(line)

    return "\n".join(pre_main_lines)


def _extract_kernel_body_for_fusion(source: str) -> str:
    """
    Extract the body of kernel_main() for fusion.

    This extracts everything inside kernel_main() { ... }, excluding the
    final closing brace of kernel_main itself.
    """
    lines = source.split("\n")
    in_kernel_main = False
    brace_depth = 0
    body_lines = []

    for line in lines:
        stripped = line.strip()

        if "void kernel_main()" in line or "KERNEL_MAIN" in line:
            in_kernel_main = True
            # Count braces on this line
            brace_depth += stripped.count("{") - stripped.count("}")
            continue

        if in_kernel_main:
            open_braces = stripped.count("{")
            close_braces = stripped.count("}")

            # Calculate new brace depth after this line
            new_brace_depth = brace_depth + open_braces - close_braces

            # Only add the line if we're still inside the function
            # (i.e., not the final closing brace that brings us to depth 0)
            if new_brace_depth > 0 or (brace_depth > 0 and new_brace_depth > 0):
                body_lines.append(line)
            elif brace_depth > 0 and new_brace_depth == 0:
                # This is the final closing brace - don't add it
                # But if there's more content on this line before the brace, keep it
                if stripped != "}":
                    # Line has content plus closing brace - need to handle carefully
                    # For now, skip the whole line as it's usually just "}"
                    pass
                break

            brace_depth = new_brace_depth

            if brace_depth == 0:
                break

    return "\n".join(body_lines)


def _apply_cb_remapping_to_source(
    source: str,
    cb_remap: Dict[int, int],
    phase_idx: int,
    compile_time_args: Optional[List[int]] = None,
) -> str:
    """
    Apply CB remapping and compile-time arg substitution to kernel source code.

    This modifies:
    1. get_compile_time_arg_val(N) → get_named_compile_time_arg_val("phaseI_ARG_NAME")
       for ALL phases, so each phase resolves its own Wt, blk, do_gamma, etc.
       independently via phase-prefixed named args.
    2. get_named_compile_time_arg_val("cb_xxx") → phase-prefixed for phase 1+
    3. Direct CB references like tt::CBIndex::c_N
    """
    result = source
    import re

    # For Phase 1+, replace positional compile-time arg calls with the
    # phase's actual values (substituted as literals).  Phase 0 keeps the
    # original get_compile_time_arg_val(N) calls which resolve correctly
    # against the fused kernel's positional args.
    if phase_idx > 0 and compile_time_args is not None:

        def replace_compile_time_arg(match):
            arg_idx = int(match.group(1))
            if arg_idx < len(compile_time_args):
                return str(compile_time_args[arg_idx])
            return match.group(0)

        result = re.sub(
            r"get_compile_time_arg_val\((\d+)\)",
            replace_compile_time_arg,
            result,
        )

    # Replace named compile-time arg references with phase-prefixed versions
    # But ONLY for phase 1+ - phase 0 keeps original names
    if phase_idx > 0:

        def replace_named_arg(match):
            name = match.group(1)
            return f'get_named_compile_time_arg_val("phase{phase_idx}_{name}")'

        result = re.sub(
            r'get_named_compile_time_arg_val\("(cb_[^"]+)"\)',
            replace_named_arg,
            result,
        )

    # Also remap direct CB index references (tt::CBIndex::c_N)
    for old_idx, new_idx in cb_remap.items():
        result = result.replace(f"tt::CBIndex::c_{old_idx}", f"tt::CBIndex::c_{new_idx}")
        result = result.replace(f"CB::c_{old_idx}", f"CB::c_{new_idx}")

    return result


# =============================================================================
# Fused Kernel Generation
# =============================================================================


class FusedKernelGenerator:
    """
    Generates fused kernel source code for chained operations.

    This class creates kernel source strings that combine multiple operations
    into single kernels, enabling true kernel fusion without intermediate
    DRAM reads/writes.

    The generator supports three fusion strategies:
    1. CB Remapping Only: Keep separate kernels but chain via shared CBs
    2. Compute Fusion: Combine compute kernels into a single kernel_main()
    3. Full Fusion: Fuse reader + compute + writer into minimal kernels

    For now, we implement CB Remapping with optimized reader/writer generation,
    which removes unnecessary DRAM accesses for intermediate results.
    """

    def __init__(self, include_paths: Optional[List[str]] = None):
        """
        Initialize the generator.

        Args:
            include_paths: Additional include paths for kernel compilation
        """
        self.include_paths = include_paths or []
        self.phases: List[Dict[str, Any]] = []  # List of phase info dicts

    def add_phase(
        self,
        compute_source: str,
        reader_source: Optional[str] = None,
        writer_source: Optional[str] = None,
        cb_remapping: Optional[Dict[int, int]] = None,
        cb_name_remapping: Optional[Dict[str, int]] = None,
        defines: Optional[Dict[str, str]] = None,
        is_first: bool = False,
        is_last: bool = False,
    ):
        """
        Add a phase to the fusion pipeline.

        Args:
            compute_source: Source code for the compute kernel
            reader_source: Source code for the reader kernel (optional)
            writer_source: Source code for the writer kernel (optional)
            cb_remapping: Mapping of original CB index -> new CB index
            cb_name_remapping: Mapping of CB name -> new CB index (for named args)
            defines: Preprocessor defines for this phase
            is_first: True if this is the first phase (needs reader)
            is_last: True if this is the last phase (needs writer)
        """
        self.phases.append(
            {
                "compute_source": compute_source,
                "reader_source": reader_source,
                "writer_source": writer_source,
                "cb_remapping": cb_remapping or {},
                "cb_name_remapping": cb_name_remapping or {},
                "defines": defines or {},
                "is_first": is_first,
                "is_last": is_last,
            }
        )

    def generate_fused_compute(self) -> str:
        """
        Generate a fused compute kernel that chains all phases.

        Returns:
            Fused compute kernel source code as a string
        """
        if not self.phases:
            raise ValueError("No phases added to generator")

        # For single phase, just return the compute source with remapping
        if len(self.phases) == 1:
            return self._apply_cb_remapping(
                self.phases[0]["compute_source"],
                self.phases[0]["cb_remapping"],
                self.phases[0]["cb_name_remapping"],
            )

        # Multi-phase: generate chained compute kernel
        return self._generate_chained_compute()

    def _generate_chained_compute(self) -> str:
        """Generate compute kernel that chains multiple phases."""
        # For now, we generate a kernel that includes all phase computations
        # A more sophisticated approach would merge the kernel_main() functions

        lines = [
            "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
            "//",
            "// SPDX-License-Identifier: Apache-2.0",
            "",
            "// Auto-generated fused compute kernel",
            f"// Fuses {len(self.phases)} phases",
            "",
        ]

        # Collect all unique includes
        includes = self._collect_includes()
        lines.extend(includes)
        lines.append("")

        # Generate the fused kernel_main
        lines.append("void kernel_main() {")

        for i, phase in enumerate(self.phases):
            lines.append(f"    // Phase {i}")
            # Extract and indent the kernel body
            body = self._extract_kernel_body(phase["compute_source"])
            remapped_body = self._apply_cb_remapping(
                body,
                phase["cb_remapping"],
                phase["cb_name_remapping"],
            )
            for line in remapped_body.split("\n"):
                lines.append(f"    {line}")
            lines.append("")

        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def generate_fused_reader(self) -> str:
        """
        Generate a fused reader kernel.

        Only the first phase needs to read from DRAM. Subsequent phases
        receive their input from the previous phase's output CB.

        Returns:
            Fused reader kernel source code
        """
        if not self.phases:
            raise ValueError("No phases added to generator")

        # Find the first phase with a reader
        for phase in self.phases:
            if phase["is_first"] and phase["reader_source"]:
                return self._apply_cb_remapping(
                    phase["reader_source"],
                    phase["cb_remapping"],
                    phase["cb_name_remapping"],
                )

        # If no reader found, return a minimal no-op reader
        return self._generate_noop_reader()

    def generate_fused_writer(self) -> str:
        """
        Generate a fused writer kernel.

        Only the last phase needs to write to DRAM.

        Returns:
            Fused writer kernel source code
        """
        if not self.phases:
            raise ValueError("No phases added to generator")

        # Find the last phase with a writer
        for phase in reversed(self.phases):
            if phase["is_last"] and phase["writer_source"]:
                return self._apply_cb_remapping(
                    phase["writer_source"],
                    phase["cb_remapping"],
                    phase["cb_name_remapping"],
                )

        # If no writer found, return a minimal no-op writer
        return self._generate_noop_writer()

    def _apply_cb_remapping(
        self,
        source: str,
        cb_remapping: Dict[int, int],
        cb_name_remapping: Dict[str, int],
    ) -> str:
        """
        Apply CB index remapping to kernel source code.

        This modifies:
        1. Named compile-time arg references: get_named_compile_time_arg_val("cb_X")
        2. Direct CB index references: tt::CBIndex::c_X, CB::c_X
        """
        result = source

        # Named CB remapping is handled via KernelDescriptor's named_compile_time_args
        # The source code itself doesn't need modification for named args
        _ = cb_name_remapping  # Mark as intentionally unused here

        # Apply direct CB index remapping
        # Handle patterns like: tt::CBIndex::c_0, CB::c_16, etc.
        for old_idx, new_idx in cb_remapping.items():
            # Replace tt::CBIndex::c_X patterns
            result = result.replace(f"tt::CBIndex::c_{old_idx}", f"tt::CBIndex::c_{new_idx}")
            # Replace CB::c_X patterns
            result = result.replace(f"CB::c_{old_idx}", f"CB::c_{new_idx}")
            # Replace numeric CB indices in common functions
            # Be careful not to replace arbitrary numbers

        return result

    def _collect_includes(self) -> List[str]:
        """Collect unique includes from all phases."""
        includes = set()
        for phase in self.phases:
            source = phase["compute_source"]
            for line in source.split("\n"):
                stripped = line.strip()
                if stripped.startswith("#include"):
                    includes.add(stripped)
        return sorted(includes)

    def _extract_kernel_body(self, source: str) -> str:
        """
        Extract the body of kernel_main() from source code.

        This is a simplified parser - a production implementation
        would use proper C++ parsing.
        """
        lines = source.split("\n")
        in_kernel_main = False
        brace_depth = 0
        body_lines = []

        for line in lines:
            stripped = line.strip()

            if "void kernel_main()" in line or "KERNEL_MAIN" in line:
                in_kernel_main = True
                # Count braces on this line (handles "void kernel_main() {" case)
                brace_depth += stripped.count("{") - stripped.count("}")
                continue

            if in_kernel_main:
                # Count braces before deciding whether to include line
                open_braces = stripped.count("{")
                close_braces = stripped.count("}")

                # If we're inside the function body, add the line
                if brace_depth > 0:
                    body_lines.append(line)

                # Update brace depth
                brace_depth += open_braces - close_braces

                # Check for end of kernel_main
                if brace_depth == 0 and body_lines:
                    break

        return "\n".join(body_lines)

    def _generate_noop_reader(self) -> str:
        """Generate a minimal no-op reader kernel."""
        return """
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Auto-generated no-op reader for fused kernel chain
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // No-op: data comes from previous phase's output CB
}
"""

    def _generate_noop_writer(self) -> str:
        """Generate a minimal no-op writer kernel."""
        return """
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Auto-generated no-op writer for fused kernel chain
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // No-op: data will be consumed by next phase
}
"""


def read_kernel_source(kernel_path: str) -> str:
    """
    Read kernel source code from a file.

    Args:
        kernel_path: Path to the kernel source file

    Returns:
        Kernel source code as a string
    """
    with open(kernel_path, "r") as f:
        return f.read()


# =============================================================================
# Convenience Functions
# =============================================================================


def chain_descriptors(
    descriptors: List[OpDescriptor],
    connections: Optional[List[Tuple[int, int, int, int]]] = None,
) -> OpDescriptor:
    """
    Convenience function to chain multiple OpDescriptors.

    Args:
        descriptors: List of OpDescriptors to chain sequentially
        connections: Optional explicit connections as (src_phase, src_cb, tgt_phase, tgt_cb)
                    If not provided, assumes linear chain: phase N output -> phase N+1 input

    Returns:
        Fused OpDescriptor

    Example:
        >>> ln1 = descriptors.normalization.layer_norm(input, weight=w1)
        >>> ln2 = descriptors.normalization.rms_norm(dummy, weight=w2)  # dummy input
        >>> fused = chain_descriptors([ln1, ln2])  # ln1 output -> ln2 input
    """
    builder = SequentialChainBuilder()

    for i, desc in enumerate(descriptors):
        builder.add_phase(desc, input_from_previous=(i > 0))

    if connections:
        for src_phase, src_cb, tgt_phase, tgt_cb in connections:
            builder.connect_phases(src_phase, src_cb, tgt_phase, tgt_cb)

    return builder.build()


def _validate_fp32_consistency(op_descriptors: List[OpDescriptor]) -> None:
    """
    Validate that all phases have consistent fp32_dest_acc_en settings.

    DST_ACCUM_MODE is a compile-time constexpr bool that controls the entire
    kernel's dest register format (fp32 vs fp16_b).  It cannot be changed
    mid-kernel, so all fused phases MUST be built with the same
    fp32_dest_acc_en setting.

    When fp32_dest_acc_en differs, the factory produces incompatible blk
    values (4 for fp32, 8 for fp16_b) and CB page sizes (4096 vs 2048).
    Forcing the hardware setting at fusion time is insufficient — the
    compile-time args and CB sizes embedded in each phase's descriptor
    must match the global DST_ACCUM_MODE.

    Raises ValueError with guidance on how to fix the mismatch.
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
        return  # All consistent

    phases_with_fp32 = [i for i, v in fp32_settings if v]
    phases_without = [i for i, v in fp32_settings if not v]

    raise ValueError(
        f"fp32_dest_acc_en mismatch: phases {phases_with_fp32} use fp32=True, "
        f"phases {phases_without} use fp32=False. "
        f"DST_ACCUM_MODE is a kernel-level hardware setting that cannot be "
        f"changed mid-kernel. All phases must use the same fp32_dest_acc_en "
        f"setting. To fix: create all descriptors with the same "
        f"compute_kernel_config. For example:\n"
        f"  config = ttnn.layernorm_default_compute_config(device.arch())\n"
        f"  rms = rms_norm.rms_norm(input, ..., compute_kernel_config=config)\n"
        f"  ln  = layer_norm.layer_norm(input, ..., compute_kernel_config=config)"
    )


def _validate_cb_capacity(phases: List[PhaseInfo]) -> None:
    """
    Validate that the fused chain won't exceed the 32 available circular buffers.

    With only 32 CBs (indices 0-31), multi-phase fusion with CB remapping can
    potentially exhaust the available CBs. This function estimates the total CB
    requirements and provides a clear error message before attempting fusion.

    The estimation is conservative:
    - Shared CBs (gamma, beta, etc.) count as 1 across all phases
    - Per-phase CBs (input, output, scratch) need unique allocations for each phase
    - Chained CBs (output of phase N = input of phase N+1) are reused

    Raises ValueError if the estimated CB count exceeds 32.
    """
    if len(phases) == 1:
        return  # Single phase can't overflow

    # CBs that are shared across phases (reader-filled constants)
    READER_FILLED_CB_INDICES = {2, 3, 5, 6}  # cb_scaler, cb_eps, cb_gamma, cb_beta

    # Track unique CBs needed
    shared_cbs = set()
    per_phase_cb_count = []

    for phase in phases:
        phase_unique_cbs = set()
        for cb_idx in phase.cb_info.keys():
            if cb_idx in READER_FILLED_CB_INDICES:
                shared_cbs.add(cb_idx)
            else:
                phase_unique_cbs.add(cb_idx)
        per_phase_cb_count.append(len(phase_unique_cbs))

    # Estimate total CBs:
    # - Shared CBs count once
    # - Each phase needs its own per-phase CBs
    # - Minus (N-1) for chained CBs (output of phase i reused as input of phase i+1)
    num_shared = len(shared_cbs)
    num_per_phase = sum(per_phase_cb_count)
    num_chained_reuse = len(phases) - 1  # Conservative: assume all phases chain

    estimated_total = num_shared + num_per_phase - num_chained_reuse

    if estimated_total > CBRemapper.NUM_CBS:
        phase_details = "\n".join(f"  Phase {i}: {count} per-phase CBs" for i, count in enumerate(per_phase_cb_count))
        raise ValueError(
            f"CB capacity exceeded: estimated {estimated_total} CBs needed, "
            f"but only {CBRemapper.NUM_CBS} available.\n"
            f"Breakdown:\n"
            f"  - Shared CBs (gamma, beta, etc.): {num_shared}\n"
            f"  - Per-phase CBs (input, output, scratch): {num_per_phase}\n"
            f"  - Chained CB reuse: -{num_chained_reuse}\n"
            f"  - Total estimate: {estimated_total}\n\n"
            f"Per-phase details:\n{phase_details}\n\n"
            f"Suggestions:\n"
            f"  - Reduce the number of phases in the chain\n"
            f"  - Fuse fewer operations at once\n"
            f"  - Split into multiple smaller chains executed sequentially"
        )


def fuse_layernorm_chain(
    op_descriptors: List[OpDescriptor],
    core_ranges: Optional[Any] = None,  # Reserved for future use
) -> OpDescriptor:
    """
    Fuse a chain of LayerNorm/RMSNorm operations into a single fused operation.

    This is a high-level convenience function specifically for fusing normalization
    operations. It handles:
    1. CB remapping to avoid conflicts
    2. Kernel fusion or chaining
    3. Removal of intermediate DRAM accesses

    Args:
        op_descriptors: List of LayerNorm/RMSNorm OpDescriptors to fuse
        core_ranges: Optional core ranges override (reserved for future use)

    Returns:
        Fused OpDescriptor that executes all operations sequentially on the same cores

    Example:
        >>> ln1 = descriptors.normalization.layer_norm(input, weight=w1)
        >>> ln2 = descriptors.normalization.rms_norm(None, weight=w2)  # Input from prev
        >>> fused = fuse_layernorm_chain([ln1, ln2])
    """
    # core_ranges is reserved for future use when we need to override core assignments
    _ = core_ranges
    if not op_descriptors:
        raise ValueError("No descriptors provided for fusion")

    if len(op_descriptors) == 1:
        return op_descriptors[0]

    # Build the sequential chain with proper CB remapping
    builder = SequentialChainBuilder()

    for i, desc in enumerate(op_descriptors):
        # LayerNorm/RMSNorm conventions:
        # - Input CB: cb_in (index 0)
        # - Output CB: cb_out (index 16)
        builder.add_phase(
            desc,
            input_from_previous=(i > 0),
            input_cb=0,  # cb_in
            output_cb=16,  # cb_out
        )

    return builder.build()


def create_parallel_chain_descriptors(
    chains: List[List[OpDescriptor]],
) -> List[OpDescriptor]:
    """
    Create fused descriptors for multiple parallel chains.

    Each chain is fused sequentially, and the resulting fused ops can be
    run in parallel using composite.launch().

    Args:
        chains: List of chains, where each chain is a list of OpDescriptors
                to be fused sequentially

    Returns:
        List of fused OpDescriptors, one per chain

    Example:
        >>> # Two parallel chains of LayerNorm -> RMSNorm
        >>> chain1 = [ln1_a, rms1_a]  # First chain
        >>> chain2 = [ln1_b, rms1_b]  # Second chain
        >>> fused = create_parallel_chain_descriptors([chain1, chain2])
        >>> # fused[0] and fused[1] can be run in parallel
    """
    fused_descriptors = []

    for chain in chains:
        if not chain:
            continue

        if len(chain) == 1:
            fused_descriptors.append(chain[0])
        else:
            fused = chain_descriptors(chain)
            fused_descriptors.append(fused)

    return fused_descriptors


__all__ = [
    # Core classes
    "SequentialChainBuilder",
    "CBRemapper",
    "PhaseInfo",
    "CBInfo",
    "PhaseConnection",
    "FusedKernelGenerator",
    # Functions
    "chain_descriptors",
    "fuse_layernorm_chain",
    "create_parallel_chain_descriptors",
    "read_kernel_source",
    "extract_cb_info",
    "extract_cb_names_from_kernel",
    "remap_kernel_cb_indices",
    "remap_cb_descriptors",
]
