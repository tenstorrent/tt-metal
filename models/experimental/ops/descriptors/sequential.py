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
    ) -> Dict[int, int]:
        """
        Allocate remapped CB indices for a phase.

        Args:
            phase: Phase information with CB requirements
            chain_from_previous: If True, connect previous output to this phase's input
            previous_output_cb: The remapped CB index holding previous phase's output
            target_input_cb: The original CB index that should receive the chained input

        Returns:
            Mapping of original CB index -> remapped CB index
        """
        remapping: Dict[int, int] = {}

        # If chaining from previous phase, reuse the output CB as this phase's input
        if chain_from_previous and previous_output_cb is not None and target_input_cb is not None:
            remapping[target_input_cb] = previous_output_cb
            # Mark this CB as in use (it holds live data)
            self.allocated.add(previous_output_cb)

        # Allocate CBs for all other requirements
        for original_idx, cb_info in phase.cb_info.items():
            if original_idx in remapping:
                continue  # Already mapped (chained input)

            # Find a free CB index
            new_idx = self._find_free_cb()
            remapping[original_idx] = new_idx
            self.allocated.add(new_idx)

        self.phase_allocations.append(remapping)
        return remapping

    def finish_phase(self, remapping: Dict[int, int], output_cb_original: Optional[int] = None):
        """
        Mark a phase as complete, freeing non-output CBs.

        Args:
            remapping: The CB remapping for this phase
            output_cb_original: The original index of the output CB (to preserve for next phase)
        """
        output_cb_remapped = remapping.get(output_cb_original) if output_cb_original else None

        for original, remapped in remapping.items():
            if output_cb_remapped is not None and remapped == output_cb_remapped:
                # Keep output CB for next phase
                self.live_data_cbs.add(remapped)
            else:
                # Free for reuse
                self.allocated.discard(remapped)

        # Clear live data CBs from previous phases (they've been consumed)
        self.live_data_cbs = {output_cb_remapped} if output_cb_remapped else set()

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


def _copy_kernel_descriptor(kernel_desc: "ttnn.KernelDescriptor") -> "ttnn.KernelDescriptor":
    """
    Create a copy of a KernelDescriptor.

    Since C++ binding objects can't be deep copied, we create a new descriptor
    and copy each field manually.
    """
    new_desc = ttnn.KernelDescriptor()

    # Copy all the fields
    new_desc.kernel_source = kernel_desc.kernel_source
    new_desc.core_ranges = kernel_desc.core_ranges
    new_desc.compile_time_args = list(kernel_desc.compile_time_args)
    new_desc.runtime_args = list(kernel_desc.runtime_args)
    new_desc.common_runtime_args = list(kernel_desc.common_runtime_args)
    new_desc.defines = list(kernel_desc.defines)
    new_desc.config = kernel_desc.config
    new_desc.source_type = kernel_desc.source_type

    if hasattr(kernel_desc, "named_compile_time_args"):
        new_desc.named_compile_time_args = list(kernel_desc.named_compile_time_args)

    return new_desc


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
        New kernel descriptor with remapped CB indices
    """
    # Try deepcopy first for mock objects in tests, fall back to manual copy
    try:
        new_desc = deepcopy(kernel_desc)
    except TypeError:
        # C++ binding objects can't be deep copied
        new_desc = _copy_kernel_descriptor(kernel_desc)

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
        # Try deepcopy first for mock objects, fall back to manual copy
        try:
            new_cb_desc = deepcopy(cb_desc)
        except TypeError:
            new_cb_desc = _copy_cb_descriptor(cb_desc)

        # Remap format descriptors
        new_formats = []
        for fmt_desc in cb_desc.format_descriptors:
            original_idx = fmt_desc.buffer_index
            if original_idx in cb_remapping:
                try:
                    new_fmt = deepcopy(fmt_desc)
                except TypeError:
                    new_fmt = _copy_format_descriptor(fmt_desc)
                new_fmt.buffer_index = cb_remapping[original_idx]
                new_formats.append(new_fmt)
            else:
                try:
                    new_formats.append(deepcopy(fmt_desc))
                except TypeError:
                    new_formats.append(_copy_format_descriptor(fmt_desc))

        new_cb_desc.format_descriptors = new_formats
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
        Build the fused descriptor with CB remapping.
        """
        remapper = CBRemapper()

        # First pass: allocate CB indices for all phases
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

            # Allocate CBs for this phase
            remapping = remapper.allocate_for_phase(
                phase,
                chain_from_previous=chain_from_prev,
                previous_output_cb=prev_output_cb,
                target_input_cb=target_input_cb,
            )
            phase.cb_remapping = remapping

            # Mark phase complete (frees non-output CBs)
            output_cb = list(phase.output_cb_indices)[0] if phase.output_cb_indices else None
            remapper.finish_phase(remapping, output_cb)

        # Second pass: create remapped kernel and CB descriptors
        all_kernels = []
        all_cbs = []
        all_semaphores = []
        all_input_tensors = []
        output_tensor = None

        for i, phase in enumerate(self.phases):
            orig_desc = phase.op_descriptor.descriptor

            # Remap CB descriptors
            remapped_cbs = remap_cb_descriptors(list(orig_desc.cbs), phase.cb_remapping)
            all_cbs.extend(remapped_cbs)

            # Remap kernel descriptors
            phase_cb_args = cb_arg_positions.get(i) if cb_arg_positions else None
            phase_cb_defs = cb_defines.get(i) if cb_defines else None

            for kernel_desc in orig_desc.kernels:
                remapped_kernel = remap_kernel_cb_indices(
                    kernel_desc,
                    phase.cb_remapping,
                    phase_cb_args,
                    phase_cb_defs,
                )
                all_kernels.append(remapped_kernel)

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

        # Create the merged ProgramDescriptor
        merged_descriptor = ttnn.ProgramDescriptor()
        merged_descriptor.kernels = all_kernels
        merged_descriptor.cbs = all_cbs
        merged_descriptor.semaphores = all_semaphores

        return OpDescriptor(
            descriptor=merged_descriptor,
            input_tensors=all_input_tensors,
            output_tensors=[output_tensor] if output_tensor else [],
        )


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
