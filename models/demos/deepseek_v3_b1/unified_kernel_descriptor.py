# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unified Kernel Descriptor for fused operations.

Creates reader (NCRISC), writer (BRISC), and compute (TRISC) kernel descriptors
from a single unified kernel source file.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import ttnn


@dataclass
class KernelGroup:
    """
    A group of kernels (NCRISC, BRISC, TRISC) for a specific core range with specific compile-time args.

    This allows callers to identify which kernel indices correspond to which logical group
    (e.g., sender vs receiver) based on the compile_time_arg_values.
    """

    core_range_set: ttnn.CoreRangeSet
    compile_time_arg_values: dict  # e.g., {"is_sender": 1}
    ncrisc_kernel_index: int
    brisc_kernel_index: int
    trisc_kernel_index: int


@dataclass
class UnifiedKernelResult:
    """
    Result of get_kernel_descriptors() containing both the kernel list and grouping metadata.

    Args:
        kernels: List of KernelDescriptor objects to pass to ProgramDescriptor
        groups: List of KernelGroup objects for identifying kernel indices by role
    """

    kernels: list
    groups: list  # List[KernelGroup]

    def get_group_by_arg(self, arg_name: str, arg_value: int) -> Optional["KernelGroup"]:
        """Find the kernel group where the named compile-time arg has the specified value."""
        for group in self.groups:
            if group.compile_time_arg_values.get(arg_name) == arg_value:
                return group
        return None


@dataclass
class PerCoreCompileTimeDescriptor:
    """
    Descriptor for a compile-time arg with unique values per core.

    Unlike UnifiedCompileTimeCoreDescriptor which uses a single value for all cores
    in a range, this allows specifying different values for each individual core.

    Use case: When each core needs a unique compile-time value (e.g., bank_id, vc)
    that cannot be computed from grid position (e.g., scattered non-rectangular grids).

    Args:
        named_compile_time_arg: Name of the compile-time arg (string)
        core_values: List of (CoreCoord, value) tuples specifying the value for each core
        other_value: Value for any cores NOT in the core_values list
    """

    named_compile_time_arg: str
    core_values: list  # List of (CoreCoord, value) tuples
    other_value: int


@dataclass
class PerCoreRuntimeArgsDescriptor:
    """
    Descriptor for per-core runtime args for each RISC processor.

    Each RISC can have different runtime args per core. This allows specifying
    runtime args that vary by both RISC and core coordinate.

    Args:
        ncrisc_args: List of (CoreCoord, args_list) tuples for NCRISC
        brisc_args: List of (CoreCoord, args_list) tuples for BRISC
        trisc_args: List of (CoreCoord, args_list) tuples for TRISC
    """

    ncrisc_args: list = field(default_factory=list)
    brisc_args: list = field(default_factory=list)
    trisc_args: list = field(default_factory=list)


@dataclass
class UnifiedCompileTimeCoreDescriptor:
    """
    Descriptor for a compile-time arg that varies across core ranges.

    This descriptor is shared across all RISCs (NCRISC, BRISC, TRISC) and specializes
    all RISC kernels by breaking them into groups with unique compile-time values.

    Each core can use these values for compile-time logic with constexpr, enabling
    dead code elimination for different code paths on different cores.

    A major use case is differentiating the role of each core in the grid. For example:
    - Is the core performing RMSNorm vs MatMul?
    - Is the core a multicast sender or receiver?
    - Is the core participating in a gather or scatter operation?

    Args:
        named_compile_time_arg: Name of the compile-time arg (string)
        core_ranges: Cores that should have the specified value (must be within full
                     core_ranges passed to UnifiedKernelDescriptor)
        value: Value of the arg for cores in core_range
        other_value: Value of the arg for all cores NOT in core_range
    """

    named_compile_time_arg: str
    core_range: Union[ttnn.CoreCoord, ttnn.CoreRange, ttnn.CoreRangeSet]
    value: int
    other_value: int


@dataclass
class UnifiedKernelDescriptor:
    """
    Descriptor for a unified kernel that compiles for NCRISC, BRISC, and TRISC.

    A unified kernel is a single .cpp file with #if defined(COMPILE_FOR_NCRISC/BRISC/TRISC)
    sections that compile to different code for each RISC processor.

    Args:
        kernel_source: Path to the unified kernel source file
        core_ranges: CoreRangeSet where the kernel will execute
        ncrisc_compile_time_args: Compile-time args for reader (NCRISC)
        brisc_compile_time_args: Compile-time args for writer (BRISC)
        trisc_compile_time_args: Compile-time args for compute (TRISC)
        ncrisc_named_compile_time_args: Named compile-time args for reader (NCRISC)
        brisc_named_compile_time_args: Named compile-time args for writer (BRISC)
        trisc_named_compile_time_args: Named compile-time args for compute (TRISC)
        ncrisc_common_runtime_args: Common runtime args for reader (shared across all cores)
        brisc_common_runtime_args: Common runtime args for writer (shared across all cores)
        trisc_common_runtime_args: Common runtime args for compute (shared across all cores)
        trisc_compute_config: Optional compute configuration (math fidelity, fp32 acc, etc.)
        unified_compile_time_core_descriptors: List of per-core-range compile-time arg overrides
        per_core_compile_time_descriptors: List of PerCoreCompileTimeDescriptor for individual core values
        per_core_runtime_args_descriptor: PerCoreRuntimeArgsDescriptor for per-core runtime args
        defines: Preprocessor definitions as list of (name, value) tuples, e.g. [("SKIP_CCL", "1")]
    """

    kernel_source: str
    core_ranges: ttnn.CoreRangeSet
    ncrisc_compile_time_args: list = field(default_factory=list)
    brisc_compile_time_args: list = field(default_factory=list)
    trisc_compile_time_args: list = field(default_factory=list)
    ncrisc_named_compile_time_args: list = field(default_factory=list)
    brisc_named_compile_time_args: list = field(default_factory=list)
    trisc_named_compile_time_args: list = field(default_factory=list)
    ncrisc_common_runtime_args: list = field(default_factory=list)
    brisc_common_runtime_args: list = field(default_factory=list)
    trisc_common_runtime_args: list = field(default_factory=list)
    trisc_compute_config: Optional[ttnn.ComputeConfigDescriptor] = None
    unified_compile_time_core_descriptors: list = field(default_factory=list)
    per_core_compile_time_descriptors: list = field(default_factory=list)  # List of PerCoreCompileTimeDescriptor
    per_core_runtime_args_descriptor: Optional[
        PerCoreRuntimeArgsDescriptor
    ] = None  # Per-core runtime args for all RISCs
    defines: list = field(default_factory=list)  # List of (name, value) tuples

    def _get_core_range_set(
        self, core_range: Union[ttnn.CoreCoord, ttnn.CoreRange, ttnn.CoreRangeSet]
    ) -> ttnn.CoreRangeSet:
        """Convert CoreCoord/CoreRange to CoreRangeSet if needed."""
        if isinstance(core_range, ttnn.CoreRangeSet):
            return core_range
        if isinstance(core_range, ttnn.CoreCoord):
            return ttnn.CoreRangeSet([ttnn.CoreRange(core_range, core_range)])
        return ttnn.CoreRangeSet([core_range])

    def _build_runtime_args(self, risc: str, core_range_set: ttnn.CoreRangeSet) -> ttnn.RuntimeArgs:
        """
        Build RuntimeArgs for a specific RISC based on runtime_args_descriptor.

        Args:
            risc: Which RISC processor ("ncrisc", "brisc", or "trisc")
            core_range_set: The cores to build args for

        Returns:
            RuntimeArgs object (empty if no descriptor or no args for this RISC)
        """
        runtime_args = ttnn.RuntimeArgs()

        if self.per_core_runtime_args_descriptor is None:
            return runtime_args

        # Get the args list for this RISC
        if risc == "ncrisc":
            core_args_list = self.per_core_runtime_args_descriptor.ncrisc_args
        elif risc == "brisc":
            core_args_list = self.per_core_runtime_args_descriptor.brisc_args
        else:  # trisc
            core_args_list = self.per_core_runtime_args_descriptor.trisc_args

        if not core_args_list:
            return runtime_args

        # Build lookup from core coord to args
        core_to_args = {}  # {(x, y): args_list}
        for core_coord, args in core_args_list:
            core_key = (core_coord.x, core_coord.y)
            if core_key not in core_to_args:
                core_to_args[core_key] = []
            core_to_args[core_key].extend(args)

        # Build RuntimeArgs for all cores in the range
        all_cores = ttnn.corerange_to_cores(core_range_set)
        for core_coord in all_cores:
            core_key = (core_coord.x, core_coord.y)
            args = core_to_args.get(core_key, [])
            runtime_args[core_coord.x][core_coord.y] = list(args)

        return runtime_args

    def get_kernel_descriptors(self) -> UnifiedKernelResult:
        """
        Generate kernel descriptors for all RISC processors.

        If both unified_compile_time_core_descriptors and per_core_compile_time_descriptors
        are empty, returns 3 descriptors (one each for NCRISC, BRISC, TRISC).

        If either is specified, returns multiple kernel descriptors per processor
        to handle different compile-time args for different core ranges.

        Returns:
            UnifiedKernelResult containing:
            - kernels: List of KernelDescriptor objects
            - groups: List of KernelGroup objects for identifying kernels by role
        """
        if not self.unified_compile_time_core_descriptors and not self.per_core_compile_time_descriptors:
            # Simple case: no per-core compile-time arg overrides
            return self._get_simple_kernel_descriptors()

        # Complex case: generate multiple kernels per processor for different core ranges
        return self._get_split_kernel_descriptors()

    def _get_simple_kernel_descriptors(self) -> UnifiedKernelResult:
        """Generate simple kernel descriptors without per-core overrides."""
        # Build runtime args from per_core_runtime_args_descriptors for each RISC
        ncrisc_runtime_args = self._build_runtime_args("ncrisc", self.core_ranges)
        brisc_runtime_args = self._build_runtime_args("brisc", self.core_ranges)
        trisc_runtime_args = self._build_runtime_args("trisc", self.core_ranges)

        # Reader kernel (NCRISC)
        reader_descriptor = ttnn.KernelDescriptor(
            kernel_source=self.kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=self.ncrisc_compile_time_args,
            named_compile_time_args=self.ncrisc_named_compile_time_args,
            defines=self.defines,
            common_runtime_args=self.ncrisc_common_runtime_args,
            runtime_args=ncrisc_runtime_args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1,
                noc=ttnn.NOC.RISCV_1_default,
                noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
            ),
        )

        # Writer kernel (BRISC)
        writer_descriptor = ttnn.KernelDescriptor(
            kernel_source=self.kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=self.brisc_compile_time_args,
            named_compile_time_args=self.brisc_named_compile_time_args,
            defines=self.defines,
            common_runtime_args=self.brisc_common_runtime_args,
            runtime_args=brisc_runtime_args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default,
                noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
            ),
        )

        # Compute kernel (TRISC)
        compute_config = self.trisc_compute_config or ttnn.ComputeConfigDescriptor()
        compute_descriptor = ttnn.KernelDescriptor(
            kernel_source=self.kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=self.trisc_compile_time_args,
            named_compile_time_args=self.trisc_named_compile_time_args,
            defines=self.defines,
            common_runtime_args=self.trisc_common_runtime_args,
            runtime_args=trisc_runtime_args,
            config=compute_config,
        )

        kernels = [reader_descriptor, writer_descriptor, compute_descriptor]
        # Single group with all cores, no special compile-time arg values
        group = KernelGroup(
            core_range_set=self.core_ranges,
            compile_time_arg_values={},
            ncrisc_kernel_index=0,
            brisc_kernel_index=1,
            trisc_kernel_index=2,
        )
        return UnifiedKernelResult(kernels=kernels, groups=[group])

    def _get_split_kernel_descriptors(self) -> UnifiedKernelResult:
        """
        Generate split kernel descriptors with per-core compile-time arg overrides.

        This method properly handles overlapping descriptors by:
        1. Enumerating all cores in the full core_ranges
        2. Computing each core's complete set of named compile-time args
        3. Grouping cores by their unique arg combinations
        4. Creating one kernel set (NCRISC/BRISC/TRISC) per unique group

        Returns:
            UnifiedKernelResult with kernels and groups for easy index lookup.
        """
        # Step 1: Enumerate all cores
        all_cores = ttnn.corerange_to_cores(self.core_ranges)

        # Preconvert desc.core_range to CoreRangeSets for all unified descriptors
        desc_core_range_sets = [
            self._get_core_range_set(desc.core_range) for desc in self.unified_compile_time_core_descriptors
        ]

        # Preprocess per-core descriptors into lookup dicts for fast access
        per_core_lookup = {}  # {arg_name: {(x, y): value, ...}}
        for desc in self.per_core_compile_time_descriptors:
            arg_lookup = {}
            for core_coord, value in desc.core_values:
                arg_lookup[(core_coord.x, core_coord.y)] = value
            per_core_lookup[desc.named_compile_time_arg] = (arg_lookup, desc.other_value)

        # Step 2: For each core, compute its complete named compile-time args
        core_to_args = {}
        for core_coord in all_cores:
            args = {}
            # Process unified compile-time core descriptors (range-based)
            for desc, desc_core_range in zip(self.unified_compile_time_core_descriptors, desc_core_range_sets):
                if desc_core_range.contains(core_coord):
                    args[desc.named_compile_time_arg] = desc.value
                else:
                    args[desc.named_compile_time_arg] = desc.other_value

            # Process per-core compile-time descriptors (individual core values)
            core_key = (core_coord.x, core_coord.y)
            for arg_name, (arg_lookup, other_value) in per_core_lookup.items():
                if core_key in arg_lookup:
                    args[arg_name] = arg_lookup[core_key]
                else:
                    args[arg_name] = other_value

            # Convert to frozen (hashable) form, use (x, y) tuple as key
            core_to_args[(core_coord.x, core_coord.y)] = tuple(sorted(args.items()))

        # Step 3: Group cores by their unique arg combinations
        args_to_cores = {}
        for core_tuple, frozen_args in core_to_args.items():
            args_to_cores.setdefault(frozen_args, []).append(core_tuple)

        # Step 4: Create kernel descriptors for each unique group
        descriptors = []
        groups = []
        compute_config = self.trisc_compute_config or ttnn.ComputeConfigDescriptor()

        for frozen_args, core_tuples in args_to_cores.items():
            # Convert frozen args to list of tuples and dict
            unified_named_args = list(frozen_args)
            arg_values_dict = dict(frozen_args)

            # Combine with common named_compile_time_args for each RISC
            ncrisc_named_compile_time_args_merged = list(self.ncrisc_named_compile_time_args) + unified_named_args
            brisc_named_compile_time_args_merged = list(self.brisc_named_compile_time_args) + unified_named_args
            trisc_named_compile_time_args_merged = list(self.trisc_named_compile_time_args) + unified_named_args

            # Convert cores to CoreRangeSet
            core_coords = [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(core_tuples)]
            core_range_set = ttnn.CoreRangeSet(core_coords)

            # Build runtime args for this core range for each RISC
            ncrisc_runtime_args = self._build_runtime_args("ncrisc", core_range_set)
            brisc_runtime_args = self._build_runtime_args("brisc", core_range_set)
            trisc_runtime_args = self._build_runtime_args("trisc", core_range_set)

            # Track kernel indices for this group
            ncrisc_idx = len(descriptors)
            descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=self.kernel_source,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_range_set,
                    compile_time_args=self.ncrisc_compile_time_args,
                    named_compile_time_args=ncrisc_named_compile_time_args_merged,
                    defines=self.defines,
                    common_runtime_args=self.ncrisc_common_runtime_args,
                    runtime_args=ncrisc_runtime_args,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1,
                        noc=ttnn.NOC.RISCV_1_default,
                        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                    ),
                )
            )

            brisc_idx = len(descriptors)
            descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=self.kernel_source,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_range_set,
                    compile_time_args=self.brisc_compile_time_args,
                    named_compile_time_args=brisc_named_compile_time_args_merged,
                    defines=self.defines,
                    common_runtime_args=self.brisc_common_runtime_args,
                    runtime_args=brisc_runtime_args,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0,
                        noc=ttnn.NOC.RISCV_0_default,
                        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                    ),
                )
            )

            trisc_idx = len(descriptors)
            descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=self.kernel_source,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_range_set,
                    compile_time_args=self.trisc_compile_time_args,
                    named_compile_time_args=trisc_named_compile_time_args_merged,
                    defines=self.defines,
                    common_runtime_args=self.trisc_common_runtime_args,
                    runtime_args=trisc_runtime_args,
                    config=compute_config,
                )
            )

            # Create group with indices
            groups.append(
                KernelGroup(
                    core_range_set=core_range_set,
                    compile_time_arg_values=arg_values_dict,
                    ncrisc_kernel_index=ncrisc_idx,
                    brisc_kernel_index=brisc_idx,
                    trisc_kernel_index=trisc_idx,
                )
            )

        return UnifiedKernelResult(kernels=descriptors, groups=groups)
