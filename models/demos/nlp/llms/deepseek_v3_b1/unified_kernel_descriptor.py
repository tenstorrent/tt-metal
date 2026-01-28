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

    def _get_core_range_set(
        self, core_range: Union[ttnn.CoreCoord, ttnn.CoreRange, ttnn.CoreRangeSet]
    ) -> ttnn.CoreRangeSet:
        """Convert CoreCoord/CoreRange to CoreRangeSet if needed."""
        if isinstance(core_range, ttnn.CoreRangeSet):
            return core_range
        if isinstance(core_range, ttnn.CoreCoord):
            return ttnn.CoreRangeSet([ttnn.CoreRange(core_range, core_range)])
        return ttnn.CoreRangeSet([core_range])

    def get_kernel_descriptors(self) -> list:
        """
        Generate kernel descriptors for all RISC processors.

        If unified_compile_time_core_descriptors is empty, returns 3 descriptors
        (one each for NCRISC, BRISC, TRISC).

        If unified_compile_time_core_descriptors is specified, returns multiple
        kernel descriptors per processor to handle different compile-time args
        for different core ranges.

        Returns:
            List of KernelDescriptor objects.
        """
        if not self.unified_compile_time_core_descriptors:
            # Simple case: no per-core compile-time arg overrides
            return self._get_simple_kernel_descriptors()

        # Complex case: generate multiple kernels per processor for different core ranges
        return self._get_split_kernel_descriptors()

    def _get_simple_kernel_descriptors(self) -> list:
        """Generate simple kernel descriptors without per-core overrides."""
        # Reader kernel (NCRISC)
        reader_descriptor = ttnn.KernelDescriptor(
            kernel_source=self.kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=self.ncrisc_compile_time_args,
            named_compile_time_args=self.ncrisc_named_compile_time_args,
            common_runtime_args=self.ncrisc_common_runtime_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Writer kernel (BRISC)
        writer_descriptor = ttnn.KernelDescriptor(
            kernel_source=self.kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=self.brisc_compile_time_args,
            named_compile_time_args=self.brisc_named_compile_time_args,
            common_runtime_args=self.brisc_common_runtime_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        # Compute kernel (TRISC)
        compute_config = self.trisc_compute_config or ttnn.ComputeConfigDescriptor()
        compute_descriptor = ttnn.KernelDescriptor(
            kernel_source=self.kernel_source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=self.trisc_compile_time_args,
            named_compile_time_args=self.trisc_named_compile_time_args,
            common_runtime_args=self.trisc_common_runtime_args,
            config=compute_config,
        )

        return [reader_descriptor, writer_descriptor, compute_descriptor]

    def _get_split_kernel_descriptors(self) -> list:
        """
        Generate split kernel descriptors with per-core compile-time arg overrides.

        This method properly handles overlapping descriptors by:
        1. Enumerating all cores in the full core_ranges
        2. Computing each core's complete set of named compile-time args
        3. Grouping cores by their unique arg combinations
        4. Creating one kernel set (NCRISC/BRISC/TRISC) per unique group
        """
        # Step 1: Enumerate all cores
        all_cores = ttnn.corerange_to_cores(self.core_ranges)

        # Preconvert desc.core_range to CoreRangeSets for all descriptors
        desc_core_range_sets = [
            self._get_core_range_set(desc.core_range) for desc in self.unified_compile_time_core_descriptors
        ]

        # Step 2: For each core, compute its complete named compile-time args
        core_to_args = {}
        for core_coord in all_cores:
            args = {}
            for desc, desc_core_range in zip(self.unified_compile_time_core_descriptors, desc_core_range_sets):
                if desc_core_range.contains(core_coord):
                    args[desc.named_compile_time_arg] = desc.value
                else:
                    args[desc.named_compile_time_arg] = desc.other_value
            # Convert to frozen (hashable) form, use (x, y) tuple as key
            core_to_args[(core_coord.x, core_coord.y)] = tuple(sorted(args.items()))

        # Step 3: Group cores by their unique arg combinations
        args_to_cores = {}
        for core_tuple, frozen_args in core_to_args.items():
            args_to_cores.setdefault(frozen_args, []).append(core_tuple)

        # Step 4: Create kernel descriptors for each unique group
        descriptors = []
        compute_config = self.trisc_compute_config or ttnn.ComputeConfigDescriptor()

        for frozen_args, core_tuples in args_to_cores.items():
            # Convert frozen args to list of tuples
            unified_named_args = list(frozen_args)

            # Combine with common named_compile_time_args for each RISC
            ncrisc_named_compile_time_args_merged = list(self.ncrisc_named_compile_time_args) + unified_named_args
            brisc_named_compile_time_args_merged = list(self.brisc_named_compile_time_args) + unified_named_args
            trisc_named_compile_time_args_merged = list(self.trisc_named_compile_time_args) + unified_named_args

            # Convert cores to CoreRangeSet
            core_coords = [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(core_tuples)]
            core_range_set = ttnn.CoreRangeSet(core_coords)

            # Create NCRISC/BRISC/TRISC kernel descriptors for this group
            descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=self.kernel_source,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_range_set,
                    compile_time_args=self.ncrisc_compile_time_args,
                    named_compile_time_args=ncrisc_named_compile_time_args_merged,
                    common_runtime_args=self.ncrisc_common_runtime_args,
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
            descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=self.kernel_source,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_range_set,
                    compile_time_args=self.brisc_compile_time_args,
                    named_compile_time_args=brisc_named_compile_time_args_merged,
                    common_runtime_args=self.brisc_common_runtime_args,
                    config=ttnn.WriterConfigDescriptor(),
                )
            )
            descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=self.kernel_source,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_range_set,
                    compile_time_args=self.trisc_compile_time_args,
                    named_compile_time_args=trisc_named_compile_time_args_merged,
                    common_runtime_args=self.trisc_common_runtime_args,
                    config=compute_config,
                )
            )

        return descriptors
