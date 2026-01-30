# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Kernel Fusion Framework for Unified Kernels

Automates the fusion of multiple ops into a single program by:
1. Allocating CB/semaphore offsets for each sub-program
2. Collecting all CB/semaphore descriptors
3. Merging named compile-time args with prefixes
4. Merging per-core runtime args
5. Creating a fused unified kernel descriptor

API Flow:
    global_program = GlobalProgram(device, core_ranges, fused_kernel_path)

    # Get offset, create sub-program, attach (updates offset internally)
    cb_offset, sem_offset, rt_offset = global_program.get_next_offsets()
    info_0 = SomeOp.create_program_info(..., cb_offset=cb_offset, sem_offset=sem_offset)
    global_program.attach(SubProgram.from_program_info("dsm0", info_0))

    cb_offset, sem_offset, rt_offset = global_program.get_next_offsets()
    info_1 = SomeOp.create_program_info(..., cb_offset=cb_offset, sem_offset=sem_offset)
    global_program.attach(SubProgram.from_program_info("dsm1", info_1))

    fused_descriptor = global_program.fuse()
    ttnn.generic_op(global_program.get_io_tensors(), fused_descriptor)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


@dataclass
class SubProgram:
    """
    Represents a sub-program extracted from an op for fusion.

    Contains all the information needed to fuse this op into a global program.
    Named compile-time args will be prefixed with the sub-program name.
    """

    name: str  # e.g., "dsm0", "dsm1" - used as prefix for named args
    program_descriptor: ttnn.ProgramDescriptor
    io_tensors: List[ttnn.Tensor]
    num_cbs: int
    num_semaphores: int
    num_runtime_args_per_core: int  # Per-core runtime args for BRISC (writer)

    # Named compile-time args per RISC (will be prefixed with name)
    ncrisc_named_compile_time_args: List[Tuple[str, int]] = field(default_factory=list)
    brisc_named_compile_time_args: List[Tuple[str, int]] = field(default_factory=list)
    trisc_named_compile_time_args: List[Tuple[str, int]] = field(default_factory=list)

    # Per-core runtime args per RISC
    brisc_runtime_args: List[Tuple[ttnn.CoreCoord, List[int]]] = field(default_factory=list)

    # Compute config (for TRISC defines like FUSE_SILU)
    trisc_compute_config: Optional[ttnn.ComputeConfigDescriptor] = None

    @classmethod
    def from_program_info(cls, name: str, program_info) -> "SubProgram":
        """
        Create a SubProgram from a program info object.

        The program_info should have:
        - program_descriptor: ttnn.ProgramDescriptor
        - io_tensors: List[ttnn.Tensor]
        - num_cbs: int
        - num_semaphores: int
        - num_runtime_args_per_core: int
        """
        # Extract named compile-time args and runtime args from kernels
        ncrisc_args = []
        brisc_args = []
        trisc_args = []
        brisc_rt_args = []
        trisc_config = None

        for kernel in program_info.program_descriptor.kernels:
            config = kernel.config
            if isinstance(config, ttnn.ReaderConfigDescriptor):
                # NCRISC
                if kernel.named_compile_time_args:
                    ncrisc_args = list(kernel.named_compile_time_args)
            elif isinstance(config, ttnn.WriterConfigDescriptor):
                # BRISC
                if kernel.named_compile_time_args:
                    brisc_args = list(kernel.named_compile_time_args)
                if kernel.runtime_args:
                    brisc_rt_args = list(kernel.runtime_args)
            elif isinstance(config, ttnn.ComputeConfigDescriptor):
                # TRISC
                if kernel.named_compile_time_args:
                    trisc_args = list(kernel.named_compile_time_args)
                trisc_config = config

        return cls(
            name=name,
            program_descriptor=program_info.program_descriptor,
            io_tensors=program_info.io_tensors,
            num_cbs=program_info.num_cbs,
            num_semaphores=program_info.num_semaphores,
            num_runtime_args_per_core=program_info.num_runtime_args_per_core,
            ncrisc_named_compile_time_args=ncrisc_args,
            brisc_named_compile_time_args=brisc_args,
            trisc_named_compile_time_args=trisc_args,
            brisc_runtime_args=brisc_rt_args,
            trisc_compute_config=trisc_config,
        )


class GlobalProgram:
    """
    Global program that fuses multiple sub-programs using unified kernels.

    Handles:
    - CB/semaphore offset allocation
    - Collecting all CB/semaphore descriptors
    - Merging named compile-time args with prefixes
    - Merging per-core runtime args
    - Creating fused unified kernel descriptor
    """

    def __init__(
        self,
        device: ttnn.Device,
        core_ranges: ttnn.CoreRangeSet,
        fused_kernel_path: str,
    ):
        """
        Create a GlobalProgram.

        Args:
            device: The device to run on
            core_ranges: The full core grid for the fused program
            fused_kernel_path: Path to the fused unified kernel file
        """
        self.device = device
        self.core_ranges = core_ranges
        self.fused_kernel_path = fused_kernel_path
        self.sub_programs: List[SubProgram] = []

        # Offset tracking
        self._next_cb_id = 0
        self._next_sem_id = 0
        self._next_rt_arg_offset = 0

    def get_next_offsets(self) -> Tuple[int, int, int]:
        """
        Get the next available offsets for CB, semaphore, and runtime args.

        Returns:
            (cb_offset, sem_offset, rt_arg_offset)
        """
        return self._next_cb_id, self._next_sem_id, self._next_rt_arg_offset

    def attach(self, sub_program: SubProgram) -> None:
        """
        Attach a sub-program to this global program.

        Updates internal offset counters based on the sub-program's resource usage.
        """
        self.sub_programs.append(sub_program)

        # Update offsets for next sub-program
        self._next_cb_id += sub_program.num_cbs
        self._next_sem_id += sub_program.num_semaphores
        self._next_rt_arg_offset += sub_program.num_runtime_args_per_core

    def _collect_cb_descriptors(self) -> List[ttnn.CBDescriptor]:
        """Collect all CB descriptors from all sub-programs."""
        cb_descriptors = []
        for sub_program in self.sub_programs:
            cb_descriptors.extend(sub_program.program_descriptor.cbs)
        return cb_descriptors

    def _collect_semaphore_descriptors(self) -> List[ttnn.SemaphoreDescriptor]:
        """Collect all semaphore descriptors from all sub-programs."""
        sem_descriptors = []
        for sub_program in self.sub_programs:
            sem_descriptors.extend(sub_program.program_descriptor.semaphores)
        return sem_descriptors

    def _prefix_named_args(self, args: List[Tuple[str, int]], prefix: str) -> List[Tuple[str, int]]:
        """Prefix named compile-time args with sub-program prefix."""
        prefixed = []
        for name, value in args:
            if name == "is_active_core":
                new_name = name  # Don't prefix is_active_core
            else:
                new_name = f"{prefix}_{name}"
            prefixed.append((new_name, value))
        return prefixed

    def _merge_named_compile_time_args(self) -> Tuple[List, List, List]:
        """
        Merge named compile-time args from all sub-programs.

        Each sub-program's args get prefixed with sub{index}_ to avoid conflicts.

        Returns:
            (ncrisc_args, brisc_args, trisc_args)
        """
        ncrisc_args = []
        brisc_args = []
        trisc_args = []

        for idx, sub_program in enumerate(self.sub_programs):
            prefix = f"sub{idx}"

            # Prefix and collect args
            ncrisc_args.extend(self._prefix_named_args(sub_program.ncrisc_named_compile_time_args, prefix))
            brisc_args.extend(self._prefix_named_args(sub_program.brisc_named_compile_time_args, prefix))
            trisc_args.extend(self._prefix_named_args(sub_program.trisc_named_compile_time_args, prefix))

        return ncrisc_args, brisc_args, trisc_args

    def _merge_runtime_args(self) -> List[Tuple[ttnn.CoreCoord, List[int]]]:
        """
        Merge per-core runtime args from all sub-programs.

        For each core, concatenates runtime args from all sub-programs in order.

        Returns:
            List of (CoreCoord, merged_args) tuples
        """
        # Build a map of core -> args
        core_args_map: Dict[Tuple[int, int], List[int]] = {}

        for sub_program in self.sub_programs:
            for core, args in sub_program.brisc_runtime_args:
                key = (core.x, core.y)
                if key not in core_args_map:
                    core_args_map[key] = []
                core_args_map[key].extend(args)

        # Convert back to list of (CoreCoord, args)
        result = []
        for (x, y), args in sorted(core_args_map.items()):
            result.append((ttnn.CoreCoord(x, y), args))
        return result

    def _generate_enable_args(self) -> List[Tuple[str, int]]:
        """
        Generate enable named compile-time args for each sub-program.

        Each sub-program gets an enable arg (e.g., "sub0_enabled" = 1).
        """
        enable_args = []
        for idx in range(len(self.sub_programs)):
            enable_args.append((f"sub{idx}_enabled", 1))

        return enable_args

    def _get_merged_trisc_config(self) -> ttnn.ComputeConfigDescriptor:
        """
        Get merged TRISC compute config.

        Uses the first sub-program's config as base and merges defines.
        """
        # Collect all defines from sub-programs
        all_defines = []
        base_config = None

        for sub_program in self.sub_programs:
            if sub_program.trisc_compute_config:
                if base_config is None:
                    base_config = sub_program.trisc_compute_config
                # Collect defines
                if hasattr(sub_program.trisc_compute_config, "defines") and sub_program.trisc_compute_config.defines:
                    all_defines.extend(sub_program.trisc_compute_config.defines)

        if base_config is None:
            return ttnn.ComputeConfigDescriptor()

        # Create new config with merged defines
        return ttnn.ComputeConfigDescriptor(
            math_fidelity=base_config.math_fidelity,
            fp32_dest_acc_en=base_config.fp32_dest_acc_en,
            math_approx_mode=base_config.math_approx_mode,
            defines=all_defines if all_defines else None,
        )

    def fuse(self) -> ttnn.ProgramDescriptor:
        """
        Fuse all attached sub-programs into a single program descriptor.

        Returns:
            A fused ProgramDescriptor ready for execution.
        """
        # Collect all CB and semaphore descriptors
        cb_descriptors = self._collect_cb_descriptors()
        sem_descriptors = self._collect_semaphore_descriptors()

        # Merge named compile-time args
        ncrisc_args, brisc_args, trisc_args = self._merge_named_compile_time_args()

        # Merge runtime args
        merged_brisc_runtime_args = self._merge_runtime_args()

        # Generate enable named compile-time args (e.g., "dsm0_enabled" = 1)
        enable_args = self._generate_enable_args()

        # Add enable args to all RISCs
        ncrisc_args = ncrisc_args + enable_args
        brisc_args = brisc_args + enable_args
        trisc_args = trisc_args + enable_args

        # Get merged compute config
        trisc_config = self._get_merged_trisc_config()

        # Create unified kernel descriptor
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=self.fused_kernel_path,
            core_ranges=self.core_ranges,
            ncrisc_named_compile_time_args=ncrisc_args,
            brisc_named_compile_time_args=brisc_args,
            trisc_named_compile_time_args=trisc_args,
            brisc_runtime_args=merged_brisc_runtime_args,
            trisc_compute_config=trisc_config,
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_active_core",
                    core_range=self.core_ranges,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        return ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=cb_descriptors,
            semaphores=sem_descriptors,
        )

    def get_io_tensors(self) -> List[ttnn.Tensor]:
        """Get all I/O tensors from all sub-programs."""
        io_tensors = []
        for sub_program in self.sub_programs:
            io_tensors.extend(sub_program.io_tensors)
        return io_tensors
