# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class PersistentLoop:
    """Manages persistent loop state for compute kernels.

    Encapsulates termination and next-iteration semaphore creation,
    compile-time arg generation, and termination signaling.  Used by
    LMHeadStage, DecoderStage, and unit tests.
    """

    def __init__(self, mesh_device, core_range_set, persistent_mode=True):
        self.persistent_mode = persistent_mode
        self.termination_semaphore = None
        self.next_iter_semaphore = None
        if persistent_mode:
            self.termination_semaphore = ttnn.create_global_semaphore(mesh_device, core_range_set, 0)
            self.next_iter_semaphore = ttnn.create_global_semaphore(mesh_device, core_range_set, 1)

    @property
    def termination_semaphore_address(self):
        if self.termination_semaphore is not None:
            return int(ttnn.get_global_semaphore_address(self.termination_semaphore))
        return 0

    @property
    def next_iter_semaphore_address(self):
        if self.next_iter_semaphore is not None:
            return int(ttnn.get_global_semaphore_address(self.next_iter_semaphore))
        return 0

    def get_compile_time_args(self):
        """Named CT args common to all persistent compute kernels."""
        return [
            ("persistent_mode", 1 if self.persistent_mode else 0),
            ("termination_semaphore_addr", self.termination_semaphore_address),
        ]

    def terminate(self):
        """Signal the persistent kernel to exit. No-op when not in persistent mode."""
        if self.termination_semaphore is not None:
            ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)

    def reset(self):
        """Reset termination semaphore for a new launch cycle."""
        if self.termination_semaphore is not None:
            ttnn.reset_global_semaphore_value(self.termination_semaphore, 0)

    KERNEL_PATH = "models/demos/deepseek_v3_b1/micro_ops/persistent_loop/kernels/persistent_loop_kernel.cpp"

    def op(self, mesh_device, core_coord, iteration_count_semaphore, max_iterations=1):
        """Dispatch a minimal test kernel that exercises PersistentLoop."""
        iteration_count_addr = int(ttnn.get_global_semaphore_address(iteration_count_semaphore))
        core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord)])

        named_ct_args = self.get_compile_time_args() + [
            ("max_iterations", max_iterations),
            ("iteration_count_addr", iteration_count_addr),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=self.KERNEL_PATH,
            core_ranges=core_range_set,
            ncrisc_named_compile_time_args=named_ct_args,
            brisc_named_compile_time_args=named_ct_args,
            trisc_named_compile_time_args=named_ct_args,
        )

        kernel_result = unified_kernel.get_kernel_descriptors()

        program = ttnn.ProgramDescriptor(
            kernels=kernel_result.kernels,
            semaphores=[],
            cbs=[],
        )

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        coord = ttnn.MeshCoordinate(0, 0)
        mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
        )
        ttnn.generic_op([dummy_tensor, dummy_tensor], mesh_program_descriptor)
