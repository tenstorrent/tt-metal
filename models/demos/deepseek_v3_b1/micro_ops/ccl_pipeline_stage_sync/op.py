# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PipelineStageSync Operation using ttnn.generic_op

This module implements a multi-device synchronization on an arbitrary mesh.

Stalling device waits until a signal from the signalling device.

Stalling device and signalling device can be the same, as long as the stalling core and signalling core are not the same.
"""


import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class PipelineStageSync:
    """
    Multi-device PipelineStageSync implementation using ttnn.generic_op.
    """

    @staticmethod
    def golden() -> None:
        # NOTE: micro op is purely for synchronization, no associated golden
        return

    @staticmethod
    def op(
        mesh_device: ttnn.MeshDevice,
        stalling_device_mesh_coord: ttnn.MeshCoordinate,
        stalling_core: ttnn.CoreCoord,
        signalling_device_mesh_coord: ttnn.MeshCoordinate,
        signalling_core: ttnn.CoreCoord,
        num_iterations: int = 1,
    ) -> None:
        """
        Execute PipelineStageSync using generic_op.

        Args:
            mesh_device: mesh_device micro op operates on
            stalling_device_mesh_coord: mesh coordinate on which the stalling_core stalls
            stalling_core: core that the stalling device stalls on
            signalling_device_mesh_coord: mesh coordinate on which the signalling_core signals
            signalling_core: core that the signalling device signals on
            num_iterations: Number of iterations to run inside the kernel

        Returns:
            None
        """

        # create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/pipeline_stage_sync/kernels/pipeline_stage_sync_kernel.cpp"

        is_stalling_device_equal_signalling_device = stalling_device_mesh_coord == signalling_device_mesh_coord
        assert not (
            is_stalling_device_equal_signalling_device and stalling_core == signalling_core
        ), f"If the stalling device is the same as the signalling device, then the stalling core must be different than the siganlling core"

        global_semaphore = ttnn.create_global_semaphore(
            mesh_device, ttnn.CoreRangeSet([ttnn.CoreRange(stalling_device_mesh_coord, stalling_device_mesh_coord)]), 0
        )
        global_semaphore_addr = ttnn.get_global_semaphore_address(global_semaphore)

        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                mesh_coord = ttnn.MeshCoordinate(row, col)
                device = mesh_device.get_device(row, col)

                # === Compile-time args ===

                # Reader (NCRISC) compile-time args
                reader_named_ct_args = [
                    ("is_stalling_device", stalling_device_mesh_coord == mesh_coord),
                    ("num_iterations", num_iterations),
                ]

                # Writer (BRISC) compile-time args
                stalling_device_fabric_node_id = mesh_device.get_fabric_node_id(stalling_device_mesh_coord)
                stalling_device_chip_id = int(stalling_device_fabric_node_id.chip_id)
                stalling_device_mesh_id = int(stalling_device_fabric_node_id.mesh_id)

                fabric_arg_base = 0
                writer_named_ct_args = [
                    ("is_signalling_device", signalling_device_mesh_coord == mesh_coord),
                    ("is_stalling_device_equal_signalling_device", is_stalling_device_equal_signalling_device),
                    ("stalling_device_chip_id", stalling_device_chip_id),
                    ("stalling_device_mesh_id", stalling_device_mesh_id),
                    ("fabric_arg_base", fabric_arg_base),
                    ("num_iterations", num_iterations),
                ]

                # === Common Runtime Args ===
                stalling_core_phys = device.worker_core_from_logical_core(stalling_core)
                stalling_device_semaphore_noc_x_addr = stalling_core_phys.x
                stalling_device_semaphore_noc_y_addr = stalling_core_phys.y
                stalling_device_semaphore_l1_addr = global_semaphore_addr

                # Reader (NCRISC) common runtime args
                reader_common_rt_args = [stalling_device_semaphore_l1_addr]

                # Reader (BRISC) common runtime args
                writer_common_rt_args = [
                    stalling_device_semaphore_noc_x_addr,
                    stalling_device_semaphore_noc_y_addr,
                    stalling_device_semaphore_l1_addr,
                    stalling_device_chip_id,
                    stalling_device_mesh_id,
                ]

                # === Unified Kernel Descriptor ===
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(stalling_core, stalling_core),
                            ttnn.CoreRange(signalling_core, signalling_core),
                        ]
                    ),
                    ncrisc_named_compile_time_args=reader_named_ct_args,
                    brisc_named_compile_time_args=writer_named_ct_args,
                    ncrisc_common_runtime_args=reader_common_rt_args,
                    brisc_common_runtime_args=writer_common_rt_args,
                )

                kernel_result = unified_kernel.get_kernel_descriptors()

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    semaphores=[],
                    cbs=[],
                )

                if mesh_coord == signalling_device_mesh_coord:
                    brisc_kernel_idx = kernel_result.groups[0].brisc_kernel_index
                    per_core_rt_args_ref = program.kernels[brisc_kernel_idx].runtime_args[signalling_core.x][
                        signalling_core.y
                    ]

                    signalling_fabric_node_id = mesh_device.get_fabric_node_id(signalling_device_mesh_coord)
                    link_index = 0
                    fabric_args = ttnn.setup_fabric_connection(
                        signalling_fabric_node_id,
                        stalling_device_fabric_node_id,
                        link_index,
                        program,
                        signalling_core,
                    )
                    per_core_rt_args_ref.extend(fabric_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(mesh_coord, mesh_coord)] = program

        # Execute pipeline_stage_sync operation
        ttnn.generic_op([], mesh_program_descriptor)
