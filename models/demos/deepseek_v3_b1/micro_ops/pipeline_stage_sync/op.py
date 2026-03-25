# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PipelineStageSync Operation using ttnn.generic_op

This module implements a multi-device synchronization on an arbitrary mesh.

Stalling device waits until a signal from the signalling device.

Stalling device and signalling device can be the same, as long as the stalling core and signalling core are not the same.
"""


import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor


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
        pseudo_input_tensor: ttnn.Tensor,
        pseudo_output_tensor: ttnn.Tensor,
        mesh_device: ttnn.MeshDevice,
        stalling_device_mesh_coord: ttnn.MeshCoordinate,
        stalling_core: ttnn.CoreCoord,
        run_stalling_kernel_on_brisc: bool,
        signalling_device_mesh_coord: ttnn.MeshCoordinate,
        signalling_core: ttnn.CoreCoord,
        run_signalling_kernel_on_brisc: bool,
        num_iterations: int = 1,
    ) -> None:
        """
        Execute PipelineStageSync using generic_op.

        Args:
            mesh_device: mesh_device micro op operates on
            stalling_device_mesh_coord: mesh coordinate on which the stalling_core stalls
            stalling_core: core that the stalling device stalls on
            run_stalling_kernel_on_brisc: whether to run the stalling kernel on brisc, if false runs on ncrisc
            signalling_device_mesh_coord: mesh coordinate on which the signalling_core signals
            signalling_core: core that the signalling device signals on
            run_signalling_kernel_on_brisc: whether to run the signalling kernel on brisc, if false runs on ncrisc
            num_iterations: Number of iterations to run inside the kernel

        Returns:
            None
        """

        # create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/pipeline_stage_sync/kernels/pipeline_stage_sync_kernel.cpp"

        is_stalling_device_equal_signalling_device = stalling_device_mesh_coord == signalling_device_mesh_coord
        is_stalling_core_equal_signalling_core = stalling_core == signalling_core
        is_stalling_kernel_and_signalling_kernel_on_same_risc = (
            run_stalling_kernel_on_brisc == run_signalling_kernel_on_brisc
        )
        assert not (
            is_stalling_device_equal_signalling_device
            and is_stalling_core_equal_signalling_core
            and is_stalling_kernel_and_signalling_kernel_on_same_risc
        ), f"If the stalling device is the same as the signalling device, and the stalling core is the same as the signalling core, then the stalling kernel must run on a different risc than the signalling kernel"

        global_semaphore = ttnn.create_global_semaphore(
            mesh_device, ttnn.CoreRangeSet([ttnn.CoreRange(stalling_core, stalling_core)]), 0
        )
        global_semaphore_addr = ttnn.get_global_semaphore_address(global_semaphore)

        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                mesh_coord = ttnn.MeshCoordinate(row, col)
                # device = mesh_device.get_device(row, col)

                # === Compile-time args ===

                # Reader (NCRISC) and Writer (BRISC) compile-time args
                stalling_device_fabric_node_id = mesh_device.get_fabric_node_id(stalling_device_mesh_coord)
                stalling_device_chip_id = int(stalling_device_fabric_node_id.chip_id)
                stalling_device_mesh_id = int(stalling_device_fabric_node_id.mesh_id)
                fabric_arg_base = 0

                reader_named_ct_args = [
                    ("is_stalling_device_equal_signalling_device", is_stalling_device_equal_signalling_device),
                    ("stalling_device_chip_id", stalling_device_chip_id),
                    ("stalling_device_mesh_id", stalling_device_mesh_id),
                    ("fabric_arg_base", fabric_arg_base),
                    ("num_iterations", num_iterations),
                ]
                writer_named_ct_args = reader_named_ct_args
                compute_name_ct_args = [("num_iterations", num_iterations)]

                # === Common Runtime Args ===
                stalling_core_phys = mesh_device.worker_core_from_logical_core(stalling_core)
                stalling_device_semaphore_noc_x_addr = stalling_core_phys.x
                stalling_device_semaphore_noc_y_addr = stalling_core_phys.y
                stalling_device_semaphore_l1_addr = global_semaphore_addr

                # Reader (NCRISC) and Writer (BRISC) common runtime args
                reader_common_rt_args = [
                    stalling_device_semaphore_noc_x_addr,
                    stalling_device_semaphore_noc_y_addr,
                    stalling_device_semaphore_l1_addr,
                ]
                writer_common_rt_args = reader_common_rt_args

                # === Unified Kernel Descriptor ===
                run_stalling_logic_on_ncrisc = (
                    mesh_coord == stalling_device_mesh_coord and not run_stalling_kernel_on_brisc
                )
                run_stalling_logic_on_brisc = mesh_coord == stalling_device_mesh_coord and run_stalling_kernel_on_brisc
                run_signalling_logic_on_ncrisc = (
                    mesh_coord == signalling_device_mesh_coord and not run_signalling_kernel_on_brisc
                )
                run_signalling_logic_on_brisc = (
                    mesh_coord == signalling_device_mesh_coord and run_signalling_kernel_on_brisc
                )

                if stalling_core == signalling_core:
                    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(stalling_core, stalling_core)])
                else:
                    core_ranges = ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(stalling_core, stalling_core),
                            ttnn.CoreRange(signalling_core, signalling_core),
                        ]
                    )
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=core_ranges,
                    ncrisc_named_compile_time_args=reader_named_ct_args,
                    trisc_named_compile_time_args=compute_name_ct_args,
                    brisc_named_compile_time_args=writer_named_ct_args,
                    ncrisc_common_runtime_args=reader_common_rt_args,
                    brisc_common_runtime_args=writer_common_rt_args,
                    per_core_compile_time_descriptors=[
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_stalling_logic_on_ncrisc",
                            core_values=[(stalling_core, run_stalling_logic_on_ncrisc), (signalling_core, 0)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_stalling_logic_on_brisc",
                            core_values=[(stalling_core, run_stalling_logic_on_brisc), (signalling_core, 0)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_signalling_logic_on_ncrisc",
                            core_values=[(stalling_core, 0), (signalling_core, run_signalling_logic_on_ncrisc)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_signalling_logic_on_brisc",
                            core_values=[(stalling_core, 0), (signalling_core, run_signalling_logic_on_brisc)],
                            other_value=0,
                        ),
                    ],
                )

                kernel_result = unified_kernel.get_kernel_descriptors()

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    semaphores=[],
                    cbs=[],
                )

                if (
                    mesh_coord == signalling_device_mesh_coord
                    and not stalling_device_mesh_coord == signalling_device_mesh_coord
                ):
                    if not run_signalling_kernel_on_brisc:
                        kernel_idx = kernel_result.get_group_by_arg(
                            "run_signalling_logic_on_ncrisc", 1
                        ).ncrisc_kernel_index
                    else:
                        kernel_idx = kernel_result.get_group_by_arg(
                            "run_signalling_logic_on_brisc", 1
                        ).brisc_kernel_index
                    program.kernels[kernel_idx].runtime_args[signalling_core.x][signalling_core.y] = []
                    per_core_rt_args_ref = program.kernels[kernel_idx].runtime_args[signalling_core.x][
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
        ttnn.generic_op([pseudo_input_tensor, pseudo_output_tensor], mesh_program_descriptor)
