# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
ColumnWisePipelineStageSync Operation using ttnn.generic_op

This module implements a multi-device synchronization on an arbitrary mesh.

"""


import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor


class ColumnWisePipelineStageSync:
    """
    Multi-device ColumnWisePipelineStageSync implementation using ttnn.generic_op.
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
        semaphores: list,
        entry_device_mesh_col: int,
        exit_device_mesh_col: int,
        entry_device_core_coord: ttnn.CoreCoord,
        run_entry_device_logic_on_ncrisc: bool,
        exit_device_core_coord: ttnn.CoreCoord,
        run_exit_device_logic_on_ncrisc: bool,
        num_iterations: int = 1,
    ) -> None:
        """
        Execute ColumnWisePipelineStageSync using generic_op.

        Returns:
            None
        """

        if entry_device_mesh_col == exit_device_mesh_col:
            raise ValueError("entry and exit mesh columns cannot be the same")

        # mesh details
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        if mesh_cols != 2 or entry_device_mesh_col >= 2 or exit_device_mesh_col >= 2:
            raise ValueError("requires operating on a 2 column mesh")

        # create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/column_wise_pipeline_stage_sync/kernels/column_wise_pipeline_stage_sync_kernel.cpp"

        # cores
        if entry_device_core_coord == exit_device_core_coord:
            all_cores_core_range_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(entry_device_core_coord, entry_device_core_coord)]
            )
        else:
            all_cores_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(entry_device_core_coord, entry_device_core_coord),
                    ttnn.CoreRange(exit_device_core_coord, exit_device_core_coord),
                ]
            )

        entry_device_core_phys = mesh_device.worker_core_from_logical_core(entry_device_core_coord)
        entry_device_core_noc_x_addr = entry_device_core_phys.x
        entry_device_core_noc_y_addr = entry_device_core_phys.y

        # semaphores
        if len(semaphores) != 3:
            raise ValueError("requires 3 semaphores")
        r1_semaphore_l1_addr = ttnn.get_global_semaphore_address(semaphores[0])
        r2_semaphore_l1_addr = ttnn.get_global_semaphore_address(semaphores[1])
        r3_semaphore_l1_addr = ttnn.get_global_semaphore_address(semaphores[2])

        # device loop
        for mesh_row in range(mesh_rows):
            for mesh_col in range(mesh_cols):
                mesh_coord = ttnn.MeshCoordinate(mesh_row, mesh_col)

                is_entry_device = mesh_col == entry_device_mesh_col

                # === Compile-time args ===

                # Reader (NCRISC) and Writer (BRISC) compile-time args
                fabric_arg_base = 0
                reader_named_ct_args = [
                    ("entry_device_core_noc_x_addr", entry_device_core_noc_x_addr),
                    ("entry_device_core_noc_y_addr", entry_device_core_noc_y_addr),
                    ("r1_semaphore_l1_addr", r1_semaphore_l1_addr),
                    ("r2_semaphore_l1_addr", r2_semaphore_l1_addr),
                    ("r3_semaphore_l1_addr", r3_semaphore_l1_addr),
                    ("fabric_arg_base", fabric_arg_base),
                    ("num_iterations", num_iterations),
                ]
                writer_named_ct_args = reader_named_ct_args
                compute_name_ct_args = [("num_iterations", num_iterations)]

                # select risc for entry and exit device cores
                if is_entry_device:
                    if run_entry_device_logic_on_ncrisc:
                        run_entry_device_logic_on_ncrisc = True
                        run_entry_device_logic_on_brisc = False
                    else:
                        run_entry_device_logic_on_ncrisc = False
                        run_entry_device_logic_on_brisc = True

                    run_exit_device_logic_on_ncrisc = False
                    run_exit_device_logic_on_brisc = False
                else:
                    if run_exit_device_logic_on_ncrisc:
                        run_exit_device_logic_on_ncrisc = True
                        run_exit_device_logic_on_brisc = False
                    else:
                        run_exit_device_logic_on_ncrisc = False
                        run_exit_device_logic_on_brisc = True

                    run_entry_device_logic_on_ncrisc = False
                    run_entry_device_logic_on_brisc = False

                # === Unified Kernel Descriptor ===
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=all_cores_core_range_set,
                    ncrisc_named_compile_time_args=reader_named_ct_args,
                    trisc_named_compile_time_args=compute_name_ct_args,
                    brisc_named_compile_time_args=writer_named_ct_args,
                    ncrisc_common_runtime_args=[],
                    brisc_common_runtime_args=[],
                    per_core_compile_time_descriptors=[
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_entry_device_logic_on_ncrisc",
                            core_values=[(entry_device_core_coord, run_entry_device_logic_on_ncrisc)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_entry_device_logic_on_brisc",
                            core_values=[(entry_device_core_coord, run_entry_device_logic_on_brisc)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_exit_device_logic_on_ncrisc",
                            core_values=[(exit_device_core_coord, run_exit_device_logic_on_ncrisc)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_exit_device_logic_on_brisc",
                            core_values=[(exit_device_core_coord, run_exit_device_logic_on_brisc)],
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

                # Setup fabric
                if is_entry_device:
                    if run_entry_device_logic_on_ncrisc:
                        kernel_idx = kernel_result.get_group_by_arg(
                            "run_entry_device_logic_on_ncrisc", 1
                        ).ncrisc_kernel_index
                    else:
                        kernel_idx = kernel_result.get_group_by_arg(
                            "run_entry_device_logic_on_brisc", 1
                        ).brisc_kernel_index
                    program.kernels[kernel_idx].runtime_args[entry_device_core_coord.x][entry_device_core_coord.y] = []
                    per_core_rt_args_ref = program.kernels[kernel_idx].runtime_args[entry_device_core_coord.x][
                        entry_device_core_coord.y
                    ]

                    fabric_src_node_id = mesh_device.get_fabric_node_id(mesh_coord)
                    fabric_dst_node_one_id = mesh_device.get_fabric_node_id(
                        ttnn.MeshCoordinate((mesh_row - 1) % mesh_rows, mesh_col)
                    )
                    fabric_dst_node_two_id = mesh_device.get_fabric_node_id(
                        ttnn.MeshCoordinate((mesh_row + 1) % mesh_rows, mesh_col)
                    )

                    link_index = 0
                    fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_src_node_id,
                        [fabric_dst_node_one_id, fabric_dst_node_two_id],
                        [link_index, link_index],
                        program,
                        kernel_idx,
                        entry_device_core_coord,
                    )
                    per_core_rt_args_ref.extend(fabric_args)

                else:
                    if run_exit_device_logic_on_ncrisc:
                        kernel_idx = kernel_result.get_group_by_arg(
                            "run_exit_device_logic_on_ncrisc", 1
                        ).ncrisc_kernel_index
                    else:
                        kernel_idx = kernel_result.get_group_by_arg(
                            "run_exit_device_logic_on_brisc", 1
                        ).brisc_kernel_index
                    program.kernels[kernel_idx].runtime_args[exit_device_core_coord.x][exit_device_core_coord.y] = []
                    per_core_rt_args_ref = program.kernels[kernel_idx].runtime_args[exit_device_core_coord.x][
                        exit_device_core_coord.y
                    ]

                    fabric_src_node_id = mesh_device.get_fabric_node_id(mesh_coord)
                    fabric_dst_node_id = mesh_device.get_fabric_node_id(
                        ttnn.MeshCoordinate(mesh_row, (mesh_col + 1) % mesh_cols)
                    )

                    link_index = 0
                    fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_src_node_id,
                        [fabric_dst_node_id],
                        [link_index],
                        program,
                        kernel_idx,
                        exit_device_core_coord,
                    )
                    per_core_rt_args_ref.extend(fabric_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(mesh_coord, mesh_coord)] = program

        # Execute pipeline_stage_sync operation
        ttnn.generic_op([pseudo_input_tensor, pseudo_output_tensor], mesh_program_descriptor)
