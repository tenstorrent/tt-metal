# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PipelineStageSync Operation using ttnn.generic_op

This module implements a multi-device synchronization on an arbitrary mesh.

"""


import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor


def build_signalling_path(src_device_mesh_coord, dst_device_mesh_coord, mesh_rows, mesh_cols):
    src_row, src_col = src_device_mesh_coord[0], src_device_mesh_coord[1]
    dst_row, dst_col = dst_device_mesh_coord[0], dst_device_mesh_coord[1]

    path = [ttnn.MeshCoordinate(src_row, src_col)]
    current_row, current_col = src_row, src_col

    # traverse across rows first (wrap around present)
    positive_row_distance = (dst_row - src_row) % mesh_rows
    negative_row_distance = (src_row - dst_row) % mesh_rows

    if positive_row_distance <= negative_row_distance:
        row_step = 1
        row_num_hops = positive_row_distance
    else:
        row_step = -1
        row_num_hops = negative_row_distance

    for _ in range(row_num_hops):
        current_row = (current_row + row_step) % mesh_rows
        path.append(ttnn.MeshCoordinate(current_row, current_col))

    # traverse across cols second (wrap around not present)
    while current_col != dst_col:
        current_col += 1 if dst_col > current_col else -1
        path.append(ttnn.MeshCoordinate(current_row, current_col))

    # signalling and intermediate signalling sets
    signalling_devices = set(path[0:-1])
    intermediate_signalling_devices = set(path[1:-1])

    # build signalling device to target device mapping
    signaller_device_mapping = {}
    for i in range(len(path) - 1):
        signaller_device_mapping[path[i]] = path[i + 1]

    return signalling_devices, intermediate_signalling_devices, signaller_device_mapping


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
        semaphore,
        src_device_mesh_coord: ttnn.MeshCoordinate,
        signalling_core: ttnn.CoreCoord,
        run_signalling_kernel_on_ncrisc: bool,
        dst_device_mesh_coord: ttnn.MeshCoordinate,
        stalling_core: ttnn.CoreCoord,
        run_stalling_kernel_on_ncrisc: bool,
        num_iterations: int = 1,
    ) -> None:
        """
        Execute PipelineStageSync using generic_op.

        Args:
            mesh_device: mesh_device micro op operates on
            src_device_mesh_coord: src mesh coordinate
            signalling_core: core that signalling logic is executed on
            run_signalling_kernel_on_ncrisc: whether to run the signalling kernel on ncrisc or brisc
            dst_device_mesh_coord: dst mesh coordinate
            stalling_core: core that stalling logic is executed on
            run_stalling_kernel_on_ncrisc: whether to run the stalling kernel on ncrisc or brisc
            num_iterations: Number of iterations to run inside the kernel

        Returns:
            None
        """

        # mesh details
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/pipeline_stage_sync/kernels/pipeline_stage_sync_kernel.cpp"

        assert not (src_device_mesh_coord == dst_device_mesh_coord), f"src and dst device cannot be the same"

        # cores
        if stalling_core == signalling_core:
            all_cores_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(stalling_core, stalling_core)])
        else:
            all_cores_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(stalling_core, stalling_core),
                    ttnn.CoreRange(signalling_core, signalling_core),
                ]
            )

        signalling_core_phys = mesh_device.worker_core_from_logical_core(signalling_core)
        signalling_core_noc_x_addr = signalling_core_phys.x
        signalling_core_noc_y_addr = signalling_core_phys.y

        stalling_core_phys = mesh_device.worker_core_from_logical_core(stalling_core)
        stalling_core_noc_x_addr = stalling_core_phys.x
        stalling_core_noc_y_addr = stalling_core_phys.y

        # semaphores
        semaphore_l1_addr = ttnn.get_global_semaphore_address(semaphore)

        # construct path
        signalling_devices, intermediate_signalling_devices, signaller_device_mapping = build_signalling_path(
            src_device_mesh_coord, dst_device_mesh_coord, mesh_rows, mesh_cols
        )

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                mesh_coord = ttnn.MeshCoordinate(row, col)
                target_device = signaller_device_mapping.get(mesh_coord, None)

                # === Compile-time args ===

                # Reader (NCRISC) and Writer (BRISC) compile-time args
                is_intermediate_signaller = mesh_coord in intermediate_signalling_devices
                is_signalling_to_intermediate_signaller = (
                    mesh_coord in signalling_devices and target_device in intermediate_signalling_devices
                )

                fabric_arg_base = 0
                reader_named_ct_args = [
                    ("is_intermediate_signaller", is_intermediate_signaller),
                    ("is_signalling_to_intermediate_signaller", is_signalling_to_intermediate_signaller),
                    ("signalling_core_noc_x_addr", signalling_core_noc_x_addr),
                    ("signalling_core_noc_y_addr", signalling_core_noc_y_addr),
                    ("stalling_core_noc_x_addr", stalling_core_noc_x_addr),
                    ("stalling_core_noc_y_addr", stalling_core_noc_y_addr),
                    ("semaphore_l1_addr", semaphore_l1_addr),
                    ("fabric_arg_base", fabric_arg_base),
                    ("num_iterations", num_iterations),
                ]
                writer_named_ct_args = reader_named_ct_args
                compute_name_ct_args = [("num_iterations", num_iterations)]

                # === Unified Kernel Descriptor ===

                # select risc for signalling cores
                if mesh_coord in signalling_devices:
                    if run_signalling_kernel_on_ncrisc:
                        run_signalling_logic_on_ncrisc = True
                        run_signalling_logic_on_brisc = False
                    else:
                        run_signalling_logic_on_ncrisc = False
                        run_signalling_logic_on_brisc = True
                else:
                    run_signalling_logic_on_ncrisc = False
                    run_signalling_logic_on_brisc = False

                # select risc for stalling cores
                if mesh_coord == dst_device_mesh_coord:
                    if run_stalling_kernel_on_ncrisc:
                        run_stalling_logic_on_ncrisc = True
                        run_stalling_logic_on_brisc = False
                    else:
                        run_stalling_logic_on_ncrisc = False
                        run_stalling_logic_on_brisc = True
                else:
                    run_stalling_logic_on_ncrisc = False
                    run_stalling_logic_on_brisc = False

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
                            named_compile_time_arg="run_signalling_logic_on_ncrisc",
                            core_values=[(signalling_core, run_signalling_logic_on_ncrisc)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_signalling_logic_on_brisc",
                            core_values=[(signalling_core, run_signalling_logic_on_brisc)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_stalling_logic_on_ncrisc",
                            core_values=[(stalling_core, run_stalling_logic_on_ncrisc)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_stalling_logic_on_brisc",
                            core_values=[(stalling_core, run_stalling_logic_on_brisc)],
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
                if mesh_coord in signalling_devices:
                    if run_signalling_kernel_on_ncrisc:
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

                    fabric_src_node_id = mesh_device.get_fabric_node_id(mesh_coord)
                    fabric_dst_node_id = mesh_device.get_fabric_node_id(target_device)

                    link_index = 0
                    fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_src_node_id,
                        [fabric_dst_node_id],
                        [link_index],
                        program,
                        kernel_idx,
                        signalling_core,
                    )
                    per_core_rt_args_ref.extend(fabric_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(mesh_coord, mesh_coord)] = program

        # Execute pipeline_stage_sync operation
        ttnn.generic_op([pseudo_input_tensor, pseudo_output_tensor], mesh_program_descriptor)
