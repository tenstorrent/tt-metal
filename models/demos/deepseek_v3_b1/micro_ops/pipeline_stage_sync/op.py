# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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
        dst_device_mesh_coord: ttnn.MeshCoordinate,
        stalling_core: ttnn.CoreCoord,
        run_stalling_kernel_on_ncrisc: bool,
        src_device_mesh_coord: ttnn.MeshCoordinate,
        signalling_core: ttnn.CoreCoord,
        run_signalling_kernel_on_ncrisc: bool,
        num_iterations: int = 1,
    ) -> None:
        """
        Execute PipelineStageSync using generic_op.

        Args:
            mesh_device: mesh_device micro op operates on
            dst_device_mesh_coord: mesh coordinate on which the stalling_core stalls
            stalling_core: core that the stalling device stalls on
            run_stalling_kernel_on_ncrisc: whether to run the stalling kernel on ncrisc or brisc
            src_device_mesh_coord: mesh coordinate on which the signalling_core signals
            signalling_core: core that the signalling device signals on
            run_signalling_kernel_on_ncrisc: whether to run the signalling kernel on ncrsic or brisc
            num_iterations: Number of iterations to run inside the kernel

        Returns:
            None
        """

        # create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/pipeline_stage_sync/kernels/pipeline_stage_sync_kernel.cpp"

        assert not (dst_device_mesh_coord == src_device_mesh_coord), f"Src and dst device cannot be the same"

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

        # semaphores
        global_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores_core_range_set, 0)
        semaphore_l1_addr = ttnn.get_global_semaphore_address(global_semaphore)

        # construct path - horizontal then vertical
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        src_row, src_col = src_device_mesh_coord[0], src_device_mesh_coord[1]
        dst_row, dst_col = dst_device_mesh_coord[0], dst_device_mesh_coord[1]

        path = []
        row, col = src_row, src_col
        path.append(ttnn.MeshCoordinate(row, col))

        forward_distance = (dst_row - src_row) % mesh_rows  # hops going +1
        backward_distance = (src_row - dst_row) % mesh_rows  # hops going -1

        if forward_distance <= backward_distance:
            row_step = 1
            row_hops = forward_distance
        else:
            row_step = -1
            row_hops = backward_distance

        for _ in range(row_hops):
            row = (row + row_step) % mesh_rows
            path.append(ttnn.MeshCoordinate(row, col))

        # Column dimension: no wrap, just step linearly
        while col != dst_col:
            col += 1 if dst_col > col else -1
            path.append(ttnn.MeshCoordinate(row, col))

        # Build lookup structures (same as before)
        signaller_devices_mapping = {}
        for i in range(len(path) - 1):
            signaller_devices_mapping[path[i]] = path[i + 1]

        signaller_devices = set(path[0:-1])
        intermediate_signaller_devices = set(path[1:-1])

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                mesh_coord = ttnn.MeshCoordinate(row, col)
                target_device = signaller_devices_mapping.get(mesh_coord, None)

                # === Compile-time args ===

                # Reader (NCRISC) and Writer (BRISC) compile-time args
                is_intermediate_signaller = mesh_coord in intermediate_signaller_devices
                is_signalling_to_intermediate_signaller = (
                    mesh_coord in signaller_devices and target_device in intermediate_signaller_devices
                )

                stalling_core_phys = mesh_device.worker_core_from_logical_core(stalling_core)
                stalling_core_noc_x_addr = stalling_core_phys.x
                stalling_core_noc_y_addr = stalling_core_phys.y

                signalling_core_phys = mesh_device.worker_core_from_logical_core(signalling_core)
                signalling_core_noc_x_addr = signalling_core_phys.x
                signalling_core_noc_y_addr = signalling_core_phys.y

                fabric_arg_base = 0
                reader_named_ct_args = [
                    ("is_intermediate_signaller", is_intermediate_signaller),
                    ("is_signalling_to_intermediate_signaller", is_signalling_to_intermediate_signaller),
                    ("stalling_core_noc_x_addr", stalling_core_noc_x_addr),
                    ("stalling_core_noc_y_addr", stalling_core_noc_y_addr),
                    ("signalling_core_noc_x_addr", signalling_core_noc_x_addr),
                    ("signalling_core_noc_y_addr", signalling_core_noc_y_addr),
                    ("semaphore_l1_addr", semaphore_l1_addr),
                    ("fabric_arg_base", fabric_arg_base),
                    ("num_iterations", num_iterations),
                ]
                writer_named_ct_args = reader_named_ct_args
                compute_name_ct_args = [("num_iterations", num_iterations)]

                # === Unified Kernel Descriptor ===

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

                # select risc for signalling cores
                if mesh_coord in signaller_devices:
                    if run_signalling_kernel_on_ncrisc:
                        run_signalling_logic_on_ncrisc = True
                        run_signalling_logic_on_brisc = False
                    else:
                        run_signalling_logic_on_ncrisc = False
                        run_signalling_logic_on_brisc = True
                else:
                    run_signalling_logic_on_ncrisc = False
                    run_signalling_logic_on_brisc = False

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
                            named_compile_time_arg="run_stalling_logic_on_ncrisc",
                            core_values=[(stalling_core, run_stalling_logic_on_ncrisc)],
                            other_value=0,
                        ),
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="run_stalling_logic_on_brisc",
                            core_values=[(stalling_core, run_stalling_logic_on_brisc)],
                            other_value=0,
                        ),
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
                    ],
                )

                kernel_result = unified_kernel.get_kernel_descriptors()

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    semaphores=[],
                    cbs=[],
                )

                # TODO: (GR)
                if mesh_coord in signaller_devices:
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

                    # signalling_fabric_node_id = mesh_device.get_fabric_node_id(src_device_mesh_coord)
                    # stalling_device_fabric_node_id = mesh_device.get_fabric_node_id(dst_device_mesh_coord)

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
