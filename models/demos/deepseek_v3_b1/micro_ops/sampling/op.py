# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class SamplingOp:
    """
    Sampling micro-op entry point.

    Current implementation supports k=1 (argmax fast path) for a single device
    with multi-core local winners reduced on a final core.
    """

    @staticmethod
    def golden(scores: torch.Tensor, indices: torch.Tensor, k: int = 1, p: float = 1.0) -> torch.Tensor:
        """
        PyTorch reference for sampling.

        For k=1, this is argmax with deterministic tie-break on lowest index.
        Returns a [1, 1] uint32 tensor containing the selected index.
        """
        if k != 1:
            raise NotImplementedError("Sampling golden currently supports only k=1")

        scores_f32 = scores.float().reshape(-1)
        indices_i64 = indices.to(torch.int64).reshape(-1)
        max_score = torch.max(scores_f32)
        tied_mask = scores_f32 == max_score
        selected_index = torch.min(indices_i64[tied_mask]).to(torch.uint32)
        return selected_index.reshape(1, 1)

    @staticmethod
    def op(
        scores_tensor,
        indices_tensor,
        output_index_tensor,
        k: int,
        p: float,
        final_core_coord=None,
        final_mesh_coord=None,
        global_semaphore=None,
        global_stage2_semaphore=None,
        fabric_scratch_tensor=None,
        mesh_axis: str = "x",
    ):
        """
        Execute sampling.

        Args:
            scores_tensor: [1, 160 * num_cores] bfloat16, WIDTH_SHARDED with shard shape [1, 160].
            indices_tensor: [1, 160 * num_cores] uint32, WIDTH_SHARDED with shard shape [1, 160].
            output_index_tensor: [1, 1] uint32, sharded on final core.
            k: sampling k; currently only k=1 supported.
            p: top-p threshold (unused for k=1).
            final_core_coord: target output core coordinate (validated, optional).
            final_mesh_coord: mesh coordinate containing final output in mesh mode.
            global_semaphore: external global semaphore handle used for stage-1 inter-device sync.
            global_stage2_semaphore: external global semaphore handle used for stage-2 inter-device sync.
            fabric_scratch_tensor: persistent L1 scratch tensor (single-core sharded) for fabric slot exchange.
            mesh_axis: reduction axis for first stage; currently only "x" is supported.
        """
        if k != 1:
            raise NotImplementedError("Sampling currently supports only k=1 (argmax fast path)")

        # p is intentionally unused in k=1 path, keep for stable API
        _ = p

        mesh_mode_requested = any(
            x is not None for x in (global_semaphore, global_stage2_semaphore, fabric_scratch_tensor)
        )
        if mesh_mode_requested:
            return SamplingOp._op_mesh_2x2_axis_x(
                scores_tensor=scores_tensor,
                indices_tensor=indices_tensor,
                output_index_tensor=output_index_tensor,
                final_core_coord=final_core_coord,
                final_mesh_coord=final_mesh_coord,
                global_semaphore=global_semaphore,
                global_stage2_semaphore=global_stage2_semaphore,
                fabric_scratch_tensor=fabric_scratch_tensor,
                mesh_axis=mesh_axis,
            )
        return SamplingOp._op_single_device(
            scores_tensor=scores_tensor,
            indices_tensor=indices_tensor,
            output_index_tensor=output_index_tensor,
            final_core_coord=final_core_coord,
            final_mesh_coord=final_mesh_coord,
        )

    @staticmethod
    def _op_single_device(
        scores_tensor,
        indices_tensor,
        output_index_tensor,
        final_core_coord=None,
        final_mesh_coord=None,
    ):
        scores_shard_spec = scores_tensor.memory_config().shard_spec
        indices_shard_spec = indices_tensor.memory_config().shard_spec
        output_shard_spec = output_index_tensor.memory_config().shard_spec

        all_cores = scores_shard_spec.grid
        num_cores = all_cores.num_cores()
        assert num_cores >= 1, "Sampling requires at least one active core"
        assert indices_shard_spec.grid == all_cores, "Scores and indices must be sharded on the same core grid"
        assert output_shard_spec.grid.num_cores() == 1, "Output tensor must be sharded on a single final core"
        assert scores_tensor.dtype == ttnn.bfloat16, "Scores tensor must be bfloat16"
        assert indices_tensor.dtype == ttnn.uint32, "Indices tensor must be uint32"
        assert output_index_tensor.dtype == ttnn.uint32, "Output index tensor must be uint32"
        assert tuple(scores_shard_spec.shape) == (
            1,
            160,
        ), f"Expected scores shard shape (1, 160), got {scores_shard_spec.shape}"
        assert tuple(indices_shard_spec.shape) == (
            1,
            160,
        ), f"Expected indices shard shape (1, 160), got {indices_shard_spec.shape}"
        assert tuple(scores_tensor.shape) == (
            1,
            160 * num_cores,
        ), f"Expected scores shape (1, {160 * num_cores}), got {scores_tensor.shape}"
        assert tuple(indices_tensor.shape) == (
            1,
            160 * num_cores,
        ), f"Expected indices shape (1, {160 * num_cores}), got {indices_tensor.shape}"
        assert tuple(output_index_tensor.shape) == (
            1,
            1,
        ), f"Expected output shape (1, 1), got {output_index_tensor.shape}"

        output_core = output_shard_spec.grid.ranges()[0].start
        if final_core_coord is not None:
            assert (
                final_core_coord.x == output_core.x and final_core_coord.y == output_core.y
            ), "final_core_coord must match output shard core"
        else:
            final_core_coord = output_core

        if final_mesh_coord is not None:
            assert (
                final_mesh_coord[0] == 0 and final_mesh_coord[1] == 0
            ), "Single-device sampling currently expects final_mesh_coord=(0, 0)"

        sender_cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
        assert any(
            core.x == final_core_coord.x and core.y == final_core_coord.y for core in sender_cores
        ), "final_core_coord must be in scores/indices shard grid"
        final_is_sender = any(core.x == output_core.x and core.y == output_core.y for core in sender_cores)
        expected_remote_incs = num_cores - 1 if final_is_sender else num_cores

        winner_cb = 2
        gather_cb = 3
        semaphore_id = 0
        winner_page_bytes = 16

        ncrisc_named_compile_time_args = [
            ("sampling_num_values", 160),
            ("sampling_winner_page_bytes", winner_page_bytes),
            ("sampling_num_senders", num_cores),
            ("sampling_expected_remote_incs", expected_remote_incs),
            ("sampling_winner_cb", winner_cb),
            ("sampling_gather_cb", gather_cb),
            ("sampling_receiver_semaphore_id", semaphore_id),
            ("sampling_mesh_mode", 0),
            ("sampling_stage1_sender", 0),
            ("sampling_stage1_receiver", 0),
            ("sampling_stage2_sender", 0),
            ("sampling_stage2_receiver", 0),
            ("sampling_stage1_remote_slot_offset", 0),
            ("sampling_stage1_local_slot_offset", 0),
            ("sampling_stage2_remote_slot_offset", 0),
            ("sampling_stage2_local_slot_offset", 0),
            ("sampling_mesh_send_slot_offset", 0),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            ncrisc_common_runtime_args=[
                int(scores_tensor.buffer_address()),
                int(indices_tensor.buffer_address()),
                int(output_index_tensor.buffer_address()),
                int(scores_tensor.device().worker_core_from_logical_core(final_core_coord).x),
                int(scores_tensor.device().worker_core_from_logical_core(final_core_coord).y),
                0,
                0,
                0,
            ],
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="sampling_is_active_core",
                    core_range=all_cores,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="sampling_is_final_core",
                    core_range=final_core_coord,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="sampling_mesh_sender_core",
                    core_range=all_cores,
                    value=0,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="sampling_sender_idx",
                    core_values=[(core, idx) for idx, core in enumerate(sender_cores)],
                    other_value=0,
                ),
            ],
        )

        winner_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=winner_cb,
            data_format=ttnn.uint32,
            page_size=winner_page_bytes,
        )
        winner_cb_descriptor = ttnn.CBDescriptor(
            total_size=winner_page_bytes,
            core_ranges=all_cores,
            format_descriptors=[winner_cb_format],
        )
        gather_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=gather_cb,
            data_format=ttnn.uint32,
            page_size=winner_page_bytes,
        )
        gather_cb_descriptor = ttnn.CBDescriptor(
            total_size=winner_page_bytes * num_cores,
            core_ranges=all_cores,
            format_descriptors=[gather_cb_format],
        )

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[winner_cb_descriptor, gather_cb_descriptor],
            semaphores=[receiver_semaphore_descriptor],
        )

        ttnn.generic_op([scores_tensor, indices_tensor, output_index_tensor], program_descriptor)
        return output_index_tensor

    @staticmethod
    def _op_mesh_2x2_axis_x(
        scores_tensor,
        indices_tensor,
        output_index_tensor,
        final_core_coord,
        final_mesh_coord,
        global_semaphore,
        global_stage2_semaphore,
        fabric_scratch_tensor,
        mesh_axis: str,
    ):
        if mesh_axis != "x":
            raise NotImplementedError("Sampling mesh mode currently supports only mesh_axis='x'")
        if global_semaphore is None:
            raise ValueError("global_semaphore is required in mesh mode")
        if global_stage2_semaphore is None:
            raise ValueError("global_stage2_semaphore is required in mesh mode")
        if fabric_scratch_tensor is None:
            raise ValueError("fabric_scratch_tensor is required in mesh mode")
        if final_mesh_coord is None:
            raise ValueError("final_mesh_coord is required in mesh mode")

        mesh_device = scores_tensor.device()
        mesh_shape = mesh_device.shape
        if tuple(mesh_shape) != (2, 2):
            raise ValueError(f"Sampling mesh mode currently requires a (2,2) mesh, got {mesh_shape}")
        if tuple(final_mesh_coord) != (1, 1):
            raise ValueError(
                f"Sampling mesh mode currently expects final_mesh_coord=(1,1) for axis-x flow, got {final_mesh_coord}"
            )

        scores_per_device = ttnn.get_device_tensors(scores_tensor)
        indices_per_device = ttnn.get_device_tensors(indices_tensor)
        output_per_device = ttnn.get_device_tensors(output_index_tensor)
        scratch_per_device = ttnn.get_device_tensors(fabric_scratch_tensor)

        if not (
            len(scores_per_device)
            == len(indices_per_device)
            == len(output_per_device)
            == len(scratch_per_device)
            == mesh_shape[0] * mesh_shape[1]
        ):
            raise ValueError("All mesh tensors must have one device tensor per mesh coordinate")

        global_sem_addr = int(ttnn.get_global_semaphore_address(global_semaphore))
        global_stage2_sem_addr = int(ttnn.get_global_semaphore_address(global_stage2_semaphore))

        winner_cb = 2
        gather_cb = 3
        semaphore_id = 0
        winner_page_bytes = 16
        stage_slot_offsets = {
            "stage1_remote_slot": 0 * winner_page_bytes,
            "stage1_local_slot": 1 * winner_page_bytes,
            "stage2_remote_slot": 2 * winner_page_bytes,
            "stage2_local_slot": 3 * winner_page_bytes,
        }

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for row in range(mesh_shape[0]):
            for col in range(mesh_shape[1]):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_shape[1] + col

                scores_tensor_device = scores_per_device[device_idx]
                indices_tensor_device = indices_per_device[device_idx]
                output_tensor_device = output_per_device[device_idx]
                scratch_tensor_device = scratch_per_device[device_idx]

                scores_shard_spec = scores_tensor_device.memory_config().shard_spec
                indices_shard_spec = indices_tensor_device.memory_config().shard_spec
                output_shard_spec = output_tensor_device.memory_config().shard_spec
                scratch_shard_spec = scratch_tensor_device.memory_config().shard_spec

                all_cores = scores_shard_spec.grid
                num_cores = all_cores.num_cores()
                if num_cores < 1:
                    raise ValueError("Sampling requires at least one active core")
                if indices_shard_spec.grid != all_cores:
                    raise ValueError("Scores and indices must be sharded on the same core grid")
                if output_shard_spec.grid.num_cores() != 1:
                    raise ValueError("Output tensor must be single-core sharded per device in mesh mode")
                if scratch_shard_spec.grid.num_cores() != 1:
                    raise ValueError("fabric_scratch_tensor must be single-core sharded per device")
                if tuple(scores_shard_spec.shape) != (1, 160) or tuple(indices_shard_spec.shape) != (1, 160):
                    raise ValueError("Mesh mode expects input shard shape (1,160) on all devices")
                if scratch_tensor_device.buffer_address() is None:
                    raise ValueError("fabric_scratch_tensor must be materialized in device memory")

                sender_cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
                output_core = output_shard_spec.grid.ranges()[0].start
                if final_core_coord is not None:
                    if final_core_coord.x != output_core.x or final_core_coord.y != output_core.y:
                        raise ValueError("final_core_coord must match output shard core on all mesh devices")
                else:
                    final_core_coord = output_core

                if not any(c.x == final_core_coord.x and c.y == final_core_coord.y for c in sender_cores):
                    raise ValueError("final_core_coord must be in scores/indices shard grid")

                final_is_sender = any(c.x == final_core_coord.x and c.y == final_core_coord.y for c in sender_cores)
                expected_remote_incs = num_cores - 1 if final_is_sender else num_cores

                is_stage1_sender = row == 0
                is_stage1_receiver = row == 1
                is_stage2_sender = row == 1 and col == 0
                is_stage2_receiver = row == final_mesh_coord[0] and col == final_mesh_coord[1]
                is_mesh_sender_core = is_stage1_sender or is_stage2_sender

                # Sender destination and slot metadata (used only for mesh sender cores).
                if is_stage1_sender:
                    dest_coord = ttnn.MeshCoordinate(1, col)
                    send_slot_offset = stage_slot_offsets["stage1_remote_slot"]
                elif is_stage2_sender:
                    dest_coord = ttnn.MeshCoordinate(final_mesh_coord[0], final_mesh_coord[1])
                    send_slot_offset = stage_slot_offsets["stage2_remote_slot"]
                else:
                    dest_coord = ttnn.MeshCoordinate(row, col)
                    send_slot_offset = 0

                ncrisc_named_compile_time_args = [
                    ("sampling_num_values", 160),
                    ("sampling_winner_page_bytes", winner_page_bytes),
                    ("sampling_num_senders", num_cores),
                    ("sampling_expected_remote_incs", expected_remote_incs),
                    ("sampling_winner_cb", winner_cb),
                    ("sampling_gather_cb", gather_cb),
                    ("sampling_receiver_semaphore_id", semaphore_id),
                    ("sampling_mesh_mode", 1),
                    ("sampling_stage1_sender", 1 if is_stage1_sender else 0),
                    ("sampling_stage1_receiver", 1 if is_stage1_receiver else 0),
                    ("sampling_stage2_sender", 1 if is_stage2_sender else 0),
                    ("sampling_stage2_receiver", 1 if is_stage2_receiver else 0),
                    ("sampling_stage1_remote_slot_offset", stage_slot_offsets["stage1_remote_slot"]),
                    ("sampling_stage1_local_slot_offset", stage_slot_offsets["stage1_local_slot"]),
                    ("sampling_stage2_remote_slot_offset", stage_slot_offsets["stage2_remote_slot"]),
                    ("sampling_stage2_local_slot_offset", stage_slot_offsets["stage2_local_slot"]),
                    ("sampling_mesh_send_slot_offset", send_slot_offset),
                ]

                # Mesh sender cores get sender metadata before fabric route args.
                per_core_ncrisc_runtime_args = []
                if is_mesh_sender_core:
                    dest_idx = dest_coord[0] * mesh_shape[1] + dest_coord[1]
                    sender_dst_sem_addr = global_sem_addr if is_stage1_sender else global_stage2_sem_addr
                    per_core_ncrisc_runtime_args.append(
                        (
                            final_core_coord,
                            [
                                int(mesh_device.get_fabric_node_id(dest_coord).mesh_id),
                                int(mesh_device.get_fabric_node_id(dest_coord).chip_id),
                                int(scratch_per_device[dest_idx].buffer_address()) + send_slot_offset,
                                sender_dst_sem_addr,
                            ],
                        )
                    )

                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp",
                    core_ranges=all_cores,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    ncrisc_common_runtime_args=[
                        int(scores_tensor_device.buffer_address()),
                        int(indices_tensor_device.buffer_address()),
                        int(output_tensor_device.buffer_address()),
                        int(scores_tensor_device.device().worker_core_from_logical_core(final_core_coord).x),
                        int(scores_tensor_device.device().worker_core_from_logical_core(final_core_coord).y),
                        int(scratch_tensor_device.buffer_address()),
                        global_sem_addr,
                        global_stage2_sem_addr,
                    ],
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="sampling_is_active_core",
                            core_range=all_cores,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="sampling_is_final_core",
                            core_range=final_core_coord,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="sampling_mesh_sender_core",
                            core_range=final_core_coord,
                            value=1 if is_mesh_sender_core else 0,
                            other_value=0,
                        ),
                    ],
                    per_core_compile_time_descriptors=[
                        PerCoreCompileTimeDescriptor(
                            named_compile_time_arg="sampling_sender_idx",
                            core_values=[(core, idx) for idx, core in enumerate(sender_cores)],
                            other_value=0,
                        ),
                    ],
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=per_core_ncrisc_runtime_args,
                    ),
                )
                kernel_result = unified_kernel.get_kernel_descriptors()

                winner_cb_descriptor = ttnn.CBDescriptor(
                    total_size=winner_page_bytes,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=winner_cb,
                            data_format=ttnn.uint32,
                            page_size=winner_page_bytes,
                        )
                    ],
                )
                gather_cb_descriptor = ttnn.CBDescriptor(
                    total_size=winner_page_bytes * num_cores,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=gather_cb,
                            data_format=ttnn.uint32,
                            page_size=winner_page_bytes,
                        )
                    ],
                )
                receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=semaphore_id,
                    core_ranges=all_cores,
                    initial_value=0,
                )

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    cbs=[winner_cb_descriptor, gather_cb_descriptor],
                    semaphores=[receiver_semaphore_descriptor],
                )

                if is_mesh_sender_core:
                    sender_group = kernel_result.get_group_by_arg("sampling_mesh_sender_core", 1)
                    sender_kernel_idx = sender_group.ncrisc_kernel_index
                    fabric_rt_args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=mesh_device.get_fabric_node_id(coord),
                        dst_fabric_node_id=mesh_device.get_fabric_node_id(dest_coord),
                        link_idx=0,
                        program_descriptor=program,
                        worker_core=final_core_coord,
                    )
                    program.kernels[sender_kernel_idx].runtime_args[final_core_coord.x][final_core_coord.y].extend(
                        fabric_rt_args
                    )

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        ttnn.generic_op(
            [
                scores_tensor,
                indices_tensor,
                output_index_tensor,
                fabric_scratch_tensor,
            ],
            mesh_program_descriptor,
        )
        return output_index_tensor
