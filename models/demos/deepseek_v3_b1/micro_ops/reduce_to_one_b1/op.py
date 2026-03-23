# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Reduce-to-Root B1 Operation using ttnn.generic_op

This module implements a multi-device reduce-to-one operation for a 4x2 mesh
where all 8 devices reduce their data to a single root device using a 3-level
reduction tree.

Mesh layout (coord = [row, col]):
    [0,0] a0  |  b0 [0,1]   row 0 - LEAF (sends to row 1)
    [1,0] a1  |  b1 [1,1]   row 1 - ROOT2 (a1) / ROOT1 (b1)
    [2,0] a2  |  b2 [2,1]   row 2 - ROOT3 (receives from row 3, sends to row 1)
    [3,0] a3  |  b3 [3,1]   row 3 - LEAF (sends to row 2)

3-Level Reduction Tree:
    Level 1: Row 0 LEAFs → Row 1 (a0→a1, b0→b1)
             Row 3 LEAFs → Row 2 (a3→a2, b3→b2)
    Level 2: ROOT3 → ROOT2/ROOT1 (a2→a1, b2→b1)
    Level 3: ROOT2 → ROOT1 (a1→b1, cross-column)

Final result at ROOT1 = sum of all 8 devices
"""

from typing import Optional

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

# Device roles matching the C++ implementation
MESH_LEAF = 0
MESH_ROOT3 = 1
MESH_ROOT2 = 2
MESH_ROOT1 = 3


def get_device_role(coord: ttnn.MeshCoordinate, root_coord: ttnn.MeshCoordinate, use_torus: bool = False) -> int:
    """Determine the role of a device based on its coordinate and the root coordinate."""
    if coord[0] == root_coord[0] and coord[1] == root_coord[1]:
        return MESH_ROOT1

    root_row = root_coord[0]
    my_row = coord[0]

    # ROOT2: same row as ROOT1, different column
    if my_row == root_row:
        return MESH_ROOT2

    # ROOT3 coord
    if use_torus:
        # Torus: root must be at corner (row 0 or 3), ROOT3 is opposite corner
        if root_row == 0:
            root3_row = 3
        elif root_row == 3:
            root3_row = 0
        else:
            raise ValueError(f"Torus mode requires root at corner row (0 or 3), got row {root_row}")
    else:
        # Linear: root must be at inner row (1 or 2), ROOT3 is the other inner row
        if root_row == 1:
            root3_row = 2
        elif root_row == 2:
            root3_row = 1
        else:
            raise ValueError(f"Linear mode requires root at inner row (1 or 2), got row {root_row}")

    if my_row == root3_row:
        return MESH_ROOT3

    return MESH_LEAF


class ReduceToOneB1:
    """
    Multi-device reduce-to-one implementation using ttnn.generic_op.

    This class implements a 3-level reduction tree across a 4x2 mesh to gather
    and sum all device data at a single root device.
    """

    @staticmethod
    def golden(input_tensors: list) -> torch.Tensor:
        """
        PyTorch reference implementation of reduce-to-one for validation.

        Args:
            input_tensors: List of input tensors (torch.Tensor), one per device

        Returns:
            Output tensor containing the sum of all inputs
        """
        return torch.sum(torch.stack(input_tensors), dim=0)

    @staticmethod
    def op(
        input_tensor_mesh: ttnn.Tensor,
        intermediate_tensor: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        semaphores: list,
        root_coord: ttnn.MeshCoordinate,
        exit_coord: Optional[ttnn.MeshCoordinate] = None,
        downstream_sockets=None,
        agg_output_size_bytes: int = 0,
        num_iterations: int = 1,
        is_torus: bool = False,
    ) -> ttnn.Tensor:
        """
        Execute reduce-to-one operation using generic_op.

        When downstream_sockets is provided, each ROOT1 worker core directly sends
        its shard through its own dedicated sender socket to the downstream
        d2d_exchange pipeline core.

        Args:
            input_tensor_mesh: Input tensor mesh (each device has its own data)
            intermediate_tensor: Pre-allocated intermediate tensor mesh (3x shard width,
                                 backing a single 3-page CB for all reduction rounds)
            output_tensor: Pre-allocated output tensor mesh (single-core sharded)
            semaphores: List of 4 global semaphores for synchronization
            root_coord: MeshCoordinate of the root device (must be row 1 or 2)
            exit_coord: Optional MeshCoordinate for exit signaling (defaults to root_coord)
            downstream_sockets: Optional list of sender sockets for each ROOT1 worker core
            agg_output_size_bytes: Total useful output bytes (unpadded) for socket aggregation
            num_iterations: Number of iterations to run inside the kernel
            is_torus: Whether to use torus topology

        Returns:
            Output tensor
        """
        # Convert root_coord to MeshCoordinate if it's a tuple
        if isinstance(root_coord, tuple):
            root_coord = ttnn.MeshCoordinate(root_coord[0], root_coord[1])

        # Convert exit_coord to MeshCoordinate if it's a tuple
        if exit_coord is None:
            exit_coord = root_coord
        elif isinstance(exit_coord, tuple):
            exit_coord = ttnn.MeshCoordinate(exit_coord[0], exit_coord[1])

        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # Validate mesh shape
        if mesh_rows != 4 or mesh_cols != 2:
            raise ValueError(f"Mesh shape must be 4x2, got {mesh_rows}x{mesh_cols}")

        use_torus = is_torus and root_coord[0] in [0, 3]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        intermediate_per_device = ttnn.get_device_tensors(intermediate_tensor)

        # Get semaphore addresses
        sem_round1_addr = ttnn.get_global_semaphore_address(semaphores[0])
        sem_round2_addr = ttnn.get_global_semaphore_address(semaphores[1])
        sem_round3_addr = ttnn.get_global_semaphore_address(semaphores[2])
        sem_exit_addr = ttnn.get_global_semaphore_address(semaphores[3])

        # Get tensor properties from sample
        input_sample = input_tensors_per_device[0]
        tile = input_sample.tile
        tile_height, tile_width = tile.tile_shape
        dtype = input_sample.dtype

        # Element size based on dtype
        element_size = 2  # bfloat16

        # Calculate page and payload sizes
        page_size_bytes = tile_height * tile_width * element_size
        shard_spec = input_sample.memory_config().shard_spec
        shard_shape = shard_spec.shape
        shard_width = shard_shape[1]
        num_pages = shard_width // tile_width
        payload_size_bytes = num_pages * page_size_bytes

        # Standard 32x32 compute tiles
        compute_tile_height = 32
        compute_tile_width = 32
        compute_tile_size_bytes = compute_tile_height * compute_tile_width * element_size
        shard_elements = shard_shape[0] * shard_shape[1]
        num_compute_tiles = (shard_elements + (compute_tile_height * compute_tile_width) - 1) // (
            compute_tile_height * compute_tile_width
        )

        packet_header_size_bytes = 96
        slot_size_bytes = packet_header_size_bytes + payload_size_bytes

        # CB indices (matching C++ implementation)
        local_cb = 0  # Input tensor
        received_cb = 1  # 3-page CB for all reduction rounds (backed by intermediate tensor)
        output_cb = 2  # Final output
        packet_cb = 3  # Packet staging
        scratch_cb = 5  # Scratch for compute

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # Kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/reduce_to_one_b1/kernels/reduce_to_one_kernel.cpp"

        # Get output core from output tensor shard spec (for ROOT1 gather)
        output_sample = output_tensors_per_device[0]
        output_shard_spec = output_sample.memory_config().shard_spec
        output_core = output_shard_spec.grid.ranges()[0].start

        shard_grid = input_sample.memory_config().shard_spec.grid
        shard_cores = ttnn.corerange_to_cores(shard_grid, row_wise=True)

        # Create global semaphores for worker→fabric signaling
        # Compute num_workers_per_column from input shard grid (same for all devices)
        sample_cores = ttnn.corerange_to_cores(shard_grid, row_wise=True)
        sample_columns = {}
        for c in sample_cores:
            sample_columns.setdefault(c.x, []).append(c)
        num_worker_fabric_sems = len(sample_columns[sorted(sample_columns.keys())[0]])
        device_grid_size = mesh_device.compute_with_storage_grid_size()
        worker_fabric_sem_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )
        worker_fabric_global_sems = [
            ttnn.create_global_semaphore(mesh_device, worker_fabric_sem_cores, 0) for _ in range(num_worker_fabric_sems)
        ]
        worker_fabric_sem_addrs = [ttnn.get_global_semaphore_address(s) for s in worker_fabric_global_sems]

        # Persistent-signal sync setup for downstream sockets
        agg_sem_addr = 0
        total_num_workers_count = len(shard_cores) if downstream_sockets is not None else 0
        if downstream_sockets is not None:
            agg_sem = ttnn.create_global_semaphore(mesh_device, worker_fabric_sem_cores, 0)
            agg_sem_addr = ttnn.get_global_semaphore_address(agg_sem)

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                role = get_device_role(coord, root_coord, use_torus)
                is_leaf = role == MESH_LEAF
                is_root3 = role == MESH_ROOT3
                is_root2 = role == MESH_ROOT2
                is_root1 = role == MESH_ROOT1

                # Get tensors for this device
                input_tensor_device = input_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]
                intermediate_device = intermediate_per_device[device_idx]

                device = input_tensor_device.device()

                # Worker cores from input shard grid
                input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                input_cores_list = ttnn.corerange_to_cores(input_shard_grid, row_wise=True)

                # Build core -> column index mapping
                column_to_cores = {}
                for core in input_cores_list:
                    x = core.x
                    if x not in column_to_cores:
                        column_to_cores[x] = []
                    column_to_cores[x].append(core)

                # Sort columns and cores within each column
                sorted_columns = sorted(column_to_cores.keys())
                for x in sorted_columns:
                    column_to_cores[x].sort(key=lambda c: c.y)

                num_columns = len(sorted_columns)
                num_workers_per_column = len(column_to_cores[sorted_columns[0]])

                # Fabric cores: one per column, placed to the right of bottom core
                # For horizontal layouts (all cores in same row), place below instead
                fabric_cores = []
                column_to_fabric_core = {}

                # Detect layout: if all workers in 1-2 rows, it's horizontal
                all_y_coords = set(core.y for core in input_cores_list)
                is_horizontal_layout = len(all_y_coords) <= 2

                for x in sorted_columns:
                    bottom_core = max(column_to_cores[x], key=lambda c: c.y)
                    if is_horizontal_layout:
                        # Horizontal layout: place fabric core below (y+1)
                        fabric_core = ttnn.CoreCoord(bottom_core.x, bottom_core.y + 1)
                    else:
                        # Vertical layout: place fabric core to the right (x+1)
                        fabric_core = ttnn.CoreCoord(bottom_core.x + 1, bottom_core.y)
                    fabric_cores.append(fabric_core)
                    column_to_fabric_core[x] = fabric_core

                # Core to slot index within column
                core_to_slot_idx = {}
                for x in sorted_columns:
                    for slot_idx, core in enumerate(column_to_cores[x]):
                        core_to_slot_idx[(core.x, core.y)] = slot_idx

                # Core to shard index (for output gather)
                core_to_shard_idx = {}
                for shard_idx, core in enumerate(input_cores_list):
                    core_to_shard_idx[(core.x, core.y)] = shard_idx

                # Create core range sets
                fabric_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in fabric_cores])
                all_cores_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in input_cores_list + fabric_cores])

                # Determine destination coordinate based on role
                if is_leaf:
                    if use_torus:
                        if row == 1:
                            dest_coord = ttnn.MeshCoordinate(row - 1, col)
                        else:  # row == 2
                            dest_coord = ttnn.MeshCoordinate(row + 1, col)
                    else:
                        if row == 0:
                            dest_coord = ttnn.MeshCoordinate(row + 1, col)
                        else:  # row == 3
                            dest_coord = ttnn.MeshCoordinate(row - 1, col)
                elif is_root3:
                    dest_coord = ttnn.MeshCoordinate(root_coord[0], col)
                elif is_root2:
                    dest_coord = root_coord
                else:
                    dest_coord = exit_coord

                # Get fabric node IDs
                fabric_node_id = mesh_device.get_fabric_node_id(coord)
                dest_fabric_node_id = mesh_device.get_fabric_node_id(dest_coord)

                # Destination L1 address: offset within the single intermediate tensor
                # Page 0 = round 1 (LEAF data), page 1 = round 2 (ROOT3), page 2 = round 3 (ROOT2)
                intermediate_base = intermediate_device.buffer_address()
                if is_leaf:
                    dst_l1_addr = intermediate_base  # page 0
                    dst_sem_addr = sem_round1_addr
                elif is_root3:
                    dst_l1_addr = intermediate_base + payload_size_bytes  # page 1
                    dst_sem_addr = sem_round2_addr
                elif is_root2:
                    dst_l1_addr = intermediate_base + 2 * payload_size_bytes  # page 2
                    dst_sem_addr = sem_round3_addr
                else:
                    dst_l1_addr = intermediate_base  # ROOT1 exit (not used for reduce)
                    dst_sem_addr = sem_exit_addr

                # Get physical coords for output core
                output_core_phys = device.worker_core_from_logical_core(output_core)

                # === Compile-time args ===
                # Reader (NCRISC) compile-time args
                reader_ct_args = [
                    ("device_role", role),
                    ("num_tiles", num_compute_tiles),
                    ("local_cb", local_cb),
                    ("received_cb", received_cb),
                    ("num_loop_iters", num_iterations),
                ]

                # Writer (BRISC) compile-time args
                device_total_num_workers = (
                    total_num_workers_count if (is_root1 and downstream_sockets is not None) else 0
                )
                device_agg_output_size = agg_output_size_bytes if (is_root1 and downstream_sockets is not None) else 0
                writer_ct_args = [
                    ("device_role", role),
                    ("num_tiles", num_compute_tiles),
                    ("payload_size_bytes", payload_size_bytes),
                    ("local_cb", local_cb),
                    ("scratch_cb", scratch_cb),
                    ("packet_cb", packet_cb),
                    ("num_hops", 1),
                    ("dst_fabric_node_chip_id", dest_fabric_node_id.chip_id),
                    ("dst_fabric_node_mesh_id", int(dest_fabric_node_id.mesh_id)),
                    ("output_core_noc_x", output_core_phys.x),
                    ("output_core_noc_y", output_core_phys.y),
                    ("num_workers", num_workers_per_column),
                    ("slot_size_bytes", slot_size_bytes),
                    ("total_num_workers", device_total_num_workers),
                    ("agg_output_size_bytes", device_agg_output_size),
                    ("num_loop_iters", num_iterations),
                ]

                # Compute (TRISC) compile-time args
                compute_ct_args = [
                    ("device_role", role),
                    ("num_tiles", num_compute_tiles),
                    ("local_cb", local_cb),
                    ("received_cb", received_cb),
                    ("output_cb", output_cb),
                    ("scratch_cb", scratch_cb),
                    ("num_loop_iters", num_iterations),
                ]

                # === Common Runtime Args ===
                # Reader common args: semaphore addresses (same for all worker cores)
                reader_common_rt_args = [
                    sem_round1_addr,
                    sem_round2_addr,
                    sem_round3_addr,
                ]

                # === Per-Core Runtime Args ===
                # Persistent-signal core setup for ROOT1 with downstream sockets
                # First worker core is the designated persistent-signal coordinator
                persistent_core_noc_x = 0
                persistent_core_noc_y = 0
                if is_root1 and downstream_sockets is not None:
                    persistent_core_phys = device.worker_core_from_logical_core(input_cores_list[0])
                    persistent_core_noc_x = persistent_core_phys.x
                    persistent_core_noc_y = persistent_core_phys.y

                # Build per-core BRISC args for worker cores
                brisc_per_core_args = []
                for core_idx, core in enumerate(input_cores_list):
                    fabric_core = column_to_fabric_core[core.x]
                    fabric_core_phys = device.worker_core_from_logical_core(fabric_core)
                    slot_idx = core_to_slot_idx[(core.x, core.y)]
                    shard_idx = core_to_shard_idx[(core.x, core.y)]

                    socket_config_addr = 0
                    if is_root1 and downstream_sockets is not None:
                        socket_config_addr = downstream_sockets[shard_idx].get_config_buffer_address()

                    worker_args = [
                        fabric_core_phys.x,
                        fabric_core_phys.y,
                        slot_idx,
                        worker_fabric_sem_addrs[slot_idx],
                        dst_l1_addr,
                        dst_sem_addr,
                        output_tensor_device.buffer_address(),
                        shard_idx,
                        socket_config_addr,
                    ]
                    if is_root1 and downstream_sockets is not None:
                        worker_args.extend([agg_sem_addr, persistent_core_noc_x, persistent_core_noc_y])

                    brisc_per_core_args.append((core, worker_args))

                # Fabric cores BRISC args: worker semaphore addresses (fabric args appended later)
                for fc in fabric_cores:
                    brisc_per_core_args.append((fc, list(worker_fabric_sem_addrs)))

                # === CB Descriptors ===
                compute_tile_desc = ttnn.TileDescriptor(compute_tile_height, compute_tile_width)

                # local_cb: backed by input tensor
                cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(local_cb, input_tensor_device)
                cb0_desc.core_ranges = all_cores_set
                cb0_desc.total_size = payload_size_bytes
                cb0_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=local_cb,
                        data_format=dtype,
                        page_size=payload_size_bytes,
                        tile=compute_tile_desc,
                    )
                ]

                # received_cb: 3-page CB backed by intermediate tensor (all reduction rounds)
                cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(received_cb, intermediate_device)
                cb1_desc.core_ranges = all_cores_set
                cb1_desc.total_size = 3 * payload_size_bytes
                cb1_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=received_cb,
                        data_format=dtype,
                        page_size=payload_size_bytes,
                        tile=compute_tile_desc,
                    )
                ]

                # output_cb: backed by output tensor
                cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor_device)
                cb2_desc.core_ranges = all_cores_set
                cb2_desc.total_size = payload_size_bytes
                cb2_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=output_cb,
                        data_format=dtype,
                        page_size=payload_size_bytes,
                        tile=compute_tile_desc,
                    )
                ]

                # packet_cb: staging buffer for workers to assemble packets
                cb3_desc = ttnn.CBDescriptor(
                    total_size=num_workers_per_column * slot_size_bytes,
                    core_ranges=all_cores_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=packet_cb,
                            data_format=dtype,
                            page_size=slot_size_bytes,
                        )
                    ],
                )

                # scratch_cb: compute scratch (not tensor-backed)
                cb_size_bytes = num_compute_tiles * compute_tile_size_bytes
                cb5_desc = ttnn.CBDescriptor(
                    total_size=cb_size_bytes,
                    core_ranges=all_cores_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=scratch_cb,
                            data_format=dtype,
                            page_size=compute_tile_size_bytes,
                            tile=compute_tile_desc,
                        )
                    ],
                )

                cb_list = [cb0_desc, cb1_desc, cb2_desc, cb3_desc, cb5_desc]

                # Build unified compile-time core descriptors
                unified_ct_core_descriptors = [
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_fabric_core",
                        core_range=fabric_core_set,
                        value=1,
                        other_value=0,
                    ),
                ]
                # === Unified Kernel Descriptor ===
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=all_cores_set,
                    ncrisc_named_compile_time_args=reader_ct_args,
                    brisc_named_compile_time_args=writer_ct_args,
                    trisc_named_compile_time_args=compute_ct_args,
                    ncrisc_common_runtime_args=reader_common_rt_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=False,
                        math_approx_mode=False,
                    ),
                    unified_compile_time_core_descriptors=unified_ct_core_descriptors,
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        brisc_args=brisc_per_core_args,
                    ),
                )

                kernel_result = unified_kernel.get_kernel_descriptors()
                fabric_group = kernel_result.get_group_by_arg("is_fabric_core", 1)

                # Worker→fabric semaphores are global (created before the loop)
                semaphore_descriptors = []

                # === Program Descriptor ===
                all_kernels = kernel_result.kernels

                program = ttnn.ProgramDescriptor(
                    kernels=all_kernels,
                    semaphores=semaphore_descriptors,
                    cbs=cb_list,
                )

                # Add fabric connection args for fabric cores (must be done after program creation)
                # ROOT1 doesn't send via fabric for exit socket, so skip that fabric setup
                if not is_root1:
                    fabric_kernel_idx = fabric_group.brisc_kernel_index
                    for fc_idx, fc in enumerate(fabric_cores):
                        col_idx = fc_idx
                        link_idx = 0 if col_idx < num_columns // 2 else 1
                        fabric_rt_args_ref = program.kernels[fabric_kernel_idx].runtime_args[fc.x][fc.y]
                        fabric_conn_args = ttnn.setup_fabric_connection(
                            fabric_node_id,
                            dest_fabric_node_id,
                            link_idx,
                            program,
                            fc,
                        )
                        fabric_rt_args_ref.extend(fabric_conn_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute reduce-to-one operation
        input_list = [
            input_tensor_mesh,
            output_tensor,
            intermediate_tensor,
        ]
        ttnn.generic_op(input_list, mesh_program_descriptor)

        return output_tensor
