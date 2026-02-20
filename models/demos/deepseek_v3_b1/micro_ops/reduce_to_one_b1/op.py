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


def get_device_role(coord: ttnn.MeshCoordinate, root_coord: ttnn.MeshCoordinate) -> int:
    """Determine the role of a device based on its coordinate and the root coordinate."""
    if coord[0] == root_coord[0] and coord[1] == root_coord[1]:
        return MESH_ROOT1

    root_row = root_coord[0]
    my_row = coord[0]

    # ROOT2: same row as ROOT1, different column
    if my_row == root_row:
        return MESH_ROOT2

    # ROOT3: the other inner row (if ROOT1 is at row 1, ROOT3 is at row 2, and vice versa)
    root3_row = 2 if root_row == 1 else 1
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
    def create_d2h_infrastructure(
        submesh_device: ttnn.MeshDevice,
        root_coord: tuple,
        shard_cores: list,
        page_size_bytes: int,
    ) -> dict:
        """
        Create D2H socket infrastructure for reduce-to-one output.

        This method allocates a D2H receiver core that doesn't overlap with
        worker or fabric cores used by reduce-to-one, creates D2D socket pairs,
        and sets up the necessary semaphores.

        Args:
            submesh_device: The mesh device
            root_coord: Tuple (row, col) of root device coordinate
            shard_cores: List of worker cores used by reduce-to-one
            page_size_bytes: Size of each page in bytes

        Returns:
            Dictionary containing:
                - d2h_socket: D2H socket for host communication
                - d2d_socket_pairs: List of 8 D2D socket pairs
                - d2h_termination_semaphore: Semaphore for D2H termination
                - d2h_core: The allocated D2H receiver core
        """
        device_coord = ttnn.MeshCoordinate(root_coord[0], root_coord[1])
        compute_grid = submesh_device.compute_with_storage_grid_size()

        # Build column structure from shard cores to determine fabric cores
        column_to_cores_map = {}
        for core in shard_cores:
            x = core.x
            if x not in column_to_cores_map:
                column_to_cores_map[x] = []
            column_to_cores_map[x].append(core)

        # Sort columns and cores within each column
        sorted_columns = sorted(column_to_cores_map.keys())
        for x in sorted_columns:
            column_to_cores_map[x].sort(key=lambda c: c.y)

        # Calculate fabric cores (one per column)
        # For horizontal layouts (all cores in same row), place below instead of to the right
        fabric_cores = []

        # Detect layout: if all workers in 1-2 rows, it's horizontal
        all_y_coords = set(core.y for core in shard_cores)
        is_horizontal_layout = len(all_y_coords) <= 2

        for x in sorted_columns:
            bottom_core = max(column_to_cores_map[x], key=lambda c: c.y)
            if is_horizontal_layout:
                # Horizontal layout: place fabric core below (y+1)
                fabric_core = ttnn.CoreCoord(bottom_core.x, bottom_core.y + 1)
            else:
                # Vertical layout: place fabric core to the right (x+1)
                fabric_core = ttnn.CoreCoord(bottom_core.x + 1, bottom_core.y)
            fabric_cores.append(fabric_core)

        # Find a D2H core that doesn't overlap with worker or fabric cores
        all_reduce_cores = set((c.x, c.y) for c in shard_cores + fabric_cores)

        # Search for a free core, starting from bottom-right
        d2h_core = None
        for y in range(compute_grid.y - 1, -1, -1):
            for x in range(compute_grid.x - 1, -1, -1):
                candidate = ttnn.CoreCoord(x, y)
                if (candidate.x, candidate.y) not in all_reduce_cores:
                    d2h_core = candidate
                    break
            if d2h_core:
                break

        if not d2h_core:
            raise RuntimeError("Could not find non-overlapping core for D2H receiver")

        d2h_socket_core = ttnn.MeshCoreCoord(device_coord, d2h_core)

        # Create D2H socket
        d2h_socket_fifo_size = page_size_bytes * 16  # Buffer for 16 pages
        d2h_socket = ttnn.D2HSocket(submesh_device, d2h_socket_core, d2h_socket_fifo_size)

        d2d_socket_buffer_size = page_size_bytes * 2  # Buffer for 2 pages per socket
        d2d_socket_pairs = []

        for worker_core in shard_cores:
            worker_core_coord = ttnn.MeshCoreCoord(device_coord, worker_core)

            socket_connection = ttnn.SocketConnection(
                worker_core_coord,  # sender
                d2h_socket_core,  # receiver
            )
            socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, d2d_socket_buffer_size)
            socket_config = ttnn.SocketConfig([socket_connection], socket_memory_config)
            socket_pair = ttnn.create_socket_pair(submesh_device, submesh_device, socket_config)

            d2d_socket_pairs.append(socket_pair)

        # Create termination semaphore for worker cores + D2H core
        shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in shard_cores])
        d2h_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(d2h_core, d2h_core)])
        worker_and_d2h_cores = shard_grid.merge(d2h_core_set)
        d2h_termination_semaphore = ttnn.create_global_semaphore(submesh_device, worker_and_d2h_cores, 0)

        return {
            "d2h_socket": d2h_socket,
            "d2d_socket_pairs": d2d_socket_pairs,
            "d2h_termination_semaphore": d2h_termination_semaphore,
            "d2h_core": d2h_core,
        }

    @staticmethod
    def create_d2h_receiver_kernel(
        d2h_infrastructure: dict,
        page_size_bytes: int,
        num_shard_cores: int,
    ):
        """
        Create D2H receiver kernel descriptor (to be added to main program).

        Args:
            d2h_infrastructure: D2H infrastructure dict from create_d2h_infrastructure()
            page_size_bytes: Size of each page in bytes
            num_shard_cores: Number of worker cores sending data

        Returns:
            KernelDescriptor for D2H receiver kernel
        """
        d2h_core = d2h_infrastructure["d2h_core"]
        d2h_socket = d2h_infrastructure["d2h_socket"]
        d2d_socket_pairs = d2h_infrastructure["d2d_socket_pairs"]
        termination_semaphore = d2h_infrastructure["d2h_termination_semaphore"]

        # Compile-time args for D2H receiver kernel
        total_page_size = num_shard_cores * page_size_bytes  # 14k = 8 cores * 896 * 2
        ct_args = [
            d2h_socket.get_config_buffer_address(),
            ttnn.get_global_semaphore_address(termination_semaphore),
            total_page_size,
            page_size_bytes,
            num_shard_cores,  # Number of upstream sockets
        ]

        # Add receiver socket config addresses for all upstream sockets
        for socket_pair in d2d_socket_pairs:
            ct_args.append(socket_pair[1].get_config_buffer_address())  # Receiver socket

        # Pad with zeros if fewer than 8 sockets
        while len(ct_args) < 13:  # 5 base args + 8 socket addresses
            ct_args.append(0)

        # Create kernel descriptor
        d2h_core_range = ttnn.CoreRangeSet([ttnn.CoreRange(d2h_core, d2h_core)])
        d2h_kernel = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/d2h_multicore_receiver.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=d2h_core_range,
            compile_time_args=ct_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        return d2h_kernel

    @staticmethod
    def op(
        input_tensor_mesh: ttnn.Tensor,
        intermediate_tensors: list,
        output_tensor: ttnn.Tensor,
        semaphores: list,
        root_coord: ttnn.MeshCoordinate,
        exit_coord: Optional[ttnn.MeshCoordinate] = None,
        enable_d2h_output: bool = False,  # Enable D2H socket output
        d2h_infrastructure: Optional[dict] = None,  # Pre-created D2H infrastructure
    ) -> tuple:
        """
        Execute reduce-to-one operation using generic_op with optional D2H socket output.

        Args:
            input_tensor_mesh: Input tensor mesh (each device has its own data)
            intermediate_tensors: List of 3 pre-allocated intermediate tensor meshes
                                  for the 3 rounds of receiving
            output_tensor: Pre-allocated output tensor mesh (single-core sharded)
            semaphores: List of 4 global semaphores for synchronization
                        [round1, round2, round3, exit]
            root_coord: MeshCoordinate of the root device (must be row 1 or 2)
            exit_coord: Optional MeshCoordinate for exit signaling (defaults to root_coord)
            enable_d2h_output: If True, creates D2H socket infrastructure for host output
            d2h_infrastructure: Optional pre-created D2H infrastructure dict
                               (if None and enable_d2h_output=True, will create it)

        Returns:
            Tuple of (output_tensor, d2h_infrastructure or None)
            - output_tensor: Output tensor with reduced data at root device
            - d2h_infrastructure: Dict with D2H socket info if enabled, else None
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

        # Validate root_coord
        if root_coord[0] not in [1, 2]:
            raise ValueError(f"Root coordinate row must be 1 or 2, got {root_coord[0]}")

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        intermediate_r1_per_device = ttnn.get_device_tensors(intermediate_tensors[0])
        intermediate_r2_per_device = ttnn.get_device_tensors(intermediate_tensors[1])
        intermediate_r3_per_device = ttnn.get_device_tensors(intermediate_tensors[2])

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
        received_cb_r1 = 1  # Round 1: LEAF → ROOT*
        output_cb = 2  # Final output
        packet_cb = 3  # Packet staging
        received_cb_r2 = 5  # Round 2: ROOT3 → ROOT2/ROOT1
        received_cb_r3 = 6  # Round 3: ROOT2 → ROOT1
        scratch_cb = 7  # Scratch for compute

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # Kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/reduce_to_one_b1/kernels/reduce_to_one_kernel.cpp"

        # Get output core from output tensor shard spec (for ROOT1 gather)
        output_sample = output_tensors_per_device[0]
        output_shard_spec = output_sample.memory_config().shard_spec
        output_core = output_shard_spec.grid.ranges()[0].start

        # Get shard cores for D2H infrastructure (if needed)
        shard_grid = input_sample.memory_config().shard_spec.grid
        shard_cores = ttnn.corerange_to_cores(shard_grid, row_wise=True)

        # Create or use provided D2H infrastructure
        d2h_infra = None
        if enable_d2h_output:
            if d2h_infrastructure is None:
                # D2H page size is based on shard width (not tile width)
                # Each worker core sends one shard worth of data
                d2h_page_size_bytes = num_pages * shard_width * element_size

                # Create D2H infrastructure
                d2h_infra = ReduceToOneB1.create_d2h_infrastructure(
                    mesh_device,
                    (root_coord[0], root_coord[1]),
                    shard_cores,
                    d2h_page_size_bytes,
                )
            else:
                d2h_infra = d2h_infrastructure

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                role = get_device_role(coord, root_coord)
                is_leaf = role == MESH_LEAF
                is_root3 = role == MESH_ROOT3
                is_root2 = role == MESH_ROOT2
                is_root1 = role == MESH_ROOT1

                # Get tensors for this device
                input_tensor_device = input_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]
                intermediate_r1_device = intermediate_r1_per_device[device_idx]
                intermediate_r2_device = intermediate_r2_per_device[device_idx]
                intermediate_r3_device = intermediate_r3_per_device[device_idx]

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

                # Destination L1 address depends on role
                if is_leaf:
                    dst_l1_addr = intermediate_r1_device.buffer_address()
                    dst_sem_addr = sem_round1_addr
                elif is_root3:
                    dst_l1_addr = intermediate_r2_device.buffer_address()
                    dst_sem_addr = sem_round2_addr
                elif is_root2:
                    dst_l1_addr = intermediate_r3_device.buffer_address()
                    dst_sem_addr = sem_round3_addr
                else:
                    dst_l1_addr = intermediate_r1_device.buffer_address()
                    dst_sem_addr = sem_exit_addr

                # Get physical coords for output core
                output_core_phys = device.worker_core_from_logical_core(output_core)

                # === Compile-time args ===
                # Reader (NCRISC) compile-time args
                reader_ct_args = [
                    ("device_role", role),
                    ("num_tiles", num_compute_tiles),
                    ("local_cb", local_cb),
                    ("received_cb_r1", received_cb_r1),
                    ("received_cb_r2", received_cb_r2),
                    ("received_cb_r3", received_cb_r3),
                ]

                # Writer (BRISC) compile-time args
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
                ]

                # Compute (TRISC) compile-time args
                compute_ct_args = [
                    ("device_role", role),
                    ("num_tiles", num_compute_tiles),
                    ("local_cb", local_cb),
                    ("received_cb_r1", received_cb_r1),
                    ("received_cb_r2", received_cb_r2),
                    ("received_cb_r3", received_cb_r3),
                    ("output_cb", output_cb),
                    ("scratch_cb", scratch_cb),
                ]

                # === Common Runtime Args ===
                # Reader common args: semaphore addresses (same for all worker cores)
                reader_common_rt_args = [
                    sem_round1_addr,
                    sem_round2_addr,
                    sem_round3_addr,
                ]

                # === Per-Core Runtime Args ===
                # Build per-core BRISC args for worker cores
                brisc_per_core_args = []
                for core_idx, core in enumerate(input_cores_list):
                    fabric_core = column_to_fabric_core[core.x]
                    fabric_core_phys = device.worker_core_from_logical_core(fabric_core)
                    slot_idx = core_to_slot_idx[(core.x, core.y)]
                    shard_idx = core_to_shard_idx[(core.x, core.y)]

                    # Get socket config address for ROOT1 cores (if D2H enabled)
                    socket_config_addr = 0
                    if is_root1 and d2h_infra is not None:
                        # d2h_infra contains socket pairs, one per worker core
                        sender_socket = d2h_infra["d2d_socket_pairs"][core_idx][0]  # Get sender from pair
                        socket_config_addr = sender_socket.get_config_buffer_address()

                    worker_args = [
                        fabric_core_phys.x,  # fabric_core_noc_x
                        fabric_core_phys.y,  # fabric_core_noc_y
                        slot_idx,  # my_slot_idx
                        slot_idx,  # worker_sem_id (same as slot_idx for simplicity)
                        dst_l1_addr,  # dst_l1_addr
                        dst_sem_addr,  # dst_sem_addr
                        output_tensor_device.buffer_address(),  # output_base_addr
                        shard_idx,  # shard_idx
                        socket_config_addr,  # socket_config_addr (for ROOT1 socket sending)
                    ]
                    brisc_per_core_args.append((core, worker_args))

                # Fabric cores BRISC args: worker semaphore IDs (fabric args appended later)
                for fc in fabric_cores:
                    fabric_args = list(range(num_workers_per_column))  # Worker sem IDs
                    brisc_per_core_args.append((fc, fabric_args))

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

                # received_cb_r1: backed by intermediate tensor r1
                cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(received_cb_r1, intermediate_r1_device)
                cb1_desc.core_ranges = all_cores_set
                cb1_desc.total_size = payload_size_bytes
                cb1_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=received_cb_r1,
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

                # received_cb_r2: backed by intermediate tensor r2
                cb5_desc = ttnn.cb_descriptor_from_sharded_tensor(received_cb_r2, intermediate_r2_device)
                cb5_desc.core_ranges = all_cores_set
                cb5_desc.total_size = payload_size_bytes
                cb5_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=received_cb_r2,
                        data_format=dtype,
                        page_size=payload_size_bytes,
                        tile=compute_tile_desc,
                    )
                ]

                # received_cb_r3: backed by intermediate tensor r3
                cb6_desc = ttnn.cb_descriptor_from_sharded_tensor(received_cb_r3, intermediate_r3_device)
                cb6_desc.core_ranges = all_cores_set
                cb6_desc.total_size = payload_size_bytes
                cb6_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=received_cb_r3,
                        data_format=dtype,
                        page_size=payload_size_bytes,
                        tile=compute_tile_desc,
                    )
                ]

                # scratch_cb: compute scratch (not tensor-backed)
                cb_size_bytes = num_compute_tiles * compute_tile_size_bytes
                cb7_desc = ttnn.CBDescriptor(
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

                cb_list = [cb0_desc, cb1_desc, cb2_desc, cb3_desc, cb5_desc, cb6_desc, cb7_desc]

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
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_fabric_core",
                            core_range=fabric_core_set,
                            value=1,
                            other_value=0,
                        ),
                    ],
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        brisc_args=brisc_per_core_args,
                    ),
                )

                kernel_result = unified_kernel.get_kernel_descriptors()
                fabric_group = kernel_result.get_group_by_arg("is_fabric_core", 1)

                # Create semaphores for worker→fabric synchronization
                semaphore_descriptors = []
                for worker_idx in range(num_workers_per_column):
                    sem_desc = ttnn.SemaphoreDescriptor(
                        id=worker_idx,
                        core_type=ttnn.CoreType.WORKER,
                        core_ranges=fabric_core_set,
                        initial_value=0,
                    )
                    semaphore_descriptors.append(sem_desc)

                # === Program Descriptor ===
                all_kernels = kernel_result.kernels

                # Add D2H receiver kernel to ROOT1 device program if enabled
                if is_root1 and d2h_infra is not None:
                    single_page_size_bytes = shard_width * element_size

                    d2h_kernel = ReduceToOneB1.create_d2h_receiver_kernel(
                        d2h_infra,
                        single_page_size_bytes,
                        len(shard_cores),
                    )
                    all_kernels = all_kernels + [d2h_kernel]

                program = ttnn.ProgramDescriptor(
                    kernels=all_kernels,
                    semaphores=semaphore_descriptors,
                    cbs=cb_list,
                )

                # Add fabric connection args for fabric cores (must be done after program creation)
                # ROOT1 doesn't send via fabric, so skip fabric setup
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

        # Execute reduce-to-one operation (D2H receiver runs on ROOT1 device if enabled)
        input_list = [
            input_tensor_mesh,
            output_tensor,
            intermediate_tensors[0],
            intermediate_tensors[1],
            intermediate_tensors[2],
        ]
        ttnn.generic_op(input_list, mesh_program_descriptor)

        if enable_d2h_output:
            return (output_tensor, d2h_infra)
        return output_tensor
