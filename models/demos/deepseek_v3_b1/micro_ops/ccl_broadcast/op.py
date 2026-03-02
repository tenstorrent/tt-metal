# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
CCL Broadcast Operation using ttnn.generic_op
This module implements a multi-device broadcast operation where a sender device
broadcasts data to all other devices in a mesh using a neighbor-exchange topology.
"""


import ttnn
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

MAX_NUM_LINKS = 2


class BroadcastConfig:
    def __init__(
        self,
        mesh_device,
        input_tensor_mesh,
        output_tensor,
        root_coord,
        semaphores,
        chunk_size_bytes=None,
        cb_start_offset=0,
        num_links=1,
        num_iterations=1,
    ):
        self.mesh_device = mesh_device
        self.input_tensor_mesh = input_tensor_mesh
        self.output_tensor = output_tensor
        self.root_row = int(root_coord[0])
        self.root_col = int(root_coord[1])
        if not isinstance(semaphores, (list, tuple)):
            semaphores = [semaphores]
        self.semaphores = list(semaphores)
        self.cb_start_offset = cb_start_offset
        self.num_links = int(num_links)
        if self.num_links <= 0:
            raise ValueError("num_links must be greater than zero")
        if self.num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links ({self.num_links}) exceeds MAX_NUM_LINKS ({MAX_NUM_LINKS})")
        if len(self.semaphores) != self.num_links:
            raise ValueError(f"Expected {self.num_links} semaphores, got {len(self.semaphores)}")
        self.num_iterations = num_iterations

        self.mesh_rows = mesh_device.shape[0]
        self.mesh_cols = mesh_device.shape[1]

        self.input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        self.output_tensors_per_device = ttnn.get_device_tensors(output_tensor)

        input_sample = self.input_tensors_per_device[0]
        tile_height, tile_width = input_sample.tile.tile_shape
        element_size = dtype_size(input_sample.dtype)
        self.tensor0_page_size = tile_height * tile_width * element_size
        shard_spec = input_sample.memory_config().shard_spec
        shard_height, shard_width = shard_spec.shape
        if shard_height % tile_height != 0 or shard_width % tile_width != 0:
            raise ValueError(
                f"Shard shape {shard_spec.shape} must be tile-aligned to tile shape ({tile_height}, {tile_width})"
            )
        self.num_pages_to_read = (shard_height // tile_height) * (shard_width // tile_width)
        self.tensor_size_bytes = self.tensor0_page_size * self.num_pages_to_read
        if self.tensor_size_bytes <= 0:
            raise ValueError("tensor_size_bytes must be greater than zero")

        self._resolve_chunk_size(chunk_size_bytes)
        self._setup_fabric_rt_arg_count = None
        self._compute_topology_and_args()

    @property
    def num_cbs_needed(self):
        return len(self.get_cb_descriptors(0, 0))

    def _resolve_chunk_size(self, chunk_size_bytes):
        max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        if chunk_size_bytes is None:
            self.chunk_size_bytes = min(self.tensor_size_bytes, max_payload)
        else:
            self.chunk_size_bytes = int(chunk_size_bytes)
            if self.chunk_size_bytes <= 0:
                raise ValueError("chunk_size_bytes must be greater than zero")
            if self.chunk_size_bytes > max_payload:
                raise ValueError(f"chunk_size_bytes ({self.chunk_size_bytes}) exceeds max_payload ({max_payload})")
            if self.chunk_size_bytes > self.tensor_size_bytes:
                raise ValueError(
                    f"chunk_size_bytes ({self.chunk_size_bytes}) exceeds tensor_size ({self.tensor_size_bytes})"
                )

        self.last_chunk_size_bytes = self.tensor_size_bytes % self.chunk_size_bytes or self.chunk_size_bytes
        self.num_chunks = (self.tensor_size_bytes + self.chunk_size_bytes - 1) // self.chunk_size_bytes

    def _compute_dst_coords(self, row, col):
        """
        Compute downstream mesh coordinates for the XY spanning tree.

        Example (4x4, root=(1,1)):
             (0,0) (0,1) (0,2) (0,3)
               ^     ^     ^     ^
             (1,0)<-(1,1)->(1,2)->(1,3)
               |     |     |     |
             (2,0) (2,1) (2,2) (2,3)
               |     |     |     |
             (3,0) (3,1) (3,2) (3,3)
        """
        dst_coords = []
        root_row = self.root_row
        root_col = self.root_col

        # Root fans out to immediate row neighbors and immediate column neighbors.
        if row == root_row and col == root_col:
            if root_col > 0:
                dst_coords.append((root_row, root_col - 1))
            if root_col < self.mesh_cols - 1:
                dst_coords.append((root_row, root_col + 1))
            if root_row > 0:
                dst_coords.append((root_row - 1, root_col))
            if root_row < self.mesh_rows - 1:
                dst_coords.append((root_row + 1, root_col))
            return dst_coords

        # Nodes in root row continue row chain away from root, and fan to up/down.
        if row == root_row:
            if col < root_col and col > 0:
                dst_coords.append((root_row, col - 1))
            if col > root_col and col < self.mesh_cols - 1:
                dst_coords.append((root_row, col + 1))
            if root_row > 0:
                dst_coords.append((root_row - 1, col))
            if root_row < self.mesh_rows - 1:
                dst_coords.append((root_row + 1, col))
            return dst_coords

        # Non-root-row nodes only propagate along column direction.
        if row < root_row and row > 0:
            dst_coords.append((row - 1, col))
        if row > root_row and row < self.mesh_rows - 1:
            dst_coords.append((row + 1, col))
        return dst_coords

    def _compute_topology_and_args(self):
        self._per_device = {}
        for row in range(self.mesh_rows):
            for col in range(self.mesh_cols):
                idx = row * self.mesh_cols + col
                input_tensor_device = self.input_tensors_per_device[idx]
                output_tensor_device = self.output_tensors_per_device[idx]

                input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                shard_grid_start = input_shard_grid.bounding_box().start
                worker_core = ttnn.CoreCoord(shard_grid_start.x, shard_grid_start.y)
                worker_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])

                data_core_physical = input_tensor_device.device().worker_core_from_logical_core(worker_core)
                my_noc_x = int(data_core_physical.x)
                my_noc_y = int(data_core_physical.y)

                dst_coords = self._compute_dst_coords(row, col)
                dst_nodes = [self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r, c in dst_coords]

                self._per_device[(row, col)] = {
                    "is_root": row == self.root_row and col == self.root_col,
                    "num_neighbors": len(dst_nodes),
                    "dst_nodes": dst_nodes,
                    "dst_mesh_ids": [int(node.mesh_id) for node in dst_nodes],
                    "dst_chip_ids": [int(node.chip_id) for node in dst_nodes],
                    "worker_core": worker_core,
                    "worker_core_set": worker_core_set,
                    "my_noc_x": my_noc_x,
                    "my_noc_y": my_noc_y,
                    "tensor_address0": int(output_tensor_device.buffer_address()),
                    "input_tensor_device": input_tensor_device,
                    "my_fabric_node_id": self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col)),
                }

    def get_named_ct_args(self, row, col):
        d = self._per_device[(row, col)]
        return [
            ("bcast_cb0_id", self.cb_start_offset),
            ("bcast_num_pages_to_read", self.num_pages_to_read),
            ("bcast_is_sender", 1 if d["is_root"] else 0),
            ("bcast_tensor0_page_size", self.tensor0_page_size),
            ("bcast_num_neighbors", d["num_neighbors"]),
            ("bcast_num_links", self.num_links),
            ("bcast_is_root", 1 if d["is_root"] else 0),
            ("bcast_chunk_size_bytes", self.chunk_size_bytes),
            ("bcast_last_chunk_size_bytes", self.last_chunk_size_bytes),
            ("bcast_num_chunks", self.num_chunks),
            ("bcast_num_iterations", self.num_iterations),
        ]

    def get_common_rt_args(self, row, col):
        d = self._per_device[(row, col)]
        sem_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in self.semaphores]
        sem_addrs += [0] * (MAX_NUM_LINKS - len(sem_addrs))
        return [
            d["tensor_address0"],  # index 0
            d["my_noc_x"],  # index 1
            d["my_noc_y"],  # index 2
            sem_addrs[0],  # index 3 (link 0)
            sem_addrs[1],  # index 4 (link 1 or dummy)
        ]

    def append_per_core_rt_args(self, row, col, program, kernel_idx, core):
        d = self._per_device[(row, col)]
        if d["num_neighbors"] == 0:
            return 0

        writer_rt_args_ref = program.kernels[kernel_idx].runtime_args[core.x][core.y]
        src_node = d["my_fabric_node_id"]
        before_len = len(writer_rt_args_ref)

        # Neighbor-blocked RT arg layout:
        # for each neighbor: append dst ids first, then all link setup args.
        for i in range(d["num_neighbors"]):
            writer_rt_args_ref.append(d["dst_mesh_ids"][i])
            writer_rt_args_ref.append(d["dst_chip_ids"][i])
            dst_node = d["dst_nodes"][i]
            for link_idx in range(self.num_links):
                setup_args = ttnn.setup_fabric_connection(src_node, dst_node, link_idx, program, core)
                if self._setup_fabric_rt_arg_count is None:
                    self._setup_fabric_rt_arg_count = len(setup_args)
                else:
                    assert (
                        len(setup_args) == self._setup_fabric_rt_arg_count
                    ), "setup_fabric_connection arg width changed across calls"
                writer_rt_args_ref.extend(setup_args)
        return len(writer_rt_args_ref) - before_len

    def get_cb_descriptors(self, row, col):
        d = self._per_device[(row, col)]
        return [ttnn.cb_descriptor_from_sharded_tensor(self.cb_start_offset, d["input_tensor_device"])]

    def get_worker_core(self, row, col):
        return self._per_device[(row, col)]["worker_core"]

    def get_worker_core_set(self, row, col):
        return self._per_device[(row, col)]["worker_core_set"]


class DeepseekMinimalBroadcast:
    """
    Multi-device broadcast implementation using ttnn.generic_op.
    This class implements broadcast from a sender device to all other devices
    in a mesh using the fabric infrastructure.
    """

    @staticmethod
    def golden(input_tensor):
        """
        PyTorch reference implementation of broadcast for validation.
        Args:
            input_tensor: Input tensor (torch.Tensor) - the data at sender
        Returns:
            Output tensor that would be on each device after broadcast
        """
        # All devices should have the sender's data
        return input_tensor

    @staticmethod
    def configure(
        mesh_device,
        input_tensor_mesh,
        output_tensor,
        sender_coord,
        semaphores=None,
        chunk_size_bytes=None,
        cb_start_offset=0,
        num_links=1,
        num_iterations=1,
    ):
        if semaphores is None:
            raise ValueError("Expected semaphore(s) via `semaphores`")
        return BroadcastConfig(
            mesh_device=mesh_device,
            input_tensor_mesh=input_tensor_mesh,
            output_tensor=output_tensor,
            root_coord=sender_coord,
            semaphores=semaphores,
            chunk_size_bytes=chunk_size_bytes,
            cb_start_offset=cb_start_offset,
            num_links=num_links,
            num_iterations=num_iterations,
        )

    @staticmethod
    def op(
        input_tensor_mesh,
        output_tensor,
        sender_coord,
        semaphores=None,
        chunk_size_bytes=None,
        num_links=1,
        num_iterations=1,
    ):
        """
        Execute broadcast operation using generic_op.
        Args:
            input_tensor_mesh: Input tensor mesh (sender has data, others have zeros)
            output_tensor: Pre-allocated output tensor mesh
            sender_coord: ttnn.MeshCoordinate of the sender device
            semaphores: Per-link global semaphores used for chunk arrival signaling
            num_links: Number of links to use (default 1)
        Returns:
            Output tensor with broadcast data on all devices
        """
        mesh_device = input_tensor_mesh.device()
        mesh_rows, mesh_cols = mesh_device.shape
        if semaphores is None:
            raise ValueError("Expected semaphore(s) via `semaphores`")

        config = DeepseekMinimalBroadcast.configure(
            mesh_device=mesh_device,
            input_tensor_mesh=input_tensor_mesh,
            output_tensor=output_tensor,
            sender_coord=sender_coord,
            semaphores=semaphores,
            chunk_size_bytes=chunk_size_bytes,
            cb_start_offset=0,
            num_links=num_links,
            num_iterations=num_iterations,
        )

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # Kernel paths
        # Use unified kernel for both reader and writer roles
        ccl_kernel_path = "models/demos/deepseek_v3_b1/micro_ops/ccl_broadcast/kernels/ccl_broadcast_kernel.cpp"

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                # Create unified kernel descriptor for CCL broadcast
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=ccl_kernel_path,
                    core_ranges=config.get_worker_core_set(row, col),
                    ncrisc_named_compile_time_args=config.get_named_ct_args(row, col),
                    brisc_named_compile_time_args=config.get_named_ct_args(row, col),
                    ncrisc_common_runtime_args=config.get_common_rt_args(row, col),
                    # Per-core runtime args: empty for NCRISC (fabric args appended later)
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(config.get_worker_core(row, col), [])],
                    ),
                )

                # Create program descriptor (only reader and writer, no compute)
                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors().kernels[:2],
                    semaphores=[],
                    cbs=config.get_cb_descriptors(row, col),
                )

                config.append_per_core_rt_args(row, col, program, kernel_idx=0, core=config.get_worker_core(row, col))

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic_op
        result = ttnn.generic_op([input_tensor_mesh, output_tensor], mesh_program_descriptor)

        return result
