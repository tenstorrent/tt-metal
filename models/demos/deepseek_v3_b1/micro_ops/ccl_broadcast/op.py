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
from models.demos.deepseek_v3_b1.utils import fabric_config_enables_torus_x, fabric_config_enables_torus_y

MAX_NUM_LINKS = 2

WRITER_COMMON_RT_KEYS = (
    "tensor_address0",
    "my_noc_x",
    "my_noc_y",
    "sem_bank_addr_0",
    "sem_bank_addr_1",
)
READER_COMMON_RT_KEYS = (
    "socket_config_addr",
    "socket_page_size",
    "socket_num_pages",
)


def _writer_common_rt_schema(tensor_address0=0, my_noc_x=0, my_noc_y=0, sem_bank_addr_0=0, sem_bank_addr_1=0):
    return {
        "tensor_address0": int(tensor_address0),
        "my_noc_x": int(my_noc_x),
        "my_noc_y": int(my_noc_y),
        "sem_bank_addr_0": int(sem_bank_addr_0),
        "sem_bank_addr_1": int(sem_bank_addr_1),
    }


def _reader_common_rt_schema(socket_config_addr=0, socket_page_size=0, socket_num_pages=0):
    return {
        "socket_config_addr": int(socket_config_addr),
        "socket_page_size": int(socket_page_size),
        "socket_num_pages": int(socket_num_pages),
    }


def _schema_to_rt_list(schema, keys):
    return [schema[k] for k in keys]


class BroadcastConfig:
    def __init__(
        self,
        mesh_device,
        input_tensor_mesh,
        output_tensor,
        root_coord,
        semaphores,
        socket=None,
        chunk_size_bytes=None,
        bcast_cb_id=None,
        num_links=1,
        fabric_config=None,
    ):
        self.mesh_device = mesh_device
        self.input_tensor_mesh = input_tensor_mesh
        self.output_tensor = output_tensor
        self.root_coord = ttnn.MeshCoordinate(int(root_coord[0]), int(root_coord[1]))
        self.root_row = int(root_coord[0])
        self.root_col = int(root_coord[1])
        self.socket = socket
        if not isinstance(semaphores, (list, tuple)):
            semaphores = [semaphores]
        self.semaphores = list(semaphores)
        if bcast_cb_id is None:
            raise ValueError("Expected explicit `bcast_cb_id`")
        self.bcast_cb_id = int(bcast_cb_id)
        self.num_links = int(num_links)
        self.fabric_config = fabric_config
        if self.num_links <= 0:
            raise ValueError("num_links must be greater than zero")
        if self.num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links ({self.num_links}) exceeds MAX_NUM_LINKS ({MAX_NUM_LINKS})")
        if len(self.semaphores) != self.num_links:
            raise ValueError(f"Expected {self.num_links} semaphores, got {len(self.semaphores)}")
        self._socket_config_addr = 0
        self._socket_page_size = 0
        self._socket_num_pages = 0

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
        if self.socket is not None:
            self._socket_config_addr = int(self.socket.get_config_buffer_address())
            self._socket_page_size = int(self.tensor_size_bytes)
            self._socket_num_pages = 1

        self._resolve_chunk_size(chunk_size_bytes)
        self._setup_fabric_rt_arg_count = None
        self._torus_x_enabled = fabric_config_enables_torus_x(self.fabric_config)
        self._torus_y_enabled = fabric_config_enables_torus_y(self.fabric_config)
        self._children_map = self._build_children_map()
        self._compute_topology_and_args()
        self.cb_ids = {"bcast_data": self.bcast_cb_id}

    @property
    def num_cbs_needed(self):
        return self.num_cb_ids_reserved

    @property
    def num_cb_ids_reserved(self):
        return 1

    @property
    def has_bypass_socket_reader(self):
        return False

    def get_kernel_defines(self, coord):
        # Returns only broadcast-owned defines.
        # If callers need additional op-specific defines in future, merge/de-dupe
        # outside this config layer (prefer a shared utils helper).
        defines = []
        if self.uses_socket(coord):
            defines.append(("ENABLE_SOCKET_READER", "1"))
        return defines

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

    @staticmethod
    def _neighbor_minus(coord, dim, torus_enabled):
        if torus_enabled:
            return (coord - 1 + dim) % dim
        return coord - 1 if coord > 0 else None

    @staticmethod
    def _neighbor_plus(coord, dim, torus_enabled):
        if torus_enabled:
            return (coord + 1) % dim
        return coord + 1 if coord < dim - 1 else None

    @staticmethod
    def _axis_distance(coord_a, coord_b, dim, torus_enabled):
        linear_distance = abs(coord_a - coord_b)
        if torus_enabled:
            return min(linear_distance, dim - linear_distance)
        return linear_distance

    def _pick_parent_1hop(self, current_coord, target_coord, dim, torus_enabled):
        if current_coord == target_coord:
            return current_coord

        coord_minus = self._neighbor_minus(current_coord, dim, torus_enabled)
        coord_plus = self._neighbor_plus(current_coord, dim, torus_enabled)

        if coord_minus is None:
            return coord_plus
        if coord_plus is None:
            return coord_minus

        distance_minus = self._axis_distance(coord_minus, target_coord, dim, torus_enabled)
        distance_plus = self._axis_distance(coord_plus, target_coord, dim, torus_enabled)
        if distance_minus < distance_plus:
            return coord_minus
        # Includes tie-break case (choose positive direction).
        return coord_plus

    def _build_parent_map(self):
        """
        Build a 1-hop parent map for NE broadcast routing.

        High-level idea:
        - If node is on root row: move along columns toward root_col (X deduction).
        - Else: move along rows toward root_row in same column (Y deduction).
        - For torus-enabled axis, both minus/plus 1-hop candidates exist and we pick
          the one with smaller wrapped distance to target; ties choose + direction.

        4x4 non-torus reference (root = R at (1,1)):

            row 0:  (0,0) <- (0,1)    (0,2) <- (0,3)
                      ^        ^        ^        ^
            row 1:  (1,0) <-  R  ->  (1,2) -> (1,3)
                      |                 |        |
            row 2:  (2,0)    (2,1)    (2,2)    (2,3)
                      |        |        |        |
            row 3:  (3,0)    (3,1)    (3,2)    (3,3)

          Parent map intuition:
          - root row (row 1): chain away from root in X (left side uses <-, right side uses ->)
          - other rows: stay in same column and move toward row 1 in Y

        8x4 example (root = R at (0,0)):

            FABRIC_2D (no wrap):
              row 0:  R -> (0,1) -> (0,2) -> (0,3)
              col c:  (0,c) v (1,c) v ... v (7,c)

            TORUS_X only:
              row 0:  R -> (0,1), and R -> (0,3) -> (0,2)
              col c:  same as FABRIC_2D (no Y wrap)

            TORUS_Y only:
              row 0:  same as FABRIC_2D (no X wrap)
              col c:  (0,c) -> (1,c) and (0,c) -> (7,c), then continue by nearest hop

            TORUS_XY:
              combine TORUS_X row behavior with TORUS_Y column behavior.
        """
        parent_map = {}
        for row in range(self.mesh_rows):
            for col in range(self.mesh_cols):
                node_coord = (row, col)
                if node_coord == (self.root_row, self.root_col):
                    parent_map[node_coord] = None
                    continue

                # Nodes on root row choose parent on root row (column axis movement).
                if row == self.root_row:
                    parent_col = self._pick_parent_1hop(col, self.root_col, self.mesh_cols, self._torus_x_enabled)
                    parent_map[node_coord] = (self.root_row, parent_col)
                # Other nodes choose parent in same column (row axis movement).
                else:
                    parent_row = self._pick_parent_1hop(row, self.root_row, self.mesh_rows, self._torus_y_enabled)
                    parent_map[node_coord] = (parent_row, col)
        return parent_map

    def _build_children_map(self):
        parent_map = self._build_parent_map()
        children_map = {(row, col): [] for row in range(self.mesh_rows) for col in range(self.mesh_cols)}
        for node_coord, parent_coord in parent_map.items():
            if parent_coord is not None:
                children_map[parent_coord].append(node_coord)
        return children_map

    def _compute_dst_coords(self, row, col):
        return self._children_map[(row, col)]

    def _compute_topology_and_args(self):
        self._per_device = {}
        for row in range(self.mesh_rows):
            for col in range(self.mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
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

                self._per_device[coord] = {
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

    def is_root(self, coord):
        return coord == self.root_coord

    def uses_socket(self, coord):
        # Root-only ingress policy.
        return self.socket is not None and self.is_root(coord)

    def get_writer_named_ct_args(self, coord):
        d = self._per_device[coord]
        return [
            ("bcast_data_cb_id", self.cb_ids["bcast_data"]),
            ("bcast_num_pages_to_read", self.num_pages_to_read),
            ("bcast_is_sender", 1 if d["is_root"] else 0),
            ("bcast_tensor0_page_size", self.tensor0_page_size),
            ("bcast_num_neighbors", d["num_neighbors"]),
            ("bcast_num_links", self.num_links),
            ("bcast_is_root", 1 if d["is_root"] else 0),
            ("bcast_chunk_size_bytes", self.chunk_size_bytes),
            ("bcast_last_chunk_size_bytes", self.last_chunk_size_bytes),
            ("bcast_num_chunks", self.num_chunks),
        ]

    def get_reader_named_ct_args(self, coord):
        d = self._per_device[coord]
        return [
            ("bcast_data_cb_id", self.cb_ids["bcast_data"]),
            ("bcast_num_pages_to_read", self.num_pages_to_read),
            ("bcast_is_sender", 1 if d["is_root"] else 0),
            ("bcast_use_socket", 1 if self.uses_socket(coord) else 0),
        ]

    def get_writer_common_rt_args(self, coord):
        d = self._per_device[coord]
        sem_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in self.semaphores]
        sem_addrs += [0] * (MAX_NUM_LINKS - len(sem_addrs))
        schema = _writer_common_rt_schema(
            tensor_address0=d["tensor_address0"],
            my_noc_x=d["my_noc_x"],
            my_noc_y=d["my_noc_y"],
            sem_bank_addr_0=sem_addrs[0],
            sem_bank_addr_1=sem_addrs[1],
        )
        return _schema_to_rt_list(schema, WRITER_COMMON_RT_KEYS)

    def get_reader_common_rt_args(self, coord):
        if self.uses_socket(coord):
            schema = _reader_common_rt_schema(
                socket_config_addr=self._socket_config_addr,
                socket_page_size=self._socket_page_size,
                socket_num_pages=self._socket_num_pages,
            )
        else:
            schema = _reader_common_rt_schema()
        return _schema_to_rt_list(schema, READER_COMMON_RT_KEYS)

    def get_writer_per_core_rt_args(self, coord, program, core):
        d = self._per_device[coord]
        src_node = d["my_fabric_node_id"]
        payload = []

        # Neighbor-blocked RT arg layout:
        # for each neighbor: append dst ids first, then all link setup args.
        for i in range(d["num_neighbors"]):
            payload.append(d["dst_mesh_ids"][i])
            payload.append(d["dst_chip_ids"][i])
            dst_node = d["dst_nodes"][i]
            for link_idx in range(self.num_links):
                setup_args = ttnn.setup_fabric_connection(src_node, dst_node, link_idx, program, core)
                if self._setup_fabric_rt_arg_count is None:
                    self._setup_fabric_rt_arg_count = len(setup_args)
                else:
                    assert (
                        len(setup_args) == self._setup_fabric_rt_arg_count
                    ), "setup_fabric_connection arg width changed across calls"
                payload.extend(setup_args)
        return [len(payload)] + payload

    def get_cb_descriptor(self, coord):
        d = self._per_device[coord]
        return ttnn.cb_descriptor_from_sharded_tensor(self.cb_ids["bcast_data"], d["input_tensor_device"])

    def get_worker_core(self, coord):
        return self._per_device[coord]["worker_core"]

    def get_worker_core_set(self, coord):
        return self._per_device[coord]["worker_core_set"]


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
    def get_num_semaphores(num_links=1):
        num_links = int(num_links)
        if num_links <= 0:
            raise ValueError("num_links must be greater than zero")
        if num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links ({num_links}) exceeds MAX_NUM_LINKS ({MAX_NUM_LINKS})")
        return num_links

    @staticmethod
    def configure(
        mesh_device,
        input_tensor_mesh,
        output_tensor,
        sender_coord,
        semaphores=None,
        socket=None,
        skip_ccl=False,
        chunk_size_bytes=None,
        bcast_cb_id=None,
        num_links=1,
        fabric_config=None,
    ):
        if bcast_cb_id is None:
            raise ValueError("Expected explicit `bcast_cb_id`")
        if skip_ccl:
            return BypassBroadcastConfig(
                mesh_device=mesh_device,
                input_tensor_mesh=input_tensor_mesh,
                root_coord=sender_coord,
                socket=socket,
                bcast_cb_id=bcast_cb_id,
            )
        if semaphores is None:
            raise ValueError("Expected semaphore(s) via `semaphores`")
        return BroadcastConfig(
            mesh_device=mesh_device,
            input_tensor_mesh=input_tensor_mesh,
            output_tensor=output_tensor,
            root_coord=sender_coord,
            semaphores=semaphores,
            socket=socket,
            chunk_size_bytes=chunk_size_bytes,
            bcast_cb_id=bcast_cb_id,
            num_links=num_links,
            fabric_config=fabric_config,
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
        fabric_config=None,
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
            socket=None,
            skip_ccl=False,
            chunk_size_bytes=chunk_size_bytes,
            bcast_cb_id=0,
            num_links=num_links,
            fabric_config=fabric_config,
        )

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # Kernel paths
        # Use unified kernel for both reader and writer roles
        ccl_kernel_path = "models/demos/deepseek_v3_b1/micro_ops/ccl_broadcast/kernels/ccl_broadcast_kernel.cpp"

        common_named_ct_args = [("bcast_num_iterations", num_iterations)]

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                # Create unified kernel descriptor for CCL broadcast
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=ccl_kernel_path,
                    core_ranges=config.get_worker_core_set(coord),
                    ncrisc_named_compile_time_args=config.get_writer_named_ct_args(coord) + common_named_ct_args,
                    brisc_named_compile_time_args=config.get_reader_named_ct_args(coord) + common_named_ct_args,
                    ncrisc_common_runtime_args=config.get_writer_common_rt_args(coord),
                    brisc_common_runtime_args=config.get_reader_common_rt_args(coord),
                    # Per-core runtime args: empty for NCRISC (fabric args appended later)
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(config.get_worker_core(coord), [])],
                    ),
                )

                # Create program descriptor (only reader and writer, no compute)
                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors().kernels[:2],
                    semaphores=[],
                    cbs=[config.get_cb_descriptor(coord)],
                )

                writer_core = config.get_worker_core(coord)
                writer_rt_args_ref = program.kernels[0].runtime_args[writer_core.x][writer_core.y]
                writer_rt_args_ref.extend(config.get_writer_per_core_rt_args(coord, program, writer_core))

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic_op
        result = ttnn.generic_op([input_tensor_mesh, output_tensor], mesh_program_descriptor)

        return result


class BypassBroadcastConfig:
    """
    Skip-CCL broadcast shim that preserves the same interface as BroadcastConfig.
    This allows fused ops to stay config-driven without branching on broadcast internals.
    """

    def __init__(
        self,
        mesh_device,
        input_tensor_mesh,
        root_coord,
        socket=None,
        bcast_cb_id=None,
    ):
        self.mesh_device = mesh_device
        self.input_tensor_mesh = input_tensor_mesh
        self.root_coord = ttnn.MeshCoordinate(int(root_coord[0]), int(root_coord[1]))
        self.socket = socket
        if bcast_cb_id is None:
            raise ValueError("Expected explicit `bcast_cb_id`")
        self.bcast_cb_id = int(bcast_cb_id)
        self.cb_ids = {"bcast_data": self.bcast_cb_id}
        self.num_links = 1
        self.chunk_size_bytes = 0
        self.last_chunk_size_bytes = 0
        self.num_chunks = 0
        self._socket_config_addr = 0
        self._socket_page_size = 0
        self._socket_num_pages = 0
        # Bypass path intentionally avoids any broadcast topology/fabric setup.
        # It only retains minimal metadata needed for helper args and placement.
        self._per_device = {}

        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        input_sample = input_tensors_per_device[0]
        tile_height, tile_width = input_sample.tile.tile_shape
        element_size = dtype_size(input_sample.dtype)
        # Needed for socket reader page sizing in skip+socket mode.
        self.tensor0_page_size = tile_height * tile_width * element_size
        shard_spec = input_sample.memory_config().shard_spec
        shard_height, shard_width = shard_spec.shape
        if shard_height % tile_height != 0 or shard_width % tile_width != 0:
            raise ValueError(
                f"Shard shape {shard_spec.shape} must be tile-aligned to tile shape ({tile_height}, {tile_width})"
            )
        self.num_pages_to_read = (shard_height // tile_height) * (shard_width // tile_width)
        self.tensor_size_bytes = self.tensor0_page_size * self.num_pages_to_read
        if self.socket is not None:
            self._socket_config_addr = int(self.socket.get_config_buffer_address())
            self._socket_page_size = int(self.tensor_size_bytes)
            self._socket_num_pages = 1

        mesh_rows = mesh_device.shape[0]
        mesh_cols = mesh_device.shape[1]
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                idx = row * mesh_cols + col
                input_tensor_device = input_tensors_per_device[idx]
                input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                shard_grid_start = input_shard_grid.bounding_box().start
                worker_core = ttnn.CoreCoord(shard_grid_start.x, shard_grid_start.y)
                worker_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])
                self._per_device[coord] = {
                    "is_root": coord == self.root_coord,
                    "worker_core": worker_core,
                    "worker_core_set": worker_core_set,
                }

    @property
    def num_cb_ids_reserved(self):
        return 1

    @property
    def num_cbs_needed(self):
        return self.num_cb_ids_reserved

    @property
    def has_bypass_socket_reader(self):
        return self.socket is not None

    def get_kernel_defines(self, coord):
        # Returns only broadcast-owned defines.
        # If callers need additional op-specific defines in future, merge/de-dupe
        # outside this config layer (prefer a shared utils helper).
        defines = [("SKIP_CCL", "1")]
        if self.uses_socket(coord):
            defines.append(("ENABLE_SOCKET_READER", "1"))
        return defines

    def is_root(self, coord):
        return coord == self.root_coord

    def uses_socket(self, coord):
        return self.socket is not None and self.is_root(coord)

    def get_writer_named_ct_args(self, coord):
        d = self._per_device[coord]
        return [
            ("bcast_data_cb_id", self.cb_ids["bcast_data"]),
            ("bcast_num_pages_to_read", self.num_pages_to_read),
            ("bcast_is_sender", 1 if d["is_root"] else 0),
            ("bcast_tensor0_page_size", self.tensor0_page_size),
            ("bcast_num_neighbors", 0),
            ("bcast_num_links", 1),
            ("bcast_is_root", 1 if d["is_root"] else 0),
            ("bcast_chunk_size_bytes", 0),
            ("bcast_last_chunk_size_bytes", 0),
            ("bcast_num_chunks", 0),
        ]

    def get_reader_named_ct_args(self, coord):
        d = self._per_device[coord]
        return [
            ("bcast_data_cb_id", self.cb_ids["bcast_data"]),
            ("bcast_num_pages_to_read", self.num_pages_to_read),
            ("bcast_is_sender", 1 if d["is_root"] else 0),
            ("bcast_use_socket", 1 if self.uses_socket(coord) else 0),
        ]

    def get_socket_reader_ct_args(self, coord, target_cb, target_num_pages=None):
        if target_cb in self.cb_ids.values():
            raise ValueError("target_cb must not collide with broadcast-private cb ids")
        d = self._per_device[coord]
        num_pages_to_read = self.num_pages_to_read if target_num_pages is None else int(target_num_pages)
        if num_pages_to_read <= 0:
            raise ValueError("target_num_pages must be greater than zero")
        return [
            ("bcast_data_cb_id", int(target_cb)),
            ("bcast_num_pages_to_read", num_pages_to_read),
            ("bcast_is_sender", 1 if d["is_root"] else 0),
            ("bcast_use_socket", 1 if self.uses_socket(coord) else 0),
        ]

    def get_socket_reader_rt_args(self, coord):
        if self.uses_socket(coord):
            schema = _reader_common_rt_schema(
                socket_config_addr=self._socket_config_addr,
                socket_page_size=self._socket_page_size,
                socket_num_pages=self._socket_num_pages,
            )
        else:
            schema = _reader_common_rt_schema()
        return _schema_to_rt_list(schema, READER_COMMON_RT_KEYS)

    def get_writer_common_rt_args(self, coord):
        return _schema_to_rt_list(_writer_common_rt_schema(), WRITER_COMMON_RT_KEYS)

    def get_reader_common_rt_args(self, coord):
        return self.get_socket_reader_rt_args(coord)

    def get_writer_per_core_rt_args(self, coord, program, core):
        return []

    def get_cb_descriptor(self, coord):
        return None

    def get_worker_core(self, coord):
        return self._per_device[coord]["worker_core"]

    def get_worker_core_set(self, coord):
        return self._per_device[coord]["worker_core_set"]
