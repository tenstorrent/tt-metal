# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
CCL All-Gather micro-op for a 4-device torused row ring.

Two-core implementation:
- Gather core: NCRISC controller (local copies, handoffs, receive waits).
  BRISC and TRISC idle.
- Transport core: BRISC = fwd sender, NCRISC = bwd sender.  TRISC idle.

Algorithm (2-round neighbor exchange):
  R0: cross-core write to transport scratch + handoff (unblocks R1 ASAP),
      then local copy to own output slot
  R1: bidirectional send of local slice; remote writes land on gather core
  R1→R2: wait for forwarded slice, copy to transport scratch[1] + handoff
  R2: pairwise exchange

Semaphores (2 global):
  handoff_sem on transport core — monotonic (0 → 1 → 2 → 0)
  recv_sem on gather core — packed cumulative per-slot counter

No circular buffers.  Raw L1 addresses from tensor shards.
"""

import math
from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

RECV_SEM_BITS_PER_SLOT = 4
RING_SIZE = 4
MAX_NUM_LINKS = 2


def _get_neighbor_coord(mesh_shape, row, col, offset, cluster_axis=0):
    if cluster_axis == 0:
        return (row + offset) % mesh_shape[0], col
    return row, (col + offset) % mesh_shape[1]


# ---------------------------------------------------------------------------
# Byte-oriented chunk config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ByteChunkConfig:
    num_chunks: int
    chunk_size_bytes: int
    last_chunk_bytes: int


def resolve_byte_chunk_config(slice_size_bytes, max_chunk_size_bytes=None):
    max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    if max_chunk_size_bytes is not None:
        max_payload = min(max_payload, int(max_chunk_size_bytes))
    if max_payload <= 0:
        raise ValueError(f"Invalid max payload: {max_payload}")

    num_chunks = math.ceil(slice_size_bytes / max_payload)
    chunk_size_bytes = min(slice_size_bytes, max_payload)
    last_chunk_bytes = slice_size_bytes - (num_chunks - 1) * chunk_size_bytes

    max_field_value = (1 << RECV_SEM_BITS_PER_SLOT) - 1
    if num_chunks > max_field_value:
        raise ValueError(
            f"num_chunks={num_chunks} exceeds packed semaphore capacity "
            f"({max_field_value}) for BitsPerSlot={RECV_SEM_BITS_PER_SLOT}"
        )

    return ByteChunkConfig(
        num_chunks=num_chunks,
        chunk_size_bytes=chunk_size_bytes,
        last_chunk_bytes=last_chunk_bytes,
    )


# ---------------------------------------------------------------------------
# Named compile-time arg schemas
# ---------------------------------------------------------------------------


def _transport_named_ct_schema(
    slice_size_bytes=0,
    num_chunks=0,
    chunk_size_bytes=0,
    last_chunk_bytes=0,
    num_links=0,
    recv_sem_bits_per_slot=RECV_SEM_BITS_PER_SLOT,
    r2_active=0,
):
    return [
        ("allgather_slice_size_bytes", int(slice_size_bytes)),
        ("allgather_num_chunks", int(num_chunks)),
        ("allgather_chunk_size_bytes", int(chunk_size_bytes)),
        ("allgather_last_chunk_bytes", int(last_chunk_bytes)),
        ("allgather_num_links", int(num_links)),
        ("allgather_recv_sem_bits_per_slot", int(recv_sem_bits_per_slot)),
        ("allgather_r2_active", int(r2_active)),
    ]


def _gather_named_ct_schema(
    slice_size_bytes=0,
    num_chunks=0,
    ring_size=RING_SIZE,
):
    return [
        ("allgather_gather_slice_size_bytes", int(slice_size_bytes)),
        ("allgather_gather_num_chunks", int(num_chunks)),
        ("allgather_ring_size", int(ring_size)),
    ]


# ---------------------------------------------------------------------------
# Common runtime arg schemas (positional common RT args)
# ---------------------------------------------------------------------------


def _gather_common_rt_schema(
    local_input_addr=0,
    output_buffer_addr=0,
    self_slot_index=0,
    transport_scratch_base_addr=0,
    transport_noc_x=0,
    transport_noc_y=0,
    handoff_sem_bank_addr=0,
    recv_sem_addr=0,
    r2_src_slot_index=0,
):
    """Gather core NCRISC common RT args."""
    return [
        int(local_input_addr),
        int(output_buffer_addr),
        int(self_slot_index),
        int(transport_scratch_base_addr),
        int(transport_noc_x),
        int(transport_noc_y),
        int(handoff_sem_bank_addr),
        int(recv_sem_addr),
        int(r2_src_slot_index),
    ]


def _transport_common_rt_schema(
    scratch_base_addr=0,
    handoff_sem_bank_addr=0,
    dest_output_base_addr=0,
    r1_dest_slot_index=0,
    dest_noc_x=0,
    dest_noc_y=0,
    dest_recv_sem_addr=0,
    r2_dest_slot_index=0,
):
    """Transport core common RT args (shared by BRISC fwd and NCRISC bwd)."""
    return [
        int(scratch_base_addr),
        int(handoff_sem_bank_addr),
        int(dest_output_base_addr),
        int(r1_dest_slot_index),
        int(dest_noc_x),
        int(dest_noc_y),
        int(dest_recv_sem_addr),
        int(r2_dest_slot_index),
    ]


# ---------------------------------------------------------------------------
# AllGatherConfig
# ---------------------------------------------------------------------------


class AllGatherConfig:
    """Standalone all-gather config for a 4-device torused ring.

    Cores are derived from tensor shard grids:
      gather_core  — from input_tensor_mesh shard grid start
      transport_core — from scratch_tensor_mesh shard grid start

    Public API mirrors the all-reduce config pattern:
      Named CT:   get_ncrisc_named_ct_args, get_brisc_named_ct_args, get_trisc_named_ct_args
      Common RT:  get_gather_ncrisc_common_rt_args, get_transport_common_rt_args
      Per-core RT: get_transport_brisc_per_core_rt_args, get_transport_ncrisc_per_core_rt_args
    """

    def __init__(
        self,
        mesh_device,
        input_tensor_mesh,
        output_tensor_mesh,
        scratch_tensor_mesh,
        semaphores,
        cluster_axis=0,
        num_links=1,
        max_chunk_size_bytes=None,
    ):
        self.mesh_device = mesh_device
        self.cluster_axis = int(cluster_axis)
        self.num_links = int(num_links)

        self._resolve_role_assignment()

        mesh_shape = mesh_device.shape
        ring_size = mesh_shape[self.cluster_axis]
        if ring_size != RING_SIZE:
            raise ValueError(f"All-gather requires ring size {RING_SIZE}, got {ring_size}")
        if len(semaphores) < 2:
            raise ValueError(f"Need 2 semaphores (handoff + recv), got {len(semaphores)}")
        if self.num_links < 1 or self.num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links must be in [1, {MAX_NUM_LINKS}], got {self.num_links}")

        input_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        output_per_device = ttnn.get_device_tensors(output_tensor_mesh)
        scratch_per_device = ttnn.get_device_tensors(scratch_tensor_mesh)
        self.input_per_device = input_per_device
        self.output_per_device = output_per_device
        self.scratch_per_device = scratch_per_device

        # Derive cores from tensor shard grids
        gather_grid_start = input_per_device[0].memory_config().shard_spec.grid.bounding_box().start
        self.gather_core = ttnn.CoreCoord(int(gather_grid_start.x), int(gather_grid_start.y))

        transport_grid_start = scratch_per_device[0].memory_config().shard_spec.grid.bounding_box().start
        self.transport_core = ttnn.CoreCoord(int(transport_grid_start.x), int(transport_grid_start.y))

        # Slice size from input tensor shard
        input_shard = input_per_device[0].memory_config().shard_spec
        element_size = 2  # bf16
        shard_elements = input_shard.shape[0] * input_shard.shape[1]
        self.slice_size_bytes = shard_elements * element_size

        self.chunk = resolve_byte_chunk_config(self.slice_size_bytes, max_chunk_size_bytes)

        # L1 addresses (uniform across devices for same tensor allocation)
        self.input_addr = input_per_device[0].buffer_address()
        self.output_addr = output_per_device[0].buffer_address()
        self.scratch_base_addr = scratch_per_device[0].buffer_address()

        # Global semaphore addresses
        self.handoff_sem_addr = ttnn.get_global_semaphore_address(semaphores[0])
        self.recv_sem_addr = ttnn.get_global_semaphore_address(semaphores[1])

        # Physical NOC coords (uniform across devices)
        ref_device = input_per_device[0].device()
        gather_phys = ref_device.worker_core_from_logical_core(self.gather_core)
        transport_phys = ref_device.worker_core_from_logical_core(self.transport_core)
        self.gather_noc_x = gather_phys.x
        self.gather_noc_y = gather_phys.y
        self.transport_noc_x = transport_phys.x
        self.transport_noc_y = transport_phys.y

        self.gather_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.gather_core, self.gather_core)])
        self.transport_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.transport_core, self.transport_core)])

        # Per-device schedule
        mesh_rows, mesh_cols = mesh_shape
        self._per_device = {}
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)

                if self.cluster_axis == 0:
                    self_rank = row
                else:
                    self_rank = col

                fwd_row, fwd_col = _get_neighbor_coord(mesh_shape, row, col, +1, self.cluster_axis)
                bwd_row, bwd_col = _get_neighbor_coord(mesh_shape, row, col, -1, self.cluster_axis)
                fwd_coord = ttnn.MeshCoordinate(fwd_row, fwd_col)
                bwd_coord = ttnn.MeshCoordinate(bwd_row, bwd_col)

                fabric_node_by_direction = {
                    +1: mesh_device.get_fabric_node_id(fwd_coord),
                    -1: mesh_device.get_fabric_node_id(bwd_coord),
                }

                is_even = self_rank % 2 == 0
                r2_direction = self._even_r2_fabric_direction if is_even else -self._even_r2_fabric_direction

                brisc_r2_active = 1 if r2_direction == self._brisc_fabric_direction else 0
                ncrisc_r2_active = 1 if r2_direction == self._ncrisc_fabric_direction else 0

                # R2 forwards data that was received from the *opposite* direction
                # in R1.  If R2 sends in direction d, the data came from -d:
                #   r2_src_slot = (rank - d) % ring_size
                # e.g. d=+1 (fwd) → src is rank-1 (bwd neighbor's data)
                #      d=-1 (bwd) → src is rank+1 (fwd neighbor's data)
                r2_src_slot = (self_rank - r2_direction) % RING_SIZE

                self._per_device[coord] = {
                    "self_rank": self_rank,
                    "fabric_node_id": mesh_device.get_fabric_node_id(coord),
                    "fabric_node_by_direction": fabric_node_by_direction,
                    "brisc_r2_active": brisc_r2_active,
                    "ncrisc_r2_active": ncrisc_r2_active,
                    "r2_src_slot_index": r2_src_slot,
                    "r1_dest_slot_index": self_rank,
                    "r2_dest_slot_index": r2_src_slot,
                }

    # ======================================================================
    # Named compile-time args
    # ======================================================================

    def get_ncrisc_named_ct_args(self, coord):
        info = self._per_device[coord]
        transport_ct = _transport_named_ct_schema(
            slice_size_bytes=self.slice_size_bytes,
            num_chunks=self.chunk.num_chunks,
            chunk_size_bytes=self.chunk.chunk_size_bytes,
            last_chunk_bytes=self.chunk.last_chunk_bytes,
            num_links=self.num_links,
            r2_active=info["ncrisc_r2_active"],
        )
        gather_ct = _gather_named_ct_schema(
            slice_size_bytes=self.slice_size_bytes,
            num_chunks=self.chunk.num_chunks,
        )
        return transport_ct + gather_ct

    def get_brisc_named_ct_args(self, coord):
        info = self._per_device[coord]
        return _transport_named_ct_schema(
            slice_size_bytes=self.slice_size_bytes,
            num_chunks=self.chunk.num_chunks,
            chunk_size_bytes=self.chunk.chunk_size_bytes,
            last_chunk_bytes=self.chunk.last_chunk_bytes,
            num_links=self.num_links,
            r2_active=info["brisc_r2_active"],
        )

    def get_trisc_named_ct_args(self, coord):
        return []

    # ======================================================================
    # Role assignment
    # ======================================================================

    def _resolve_role_assignment(self):
        """Assign fabric directions to transport RISCs and define R2 parity.

        Direction encoding: +1 = forward (next neighbor), -1 = backward (prev).

        To swap which RISC sends in which direction: swap the two direction values.
        To flip which parity group handles R2: negate _even_r2_fabric_direction.
        """
        self._brisc_fabric_direction = +1  # BRISC sends forward
        self._ncrisc_fabric_direction = -1  # NCRISC sends backward
        self._even_r2_fabric_direction = +1  # even ranks do R2 in fwd direction

    # ======================================================================
    # Compile-time core descriptors
    # ======================================================================

    def get_compile_time_core_descriptors(self):
        """Return the list of UnifiedCompileTimeCoreDescriptor for all-gather cores."""
        return [
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_allgather_gather_core",
                core_range=self.gather_core_set,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_allgather_transport_core",
                core_range=self.transport_core_set,
                value=1,
                other_value=0,
            ),
        ]

    # ======================================================================
    # Common runtime args
    # ======================================================================

    def get_gather_ncrisc_common_rt_args(self, coord):
        info = self._per_device[coord]
        return _gather_common_rt_schema(
            local_input_addr=self.input_addr,
            output_buffer_addr=self.output_addr,
            self_slot_index=info["self_rank"],
            transport_scratch_base_addr=self.scratch_base_addr,
            transport_noc_x=self.transport_noc_x,
            transport_noc_y=self.transport_noc_y,
            handoff_sem_bank_addr=self.handoff_sem_addr,
            recv_sem_addr=self.recv_sem_addr,
            r2_src_slot_index=info["r2_src_slot_index"],
        )

    def get_transport_common_rt_args(self, coord):
        """Common RT args shared by both transport BRISC (fwd) and NCRISC (bwd).

        Destination NOC coords and L1 addresses are direction-independent
        because all devices share the same physical layout.  The actual
        fabric routing is determined by per-core RT args (fabric connection).
        """
        info = self._per_device[coord]
        return _transport_common_rt_schema(
            scratch_base_addr=self.scratch_base_addr,
            handoff_sem_bank_addr=self.handoff_sem_addr,
            dest_output_base_addr=self.output_addr,
            r1_dest_slot_index=info["r1_dest_slot_index"],
            dest_noc_x=self.gather_noc_x,
            dest_noc_y=self.gather_noc_y,
            dest_recv_sem_addr=self.recv_sem_addr,
            r2_dest_slot_index=info["r2_dest_slot_index"],
        )

    # ======================================================================
    # Per-core runtime args (fabric connections, one per link)
    # ======================================================================

    def _build_transport_per_core_rt_args(self, coord, program, core, dst_fabric_node_id):
        """Build per-core RT args for one transport RISC direction.

        Layout: [dst_mesh_id, dst_chip_id, <link 0 fabric args>, <link 1 fabric args>, ...]
        The kernel reads dst_mesh_id / dst_chip_id once, then builds num_links
        connections by iterating over the remaining args.
        """
        info = self._per_device[coord]
        out = [
            int(dst_fabric_node_id.mesh_id),
            int(dst_fabric_node_id.chip_id),
        ]
        for link_idx in range(self.num_links):
            out.extend(
                ttnn.setup_fabric_connection(
                    src_fabric_node_id=info["fabric_node_id"],
                    dst_fabric_node_id=dst_fabric_node_id,
                    link_idx=link_idx,
                    program_descriptor=program,
                    worker_core=core,
                )
            )
        return out

    def get_transport_brisc_per_core_rt_args(self, coord, program, core):
        info = self._per_device[coord]
        dst = info["fabric_node_by_direction"][self._brisc_fabric_direction]
        return self._build_transport_per_core_rt_args(coord, program, core, dst)

    def get_transport_ncrisc_per_core_rt_args(self, coord, program, core):
        info = self._per_device[coord]
        dst = info["fabric_node_by_direction"][self._ncrisc_fabric_direction]
        return self._build_transport_per_core_rt_args(coord, program, core, dst)


# ---------------------------------------------------------------------------
# DeepseekMinimalAllGather
# ---------------------------------------------------------------------------


class DeepseekMinimalAllGather:
    @staticmethod
    def golden(input_tensors_per_device):
        """Concatenate per-device inputs along the last dimension in rank order."""
        return torch.cat(input_tensors_per_device, dim=-1)

    @staticmethod
    def get_num_semaphores():
        """handoff_sem + recv_sem."""
        return 2

    @staticmethod
    def create_semaphores(mesh_device):
        num_semaphores = DeepseekMinimalAllGather.get_num_semaphores()
        device_grid_size = mesh_device.compute_with_storage_grid_size()
        available_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )
        return [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(num_semaphores)]

    @staticmethod
    def op(
        input_tensor_mesh,
        output_tensor_mesh,
        scratch_tensor_mesh,
        semaphores,
        cluster_axis=0,
        num_links=1,
        max_chunk_size_bytes=None,
    ):
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape

        config = AllGatherConfig(
            mesh_device=mesh_device,
            input_tensor_mesh=input_tensor_mesh,
            output_tensor_mesh=output_tensor_mesh,
            scratch_tensor_mesh=scratch_tensor_mesh,
            semaphores=semaphores,
            cluster_axis=cluster_axis,
            num_links=num_links,
            max_chunk_size_bytes=max_chunk_size_bytes,
        )

        gather_core = config.gather_core
        transport_core = config.transport_core
        gather_grid = config.gather_core_set
        transport_grid = config.transport_core_set
        combined_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(gather_core, gather_core), ttnn.CoreRange(transport_core, transport_core)]
        )

        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/ccl_all_gather/kernels/all_gather_kernel.cpp"

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)

                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=combined_grid,
                    ncrisc_named_compile_time_args=config.get_ncrisc_named_ct_args(coord),
                    brisc_named_compile_time_args=config.get_brisc_named_ct_args(coord),
                    trisc_named_compile_time_args=config.get_trisc_named_ct_args(coord),
                    unified_compile_time_core_descriptors=config.get_compile_time_core_descriptors(),
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(gather_core, []), (transport_core, [])],
                        brisc_args=[(gather_core, []), (transport_core, [])],
                        trisc_args=[],
                    ),
                    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                )
                kernel_result = unified_kernel.get_kernel_descriptors()
                gather_group = kernel_result.get_group_by_arg("is_allgather_gather_core", 1)
                transport_group = kernel_result.get_group_by_arg("is_allgather_gather_core", 0)
                if gather_group is None or transport_group is None:
                    raise ValueError("Expected gather and transport kernel groups")

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    semaphores=[],
                    cbs=[],
                )

                # Gather core common RT args
                program.kernels[
                    gather_group.ncrisc_kernel_index
                ].common_runtime_args = config.get_gather_ncrisc_common_rt_args(coord)
                program.kernels[gather_group.brisc_kernel_index].common_runtime_args = []
                program.kernels[gather_group.trisc_kernel_index].common_runtime_args = []

                # Transport core common RT args
                transport_rt = config.get_transport_common_rt_args(coord)
                program.kernels[transport_group.ncrisc_kernel_index].common_runtime_args = transport_rt
                program.kernels[transport_group.brisc_kernel_index].common_runtime_args = transport_rt
                program.kernels[transport_group.trisc_kernel_index].common_runtime_args = []

                # Append fabric per-core RT args (one connection per link per direction)
                brisc_per_core = program.kernels[transport_group.brisc_kernel_index].runtime_args[transport_core.x][
                    transport_core.y
                ]
                brisc_per_core.extend(config.get_transport_brisc_per_core_rt_args(coord, program, transport_core))

                ncrisc_per_core = program.kernels[transport_group.ncrisc_kernel_index].runtime_args[transport_core.x][
                    transport_core.y
                ]
                ncrisc_per_core.extend(config.get_transport_ncrisc_per_core_rt_args(coord, program, transport_core))

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        io_tensors = [input_tensor_mesh, output_tensor_mesh, scratch_tensor_mesh]
        ttnn.generic_op(io_tensors, mesh_program_descriptor)
        return output_tensor_mesh
