# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CCL All-Reduce Operation using ttnn.generic_op.

Two-core implementation:
- Sender core: NCRISC + BRISC fabric writers, ``DM_DYNAMIC_NOC``, one link per RISC when ``num_links==2``.
  Per-link fabric sem bank addresses are passed in per-core runtime args.
- Receiver core: dedicated L1 CB for the local operand (NOC read from sender),
  fabric receive + streaming remote CB pushes, TRISC reduction.
- BRISC signals ``local_ready`` after confirming local data presence;
  the receiver waits once before NOC-reading the sender.
- Semaphores: ``num_links`` for fabric plus one extra for ``local_ready``.
- Input tensor must be L1-sharded on the sender core; intermediate, output, and optional residual on
  the receiver core.  ``persistent_output_tensor`` is required.

Link assignment (fixed):
  BRISC  -> link 0, signal_local_ready=1
  NCRISC -> link 1, signal_local_ready=0
  When num_links=1, NCRISC is automatically disabled (link_index >= num_links).

CB ID layout (each ID is unique per logical buffer):
  0  local_data_cb_id       sender core   input tensor shard
  1  recv_local_data_cb_id  receiver core NOC-read copy of local data
  2  remote_data_cb_id      receiver core fabric-received remote data
  3  output_cb_id           receiver core reduction output
  4  residual_cb_id         receiver core optional residual
"""

from dataclasses import dataclass

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

MAX_NUM_LINKS = 2
CCL_TILE_H = 32
CCL_TILE_W = 32


# ---------------------------------------------------------------------------
# Compile-time named arg schemas
# ---------------------------------------------------------------------------


def _writer_named_ct_schema(
    local_data_cb_id=0,
    input_num_tiles=0,
    page_size_bytes=0,
    tiles_per_chunk=0,
    last_chunk_tiles=0,
    num_chunks=0,
    num_links=0,
    writer_link_index=0,
    writer_signal_local_ready=0,
    skip_local_push=0,
):
    """Writer CT args: per-link fabric sender on the sender core."""
    return [
        ("allreduce_local_data_cb_id", int(local_data_cb_id)),
        ("allreduce_input_num_tiles", int(input_num_tiles)),
        ("allreduce_page_size_bytes", int(page_size_bytes)),
        ("allreduce_tiles_per_chunk", int(tiles_per_chunk)),
        ("allreduce_last_chunk_tiles", int(last_chunk_tiles)),
        ("allreduce_num_chunks", int(num_chunks)),
        ("allreduce_num_links", int(num_links)),
        ("allreduce_writer_link_index", int(writer_link_index)),
        ("allreduce_writer_signal_local_ready", int(writer_signal_local_ready)),
        ("allreduce_skip_local_push", int(skip_local_push)),
    ]


def _reader_named_ct_schema(
    recv_local_data_cb_id=0,
    remote_data_cb_id=0,
    residual_cb_id=0,
    has_residual=0,
    total_num_tiles=0,
    page_size_bytes=0,
    tiles_per_chunk=0,
    last_chunk_tiles=0,
    num_chunks=0,
    num_links=0,
):
    return [
        ("allreduce_recv_local_data_cb_id", int(recv_local_data_cb_id)),
        ("allreduce_remote_data_cb_id", int(remote_data_cb_id)),
        ("allreduce_residual_cb_id", int(residual_cb_id)),
        ("allreduce_has_residual", int(has_residual)),
        ("allreduce_total_num_tiles", int(total_num_tiles)),
        ("allreduce_page_size_bytes", int(page_size_bytes)),
        ("allreduce_tiles_per_chunk", int(tiles_per_chunk)),
        ("allreduce_last_chunk_tiles", int(last_chunk_tiles)),
        ("allreduce_num_chunks", int(num_chunks)),
        ("allreduce_num_links", int(num_links)),
    ]


def _compute_named_ct_schema(
    cb_remote=0,
    cb_local=0,
    cb_out=0,
    cb_residual=0,
    has_residual=0,
    num_tiles=0,
):
    return [
        ("allreduce_cb_remote", int(cb_remote)),
        ("allreduce_cb_local", int(cb_local)),
        ("allreduce_cb_out", int(cb_out)),
        ("allreduce_cb_residual", int(cb_residual)),
        ("allreduce_has_residual", int(has_residual)),
        ("allreduce_num_tiles", int(num_tiles)),
    ]


# ---------------------------------------------------------------------------
# Runtime arg schemas (positional common RT args)
# ---------------------------------------------------------------------------


def _sender_writer_common_rt_schema(
    intermediate_buffer_address=0,
    dest_noc_x=0,
    dest_noc_y=0,
):
    """Sender core common RT args: neighbor intermediate buffer base + dest NOC coordinates."""
    return [
        int(intermediate_buffer_address),
        int(dest_noc_x),
        int(dest_noc_y),
    ]


def _receiver_common_rt_schema(
    sem_bank_addr_0=0,
    sem_bank_addr_1=0,
    sender_noc_x=0,
    sender_noc_y=0,
    sender_local_data_l1_addr=0,
    local_ready_sem_bank_addr=0,
):
    """Receiver core common RT args: fabric semaphores, sender physical coords, local data addr, local_ready sem."""
    return [
        int(sem_bank_addr_0),
        int(sem_bank_addr_1),
        int(sender_noc_x),
        int(sender_noc_y),
        int(sender_local_data_l1_addr),
        int(local_ready_sem_bank_addr),
    ]


# ---------------------------------------------------------------------------
# Chunk config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkConfig:
    tiles_per_chunk: int
    last_chunk_tiles: int
    num_chunks: int


def resolve_chunk_config(total_num_tiles, tile_size_bytes, chunk_num_tiles=None):
    if total_num_tiles <= 0:
        raise ValueError(f"total_num_tiles must be > 0, got {total_num_tiles}")

    max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    max_tiles_per_packet = max_payload // tile_size_bytes
    if max_tiles_per_packet <= 0:
        raise ValueError(f"Invalid max payload/tile size: max_payload={max_payload}, tile_size_bytes={tile_size_bytes}")

    if chunk_num_tiles is None:
        tiles_per_chunk = min(total_num_tiles, max_tiles_per_packet)
    else:
        tiles_per_chunk = int(chunk_num_tiles)
        if tiles_per_chunk <= 0:
            raise ValueError("chunk_num_tiles must be > 0")
        tiles_per_chunk = min(tiles_per_chunk, max_tiles_per_packet, total_num_tiles)

    num_chunks = (total_num_tiles + tiles_per_chunk - 1) // tiles_per_chunk
    last_chunk_tiles = total_num_tiles - (num_chunks - 1) * tiles_per_chunk
    return ChunkConfig(
        tiles_per_chunk=tiles_per_chunk,
        last_chunk_tiles=last_chunk_tiles,
        num_chunks=num_chunks,
    )


# ---------------------------------------------------------------------------
# AllReduceConfig
# ---------------------------------------------------------------------------


class AllReduceConfig:
    """Two-core standalone all-reduce: dual fabric senders (NCRISC+BRISC) on sender core.

    Sender/receiver logical cores are derived from input vs output L1 shard grid starts.

    Link assignment is fixed (see ``_resolve_link_assignment``):
      BRISC  -> link 0, signal_local_ready=1
      NCRISC -> link 1, signal_local_ready=0
    When ``num_links=1``, NCRISC is automatically disabled.

    Public API is RISC-type based for easy consumption by fused ops:
      Named CT args: get_ncrisc_named_ct_args, get_brisc_named_ct_args, get_trisc_named_ct_args
      Common RT:     get_sender_ncrisc_common_rt_args, get_sender_brisc_common_rt_args,
                     get_receiver_brisc_common_rt_args
      Per-core RT:   get_ncrisc_per_core_rt_args, get_brisc_per_core_rt_args
      CB descriptors: get_cb_descriptors
    """

    def __init__(
        self,
        mesh_device,
        input_tensor_mesh,
        intermediate_tensor,
        output_tensor,
        semaphores,
        cluster_axis=0,
        num_links=1,
        chunk_num_tiles=None,
        local_data_cb_id=0,
        recv_local_data_cb_id=1,
        remote_data_cb_id=2,
        output_cb_id=3,
        residual_tensor_mesh=None,
        residual_cb_id=4,
        skip_local_push=False,
    ):
        self.mesh_device = mesh_device
        self.cluster_axis = int(cluster_axis)
        self.num_links = int(num_links)
        self.local_data_cb_id = int(local_data_cb_id)
        self.recv_local_data_cb_id = int(recv_local_data_cb_id)
        self.remote_data_cb_id = int(remote_data_cb_id)
        self.output_cb_id = int(output_cb_id)
        self.residual_cb_id = int(residual_cb_id)
        self.skip_local_push = bool(skip_local_push)
        self.has_residual = residual_tensor_mesh is not None

        if self.num_links < 1 or self.num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links must be in [1, {MAX_NUM_LINKS}], got {self.num_links}")
        if len(semaphores) < self.num_links + 1:
            raise ValueError(
                f"Need num_links fabric semaphores plus one local_ready semaphore; "
                f"got {len(semaphores)} < {self.num_links + 1}"
            )

        self._resolve_link_assignment()

        mesh_shape = mesh_device.shape
        axis_size = mesh_shape[self.cluster_axis]
        if axis_size != 2:
            raise ValueError(
                f"All-reduce currently supports exactly 2 devices along cluster_axis={self.cluster_axis}, got {axis_size}"
            )

        self.input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        self.intermediate_tensors_per_device = ttnn.get_device_tensors(intermediate_tensor)
        self.output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        self.residual_tensors_per_device = ttnn.get_device_tensors(residual_tensor_mesh) if self.has_residual else None

        input_sample = self.input_tensors_per_device[0]
        shard_width = input_sample.memory_config().shard_spec.shape[1]
        tiny_tile_w = input_sample.tile.tile_shape[1]
        input_num_pages = shard_width // tiny_tile_w
        if input_num_pages % 32 != 0:
            raise ValueError(
                f"Input tiny tile count must be divisible by 32 for 32x32 reinterpretation, got {input_num_pages}"
            )

        self.total_num_tiles = input_num_pages // 32
        self.element_size = 2
        self.tile_size_bytes = CCL_TILE_H * CCL_TILE_W * self.element_size
        self.chunk = resolve_chunk_config(self.total_num_tiles, self.tile_size_bytes, chunk_num_tiles)

        self.data_format = input_sample.dtype
        self.standard_tile_descriptor = ttnn.TileDescriptor(CCL_TILE_H, CCL_TILE_W)

        input_grid_start = input_sample.memory_config().shard_spec.grid.bounding_box().start
        out_sample = self.output_tensors_per_device[0]
        out_grid_start = out_sample.memory_config().shard_spec.grid.bounding_box().start

        self.sender_core = ttnn.CoreCoord(int(input_grid_start.x), int(input_grid_start.y))
        self.receiver_core = ttnn.CoreCoord(int(out_grid_start.x), int(out_grid_start.y))
        im_grid_start = self.intermediate_tensors_per_device[0].memory_config().shard_spec.grid.bounding_box().start
        if im_grid_start != self.receiver_core:
            raise ValueError(
                f"Intermediate must be L1-sharded on receiver core {self.receiver_core}, got {im_grid_start}"
            )
        if self.has_residual:
            res_start = self.residual_tensors_per_device[0].memory_config().shard_spec.grid.bounding_box().start
            if res_start != self.receiver_core:
                raise ValueError(f"Residual must be L1-sharded on receiver core {self.receiver_core}, got {res_start}")

        sem_addrs = [ttnn.get_global_semaphore_address(semaphores[i]) for i in range(self.num_links)]
        while len(sem_addrs) < MAX_NUM_LINKS:
            sem_addrs.append(0)
        self.local_ready_sem_addr = ttnn.get_global_semaphore_address(semaphores[self.num_links])

        self._per_device = {}
        mesh_rows, mesh_cols = mesh_shape
        self.receiver_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.receiver_core, self.receiver_core)])
        self.sender_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.sender_core, self.sender_core)])

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col
                device = self.input_tensors_per_device[device_idx].device()

                if self.cluster_axis == 0:
                    neighbor_coord = ttnn.MeshCoordinate(1 - row, col)
                else:
                    neighbor_coord = ttnn.MeshCoordinate(row, 1 - col)
                neighbor_idx = int(neighbor_coord[0]) * mesh_cols + int(neighbor_coord[1])
                neighbor_device = self.input_tensors_per_device[neighbor_idx].device()
                neighbor_recv_phys = neighbor_device.worker_core_from_logical_core(self.receiver_core)
                neighbor_intermediate_addr = self.intermediate_tensors_per_device[neighbor_idx].buffer_address()

                sender_phys = device.worker_core_from_logical_core(self.sender_core)
                receiver_phys = device.worker_core_from_logical_core(self.receiver_core)
                input_l1_addr = self.input_tensors_per_device[device_idx].buffer_address()

                self._per_device[coord] = {
                    "device_idx": device_idx,
                    "device": device,
                    "receiver_core_physical": receiver_phys,
                    "sender_core_physical": sender_phys,
                    "sem_addrs": sem_addrs,
                    "fabric_node_id": mesh_device.get_fabric_node_id(coord),
                    "neighbor_fabric_node_id": mesh_device.get_fabric_node_id(neighbor_coord),
                    "dest_noc_x": neighbor_recv_phys.x,
                    "dest_noc_y": neighbor_recv_phys.y,
                    "neighbor_intermediate_buffer_address": neighbor_intermediate_addr,
                    "sender_local_data_l1_addr": input_l1_addr,
                    "local_ready_sem_addr": self.local_ready_sem_addr,
                }

    # ======================================================================
    # Link assignment
    # ======================================================================

    def _resolve_link_assignment(self):
        """Assign fabric links to sender RISCs.

        BRISC  -> link 0, signal_local_ready=1
        NCRISC -> link 1, signal_local_ready=0

        When num_links=1, NCRISC's link_index (1) >= num_links -> C++ writer
        early-returns via ``if constexpr (CT::link_index >= CT::num_links)``.
        """
        self._brisc_link_index = 0
        self._brisc_signal_local_ready = 1
        self._ncrisc_link_index = 1
        self._ncrisc_signal_local_ready = 0
        self._ncrisc_writer_active = self.num_links >= 2
        self._brisc_writer_active = True

    # ======================================================================
    # Public RISC-type based API
    # ======================================================================

    # -- Named CT args (per RISC, passed to UnifiedKernelDescriptor) -------

    def get_ncrisc_named_ct_args(self, coord):
        return self._sender_writer_named_ct_args(
            link_index=self._ncrisc_link_index,
            signal_local_ready=self._ncrisc_signal_local_ready,
        )

    def get_brisc_named_ct_args(self, coord):
        writer_ct = self._sender_writer_named_ct_args(
            link_index=self._brisc_link_index,
            signal_local_ready=self._brisc_signal_local_ready,
        )
        return writer_ct + self._receiver_reader_named_ct_args(coord)

    def get_trisc_named_ct_args(self, coord):
        return self._receiver_compute_named_ct_args(coord)

    # -- Common RT args (per kernel group per RISC) ------------------------

    def get_sender_ncrisc_common_rt_args(self, coord):
        return self._sender_writer_common_rt(coord)

    def get_sender_brisc_common_rt_args(self, coord):
        return self._sender_writer_common_rt(coord)

    def get_receiver_brisc_common_rt_args(self, coord):
        return self._receiver_common_rt(coord)

    # -- Per-core RT args (fabric connection) ------------------------------

    def get_ncrisc_per_core_rt_args(self, coord, program, core):
        if not self._ncrisc_writer_active:
            return []
        return self._sender_per_core_rt_args(
            coord,
            program,
            core,
            link_index=self._ncrisc_link_index,
            signal_local_ready=self._ncrisc_signal_local_ready,
        )

    def get_brisc_per_core_rt_args(self, coord, program, core):
        if not self._brisc_writer_active:
            return []
        return self._sender_per_core_rt_args(
            coord,
            program,
            core,
            link_index=self._brisc_link_index,
            signal_local_ready=self._brisc_signal_local_ready,
        )

    # -- CB descriptors ----------------------------------------------------

    def get_cb_descriptors(self, coord):
        """Return all CB descriptors: sender-side local data CB + receiver-side CBs."""
        info = self._per_device[coord]
        idx = info["device_idx"]

        sender_local_cb = ttnn.cb_descriptor_from_sharded_tensor(
            self.local_data_cb_id, self.input_tensors_per_device[idx]
        )
        sender_local_cb.core_ranges = self.sender_core_set
        sender_local_cb.total_size = self.total_num_tiles * self.tile_size_bytes
        sender_local_cb.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=self.local_data_cb_id,
                data_format=self.data_format,
                page_size=self.tile_size_bytes,
                tile=self.standard_tile_descriptor,
            )
        ]

        receiver_core_set = self.receiver_core_set

        recv_local_cb = ttnn.CBDescriptor(
            total_size=self.total_num_tiles * self.tile_size_bytes,
            core_ranges=receiver_core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=self.recv_local_data_cb_id,
                    data_format=self.data_format,
                    page_size=self.tile_size_bytes,
                    tile=self.standard_tile_descriptor,
                )
            ],
        )

        remote_cb = ttnn.cb_descriptor_from_sharded_tensor(
            self.remote_data_cb_id, self.intermediate_tensors_per_device[idx]
        )
        remote_cb.core_ranges = receiver_core_set
        remote_cb.total_size = self.total_num_tiles * self.tile_size_bytes
        remote_cb.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=self.remote_data_cb_id,
                data_format=self.data_format,
                page_size=self.tile_size_bytes,
                tile=self.standard_tile_descriptor,
            )
        ]

        output_cb = ttnn.cb_descriptor_from_sharded_tensor(self.output_cb_id, self.output_tensors_per_device[idx])
        output_cb.core_ranges = receiver_core_set
        output_cb.total_size = self.total_num_tiles * self.tile_size_bytes
        output_cb.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=self.output_cb_id,
                data_format=self.data_format,
                page_size=self.tile_size_bytes,
                tile=self.standard_tile_descriptor,
            )
        ]

        cbs = [sender_local_cb, recv_local_cb, remote_cb, output_cb]
        if self.has_residual:
            residual_cb = ttnn.cb_descriptor_from_sharded_tensor(
                self.residual_cb_id, self.residual_tensors_per_device[idx]
            )
            residual_cb.core_ranges = receiver_core_set
            residual_cb.total_size = self.total_num_tiles * self.tile_size_bytes
            residual_cb.format_descriptors = [
                ttnn.CBFormatDescriptor(
                    buffer_index=self.residual_cb_id,
                    data_format=self.data_format,
                    page_size=self.tile_size_bytes,
                    tile=self.standard_tile_descriptor,
                )
            ]
            cbs.append(residual_cb)
        return cbs

    # ======================================================================
    # Private helpers
    # ======================================================================

    def _sender_writer_named_ct_args(self, link_index, signal_local_ready):
        return _writer_named_ct_schema(
            local_data_cb_id=self.local_data_cb_id,
            input_num_tiles=self.total_num_tiles,
            page_size_bytes=self.tile_size_bytes,
            tiles_per_chunk=self.chunk.tiles_per_chunk,
            last_chunk_tiles=self.chunk.last_chunk_tiles,
            num_chunks=self.chunk.num_chunks,
            num_links=self.num_links,
            writer_link_index=link_index,
            writer_signal_local_ready=signal_local_ready,
            skip_local_push=1 if self.skip_local_push else 0,
        )

    def _receiver_reader_named_ct_args(self, coord):
        return _reader_named_ct_schema(
            recv_local_data_cb_id=self.recv_local_data_cb_id,
            remote_data_cb_id=self.remote_data_cb_id,
            residual_cb_id=self.residual_cb_id,
            has_residual=1 if self.has_residual else 0,
            total_num_tiles=self.total_num_tiles,
            page_size_bytes=self.tile_size_bytes,
            tiles_per_chunk=self.chunk.tiles_per_chunk,
            last_chunk_tiles=self.chunk.last_chunk_tiles,
            num_chunks=self.chunk.num_chunks,
            num_links=self.num_links,
        )

    def _receiver_compute_named_ct_args(self, coord):
        return _compute_named_ct_schema(
            cb_remote=self.remote_data_cb_id,
            cb_local=self.recv_local_data_cb_id,
            cb_out=self.output_cb_id,
            cb_residual=self.residual_cb_id,
            has_residual=1 if self.has_residual else 0,
            num_tiles=self.total_num_tiles,
        )

    def _sender_writer_common_rt(self, coord):
        info = self._per_device[coord]
        return _sender_writer_common_rt_schema(
            intermediate_buffer_address=info["neighbor_intermediate_buffer_address"],
            dest_noc_x=info["dest_noc_x"],
            dest_noc_y=info["dest_noc_y"],
        )

    def _receiver_common_rt(self, coord):
        info = self._per_device[coord]
        sender_phys = info["sender_core_physical"]
        return _receiver_common_rt_schema(
            sem_bank_addr_0=info["sem_addrs"][0],
            sem_bank_addr_1=info["sem_addrs"][1],
            sender_noc_x=sender_phys.x,
            sender_noc_y=sender_phys.y,
            sender_local_data_l1_addr=info["sender_local_data_l1_addr"],
            local_ready_sem_bank_addr=info["local_ready_sem_addr"],
        )

    def _sender_per_core_rt_args(self, coord, program, core, link_index, signal_local_ready):
        info = self._per_device[coord]
        out = [
            int(info["neighbor_fabric_node_id"].mesh_id),
            int(info["neighbor_fabric_node_id"].chip_id),
            int(info["sem_addrs"][link_index]),
        ]
        if signal_local_ready:
            receiver_phys = info["receiver_core_physical"]
            out.extend(
                [
                    int(receiver_phys.x),
                    int(receiver_phys.y),
                    int(info["local_ready_sem_addr"]),
                ]
            )
        out.extend(
            ttnn.setup_fabric_connection(
                src_fabric_node_id=info["fabric_node_id"],
                dst_fabric_node_id=info["neighbor_fabric_node_id"],
                link_idx=link_index,
                program_descriptor=program,
                worker_core=core,
            )
        )
        return out


# ---------------------------------------------------------------------------
# BypassAllReduceConfig
# ---------------------------------------------------------------------------


class BypassAllReduceConfig:
    """Skip-CCL shim returning default (zero) arguments matching AllReduceConfig interface."""

    def __init__(
        self,
        mesh_device,
        input_tensor_mesh,
        local_data_cb_id=0,
        recv_local_data_cb_id=1,
        remote_data_cb_id=2,
        output_cb_id=3,
        residual_cb_id=4,
        num_links=1,
    ):
        self.mesh_device = mesh_device
        self.input_tensor_mesh = input_tensor_mesh
        self.num_links = int(num_links)
        self.local_data_cb_id = int(local_data_cb_id)
        self.recv_local_data_cb_id = int(recv_local_data_cb_id)
        self.remote_data_cb_id = int(remote_data_cb_id)
        self.output_cb_id = int(output_cb_id)
        self.residual_cb_id = int(residual_cb_id)

        if self.num_links < 1 or self.num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links must be in [1, {MAX_NUM_LINKS}], got {self.num_links}")

        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        input_grid_start = input_tensors_per_device[0].memory_config().shard_spec.grid.bounding_box().start
        self.sender_core = ttnn.CoreCoord(int(input_grid_start.x), int(input_grid_start.y))
        self.receiver_core = self.sender_core

        self._per_device = {}
        mesh_rows, mesh_cols = mesh_device.shape
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                idx = row * mesh_cols + col
                device = input_tensors_per_device[idx].device()
                shard_grid_start = input_tensors_per_device[idx].memory_config().shard_spec.grid.bounding_box().start
                worker_core = ttnn.CoreCoord(shard_grid_start.x, shard_grid_start.y)
                worker_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])
                worker_core_physical = device.worker_core_from_logical_core(worker_core)
                self._per_device[coord] = {
                    "worker_core": worker_core,
                    "worker_core_set": worker_core_set,
                    "worker_core_physical": worker_core_physical,
                }

    def get_ncrisc_named_ct_args(self, coord):
        return _writer_named_ct_schema(num_links=self.num_links)

    def get_brisc_named_ct_args(self, coord):
        return _writer_named_ct_schema(num_links=self.num_links) + _reader_named_ct_schema(num_links=self.num_links)

    def get_trisc_named_ct_args(self, coord):
        return _compute_named_ct_schema()

    def get_sender_ncrisc_common_rt_args(self, coord):
        return _sender_writer_common_rt_schema()

    def get_sender_brisc_common_rt_args(self, coord):
        return _sender_writer_common_rt_schema()

    def get_receiver_brisc_common_rt_args(self, coord):
        return _receiver_common_rt_schema()

    def get_ncrisc_per_core_rt_args(self, coord, program, core):
        return []

    def get_brisc_per_core_rt_args(self, coord, program, core):
        return []

    def get_cb_descriptors(self, coord):
        return []


# ---------------------------------------------------------------------------
# DeepseekMinimalAllReduce
# ---------------------------------------------------------------------------


class DeepseekMinimalAllReduce:
    @staticmethod
    def golden(input_tensors, residual_tensor=None):
        result = torch.sum(torch.stack(input_tensors), dim=0)
        if residual_tensor is not None:
            result += residual_tensor
        return result

    @staticmethod
    def get_num_semaphores(num_links=1):
        """Fabric per-link semaphores (``num_links``) plus one ``local_ready`` semaphore."""
        num_links = int(num_links)
        if num_links < 1 or num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links must be in [1, {MAX_NUM_LINKS}], got {num_links}")
        return num_links + 1

    @staticmethod
    def configure(
        mesh_device,
        input_tensor_mesh,
        intermediate_tensor=None,
        output_tensor=None,
        semaphores=None,
        cluster_axis=0,
        num_links=1,
        chunk_num_tiles=None,
        local_data_cb_id=0,
        recv_local_data_cb_id=1,
        remote_data_cb_id=2,
        output_cb_id=3,
        residual_tensor_mesh=None,
        residual_cb_id=4,
        skip_local_push=False,
        skip_ccl=False,
    ):
        if skip_ccl:
            return BypassAllReduceConfig(
                mesh_device=mesh_device,
                input_tensor_mesh=input_tensor_mesh,
                local_data_cb_id=local_data_cb_id,
                recv_local_data_cb_id=recv_local_data_cb_id,
                remote_data_cb_id=remote_data_cb_id,
                output_cb_id=output_cb_id,
                residual_cb_id=residual_cb_id,
                num_links=num_links,
            )

        if semaphores is None:
            raise ValueError("Expected semaphore(s) via `semaphores` for non-skip all-reduce")
        if intermediate_tensor is None or output_tensor is None:
            raise ValueError("Expected `intermediate_tensor` and `output_tensor` for non-skip all-reduce")

        return AllReduceConfig(
            mesh_device=mesh_device,
            input_tensor_mesh=input_tensor_mesh,
            intermediate_tensor=intermediate_tensor,
            output_tensor=output_tensor,
            semaphores=semaphores,
            cluster_axis=cluster_axis,
            num_links=num_links,
            chunk_num_tiles=chunk_num_tiles,
            local_data_cb_id=local_data_cb_id,
            recv_local_data_cb_id=recv_local_data_cb_id,
            remote_data_cb_id=remote_data_cb_id,
            output_cb_id=output_cb_id,
            residual_tensor_mesh=residual_tensor_mesh,
            residual_cb_id=residual_cb_id,
            skip_local_push=skip_local_push,
        )

    @staticmethod
    def op(
        input_tensor_mesh,
        intermediate_tensor,
        semaphores,
        cluster_axis=0,
        num_links=1,
        chunk_num_tiles=None,
        persistent_output_tensor=None,
        residual_tensor_mesh=None,
    ):
        if persistent_output_tensor is None:
            raise ValueError("persistent_output_tensor is required (receiver-core shard)")

        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape

        output_tensor = persistent_output_tensor

        allreduce_config = DeepseekMinimalAllReduce.configure(
            mesh_device=mesh_device,
            input_tensor_mesh=input_tensor_mesh,
            intermediate_tensor=intermediate_tensor,
            output_tensor=output_tensor,
            semaphores=semaphores,
            cluster_axis=cluster_axis,
            num_links=num_links,
            chunk_num_tiles=chunk_num_tiles,
            residual_tensor_mesh=residual_tensor_mesh,
            skip_local_push=False,
            skip_ccl=False,
        )

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/ccl_all_reduce/kernels/all_reduce_kernel.cpp"

        sender_core = allreduce_config.sender_core
        receiver_core = allreduce_config.receiver_core
        sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
        receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(receiver_core, receiver_core)])
        combined_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(sender_core, sender_core), ttnn.CoreRange(receiver_core, receiver_core)]
        )

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)

                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=combined_grid,
                    ncrisc_named_compile_time_args=allreduce_config.get_ncrisc_named_ct_args(coord),
                    brisc_named_compile_time_args=allreduce_config.get_brisc_named_ct_args(coord),
                    trisc_named_compile_time_args=allreduce_config.get_trisc_named_ct_args(coord),
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        math_approx_mode=False,
                    ),
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_allreduce_sender_core",
                            core_range=sender_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_allreduce_receiver_core",
                            core_range=receiver_grid,
                            value=1,
                            other_value=0,
                        ),
                    ],
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(sender_core, []), (receiver_core, [])],
                        brisc_args=[(sender_core, []), (receiver_core, [])],
                        trisc_args=[(receiver_core, [])],
                    ),
                    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                )
                kernel_result = unified_kernel.get_kernel_descriptors()
                sender_group = kernel_result.get_group_by_arg("is_allreduce_sender_core", 1)
                receiver_group = kernel_result.get_group_by_arg("is_allreduce_receiver_core", 1)
                if sender_group is None or receiver_group is None:
                    raise ValueError("Expected sender and receiver kernel groups")

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    semaphores=[],
                    cbs=allreduce_config.get_cb_descriptors(coord),
                )
                program.kernels[
                    sender_group.ncrisc_kernel_index
                ].common_runtime_args = allreduce_config.get_sender_ncrisc_common_rt_args(coord)
                program.kernels[
                    sender_group.brisc_kernel_index
                ].common_runtime_args = allreduce_config.get_sender_brisc_common_rt_args(coord)
                program.kernels[sender_group.trisc_kernel_index].common_runtime_args = []
                program.kernels[receiver_group.ncrisc_kernel_index].common_runtime_args = []
                program.kernels[
                    receiver_group.brisc_kernel_index
                ].common_runtime_args = allreduce_config.get_receiver_brisc_common_rt_args(coord)
                program.kernels[receiver_group.trisc_kernel_index].common_runtime_args = []

                ncrisc_per_core_rt = program.kernels[sender_group.ncrisc_kernel_index].runtime_args[sender_core.x][
                    sender_core.y
                ]
                ncrisc_per_core_rt.extend(allreduce_config.get_ncrisc_per_core_rt_args(coord, program, sender_core))
                brisc_per_core_rt = program.kernels[sender_group.brisc_kernel_index].runtime_args[sender_core.x][
                    sender_core.y
                ]
                brisc_per_core_rt.extend(allreduce_config.get_brisc_per_core_rt_args(coord, program, sender_core))

                logger.info(f"adding program for coord: {coord}")
                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        input_list = [input_tensor_mesh, output_tensor, intermediate_tensor]
        if residual_tensor_mesh is not None:
            input_list.append(residual_tensor_mesh)

        ttnn.generic_op(input_list, mesh_program_descriptor)
        return output_tensor
