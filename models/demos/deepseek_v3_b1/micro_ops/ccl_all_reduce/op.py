# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CCL All-Reduce Operation using ttnn.generic_op.

Phase-1 standalone implementation:
- Single-core all-reduce (NCRISC writer, BRISC reader, TRISC compute) on the data core
- Transfer streaming via chunked fabric send/receive
- Full-buffer TRISC compute with sync_cb pop-safety fence

Optional standalone dual-core sender (HLD §11.7, ``dual_core_sender=True``):
- Sender / receiver logical cores are **only** from tensor placement: input L1 shard grid start → sender
  core; output L1 shard grid start → receiver core (same pattern as single-core ``AllReduceConfig`` deriving
  ``worker_core`` from the input shard). No separate core overrides — that would invite mismatch with tensors.
- Sender core: NCRISC + BRISC fabric writers, ``DM_DYNAMIC_NOC``, one link per RISC when ``num_links==2``.
  Per-link fabric sem bank addresses are passed in per-core runtime args (not common).
- Receiver core: dedicated L1 CB for the local operand (NOC read from sender),
  fabric receive + streaming remote CB pushes, TRISC reduction. No ``sync_cb``: local is not shared with
  another RISC on the same core.
- NCRISC signals ``local_ready`` (one global semaphore on the receiver core) after ``cb_wait_front`` on
  the sender local CB; the receiver waits once before NOC-reading the sender. BRISC does not signal.
- Semaphores: ``num_links`` for fabric plus **one** extra for ``local_ready`` (see
  ``DeepseekMinimalAllReduce.get_num_semaphores_dual_core``).
- Input tensor must be L1-sharded on the sender core; intermediate, output, and optional residual on the
  receiver core. ``persistent_output_tensor`` is required for the dual-core path.
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


WRITER_COMMON_RT_KEYS = (
    "intermediate_buffer_address",
    "my_noc_x",
    "my_noc_y",
    "sem_bank_addr_0",
    "sem_bank_addr_1",
)

READER_COMMON_RT_KEYS = (
    "sem_bank_addr_0",
    "sem_bank_addr_1",
)


def _writer_common_rt_schema(
    intermediate_buffer_address=0, my_noc_x=0, my_noc_y=0, sem_bank_addr_0=0, sem_bank_addr_1=0
):
    return {
        "intermediate_buffer_address": int(intermediate_buffer_address),
        "my_noc_x": int(my_noc_x),
        "my_noc_y": int(my_noc_y),
        "sem_bank_addr_0": int(sem_bank_addr_0),
        "sem_bank_addr_1": int(sem_bank_addr_1),
    }


def _reader_common_rt_schema(sem_bank_addr_0=0, sem_bank_addr_1=0):
    return {
        "sem_bank_addr_0": int(sem_bank_addr_0),
        "sem_bank_addr_1": int(sem_bank_addr_1),
    }


def _schema_to_rt_list(schema, keys):
    return [schema[k] for k in keys]


def _writer_named_ct_schema(
    local_data_cb_id=0,
    sync_cb_id=0,
    input_num_tiles=0,
    page_size_bytes=0,
    tiles_per_chunk=0,
    last_chunk_tiles=0,
    num_chunks=0,
    num_links=0,
):
    return [
        ("allreduce_local_data_cb_id", int(local_data_cb_id)),
        ("allreduce_sync_cb_id", int(sync_cb_id)),
        ("allreduce_input_num_tiles", int(input_num_tiles)),
        ("allreduce_page_size_bytes", int(page_size_bytes)),
        ("allreduce_tiles_per_chunk", int(tiles_per_chunk)),
        ("allreduce_last_chunk_tiles", int(last_chunk_tiles)),
        ("allreduce_num_chunks", int(num_chunks)),
        ("allreduce_num_links", int(num_links)),
    ]


def _reader_named_ct_schema(
    local_data_cb_id=0,
    remote_data_cb_id=0,
    residual_cb_id=0,
    has_residual=0,
    skip_local_push=0,
    total_num_tiles=0,
    tiles_per_chunk=0,
    last_chunk_tiles=0,
    num_chunks=0,
    num_links=0,
):
    return [
        ("allreduce_local_data_cb_id", int(local_data_cb_id)),
        ("allreduce_remote_data_cb_id", int(remote_data_cb_id)),
        ("allreduce_residual_cb_id", int(residual_cb_id)),
        ("allreduce_has_residual", int(has_residual)),
        ("allreduce_skip_local_push", int(skip_local_push)),
        ("allreduce_total_num_tiles", int(total_num_tiles)),
        ("allreduce_tiles_per_chunk", int(tiles_per_chunk)),
        ("allreduce_last_chunk_tiles", int(last_chunk_tiles)),
        ("allreduce_num_chunks", int(num_chunks)),
        ("allreduce_num_links", int(num_links)),
    ]


def _writer_dual_named_ct_schema(
    local_data_cb_id=0,
    input_num_tiles=0,
    page_size_bytes=0,
    tiles_per_chunk=0,
    last_chunk_tiles=0,
    num_chunks=0,
    num_links=0,
    writer_link_index=0,
    writer_signal_local_ready=0,
):
    """Writer CT args for dual-core path (no sync_cb).

    ``writer_link_index`` / fabric ``link_idx`` / ``sem_addrs[i]`` must stay consistent per RISC.
    ``writer_signal_local_ready``: 1 on exactly one sender RISC (after ``cb_wait_front``), 0 on others.
    """
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
    ]


def _reader_two_core_named_ct_schema(
    local_data_cb_id=0,
    remote_data_cb_id=0,
    residual_cb_id=0,
    has_residual=0,
    skip_local_push=0,
    total_num_tiles=0,
    page_size_bytes=0,
    tiles_per_chunk=0,
    last_chunk_tiles=0,
    num_chunks=0,
    num_links=0,
):
    return [
        ("allreduce_local_data_cb_id", int(local_data_cb_id)),
        ("allreduce_remote_data_cb_id", int(remote_data_cb_id)),
        ("allreduce_residual_cb_id", int(residual_cb_id)),
        ("allreduce_has_residual", int(has_residual)),
        ("allreduce_skip_local_push", int(skip_local_push)),
        ("allreduce_total_num_tiles", int(total_num_tiles)),
        ("allreduce_page_size_bytes", int(page_size_bytes)),
        ("allreduce_tiles_per_chunk", int(tiles_per_chunk)),
        ("allreduce_last_chunk_tiles", int(last_chunk_tiles)),
        ("allreduce_num_chunks", int(num_chunks)),
        ("allreduce_num_links", int(num_links)),
    ]


def _compute_dual_named_ct_schema(
    cb_remote=0,
    cb_local=0,
    cb_out=0,
    cb_residual=0,
    cb_temp=0,
    has_residual=0,
    num_tiles=0,
):
    """TRISC CT args for dual-core path (no sync_cb). Chunking is reader/writer only; compute is one full-tensor add."""
    return [
        ("allreduce_cb_remote", int(cb_remote)),
        ("allreduce_cb_local", int(cb_local)),
        ("allreduce_cb_out", int(cb_out)),
        ("allreduce_cb_residual", int(cb_residual)),
        ("allreduce_cb_temp", int(cb_temp)),
        ("allreduce_has_residual", int(has_residual)),
        ("allreduce_num_tiles", int(num_tiles)),
    ]


def _compute_named_ct_schema(
    cb_remote=0,
    cb_local=0,
    cb_out=0,
    sync_cb_id=0,
    cb_residual=0,
    cb_temp=0,
    has_residual=0,
    num_tiles=0,
    num_chunks=0,
    tiles_per_chunk=0,
    last_chunk_tiles=0,
):
    return [
        ("allreduce_cb_remote", int(cb_remote)),
        ("allreduce_cb_local", int(cb_local)),
        ("allreduce_cb_out", int(cb_out)),
        ("allreduce_sync_cb_id", int(sync_cb_id)),
        ("allreduce_cb_residual", int(cb_residual)),
        ("allreduce_cb_temp", int(cb_temp)),
        ("allreduce_has_residual", int(has_residual)),
        ("allreduce_num_tiles", int(num_tiles)),
        ("allreduce_num_chunks", int(num_chunks)),
        ("allreduce_tiles_per_chunk", int(tiles_per_chunk)),
        ("allreduce_last_chunk_tiles", int(last_chunk_tiles)),
    ]


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


class AllReduceConfig:
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
        remote_data_cb_id=1,
        output_cb_id=3,
        sync_cb_id=4,
        residual_tensor_mesh=None,
        residual_cb_id=5,
        temp_cb_id=6,
        skip_local_push=False,
    ):
        self.mesh_device = mesh_device
        self.cluster_axis = int(cluster_axis)
        self.num_links = int(num_links)
        self.local_data_cb_id = int(local_data_cb_id)
        self.remote_data_cb_id = int(remote_data_cb_id)
        self.output_cb_id = int(output_cb_id)
        self.sync_cb_id = int(sync_cb_id)
        self.residual_cb_id = int(residual_cb_id)
        self.temp_cb_id = int(temp_cb_id)
        self.skip_local_push = bool(skip_local_push)
        self.has_residual = residual_tensor_mesh is not None

        if self.num_links < 1 or self.num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links must be in [1, {MAX_NUM_LINKS}], got {self.num_links}")
        if len(semaphores) < self.num_links:
            raise ValueError(f"Need at least num_links semaphores, got {len(semaphores)} < {self.num_links}")

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

        # Validate/derive 32x32 reinterpretation geometry.
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

        # Global semaphore addresses are identical across devices.
        sem_addrs = [ttnn.get_global_semaphore_address(semaphores[i]) for i in range(self.num_links)]
        while len(sem_addrs) < MAX_NUM_LINKS:
            sem_addrs.append(0)

        self._per_device = {}
        mesh_rows, mesh_cols = mesh_shape
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col
                device = self.input_tensors_per_device[device_idx].device()

                input_shard_grid = self.input_tensors_per_device[device_idx].memory_config().shard_spec.grid
                shard_grid_start = input_shard_grid.bounding_box().start
                worker_core = ttnn.CoreCoord(shard_grid_start.x, shard_grid_start.y)
                worker_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])
                worker_core_physical = device.worker_core_from_logical_core(worker_core)

                if self.cluster_axis == 0:
                    neighbor = ttnn.MeshCoordinate(1 - row, col)
                else:
                    neighbor = ttnn.MeshCoordinate(row, 1 - col)

                self._per_device[coord] = {
                    "device_idx": device_idx,
                    "device": device,
                    "worker_core": worker_core,
                    "worker_core_set": worker_core_set,
                    "worker_core_physical": worker_core_physical,
                    "sem_addrs": sem_addrs,
                    "fabric_node_id": mesh_device.get_fabric_node_id(coord),
                    "neighbor_fabric_node_id": mesh_device.get_fabric_node_id(neighbor),
                }

    def get_brisc_ct_args(self, coord):
        return _reader_named_ct_schema(
            local_data_cb_id=self.local_data_cb_id,
            remote_data_cb_id=self.remote_data_cb_id,
            residual_cb_id=self.residual_cb_id,
            has_residual=1 if self.has_residual else 0,
            skip_local_push=1 if self.skip_local_push else 0,
            total_num_tiles=self.total_num_tiles,
            tiles_per_chunk=self.chunk.tiles_per_chunk,
            last_chunk_tiles=self.chunk.last_chunk_tiles,
            num_chunks=self.chunk.num_chunks,
            num_links=self.num_links,
        )

    def get_ncrisc_ct_args(self, coord):
        return _writer_named_ct_schema(
            local_data_cb_id=self.local_data_cb_id,
            sync_cb_id=self.sync_cb_id,
            input_num_tiles=self.total_num_tiles,
            page_size_bytes=self.tile_size_bytes,
            tiles_per_chunk=self.chunk.tiles_per_chunk,
            last_chunk_tiles=self.chunk.last_chunk_tiles,
            num_chunks=self.chunk.num_chunks,
            num_links=self.num_links,
        )

    def get_trisc_ct_args(self, coord):
        return _compute_named_ct_schema(
            cb_remote=self.remote_data_cb_id,
            cb_local=self.local_data_cb_id,
            cb_out=self.output_cb_id,
            sync_cb_id=self.sync_cb_id,
            cb_residual=self.residual_cb_id,
            cb_temp=self.temp_cb_id,
            has_residual=1 if self.has_residual else 0,
            num_tiles=self.total_num_tiles,
            num_chunks=self.chunk.num_chunks,
            tiles_per_chunk=self.chunk.tiles_per_chunk,
            last_chunk_tiles=self.chunk.last_chunk_tiles,
        )

    def get_brisc_common_rt_args(self, coord):
        info = self._per_device[coord]
        schema = _reader_common_rt_schema(
            sem_bank_addr_0=info["sem_addrs"][0],
            sem_bank_addr_1=info["sem_addrs"][1],
        )
        return _schema_to_rt_list(schema, READER_COMMON_RT_KEYS)

    def get_ncrisc_common_rt_args(self, coord):
        info = self._per_device[coord]
        schema = _writer_common_rt_schema(
            intermediate_buffer_address=self.intermediate_tensors_per_device[info["device_idx"]].buffer_address(),
            my_noc_x=info["worker_core_physical"].x,
            my_noc_y=info["worker_core_physical"].y,
            sem_bank_addr_0=info["sem_addrs"][0],
            sem_bank_addr_1=info["sem_addrs"][1],
        )
        return _schema_to_rt_list(schema, WRITER_COMMON_RT_KEYS)

    def get_ncrisc_per_core_rt_args(self, coord, program, core):
        info = self._per_device[coord]
        fabric_args = [
            int(info["neighbor_fabric_node_id"].mesh_id),
            int(info["neighbor_fabric_node_id"].chip_id),
        ]
        for link_idx in range(self.num_links):
            fabric_args.extend(
                ttnn.setup_fabric_connection(
                    src_fabric_node_id=info["fabric_node_id"],
                    dst_fabric_node_id=info["neighbor_fabric_node_id"],
                    link_idx=link_idx,
                    program_descriptor=program,
                    worker_core=core,
                )
            )
        return fabric_args

    def get_cb_descriptors(self, coord):
        info = self._per_device[coord]
        core_set = info["worker_core_set"]
        idx = info["device_idx"]

        remote_cb = ttnn.cb_descriptor_from_sharded_tensor(
            self.remote_data_cb_id, self.intermediate_tensors_per_device[idx]
        )
        remote_cb.core_ranges = core_set
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
        output_cb.core_ranges = core_set
        output_cb.total_size = self.total_num_tiles * self.tile_size_bytes
        output_cb.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=self.output_cb_id,
                data_format=self.data_format,
                page_size=self.tile_size_bytes,
                tile=self.standard_tile_descriptor,
            )
        ]

        sync_cb = ttnn.CBDescriptor(
            total_size=self.tile_size_bytes,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=self.sync_cb_id,
                    data_format=self.data_format,
                    page_size=self.tile_size_bytes,
                    tile=self.standard_tile_descriptor,
                )
            ],
        )

        cbs = [remote_cb, output_cb, sync_cb]
        if self.has_residual:
            residual_cb = ttnn.cb_descriptor_from_sharded_tensor(
                self.residual_cb_id, self.residual_tensors_per_device[idx]
            )
            residual_cb.core_ranges = core_set
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

    def get_worker_core(self, coord):
        return self._per_device[coord]["worker_core"]

    def get_worker_core_set(self, coord):
        return self._per_device[coord]["worker_core_set"]


class AllReduceDualCoreConfig:
    """Two-core standalone all-reduce: dual fabric senders (NCRISC+BRISC) on sender core.

    Sender/receiver logical cores are derived only from input vs output L1 shard grid starts (same idea as
    ``AllReduceConfig`` deriving ``worker_core`` from the input shard).
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
        remote_data_cb_id=1,
        output_cb_id=3,
        residual_tensor_mesh=None,
        residual_cb_id=5,
        temp_cb_id=6,
        skip_local_push=False,
        sender_ncrisc_link_index=0,
        sender_ncrisc_signal_local_ready=1,
        sender_brisc_link_index=1,
        sender_brisc_signal_local_ready=0,
    ):
        self.mesh_device = mesh_device
        self.cluster_axis = int(cluster_axis)
        self.num_links = int(num_links)
        self.local_data_cb_id = int(local_data_cb_id)
        self.remote_data_cb_id = int(remote_data_cb_id)
        self.output_cb_id = int(output_cb_id)
        self.residual_cb_id = int(residual_cb_id)
        self.temp_cb_id = int(temp_cb_id)
        self.skip_local_push = bool(skip_local_push)
        self.has_residual = residual_tensor_mesh is not None

        if self.num_links < 1 or self.num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links must be in [1, {MAX_NUM_LINKS}], got {self.num_links}")
        if len(semaphores) < self.num_links + 1:
            raise ValueError(
                f"dual_core_sender needs num_links fabric semaphores plus one local_ready semaphore; "
                f"got {len(semaphores)} < {self.num_links + 1}"
            )

        self.sender_ncrisc_link_index = int(sender_ncrisc_link_index)
        self.sender_ncrisc_signal_local_ready = int(sender_ncrisc_signal_local_ready)
        self.sender_brisc_link_index = int(sender_brisc_link_index)
        self.sender_brisc_signal_local_ready = int(sender_brisc_signal_local_ready)
        if not (0 <= self.sender_ncrisc_link_index < self.num_links):
            raise ValueError(f"sender_ncrisc_link_index must be in [0, num_links), got {self.sender_ncrisc_link_index}")
        if self.num_links >= 2:
            if not (0 <= self.sender_brisc_link_index < self.num_links):
                raise ValueError(
                    f"sender_brisc_link_index must be in [0, num_links) when num_links >= 2, "
                    f"got {self.sender_brisc_link_index}"
                )

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
                f"dual_core_sender: intermediate must be L1-sharded on receiver core {self.receiver_core}, got {im_grid_start}"
            )
        if self.has_residual:
            res_start = self.residual_tensors_per_device[0].memory_config().shard_spec.grid.bounding_box().start
            if res_start != self.receiver_core:
                raise ValueError(
                    f"dual_core_sender: residual must be L1-sharded on receiver core {self.receiver_core}, got {res_start}"
                )

        sem_addrs = [ttnn.get_global_semaphore_address(semaphores[i]) for i in range(self.num_links)]
        while len(sem_addrs) < MAX_NUM_LINKS:
            sem_addrs.append(0)
        self.local_ready_sem_addr = ttnn.get_global_semaphore_address(semaphores[self.num_links])

        self._per_device = {}
        mesh_rows, mesh_cols = mesh_shape
        receiver_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.receiver_core, self.receiver_core)])
        sender_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.sender_core, self.sender_core)])

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
                recv_phys = device.worker_core_from_logical_core(self.receiver_core)
                input_l1_addr = self.input_tensors_per_device[device_idx].buffer_address()

                self._per_device[coord] = {
                    "device_idx": device_idx,
                    "device": device,
                    "receiver_core": self.receiver_core,
                    "sender_core": self.sender_core,
                    "receiver_core_set": receiver_core_set,
                    "sender_core_set": sender_core_set,
                    "receiver_core_physical": recv_phys,
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

    def get_sender_writer_common_rt_args(self, coord):
        info = self._per_device[coord]
        return [
            int(info["neighbor_intermediate_buffer_address"]),
            int(info["dest_noc_x"]),
            int(info["dest_noc_y"]),
        ]

    def get_receiver_common_rt_args(self, coord):
        info = self._per_device[coord]
        s = info["sender_core_physical"]
        return [
            int(info["sem_addrs"][0]),
            int(info["sem_addrs"][1]),
            int(s.x),
            int(s.y),
            int(info["sender_local_data_l1_addr"]),
            int(info["local_ready_sem_addr"]),
        ]

    def get_sender_ncrisc_writer_named_args(self, coord):
        return _writer_dual_named_ct_schema(
            local_data_cb_id=self.local_data_cb_id,
            input_num_tiles=self.total_num_tiles,
            page_size_bytes=self.tile_size_bytes,
            tiles_per_chunk=self.chunk.tiles_per_chunk,
            last_chunk_tiles=self.chunk.last_chunk_tiles,
            num_chunks=self.chunk.num_chunks,
            num_links=self.num_links,
            writer_link_index=self.sender_ncrisc_link_index,
            writer_signal_local_ready=self.sender_ncrisc_signal_local_ready,
        )

    def get_sender_brisc_writer_named_args(self, coord):
        # When num_links < 2, BRISC does not supply fabric RT args; compile writer with link_index >= num_links
        # so WriterSingleLink returns immediately (avoids reading empty per-core args).
        brisc_li = self.sender_brisc_link_index if self.num_links >= 2 else self.num_links
        logger.info(f"BRISC writer link index: {brisc_li} (num_links={self.num_links})")
        return _writer_dual_named_ct_schema(
            local_data_cb_id=self.local_data_cb_id,
            input_num_tiles=self.total_num_tiles,
            page_size_bytes=self.tile_size_bytes,
            tiles_per_chunk=self.chunk.tiles_per_chunk,
            last_chunk_tiles=self.chunk.last_chunk_tiles,
            num_chunks=self.chunk.num_chunks,
            num_links=self.num_links,
            writer_link_index=brisc_li,
            writer_signal_local_ready=self.sender_brisc_signal_local_ready,
        )

    def get_receiver_named_args(self, coord):
        return _reader_two_core_named_ct_schema(
            local_data_cb_id=self.local_data_cb_id,
            remote_data_cb_id=self.remote_data_cb_id,
            residual_cb_id=self.residual_cb_id,
            has_residual=1 if self.has_residual else 0,
            skip_local_push=1 if self.skip_local_push else 0,
            total_num_tiles=self.total_num_tiles,
            page_size_bytes=self.tile_size_bytes,
            tiles_per_chunk=self.chunk.tiles_per_chunk,
            last_chunk_tiles=self.chunk.last_chunk_tiles,
            num_chunks=self.chunk.num_chunks,
            num_links=self.num_links,
        )

    def get_receiver_trisc_named_args(self, coord):
        return _compute_dual_named_ct_schema(
            cb_remote=self.remote_data_cb_id,
            cb_local=self.local_data_cb_id,
            cb_out=self.output_cb_id,
            cb_residual=self.residual_cb_id,
            cb_temp=self.temp_cb_id,
            has_residual=1 if self.has_residual else 0,
            num_tiles=self.total_num_tiles,
        )

    def get_ncrisc_per_core_rt_args(self, coord, program, core):
        info = self._per_device[coord]
        rp = info["receiver_core_physical"]
        li = self.sender_ncrisc_link_index
        # Order matches WriterSingleLink: mesh_id, chip_id, link_sem_bank; if signal_local_ready (named CT): …
        out = [
            int(info["neighbor_fabric_node_id"].mesh_id),
            int(info["neighbor_fabric_node_id"].chip_id),
            int(info["sem_addrs"][li]),
        ]
        if self.sender_ncrisc_signal_local_ready:
            out.extend(
                [
                    int(rp.x),
                    int(rp.y),
                    int(info["local_ready_sem_addr"]),
                ]
            )
        out.extend(
            ttnn.setup_fabric_connection(
                src_fabric_node_id=info["fabric_node_id"],
                dst_fabric_node_id=info["neighbor_fabric_node_id"],
                link_idx=li,
                program_descriptor=program,
                worker_core=core,
            )
        )
        return out

    def get_brisc_per_core_rt_args(self, coord, program, core):
        info = self._per_device[coord]
        if self.num_links < 2:
            return []
        li = self.sender_brisc_link_index
        out = [
            int(info["neighbor_fabric_node_id"].mesh_id),
            int(info["neighbor_fabric_node_id"].chip_id),
            int(info["sem_addrs"][li]),
        ]
        if self.sender_brisc_signal_local_ready:
            rp = info["receiver_core_physical"]
            out.extend(
                [
                    int(rp.x),
                    int(rp.y),
                    int(info["local_ready_sem_addr"]),
                ]
            )
        out.extend(
            ttnn.setup_fabric_connection(
                src_fabric_node_id=info["fabric_node_id"],
                dst_fabric_node_id=info["neighbor_fabric_node_id"],
                link_idx=li,
                program_descriptor=program,
                worker_core=core,
            )
        )
        return out

    def get_cb_descriptors(self, coord):
        info = self._per_device[coord]
        core_set = info["receiver_core_set"]
        idx = info["device_idx"]

        local_recv_cb = ttnn.CBDescriptor(
            total_size=self.total_num_tiles * self.tile_size_bytes,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=self.local_data_cb_id,
                    data_format=self.data_format,
                    page_size=self.tile_size_bytes,
                    tile=self.standard_tile_descriptor,
                )
            ],
        )

        remote_cb = ttnn.cb_descriptor_from_sharded_tensor(
            self.remote_data_cb_id, self.intermediate_tensors_per_device[idx]
        )
        remote_cb.core_ranges = core_set
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
        output_cb.core_ranges = core_set
        output_cb.total_size = self.total_num_tiles * self.tile_size_bytes
        output_cb.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=self.output_cb_id,
                data_format=self.data_format,
                page_size=self.tile_size_bytes,
                tile=self.standard_tile_descriptor,
            )
        ]

        cbs = [local_recv_cb, remote_cb, output_cb]
        if self.has_residual:
            residual_cb = ttnn.cb_descriptor_from_sharded_tensor(
                self.residual_cb_id, self.residual_tensors_per_device[idx]
            )
            residual_cb.core_ranges = core_set
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


class BypassAllReduceConfig:
    """Skip-CCL shim that preserves the AllReduceConfig interface."""

    def __init__(
        self,
        mesh_device,
        input_tensor_mesh,
        local_data_cb_id=0,
        remote_data_cb_id=1,
        output_cb_id=3,
        sync_cb_id=4,
        residual_cb_id=5,
        temp_cb_id=6,
        num_links=1,
    ):
        self.mesh_device = mesh_device
        self.input_tensor_mesh = input_tensor_mesh
        self.num_links = int(num_links)
        self.local_data_cb_id = int(local_data_cb_id)
        self.remote_data_cb_id = int(remote_data_cb_id)
        self.output_cb_id = int(output_cb_id)
        self.sync_cb_id = int(sync_cb_id)
        self.residual_cb_id = int(residual_cb_id)
        self.temp_cb_id = int(temp_cb_id)

        if self.num_links < 1 or self.num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links must be in [1, {MAX_NUM_LINKS}], got {self.num_links}")

        self._per_device = {}
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        mesh_rows, mesh_cols = mesh_device.shape
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                idx = row * mesh_cols + col
                device = input_tensors_per_device[idx].device()
                input_shard_grid = input_tensors_per_device[idx].memory_config().shard_spec.grid
                shard_grid_start = input_shard_grid.bounding_box().start
                worker_core = ttnn.CoreCoord(shard_grid_start.x, shard_grid_start.y)
                worker_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])
                worker_core_physical = device.worker_core_from_logical_core(worker_core)
                self._per_device[coord] = {
                    "worker_core": worker_core,
                    "worker_core_set": worker_core_set,
                    "worker_core_physical": worker_core_physical,
                }

    def get_brisc_ct_args(self, coord):
        return _reader_named_ct_schema(num_links=self.num_links)

    def get_ncrisc_ct_args(self, coord):
        return _writer_named_ct_schema(num_links=self.num_links)

    def get_trisc_ct_args(self, coord):
        return _compute_named_ct_schema()

    def get_brisc_common_rt_args(self, coord):
        return _schema_to_rt_list(_reader_common_rt_schema(), READER_COMMON_RT_KEYS)

    def get_ncrisc_common_rt_args(self, coord):
        return _schema_to_rt_list(_writer_common_rt_schema(), WRITER_COMMON_RT_KEYS)

    def get_ncrisc_per_core_rt_args(self, coord, program, core):
        return []

    def get_cb_descriptors(self, coord):
        return []

    def get_worker_core(self, coord):
        return self._per_device[coord]["worker_core"]

    def get_worker_core_set(self, coord):
        return self._per_device[coord]["worker_core_set"]


class DeepseekMinimalAllReduce:
    @staticmethod
    def golden(input_tensors, residual_tensor=None):
        result = torch.sum(torch.stack(input_tensors), dim=0)
        if residual_tensor is not None:
            result += residual_tensor
        return result

    @staticmethod
    def get_num_semaphores(num_links=1):
        num_links = int(num_links)
        if num_links < 1 or num_links > MAX_NUM_LINKS:
            raise ValueError(f"num_links must be in [1, {MAX_NUM_LINKS}], got {num_links}")
        return num_links

    @staticmethod
    def get_num_semaphores_dual_core(num_links=1):
        """Fabric per-link semaphores (``num_links``) plus one ``local_ready`` semaphore (NCRISC → receiver)."""
        return DeepseekMinimalAllReduce.get_num_semaphores(num_links) + 1

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
        remote_data_cb_id=1,
        output_cb_id=3,
        sync_cb_id=4,
        residual_tensor_mesh=None,
        residual_cb_id=5,
        temp_cb_id=6,
        skip_local_push=False,
        skip_ccl=False,
        dual_core_sender=False,
        dual_sender_ncrisc_link_index=1,
        dual_sender_ncrisc_signal_local_ready=0,
        dual_sender_brisc_link_index=0,
        dual_sender_brisc_signal_local_ready=1,
    ):
        if skip_ccl:
            return BypassAllReduceConfig(
                mesh_device=mesh_device,
                input_tensor_mesh=input_tensor_mesh,
                local_data_cb_id=local_data_cb_id,
                remote_data_cb_id=remote_data_cb_id,
                output_cb_id=output_cb_id,
                sync_cb_id=sync_cb_id,
                residual_cb_id=residual_cb_id,
                temp_cb_id=temp_cb_id,
                num_links=num_links,
            )

        if semaphores is None:
            raise ValueError("Expected semaphore(s) via `semaphores` for non-skip all-reduce")
        if intermediate_tensor is None or output_tensor is None:
            raise ValueError("Expected `intermediate_tensor` and `output_tensor` for non-skip all-reduce")

        if dual_core_sender:
            return AllReduceDualCoreConfig(
                mesh_device=mesh_device,
                input_tensor_mesh=input_tensor_mesh,
                intermediate_tensor=intermediate_tensor,
                output_tensor=output_tensor,
                semaphores=semaphores,
                cluster_axis=cluster_axis,
                num_links=num_links,
                chunk_num_tiles=chunk_num_tiles,
                local_data_cb_id=local_data_cb_id,
                remote_data_cb_id=remote_data_cb_id,
                output_cb_id=output_cb_id,
                residual_tensor_mesh=residual_tensor_mesh,
                residual_cb_id=residual_cb_id,
                temp_cb_id=temp_cb_id,
                skip_local_push=skip_local_push,
                sender_ncrisc_link_index=dual_sender_ncrisc_link_index,
                sender_ncrisc_signal_local_ready=dual_sender_ncrisc_signal_local_ready,
                sender_brisc_link_index=dual_sender_brisc_link_index,
                sender_brisc_signal_local_ready=dual_sender_brisc_signal_local_ready,
            )

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
            remote_data_cb_id=remote_data_cb_id,
            output_cb_id=output_cb_id,
            sync_cb_id=sync_cb_id,
            residual_tensor_mesh=residual_tensor_mesh,
            residual_cb_id=residual_cb_id,
            temp_cb_id=temp_cb_id,
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
        dual_core_sender=False,
        dual_sender_ncrisc_link_index=1,
        dual_sender_ncrisc_signal_local_ready=0,
        dual_sender_brisc_link_index=0,
        dual_sender_brisc_signal_local_ready=1,
    ):
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape

        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        input_sample = input_tensors_per_device[0]
        dtype = input_sample.dtype
        tile_h, tile_w = input_sample.tile.tile_shape
        element_size = 2
        shard_width = input_sample.memory_config().shard_spec.shape[1]
        input_num_pages = shard_width // tile_w
        if input_num_pages % 32 != 0:
            raise ValueError(f"Input tile count must be divisible by 32, got {input_num_pages}")
        total_num_tiles = input_num_pages // 32
        standard_tile_size_bytes = 32 * 32 * element_size

        if persistent_output_tensor is not None:
            output_tensor = persistent_output_tensor
        elif dual_core_sender:
            raise ValueError(
                "dual_core_sender requires persistent_output_tensor with receiver-core shard (see op docstring)"
            )
        else:
            output_tensor = ttnn.allocate_tensor_on_device(input_sample.spec, mesh_device)

        # CB IDs (standalone)
        local_data_cb_id = 0
        remote_data_cb_id = 1
        output_cb_id = 3
        sync_cb_id = 4
        residual_cb_id = 5
        temp_cb_id = 6

        allreduce_config = DeepseekMinimalAllReduce.configure(
            mesh_device=mesh_device,
            input_tensor_mesh=input_tensor_mesh,
            intermediate_tensor=intermediate_tensor,
            output_tensor=output_tensor,
            semaphores=semaphores,
            cluster_axis=cluster_axis,
            num_links=num_links,
            chunk_num_tiles=chunk_num_tiles,
            local_data_cb_id=local_data_cb_id,
            remote_data_cb_id=remote_data_cb_id,
            output_cb_id=output_cb_id,
            sync_cb_id=sync_cb_id,
            residual_tensor_mesh=residual_tensor_mesh,
            residual_cb_id=residual_cb_id,
            temp_cb_id=temp_cb_id,
            skip_local_push=False,
            skip_ccl=False,
            dual_core_sender=dual_core_sender,
            dual_sender_ncrisc_link_index=dual_sender_ncrisc_link_index,
            dual_sender_ncrisc_signal_local_ready=dual_sender_ncrisc_signal_local_ready,
            dual_sender_brisc_link_index=dual_sender_brisc_link_index,
            dual_sender_brisc_signal_local_ready=dual_sender_brisc_signal_local_ready,
        )

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        dual_kernel_path = (
            "models/demos/deepseek_v3_b1/micro_ops/ccl_all_reduce/kernels/all_reduce_dual_core_kernel.cpp"
        )
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/ccl_all_reduce/kernels/all_reduce_kernel.cpp"

        if dual_core_sender:
            sc = allreduce_config.sender_core
            rc = allreduce_config.receiver_core
            sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sc, sc)])
            receiver_grid = ttnn.CoreRangeSet([ttnn.CoreRange(rc, rc)])
            dual_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sc, sc), ttnn.CoreRange(rc, rc)])

            for row in range(mesh_rows):
                for col in range(mesh_cols):
                    coord = ttnn.MeshCoordinate(row, col)
                    cfg = allreduce_config
                    sender_core = cfg._per_device[coord]["sender_core"]
                    receiver_core = cfg._per_device[coord]["receiver_core"]
                    device_idx = row * mesh_cols + col

                    local_cb = ttnn.cb_descriptor_from_sharded_tensor(
                        local_data_cb_id, input_tensors_per_device[device_idx]
                    )
                    local_cb.core_ranges = sender_grid
                    local_cb.total_size = total_num_tiles * standard_tile_size_bytes
                    local_cb.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=local_data_cb_id,
                            data_format=dtype,
                            page_size=standard_tile_size_bytes,
                            tile=ttnn.TileDescriptor(32, 32),
                        )
                    ]
                    cb_list = [local_cb] + cfg.get_cb_descriptors(coord)

                    w_named_nc = cfg.get_sender_ncrisc_writer_named_args(coord)
                    w_named_br = cfg.get_sender_brisc_writer_named_args(coord)
                    r_br = cfg.get_receiver_named_args(coord)
                    r_tr = cfg.get_receiver_trisc_named_args(coord)
                    unified_kernel = UnifiedKernelDescriptor(
                        kernel_source=dual_kernel_path,
                        core_ranges=dual_grid,
                        ncrisc_named_compile_time_args=w_named_nc,
                        brisc_named_compile_time_args=w_named_br + r_br,
                        trisc_named_compile_time_args=r_tr,
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
                        # noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                    )
                    kernel_result = unified_kernel.get_kernel_descriptors()
                    sender_group = kernel_result.get_group_by_arg("is_allreduce_sender_core", 1)
                    receiver_group = kernel_result.get_group_by_arg("is_allreduce_receiver_core", 1)
                    if sender_group is None or receiver_group is None:
                        raise ValueError("dual_core_sender expected sender and receiver kernel groups")

                    program = ttnn.ProgramDescriptor(
                        kernels=kernel_result.kernels,
                        semaphores=[],
                        cbs=cb_list,
                    )
                    sender_w_rt = cfg.get_sender_writer_common_rt_args(coord)
                    recv_br_rt = cfg.get_receiver_common_rt_args(coord)
                    program.kernels[sender_group.ncrisc_kernel_index].common_runtime_args = sender_w_rt
                    program.kernels[sender_group.brisc_kernel_index].common_runtime_args = sender_w_rt
                    program.kernels[sender_group.trisc_kernel_index].common_runtime_args = []
                    program.kernels[receiver_group.ncrisc_kernel_index].common_runtime_args = []
                    program.kernels[receiver_group.brisc_kernel_index].common_runtime_args = recv_br_rt
                    program.kernels[receiver_group.trisc_kernel_index].common_runtime_args = []

                    n0 = program.kernels[sender_group.ncrisc_kernel_index].runtime_args[sender_core.x][sender_core.y]
                    n0.extend(cfg.get_ncrisc_per_core_rt_args(coord, program, sender_core))
                    n1 = program.kernels[sender_group.brisc_kernel_index].runtime_args[sender_core.x][sender_core.y]
                    n1.extend(cfg.get_brisc_per_core_rt_args(coord, program, sender_core))

                    logger.info(f"adding program for coord: {coord}")
                    mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

            input_list = [input_tensor_mesh, output_tensor, intermediate_tensor]
            if residual_tensor_mesh is not None:
                input_list.append(residual_tensor_mesh)

            logger.info("running dual-core all-reduce")
            ttnn.generic_op(input_list, mesh_program_descriptor)
            return output_tensor

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                core = allreduce_config.get_worker_core(coord)
                core_set = allreduce_config.get_worker_core_set(coord)
                device_idx = row * mesh_cols + col

                # Local data CB backed by input tensor, format-overridden to 32x32.
                local_cb = ttnn.cb_descriptor_from_sharded_tensor(
                    local_data_cb_id, input_tensors_per_device[device_idx]
                )
                local_cb.core_ranges = core_set
                local_cb.total_size = total_num_tiles * standard_tile_size_bytes
                local_cb.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=local_data_cb_id,
                        data_format=dtype,
                        page_size=standard_tile_size_bytes,
                        tile=ttnn.TileDescriptor(32, 32),
                    )
                ]

                cb_list = [local_cb] + allreduce_config.get_cb_descriptors(coord)

                ncrisc_ct = allreduce_config.get_ncrisc_ct_args(coord)
                brisc_ct = allreduce_config.get_brisc_ct_args(coord)
                trisc_ct = allreduce_config.get_trisc_ct_args(coord)

                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=core_set,
                    ncrisc_named_compile_time_args=ncrisc_ct,
                    brisc_named_compile_time_args=brisc_ct,
                    trisc_named_compile_time_args=trisc_ct,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        math_approx_mode=False,
                    ),
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_allreduce_core",
                            core_range=core_set,
                            value=1,
                            other_value=0,
                        ),
                    ],
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(core, [])],
                    ),
                )

                kernel_result = unified_kernel.get_kernel_descriptors()
                group = kernel_result.get_group_by_arg("is_allreduce_core", 1)

                kernel_result.kernels[
                    group.ncrisc_kernel_index
                ].common_runtime_args = allreduce_config.get_ncrisc_common_rt_args(coord)
                kernel_result.kernels[
                    group.brisc_kernel_index
                ].common_runtime_args = allreduce_config.get_brisc_common_rt_args(coord)

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    semaphores=[],
                    cbs=cb_list,
                )

                ncrisc_rt = program.kernels[group.ncrisc_kernel_index].runtime_args[core.x][core.y]
                ncrisc_rt.extend(allreduce_config.get_ncrisc_per_core_rt_args(coord, program, core))

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        input_list = [input_tensor_mesh, output_tensor, intermediate_tensor]
        if residual_tensor_mesh is not None:
            input_list.append(residual_tensor_mesh)

        ttnn.generic_op(input_list, mesh_program_descriptor)
        return output_tensor
