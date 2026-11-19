# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TP collectives for Muse-Glimmer (1 x TP mesh, cluster_axis=1)."""

import os

from loguru import logger

import ttnn
from models.demos.muse_glimmer_30b.tt.common import decode_memcfg, is_decode

# 1x4 systems that are ring-wired
RING_MESH_DEVICES = {"P300x2", "QB2"}


def is_blackhole_device(mesh_device) -> bool:
    return mesh_device.arch() == ttnn.device.Arch.BLACKHOLE


def default_topology(tp: int):
    """Ring if tp >= 8 else Linear."""
    return ttnn.Topology.Ring if tp >= 8 else ttnn.Topology.Linear


def default_num_links(mesh_device) -> int:
    """2 links on Blackhole, 1 otherwise."""
    return 2 if is_blackhole_device(mesh_device) else 1


def fabric_config():
    """FABRIC_1D_RING on ring-wired systems (per MESH_DEVICE), else FABRIC_1D."""
    ring = os.environ.get("MESH_DEVICE") in RING_MESH_DEVICES
    return ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D


def fabric_config_for(tp: int):
    """Fabric config to set BEFORE opening the mesh (None when tp == 1)."""
    if tp <= 1:
        return None
    return fabric_config()


class MuseCCL:
    """Synchronous all_reduce / all_gather along mesh axis 1.

    ``topology``/``num_links`` are not passed to the synchronous ops: the fabric config decides the
    routing. They are kept for logging and the sampling all-gather.

    On a ring fabric, decode all_reduce (32 rows) uses the fused all_reduce_async with a persistent L1 buffer and
    ping-pong global semaphores; prefill and other meshes use the synchronous all_reduce.
    """

    def __init__(self, mesh_device, topology=None, num_links=None):
        self.mesh_device = mesh_device
        self.tp = int(mesh_device.shape[1])
        self.cluster_axis = 1
        self.ring = self.tp > 1 and ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_1D_RING
        if topology is None:
            topology = ttnn.Topology.Ring if self.ring else default_topology(self.tp)
        self.topology = topology
        self.num_links = num_links if num_links is not None else default_num_links(mesh_device)
        if self.ring:
            grid = mesh_device.compute_with_storage_grid_size()
            cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
            self._ar_sems = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]
            self._ar_idx = 0
            self._ar_buffers = {}
        logger.info(f"MuseCCL: tp={self.tp} topology={self.topology} num_links={self.num_links} ring={self.ring}")

    def decode_buffer(self, shape, dtype):
        """Persistent L1 all_reduce_async buffer for decode inputs of `shape` / `dtype`, created on first use."""
        key = (tuple(shape), dtype)
        if key not in self._ar_buffers:
            shard = decode_memcfg(shape[-1], self.mesh_device).shard_spec
            buf_spec = ttnn.ShardSpec(shard.grid, [32, shard.shape[1] * self.tp], ttnn.ShardOrientation.ROW_MAJOR)
            self._ar_buffers[key] = ttnn.empty(
                list(shape)[:-1] + [shape[-1] * self.tp],
                device=self.mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, buf_spec),
            )
        return self._ar_buffers[key]

    def all_reduce(self, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        if self.tp == 1:
            return x
        if self.ring and is_decode(x):
            mem = decode_memcfg(x.shape[-1], self.mesh_device)
            own = x.memory_config() != mem
            xs = ttnn.to_memory_config(x, mem) if own else x
            out = ttnn.experimental.all_reduce_async(
                xs,
                self.decode_buffer(x.shape, x.dtype),
                cluster_axis=self.cluster_axis,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=self._ar_sems[self._ar_idx],
                memory_config=mem,
                dtype=x.dtype,
                topology=self.topology,
                num_links=self.num_links,
            )
            self._ar_idx = 1 - self._ar_idx
            if own:
                xs.deallocate(True)
            if memory_config.is_sharded():
                return out
            res = ttnn.sharded_to_interleaved(out, memory_config)
            out.deallocate(True)
            return res
        return ttnn.all_reduce(x, cluster_axis=self.cluster_axis, memory_config=memory_config)

    def all_gather(self, x, dim: int, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        if self.tp == 1:
            return x
        return ttnn.all_gather(x, dim=dim, cluster_axis=self.cluster_axis, memory_config=memory_config)
