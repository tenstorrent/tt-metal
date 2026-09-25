# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh parallelization and collectives for Mistral-Medium-3.5 prefill on the 8x4 Blackhole Galaxy.

Structural copy of ``models/demos/minimax_m3/config.py::MeshConfig`` and
``models/demos/minimax_m3/tt/ccl.py::CCLManager`` (measured there at hidden 6144, chunk 5120, SP8 x TP4):
semaphore ping-pong for ``reduce_scatter_minimal_async`` / ``all_gather_async``, the ring-attention
CCL column carved off the compute grid, and persistent ring-gather scratch for ring-joint SDPA.
Kept local (not imported) so this package does not depend on another model's package.

Mesh axes: rows (axis 0) = SP (sequence), cols (axis 1) = TP (features). The spec fixes SP=8, TP=4.
"""

import torch

import ttnn

# Same L1_SMALL reservation minimax_m3 / DeepSeek open the mesh with.
L1_SMALL_SIZE = 1152


def get_default_num_links(mesh_device) -> int:
    """Blackhole Galaxy exposes 2 fabric links per chip on each mesh axis."""
    return 1 if mesh_device.shape[0] == 1 else 2


class MeshConfig:
    """SP over mesh rows, TP over mesh cols. ``sp`` / ``tp`` are the axis extents."""

    def __init__(self, mesh_shape, sp_axis: int = 0, tp_axis: int = 1):
        self.mesh_shape = tuple(mesh_shape)
        assert {sp_axis, tp_axis} == {0, 1}, f"sp_axis/tp_axis must be 0/1, got {sp_axis}/{tp_axis}"
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.sp = self.mesh_shape[sp_axis]
        self.tp = self.mesh_shape[tp_axis]

    def _dims(self, sp_dim=None, tp_dim=None):
        dims = [None, None]
        dims[self.sp_axis] = sp_dim
        dims[self.tp_axis] = tp_dim
        return tuple(dims)

    def mapper(self, mesh_device, sp_dim=None, tp_dim=None):
        """ShardTensor2dMesh with tensor dim ``sp_dim`` split over SP rows and ``tp_dim`` over TP cols."""
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=self._dims(sp_dim, tp_dim))

    def column_parallel(self, mesh_device):
        """Weight [.., in, out] sharded on ``out`` across TP, replicated across SP."""
        return self.mapper(mesh_device, tp_dim=-1)

    def row_parallel(self, mesh_device):
        """Weight [.., in, out] sharded on ``in`` across TP, replicated across SP."""
        return self.mapper(mesh_device, tp_dim=-2)

    def shard_size(self, total: int) -> int:
        assert total % self.tp == 0, f"{total} not divisible by tp={self.tp}"
        return total // self.tp

    def allgather(self, tensor, ccl_manager, axis, dim, memory_config=None):
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

    def reduce_scatter(self, tensor, ccl_manager, axis, dim, memory_config=None):
        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=dim,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            topology=ccl_manager.topology,
            cluster_axis=axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

    def allreduce(self, tensor, ccl_manager, axis, dim=3):
        """Reduce-scatter + all-gather. Frees ``tensor`` between the two halves to bound peak DRAM."""
        scattered = self.reduce_scatter(tensor, ccl_manager, axis=axis, dim=dim)
        tensor.deallocate(True)
        gathered = self.allgather(scattered, ccl_manager, axis=axis, dim=dim)
        scattered.deallocate(True)
        return gathered

    def __repr__(self):
        return f"MeshConfig({self.mesh_shape}, sp={self.sp}, tp={self.tp})"


class CCLManager:
    """Global semaphores, topology and persistent scratch for every collective in the model."""

    def __init__(self, mesh_device, num_links=None, topology=ttnn.Topology.Linear):
        self.mesh_device = mesh_device
        self.num_links = get_default_num_links(mesh_device) if num_links is None else num_links
        self.topology = topology

        grid = mesh_device.compute_with_storage_grid_size()
        self.compute_grid_size = grid
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        # Ring-joint SDPA runs its CCL workers in the LAST compute column; SDPA compute uses the rest.
        self.ring_attention_ccl_core_grid_offset = (grid.x - 1, 0)

        def sems(n):
            return [ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(n)]

        self.rs_ping_pong_semaphores = sems(3 * 2)
        self.ag_ping_pong_semaphores = sems(2 * 2)
        self.barrier_semaphores = sems(2)
        self.ring_attention_ccl_semaphore_handles = sems(2)
        self.rs_idx = 0
        self.ag_idx = 0
        self.barrier_idx = 0
        self._ring_gather_buffers = {}

    def get_rs_ping_pong_semaphore(self):
        i = self.rs_idx
        self.rs_idx = (i + 1) % 2
        return self.rs_ping_pong_semaphores[i * 3 : (i + 1) * 3]

    def get_ag_ping_pong_semaphore(self):
        i = self.ag_idx
        self.ag_idx = (i + 1) % 2
        return self.ag_ping_pong_semaphores[i * 2 : (i + 1) * 2]

    def get_barrier_semaphore(self):
        i = self.barrier_idx
        self.barrier_idx = (i + 1) % 2
        return self.barrier_semaphores[i]

    def get_ring_gather_buffer(self, key, n_kv, seq, head_dim, dtype):
        """Persistent ring-gather scratch for ring-joint SDPA, allocated once per (key, shape, dtype).

        Heads shard on the TP cols, sequence replicated across the SP rows (the layout the ring op
        reconstructs into). The op fills the gathered region and masks the invalid tail, so reuse
        without re-zeroing is safe.
        """
        cache_key = (key, n_kv, seq, head_dim, str(dtype))
        if cache_key not in self._ring_gather_buffers:
            dims = [None, None]
            dims[1] = 1  # heads over the TP cols (axis 1)
            self._ring_gather_buffers[cache_key] = ttnn.from_torch(
                torch.zeros(1, n_kv, seq, head_dim),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=self.mesh_device.shape, dims=dims),
            )
        return self._ring_gather_buffers[cache_key]
