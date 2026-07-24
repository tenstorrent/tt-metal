# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MeshConfig — prefill-time mesh parallelization for the MiniMax-M3 MoE model.

TP shards features across one mesh axis (default: cols); the other axis (rows) carries
sequence-parallel prefill (SP = number of rows) and the expert-parallel MoE. Those degrees
are derived from the mesh shape + tp_axis, so the only knob is TP.
"""

from loguru import logger

import ttnn

# The single configuration validated on hardware (Blackhole Galaxy): (8,4), TP=4 -> SP=8, EP=32.
_VALIDATED_MESH_SHAPE = (8, 4)
_VALIDATED_TP = 4


class MeshConfig:
    """Prefill mesh parallelization. TP is the only knob; SP/EP follow from the mesh shape."""

    def __init__(self, mesh_shape, tp, tp_axis: int = 1):
        """
        Args:
            mesh_shape: (rows, cols) - any mesh size
            tp: tensor-parallel size (shards features along tp_axis)
            tp_axis: which mesh axis is TP (0=rows, 1=cols, default: 1). The other axis
                carries sequence-parallel prefill (SP = size of that axis) and the EP MoE.
        """
        self.mesh_shape = tuple(mesh_shape)
        self.tp = tp
        self.tp_axis = tp_axis
        self.ep_axis = 0 if tp_axis == 1 else 1
        self.sp_axis = self.ep_axis
        self.total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        self._validate()

    def _validate(self):
        tp_dim_size = self.mesh_shape[self.tp_axis]
        if self.tp > tp_dim_size:
            raise ValueError(f"TP({self.tp}) > mesh_{self.tp_axis}_size({tp_dim_size})")
        if self.total_devices % self.tp != 0:
            raise ValueError(f"TP({self.tp}) does not divide total_devices({self.total_devices})")
        if (self.mesh_shape, self.tp) != (_VALIDATED_MESH_SHAPE, _VALIDATED_TP):
            logger.warning(
                f"MeshConfig(mesh_shape={self.mesh_shape}, tp={self.tp}) is untested — only "
                f"mesh_shape={_VALIDATED_MESH_SHAPE}, tp={_VALIDATED_TP} (SP=8, EP=32) is validated on hardware."
            )

    def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None):
        """Unified 2D sharding - replaces all individual mappers"""
        if mesh_dims is None:
            # Default: shard along TP axis only
            mesh_dims = (None, tensor_dim) if self.tp_axis == 1 else (tensor_dim, None)

        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=mesh_dims)

    # Clean semantic helpers (all use unified shard_mapper)
    def column_parallel(self, mesh_device):
        """Column-parallel weights (feature dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Row-parallel weights (sequence/batch dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def sequence_parallel(self, mesh_device):
        """Sequence sharding (for KV cache)"""
        return self.shard_mapper(mesh_device, tensor_dim=-3)

    def shard_size(self, total_size):
        """Size per device for tensor parallel sharding"""
        return total_size // self.tp

    def allreduce(self, tensor, ccl_manager, memory_config=None, pad_size=None, axis=0):
        """
        General tensor parallel allreduce (reduce-scatter + all-gather)

        Note: Caller should check if communication is needed before calling
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        # Optional performance padding (caller specifies, no magic numbers)
        padded = False
        if pad_size and tensor.shape[-2] >= 32:
            tensor_padded = ttnn.pad(tensor, [(0, 0), (0, 0), (0, 0), (0, pad_size)], 0)
            tensor.deallocate(True)
            tensor = tensor_padded
            padded = True

        # Reduce-scatter along TP axis
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            topology=ccl_manager.topology,
            cluster_axis=axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )
        # Free the full-size input (~94 MiB at ISL=16384) before the
        # all-gather allocates its full-size output. Without this, peak
        # live memory inside allreduce is tensor + scattered + gathered
        # (~200 MiB at ISL=16384) which fragments DRAM under
        # long-context prefill — see tt-shield run 26440169327 OOM.
        # Callers must NOT use `tensor` after this returns (they don't:
        # apply_allreduce assigns the return value and deallocates the
        # original handle, which becomes a no-op).
        tensor.deallocate(True)

        # All-gather back
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            dim=3,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )
        scattered.deallocate(True)

        # Remove padding if applied
        if padded:
            gathered_sliced = gathered[:, :, :, :-pad_size]
            gathered.deallocate(True)
            gathered = gathered_sliced
        return gathered

    def allgather(self, tensor, ccl_manager, memory_config=None, axis=0, dim=3, linear=False):
        """
        All-gather operation for tensor parallel communication

        Note: Caller should check if communication is needed before calling
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ttnn.Topology.Linear if linear else ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

    def __repr__(self):
        sp = self.mesh_shape[self.sp_axis]
        return f"MeshConfig({self.mesh_shape}, tp={self.tp}, sp={sp}, tp_axis={self.tp_axis})"
