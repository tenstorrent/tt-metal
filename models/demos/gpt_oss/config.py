# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the MeshConfig class which manages parallelization strategies
across a mesh of devices for the GPT-OSS MoE model.
"""

import ttnn


class MeshConfig:
    """General mesh parallelization for any configuration"""

    def __init__(self, mesh_shape, tp, ep=4, tp_axis=1):
        """
        Args:
            mesh_shape: (rows, cols) - any mesh size
            tp: Tensor parallel size (must specify - no defaults)
            ep: Expert parallel size (default: 4)
            tp_axis: Which mesh axis is TP (0=rows, 1=cols, default: 1)
        """

        self.mesh_shape = tuple(mesh_shape)
        self.tp = tp
        self.ep = ep
        self.tp_axis = tp_axis
        self.ep_axis = 0 if tp_axis == 1 else 1

        total_devices = mesh_shape[0] * mesh_shape[1]
        self.dp = total_devices // (tp * ep)

        if self.tp * self.dp * self.ep != total_devices:
            raise ValueError(f"TP({tp}) × DP({self.dp}) × EP({ep}) != total_devices({total_devices})")

        # Validate TP fits in mesh
        tp_dim_size = mesh_shape[tp_axis]
        if tp > tp_dim_size:
            raise ValueError(f"TP({tp}) > mesh_{tp_axis}_size({tp_dim_size})")

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
        """General tensor parallel allreduce (no hardcoded hacks)"""
        if self.tp <= 1:
            return tensor

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
            num_links=1,
            memory_config=memory_config,
            topology=ccl_manager.topology,
            cluster_axis=axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

        # All-gather back
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            dim=3,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=1,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

        # Remove padding if applied
        if padded:
            gathered_sliced = gathered[:, :, :, :-pad_size]
            gathered.deallocate(True)
            gathered = gathered_sliced

        return gathered

    def __repr__(self):
        return f"MeshConfig({self.mesh_shape}, TP={self.tp}@axis{self.tp_axis}, DP={self.dp}, EP={self.ep}@axis{self.ep_axis})"


# Convenience factory functions for common configurations
def mesh_2x4():
    return MeshConfig((2, 4), tp=4)  # (2,4) TP=4, DP=2


def mesh_4x8():
    return MeshConfig((4, 8), tp=8, ep=4)  # (4,8) TP=8, DP=4


def mesh_4x4():
    return MeshConfig((4, 4), tp=4)  # (4,4) TP=4, DP=4


def mesh_1x8():
    return MeshConfig((1, 8), tp=8)  # (1,8) TP=8, DP=1
