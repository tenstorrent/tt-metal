# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parallelism configuration for Gemma4 prefill on one Galaxy."""

import ttnn

GALAXY_MESH_SHAPES = ((8, 4), (4, 8))


def validate_galaxy_mesh(mesh_shape):
    """Require a full 32-device Galaxy with CP rows and TP columns."""
    if tuple(mesh_shape) not in GALAXY_MESH_SHAPES:
        raise ValueError(f"Gemma4 P/D requires a Galaxy mesh {GALAXY_MESH_SHAPES}, got {tuple(mesh_shape)}")


class MeshConfig:
    """Bind an open device mesh to CP rows and TP columns; the caller owns its lifetime."""

    def __init__(self, mesh_device):
        validate_galaxy_mesh(mesh_device.shape)
        self.device = mesh_device
        self.tp_axis = 1
        self.cp_axis = 0
        self.total_devices = 32

    @property
    def mesh_shape(self):
        return tuple(self.device.shape)

    @property
    def cp_degree(self):
        """Number of context-parallel ranks along the CP mesh axis."""
        return self.mesh_shape[self.cp_axis]

    @property
    def tp_degree(self):
        """Number of tensor-parallel ranks along the TP mesh axis."""
        return self.mesh_shape[self.tp_axis]

    def shard_mapper(self, tensor_dim=None, mesh_dims=None):
        if mesh_dims is None:
            mesh_dims = [None, None]
            mesh_dims[self.tp_axis] = tensor_dim
        return ttnn.ShardTensor2dMesh(self.device, self.mesh_shape, dims=mesh_dims)

    def column_parallel(self):
        return self.shard_mapper(tensor_dim=-1)

    def row_parallel(self):
        return self.shard_mapper(tensor_dim=-2)

    def __repr__(self):
        return f"MeshConfig({self.mesh_shape}, CP={self.cp_degree}, TP={self.tp_degree})"
