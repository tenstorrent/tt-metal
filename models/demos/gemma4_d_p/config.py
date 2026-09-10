# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parallelism configuration for Gemma4 prefill on one Galaxy."""

from dataclasses import dataclass

import ttnn

GALAXY_MESH_SHAPES = ((8, 4), (4, 8))


def validate_galaxy_mesh(mesh_shape):
    """Require a full 32-device Galaxy with CP rows and TP columns."""
    if tuple(mesh_shape) not in GALAXY_MESH_SHAPES:
        raise ValueError(f"Gemma4 P/D requires a Galaxy mesh {GALAXY_MESH_SHAPES}, got {tuple(mesh_shape)}")


@dataclass(frozen=True)
class ModeConfig:
    """Prefill parallelism across Galaxy rows and columns."""

    tp: int
    sp: int
    ep: int = 1


class MeshConfig:
    """Use all Galaxy rows for CP and columns for TP."""

    def __init__(self, mesh_shape, prefill=None):
        validate_galaxy_mesh(mesh_shape)
        self.mesh_shape = tuple(mesh_shape)
        self.tp_axis = 1
        self.sp_axis = 0
        self.ep_axis = 0
        self.total_devices = 32
        self.tp = mesh_shape[1]
        self.prefill = prefill or ModeConfig(tp=self.tp, sp=mesh_shape[0])
        if self.prefill != ModeConfig(tp=self.tp, sp=mesh_shape[0]):
            raise ValueError("Prefill must use all Galaxy rows for CP and columns for TP")

    def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None):
        return ttnn.ShardTensor2dMesh(
            mesh_device, mesh_device.shape, dims=mesh_dims if mesh_dims is not None else (None, tensor_dim)
        )

    def column_parallel(self, mesh_device):
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def __repr__(self):
        return f"MeshConfig({self.mesh_shape}, CP={self.prefill.sp}, TP={self.tp})"
