# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh parallelization for Qwen3.8-27B prefill.

TP shards features along one mesh axis (the columns by default); the other axis carries
sequence-parallel prefill (SP = the number of rows). The spec's graded shape is ``(8, 4)``,
SP=8 x TP=4 on a Blackhole Galaxy — but **nothing here is fixed to it**. Every SP/TP-dependent
quantity is derived from the mesh actually opened, so the same code runs the qb/lb-shaped coverage
pairs the recipe asks for (recipe section 4, "support multiple mesh topologies by default").
"""

from __future__ import annotations

from loguru import logger

import ttnn

from .spec import load_spec


class MeshConfig:
    """Prefill mesh parallelization. TP is the only knob; SP follows from the mesh shape."""

    def __init__(self, mesh_shape: tuple[int, int], tp: int, tp_axis: int = 1) -> None:
        self.mesh_shape: tuple[int, int] = tuple(mesh_shape)
        self.tp = tp
        self.tp_axis = tp_axis
        self.sp_axis = 1 - tp_axis
        self.total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        self._validate()

    @property
    def sp(self) -> int:
        return self.mesh_shape[self.sp_axis]

    def _validate(self) -> None:
        tp_dim = self.mesh_shape[self.tp_axis]
        if self.tp != tp_dim:
            raise ValueError(
                f"tp={self.tp} must equal the mesh's tp-axis size ({tp_dim}); a partial TP split "
                f"would leave idle columns rather than a smaller model"
            )
        graded = load_spec()
        if (self.mesh_shape, self.tp) != (graded.mesh_shape, graded.tp):
            logger.warning(
                f"MeshConfig(mesh_shape={self.mesh_shape}, tp={self.tp}) is a COVERAGE shape; the "
                f"graded configuration is mesh_shape={graded.mesh_shape}, tp={graded.tp} (sp={graded.sp})."
            )

    # --- mesh mappers -------------------------------------------------------------------
    def shard_mapper(self, mesh_device, tensor_dim: int | None = None, mesh_dims: tuple | None = None):
        if mesh_dims is None:
            mesh_dims = (None, tensor_dim) if self.tp_axis == 1 else (tensor_dim, None)
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=mesh_dims)

    def column_parallel(self, mesh_device):
        """Shard a weight's OUTPUT (feature) dim across TP — gate/up/qkv projections."""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Shard a weight's INPUT dim across TP — down/o projections; needs a closing TP reduction."""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def replicate(self, mesh_device):
        return ttnn.ReplicateTensorToMesh(mesh_device)

    def sequence_parallel(self, mesh_device, seq_dim: int = -2):
        """Shard an activation's sequence dim across the SP rows, replicated across the TP cols."""
        dims: list[int | None] = [None, None]
        dims[self.sp_axis] = seq_dim
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=tuple(dims))

    def shard_size(self, total: int) -> int:
        assert total % self.tp == 0, f"{total} does not split evenly across tp={self.tp}"
        return total // self.tp

    # --- collectives --------------------------------------------------------------------
    def allgather(self, tensor, ccl_manager, *, axis: int, dim: int = 3, memory_config=None):
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

    def reduce_scatter(self, tensor, ccl_manager, *, axis: int, dim: int = 3, memory_config=None):
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

    def allreduce(self, tensor, ccl_manager, *, axis: int, dim: int = 3, memory_config=None, fp32_reduce=True):
        """reduce-scatter + all-gather. The input is freed between the two halves: holding it
        alongside both the scattered and the gathered tensor is what fragments DRAM at long ISL.

        ``fp32_reduce`` widens the **reduction** to fp32 and rounds back to the input dtype after.
        This is a fidelity choice of the same kind as HiFi4 on the matmuls, not a change to the
        spec's ``bfloat16`` activations: what it widens is the sum of ``tp`` row-parallel partial
        products, which is where a deep residual stream loses the most. It doubles the collective's
        bytes on the wire; measured, and kept, because 64 layers of bf16 partial sums is a
        measurable share of this model's depth drift (see the README PCC table).
        """
        source_dtype = tensor.dtype
        widen = fp32_reduce and source_dtype != ttnn.float32
        if widen:
            wide = ttnn.typecast(tensor, ttnn.float32)
            tensor.deallocate(True)
            tensor = wide
        scattered = self.reduce_scatter(tensor, ccl_manager, axis=axis, dim=dim, memory_config=memory_config)
        tensor.deallocate(True)
        gathered = self.allgather(scattered, ccl_manager, axis=axis, dim=dim, memory_config=memory_config)
        scattered.deallocate(True)
        if widen:
            narrow = ttnn.typecast(gathered, source_dtype)
            gathered.deallocate(True)
            gathered = narrow
        return gathered

    def __repr__(self) -> str:
        return f"MeshConfig({self.mesh_shape}, tp={self.tp}, sp={self.sp}, tp_axis={self.tp_axis})"
