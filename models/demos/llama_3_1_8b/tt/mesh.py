# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh parallelization for Llama-3.1-8B prefill: TP on the columns, SP on the rows.

The prefill spec binds ``tp=4`` and ``sp=8`` on a ``bh_galaxy``, i.e. an ``(8, 4)`` mesh with
``tp_axis=1``. TP shards features (the q/k/v/gate/up output dim, the o_proj/down_proj input dim, the
LM-head vocab dim); SP shards the sequence, so each mesh row holds ``chunk_size / 8`` tokens of the
residual stream and its slice of the KV cache.

Residual-stream layout is the **replicated** one: the hidden state is full ``hidden_size`` on every
TP column, so the two row-parallel matmuls (attention o_proj, MLP down_proj) close with a full
all-reduce. MiniMax-M3 also offers an ``emb/tp``-sharded residual that replaces each all-reduce with
a reduce-scatter; that is a perf lever (out of scope here) and it is deliberately NOT carried over —
it changes the norm sharding, the embedding tail and the attention/MLP epilogues all at once, which
is three ways to be subtly wrong for zero correctness gain.
"""

from __future__ import annotations

import ttnn

# The one configuration this spec binds and this package is measured at.
SPEC_MESH_SHAPE = (8, 4)
SPEC_TP = 4


class MeshConfig:
    """(rows, cols) mesh with TP on ``tp_axis``; SP is whatever the other axis is."""

    def __init__(self, mesh_shape, tp: int, tp_axis: int = 1):
        self.mesh_shape = tuple(mesh_shape)
        self.tp = tp
        self.tp_axis = tp_axis
        self.sp_axis = 1 - tp_axis
        self.total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        if self.tp > self.mesh_shape[self.tp_axis]:
            raise ValueError(f"tp({tp}) > mesh axis {tp_axis} size {self.mesh_shape[self.tp_axis]}")

    @property
    def sp(self) -> int:
        return self.mesh_shape[self.sp_axis]

    def shard_size(self, total: int) -> int:
        assert total % self.tp == 0, f"{total} is not divisible by tp={self.tp}"
        return total // self.tp

    # --- mesh mappers -------------------------------------------------------------------------
    def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None):
        if mesh_dims is None:
            mesh_dims = (None, tensor_dim) if self.tp_axis == 1 else (tensor_dim, None)
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=tuple(mesh_dims))

    def column_parallel(self, mesh_device):
        """Shard a ``[.., in, out]`` weight on ``out`` across TP (q/k/v, gate/up, lm_head)."""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Shard a ``[.., in, out]`` weight on ``in`` across TP (o_proj, down_proj)."""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def sequence_parallel(self, mesh_device, seq_dim: int = 2):
        """Shard an activation's sequence dim across the SP axis, replicated across TP."""
        dims = [None, None]
        dims[self.sp_axis] = seq_dim
        return self.shard_mapper(mesh_device, mesh_dims=dims)

    def replicate(self, mesh_device):
        return ttnn.ReplicateTensorToMesh(mesh_device)

    # --- collectives --------------------------------------------------------------------------
    def all_gather(self, tensor, ccl, *, axis=None, dim=3, memory_config=None):
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            cluster_axis=self.tp_axis if axis is None else axis,
            mesh_device=ccl.mesh_device,
            topology=ccl.topology,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            num_links=ccl.num_links,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )

    def reduce_scatter(self, tensor, ccl, *, dim=3, axis=None, memory_config=None):
        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=dim,
            multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
            num_links=ccl.num_links,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            topology=ccl.topology,
            cluster_axis=self.tp_axis if axis is None else axis,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )

    def all_reduce(self, tensor, ccl, *, axis=None, memory_config=None):
        """Reduce-scatter + all-gather on the last dim across ``axis`` (TP by default).

        The input handle is freed between the two halves: at chunk 5120 / hidden 4096 the residual is
        ~40 MiB and holding input + scattered + gathered live at once is what fragments DRAM on a
        long prefill (the same failure M3's ``allreduce`` comments record). Callers must treat
        ``tensor`` as dead after this returns.
        """
        if self.tp <= 1:
            return tensor
        scattered = self.reduce_scatter(tensor, ccl, dim=3, axis=axis, memory_config=memory_config)
        tensor.deallocate(True)
        gathered = self.all_gather(scattered, ccl, axis=axis, dim=3, memory_config=memory_config)
        scattered.deallocate(True)
        return gathered

    def __repr__(self):
        return f"MeshConfig({self.mesh_shape}, tp={self.tp}, sp={self.sp}, tp_axis={self.tp_axis})"
