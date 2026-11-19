# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh plan and helpers shared by the Hy4-preview TTNN modules.

SC1 parallel plan (Blackhole Galaxy, tenstorrent/tt-metal#65884): one engine on all 32 chips, every
weight sharded once. #65884 draws the mesh as 4 rows x 8 columns; the Galaxy's fabric is 8x4 and the
plan runs on it as it is, the same mesh and axes prefill uses (#65935: sequence and heads on axis 0,
hidden on axis 1), so the issue's 4 rows are mesh axis 1 and its 8 columns are mesh axis 0:

* Hidden axis (``hidden_axis=1``, 4 chips): the residual streams' hidden dimension, and the input
  rank of every attention projection, are split over it and replicated across the other axis.
  Collectives that sum over the hidden split run on ``cluster_axis=hidden_axis``.
* Head axis (the other axis, mesh axis 0, 8 chips): attention heads, 8 per chip. A projection whose
  input is split over the hidden axis and whose output is split over the head axis puts 1/32 of its
  weight on each chip.

Chips are counted row-major over the 8x4 mesh, ``chip = head_coord * 4 + hidden_coord``: the order
ttnn.get_device_tensors returns them in, and the linear chip of the KV stripe prefill writes
(#65886: ``sp_coord * tp + tp_coord``), so tt/kv_layout.py addresses the same rows.
"""

import torch

import ttnn


def compute_config(fidelity=ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False):
    # WormholeComputeKernelConfig is the compute config type for Blackhole too.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


class MeshPlan:
    """Where the hidden dimension lives on the mesh.

    ``hidden_axis`` is the mesh axis the hidden dimension is split over (1 on SC1), or None when
    nothing is split (a single chip). ``num_links`` / ``topology`` left as None let each collective
    pick them from the active fabric.
    """

    def __init__(self, mesh_device, hidden_axis=1, num_links=None, topology=None):
        assert hidden_axis in (None, 0, 1), f"hidden_axis must be None, 0 or 1, got {hidden_axis}"
        self.mesh_device = mesh_device
        self.hidden_axis = hidden_axis
        self.mesh_shape = tuple(mesh_device.shape) if hidden_axis is not None else None
        self.head_axis = 1 - hidden_axis if hidden_axis is not None else None
        self.hidden_tp = self.mesh_shape[hidden_axis] if hidden_axis is not None else 1
        self.head_tp = self.mesh_shape[self.head_axis] if hidden_axis is not None else 1
        self.num_links = num_links
        self.topology = topology

    def _tp(self, axis):
        return 1 if axis is None else self.mesh_shape[axis]

    def shard(self, hidden_dim=None, head_dim=None):
        """Mesh mapper: shard tensor dim ``hidden_dim`` across the hidden axis and ``head_dim`` across
        the head axis; a dim left as None is replicated across its axis."""
        if self.hidden_axis is None or (hidden_dim is None and head_dim is None):
            return None  # ttnn.from_torch replicates
        dims = [None, None]
        dims[self.hidden_axis], dims[self.head_axis] = hidden_dim, head_dim
        return ttnn.ShardTensor2dMesh(self.mesh_device, dims=tuple(dims), mesh_shape=self.mesh_shape)

    def shard_hidden(self, dim):
        """Mesh mapper: shard tensor ``dim`` across the hidden axis, replicate across the other axis."""
        return self.shard(hidden_dim=dim)

    def all_reduce(self, x, axis, memory_config=None):
        """Sum ``x`` across mesh ``axis``, leaving the result on every chip along it. Frees ``x``."""
        if self._tp(axis) == 1:
            return x
        out = ttnn.all_reduce(
            x,
            cluster_axis=axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        x.deallocate(True)
        return out

    def all_reduce_hidden(self, x, memory_config=None):
        """Sum ``x`` across the hidden axis, leaving the result on every chip."""
        return self.all_reduce(x, self.hidden_axis, memory_config)

    def all_gather(self, x, dim, axis, memory_config=None):
        """Concatenate the shards of ``x`` along mesh ``axis`` on tensor ``dim``, on every chip along it."""
        if self._tp(axis) == 1:
            return x
        return ttnn.all_gather(x, dim=dim, cluster_axis=axis, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)

    def partition(self, x, dim, axis):
        """The inverse of all_gather, without the fabric: chip i along mesh ``axis`` keeps its i-th
        slice of ``x`` on tensor ``dim``."""
        if self._tp(axis) == 1:
            return x
        return ttnn.mesh_partition(x, dim, cluster_axis=axis)

    def to_torch_concat(self, x, dim, axis):
        """Host readback of a tensor sharded along mesh ``axis``: concat the shards on ``dim``, taken
        from replica 0 of the other axis."""
        shards = ttnn.get_device_tensors(x)
        tp = self._tp(axis)
        if tp == 1:
            return ttnn.to_torch(shards[0])
        cols = self.mesh_shape[1]
        # get_device_tensors is row-major over the mesh
        picks = [i * cols for i in range(tp)] if axis == 0 else range(tp)
        return torch.cat([ttnn.to_torch(shards[i]) for i in picks], dim=dim)

    def to_torch_hidden_concat(self, x, dim):
        return self.to_torch_concat(x, dim, self.hidden_axis)

    def to_torch_head_concat(self, x, dim):
        return self.to_torch_concat(x, dim, self.head_axis)

    def to_torch_replicated(self, x):
        return ttnn.to_torch(ttnn.get_device_tensors(x)[0])
