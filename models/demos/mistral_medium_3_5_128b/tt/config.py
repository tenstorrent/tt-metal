# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MeshConfig — prefill mesh parallelization for Mistral-Medium-3.5-128B.

Structure ported from ``gpt_oss_d_p/tt/config.py`` (measured at 4x8 / TP=8); the target here is the
transpose: **8x4 Blackhole Galaxy, TP=4 (cols), SP=8 (rows)**, per the binding spec's
``parallelism: {tp: 4, sp: 8}``.

TP shards features on one axis (cols); the other axis (rows) carries sequence-parallel prefill
(SP = #rows). TP is the only knob — SP follows from the mesh shape. This model is dense, so unlike
the GPT-OSS source there is no EP axis; ``sp_axis`` is the whole story for the non-TP axis.

Per-chip shapes at the target (hidden 12288, 96 Q / 8 KV heads, head_dim 128, intermediate 28672):

    Q heads/chip   96 / 4 = 24
    KV heads/chip   8 / 4 = 2      <- note: NOT 1, unlike gpt_oss_d_p and minimax_m3
    hidden/chip 12288 / 4 = 3072
    inter/chip  28672 / 4 = 7168
    seq_local   10240 / 8 = 1280

The 2-KV-heads-per-chip point is the one place this model leaves both borrowed packages' measured
envelope (both have n_kv == tp, i.e. exactly one KV head per chip). It is supported by the ops we
rely on — ``update_padded_kv_cache`` only requires ``dim1 == 1`` when the cache is TP-deduped
(``tp_factor > 1``, which is not our call shape), and the ring joint SDPA explicitly handles grouped
GQA KV (``NKH == NVH < NQH && NQH % NKH == 0``) — but it is untested territory in this repo, so the
assertion lives in ``attention/kv_cache.py`` rather than being assumed here.
"""

from loguru import logger

import ttnn

# The single configuration targeted on hardware (Blackhole Galaxy): (8,4), TP=4 -> SP=8.
_VALIDATED_MESH_SHAPE = (8, 4)
_VALIDATED_TP = 4


class MeshConfig:
    """Prefill mesh parallelization. TP is the only knob; SP follows from the mesh shape."""

    def __init__(self, mesh_shape, tp, tp_axis: int = 1):
        """
        Args:
            mesh_shape: (rows, cols) - any mesh size
            tp: tensor-parallel size (shards features along tp_axis)
            tp_axis: which mesh axis is TP (0=rows, 1=cols, default: 1). The other axis
                carries sequence-parallel prefill (SP = size of that axis).
        """
        self.mesh_shape = tuple(mesh_shape)
        self.tp = tp
        self.tp_axis = tp_axis
        self.sp_axis = 0 if tp_axis == 1 else 1
        self.total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        self._validate()

    def _validate(self):
        tp_dim_size = self.mesh_shape[self.tp_axis]
        # shard_mapper always shards a tensor across the ENTIRE tp_axis, so TP must span the whole
        # axis. A smaller TP would build head/feature counts from `tp` while the mapper still splits
        # across all `tp_dim_size` devices, giving inconsistent per-device shapes.
        if self.tp != tp_dim_size:
            raise ValueError(
                f"TP({self.tp}) must equal mesh_{self.tp_axis}_size({tp_dim_size}); "
                f"sub-axis TP is unsupported (shard_mapper shards the full axis)."
            )
        if (self.mesh_shape, self.tp) != (_VALIDATED_MESH_SHAPE, _VALIDATED_TP):
            logger.warning(
                f"MeshConfig(mesh_shape={self.mesh_shape}, tp={self.tp}) is untested; only "
                f"mesh_shape={_VALIDATED_MESH_SHAPE}, tp={_VALIDATED_TP} (SP=8) is the "
                f"Mistral-Medium-3.5 target."
            )

    @property
    def sp(self) -> int:
        """Sequence-parallel degree (size of the non-TP axis)."""
        return self.mesh_shape[self.sp_axis]

    def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None):
        """Unified 2D sharding - replaces all individual mappers"""
        if mesh_dims is None:
            # Default: shard along TP axis only
            mesh_dims = (None, tensor_dim) if self.tp_axis == 1 else (tensor_dim, None)

        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=mesh_dims)

    # Clean semantic helpers (all use unified shard_mapper)
    def column_parallel(self, mesh_device):
        """Column-parallel weights (output/feature dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Row-parallel weights (input dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def shard_heads_tp(self, mesh_device):
        """Shard dim -3 (heads) across the TP axis. For [1, n_heads, seq, head_dim] tensors.

        This is what ``gpt_oss_d_p``'s confusingly-named ``sequence_parallel`` actually does — it
        shards heads on the TP axis, not sequence on the SP axis. Renamed here so the axis is
        readable at the call site.
        """
        return self.shard_mapper(mesh_device, tensor_dim=-3)

    def shard_activation(self, mesh_device, seq_dim=-2, feat_dim=-1):
        """The prefill activation layout: sequence sharded on the SP axis AND features on the TP
        axis, i.e. a genuine 2D shard of a [1, 1, seq, hidden] residual stream."""
        dims = [None, None]
        dims[self.sp_axis] = seq_dim
        dims[self.tp_axis] = feat_dim
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=tuple(dims))

    def shard_seq_sp(self, mesh_device, seq_dim=-2):
        """Sequence sharded on the SP axis only; replicated across TP."""
        dims = [None, None]
        dims[self.sp_axis] = seq_dim
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=tuple(dims))

    def shard_size(self, total_size):
        """Size per device for tensor parallel sharding"""
        return total_size // self.tp

    def reduce_scatter(self, tensor, ccl_manager, dim=3, axis=None, memory_config=None):
        """TP reduce-scatter (row-parallel matmul tail when the residual stream is sharded)."""
        axis = self.tp_axis if axis is None else axis
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=dim,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            topology=ccl_manager.topology,
            cluster_axis=axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

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
        # Free the full-size input before the all-gather allocates its full-size output, to keep peak
        # live DRAM bounded under long-context prefill. Callers must NOT use `tensor` after this
        # returns (apply_allreduce assigns the return value).
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
        All-gather operation for tensor parallel / sequence parallel communication

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
        return f"MeshConfig({self.mesh_shape}, tp={self.tp}, sp={self.sp}, tp_axis={self.tp_axis})"
