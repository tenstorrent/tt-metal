# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MeshConfig — prefill mesh parallelisation for Llama-3.1-8B-Instruct.

TP shards features on one mesh axis (default cols); the other axis (rows) carries SP prefill
(SP = #rows). Both are derived from ``mesh_shape`` + ``tp_axis``, so **TP is the only knob**.
Target: 4x8 Blackhole Galaxy, TP=8 (32 Q / 8 KV heads across 8 cols), SP=4 (sequence across 4 rows)
— ``DEC-002``.

Nothing here is model-specific.

This file is the **union** of the two in-repo copies (``DEC-019``, member-by-member table in
``bringup_log/04_CCL_PLAN.md`` §3), because neither is a superset:

* base + strict ``_validate`` + the ``sp`` property: ``models/demos/gpt_oss_d_p/tt/config.py:19``
* ``reduce_scatter``: ``models/demos/minimax_m3/config.py:155`` (gpt-oss has none)
* the DRAM-lifetime comment inside ``allreduce``: ``models/demos/minimax_m3/config.py:104-111``

Deleted vs the gpt-oss copy: ``ep_axis``. Llama has no MoE, so an expert-parallel alias would be
dead vocabulary (``00_MODEL_CARD.md`` §3); ``sp_axis`` is the only non-TP axis name — ``DEC-022``.
"""

from loguru import logger

import ttnn

# The single configuration targeted on hardware (Blackhole Galaxy): (4,8), TP=8 -> SP=4. DEC-002.
_VALIDATED_MESH_SHAPE = (4, 8)
_VALIDATED_TP = 8


class MeshConfig:
    """Prefill mesh parallelisation. TP is the only knob; SP follows from the mesh shape."""

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
        # across all `tp_dim_size` devices, giving inconsistent per-device shapes. Require equality
        # until sub-axis (subgroup) mapping is implemented.
        if self.tp != tp_dim_size:
            raise ValueError(
                f"TP({self.tp}) must equal mesh_{self.tp_axis}_size({tp_dim_size}); "
                f"sub-axis TP is unsupported (shard_mapper shards the full axis)."
            )
        if (self.mesh_shape, self.tp) != (_VALIDATED_MESH_SHAPE, _VALIDATED_TP):
            logger.warning(
                f"MeshConfig(mesh_shape={self.mesh_shape}, tp={self.tp}) is untested; only "
                f"mesh_shape={_VALIDATED_MESH_SHAPE}, tp={_VALIDATED_TP} (SP=4) is the Llama-3.1-8B target."
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
        """Column-parallel weights (feature dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Row-parallel weights (sequence/batch dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def sequence_parallel(self, mesh_device):
        """Sequence sharding (for the KV cache and the whole-cache cos/sin)."""
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

    def reduce_scatter(self, tensor, ccl_manager, dim=3, axis=0, memory_config=None):
        """Reduce-scatter along mesh `axis`: sum across the devices on that axis, scattering the result
        on tensor `dim`. (allreduce = reduce_scatter + all_gather; this exposes the scatter half alone.)

        Note: Caller should check if communication is needed before calling
        """
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

    def __repr__(self):
        return f"MeshConfig({self.mesh_shape}, tp={self.tp}, sp={self.sp}, tp_axis={self.tp_axis})"
