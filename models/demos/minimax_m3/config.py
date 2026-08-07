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

    def reduce_scatter(
        self, tensor, ccl_manager, dim=3, axis=0, memory_config=None, subdevice_id=None, overlapped=False
    ):
        """Reduce-scatter along mesh `axis`: sum across the devices on that axis, scattering the result
        on tensor `dim`. (allreduce = reduce_scatter + all_gather; this exposes the scatter half alone.)

        ``subdevice_id`` / ``overlapped`` are for the shared-expert||dispatch overlap, where this op is
        enqueued on one sub-device while dispatch runs concurrently on another. Both must be set there,
        and neither may be set elsewhere:

          * ``subdevice_id`` confines the op's workers to that sub-device.
          * ``overlapped=True`` switches to an OWNED, stable-address intermediate accumulator
            (``ccl_manager.get_shared_rs_intermediate``) and pins this call's INPUT alive until the
            next overlapped call (``set_shared_rs_input_keepalive``).

        The second part is not an optimization, it is a correctness requirement. This op is async: its
        kernels keep touching its DRAM after the Python call returns, while the concurrent dispatch
        allocates its own large buffers. Any buffer freed before those kernels finish can be re-handed
        to dispatch and overwritten mid-flight. DeepSeek observed exactly that — both the RS input and
        its intermediate re-handed to dispatch's `metadata` / `dispatched_buffer` at the SAME DRAM
        address, the input alias giving a catastrophic period-2 failure and the intermediate alias
        nondeterministic PCC (models/demos/deepseek_v3_d_p/tt/moe/tt_shared_expert.py:473-507). Owning
        the intermediate at a fixed address additionally makes the fabric reduction order identical
        every iteration, i.e. bit-exact determinism.

        Note: Caller should check if communication is needed before calling
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        if overlapped:
            assert subdevice_id is not None, "overlapped reduce_scatter needs the sub-device it runs on"
            intermediate = ccl_manager.get_shared_rs_intermediate(tensor, ccl_manager.topology)
            out = ttnn.experimental.reduce_scatter_minimal_async(
                tensor,
                persistent_output_buffers=[intermediate],
                dim=dim,
                multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
                barrier_semaphore=ccl_manager.get_barrier_semaphore(),
                num_links=ccl_manager.num_links,
                memory_config=memory_config,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ccl_manager.topology,
                cluster_axis=axis,
                subdevice_id=subdevice_id,
            )
            # Hold the (fresh-per-iteration) input until the next overlapped call, so the concurrent
            # dispatch cannot reuse its slot mid-flight. One slot on the CCL manager, not per-layer.
            ccl_manager.set_shared_rs_input_keepalive(tensor)
            return out

        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=dim,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            topology=ccl_manager.topology,
            cluster_axis=axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            **({"subdevice_id": subdevice_id} if subdevice_id is not None else {}),
        )

    def __repr__(self):
        sp = self.mesh_shape[self.sp_axis]
        return f"MeshConfig({self.mesh_shape}, tp={self.tp}, sp={sp}, tp_axis={self.tp_axis})"
