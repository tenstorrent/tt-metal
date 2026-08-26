# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MeshConfig — prefill mesh parallelization for Mistral-Medium-3.5-128B.

TP shards features along ``tp_axis`` (default cols); the other axis carries sequence-parallel
prefill (SP = size of that axis). Those follow from the mesh shape, so **TP is the only knob**.

**Target: (8,4) Blackhole Galaxy, TP=4 (cols), SP=8 (rows)** — the same split every other Galaxy
prefill model in the repo runs (deepseek_v3/v3.2, kimi_k2_6/k2_7/k3, glm_5_1/5_2, minimax_m3).

Why TP=4 and not TP=8 (which ``num_key_value_heads == 8`` would also allow): the TP all-reduce
volume per chip is

    2·(T−1)/T · s_loc·H·2   with   s_loc = S/sp = S·T/32     ->    (S·H·4/32)·(T−1)

i.e. **linear in (T−1)** — the ``1/T`` ring-efficiency gain is exactly cancelled by the per-chip
sequence shard growing with T. At 128K over 88 layers that is 231 GB/chip at TP=8 versus 99 GB/chip
at TP=4. The countervailing term, the ring-SDPA KV gather, grows as TP shrinks but is only
2.1 -> 4.8 GB because GQA has just 8 KV heads. Net: TP=4 moves 2.25x less fabric traffic.

Per-chip shapes at TP=4 (all tile-aligned; no padding anywhere):

    hidden 12288/4 = 3072       ffn 28672/4 = 7168        Q heads 96/4 = 24
    KV heads 8/4 = 2            fused QKV (24+2+2)*128 = 3584
    o_proj K 24*128 = 3072      fused gate|up 2*7168 = 14336

Note ``n_kv_local = 2``: every other GQA model in the repo lands on exactly one KV head per chip
(minimax_m3 4/4, gpt_oss 8/8). Two is legal — ``update_padded_kv_cache`` only requires cache and
input head dims to match, and ring-joint SDPA supports grouped GQA ``NKH == NVH < NQH`` — but it is
unexercised, so ``tests/unit/test_ring_joint_sp_vs_ref.py`` drives the op at 2/2/24 to pin it.
"""

from loguru import logger

import ttnn

# The single configuration targeted on hardware (Blackhole Galaxy): (8,4), TP=4 -> SP=8.
_VALIDATED_MESH_SHAPE = (8, 4)
_VALIDATED_TP = 4


class MeshConfig:
    """Prefill mesh parallelization. TP is the only knob; SP follows from the mesh shape."""

    def __init__(self, mesh_shape, tp, tp_axis: int = 1, tp_slice: int = 0):
        """
        Args:
            mesh_shape: (rows, cols) - any mesh size
            tp: tensor-parallel DEGREE - the number of feature shards. Drives every per-device
                shape (`shard_size`, head counts, `hidden/tp`), independent of how many devices
                actually sit on tp_axis.
            tp_axis: which mesh axis is TP (0=rows, 1=cols, default: 1). The other axis carries
                sequence-parallel prefill (SP = size of that axis).
            tp_slice: which of the `tp` shards to materialize when the mesh has a SINGLE device on
                tp_axis - see `is_tp_slice`.
        """
        self.mesh_shape = tuple(mesh_shape)
        self.tp = tp
        self.tp_axis = tp_axis
        self.sp_axis = 0 if tp_axis == 1 else 1
        self.tp_devices = self.mesh_shape[self.tp_axis]
        self.tp_slice = tp_slice
        self.total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        self._validate()

    def _validate(self):
        tp_dim_size = self.mesh_shape[self.tp_axis]
        # shard_mapper always shards across the ENTIRE tp_axis, so TP must span the whole axis: a
        # smaller TP would build per-device feature counts from `tp` while the mapper still split
        # across all `tp_dim_size` devices, giving inconsistent shapes.
        #
        # The ONE exception is the TP-slice rig (`tp_devices == 1 < tp`): with nothing to shard
        # across, `take_tp_slice` narrows the host tensor to a single shard and the mapper
        # replicates, so per-device shapes match a real TP=`tp` mesh. That is what lets a 4-chip
        # box drive the SP ring at Galaxy per-device sizes. See `is_tp_slice` for the caveats.
        if self.tp != tp_dim_size and not (tp_dim_size == 1 and self.tp > 1):
            raise ValueError(
                f"TP({self.tp}) must equal mesh_{self.tp_axis}_size({tp_dim_size}), or that axis "
                f"must be 1 device wide (the TP-slice rig); sub-axis TP is unsupported "
                f"(shard_mapper shards the full axis)."
            )
        if not 0 <= self.tp_slice < self.tp:
            raise ValueError(f"tp_slice({self.tp_slice}) out of range for tp({self.tp})")
        if (self.mesh_shape, self.tp) != (_VALIDATED_MESH_SHAPE, _VALIDATED_TP):
            logger.warning(
                f"MeshConfig(mesh_shape={self.mesh_shape}, tp={self.tp}) is untested; only "
                f"mesh_shape={_VALIDATED_MESH_SHAPE}, tp={_VALIDATED_TP} (SP=8) is the Mistral-Medium target."
            )

    @property
    def sp(self) -> int:
        """Sequence-parallel degree (size of the non-TP axis)."""
        return self.mesh_shape[self.sp_axis]

    @property
    def is_tp_slice(self) -> bool:
        """True when the mesh is ONE device wide on the TP axis but `tp` > 1.

        Per-device shapes then match a real TP=`tp` mesh while only shard `tp_slice` exists. Two
        consequences: the TP close is a no-op (nothing to reduce across), and any ROW-parallel
        output is a partial sum whose remaining `tp - 1` shards are simply absent - so a reference
        must be computed over the same slice. COLUMN-parallel outputs (per-head tensors, which is
        what the SP ring consumes) are exact.
        """
        return self.tp_devices == 1 and self.tp > 1

    def take_tp_slice(self, t, dim: int):
        """Narrow `t` to TP shard `tp_slice` along `dim`. No-op unless `is_tp_slice`.

        A contiguous narrow is exactly what sharding across the TP axis would have handed device
        `tp_slice`, so callers get Galaxy per-device shapes without a `tp`-wide axis.
        """
        if not self.is_tp_slice:
            return t
        n = t.shape[dim] // self.tp
        return t.narrow(dim, self.tp_slice * n, n)

    def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None):
        """2D mesh sharding. Default: shard `tensor_dim` along the TP axis, replicate along SP."""
        if mesh_dims is None:
            mesh_dims = (None, tensor_dim) if self.tp_axis == 1 else (tensor_dim, None)
        if self.is_tp_slice:
            # Nothing to shard on a 1-wide axis; the caller already narrowed via take_tp_slice.
            mesh_dims = tuple(None if i == self.tp_axis else d for i, d in enumerate(mesh_dims))
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=mesh_dims)

    def column_parallel(self, mesh_device):
        """Column-parallel weights: shard the OUTPUT/feature dim. Needs the full contraction dim."""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Row-parallel weights: shard the CONTRACTION dim. Output is a partial sum, needs reducing."""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def sequence_parallel(self, mesh_device):
        """Shard dim -3 (heads / sequence-major tensors)."""
        return self.shard_mapper(mesh_device, tensor_dim=-3)

    def shard_size(self, total_size):
        """Per-device size under TP sharding."""
        return total_size // self.tp

    def reduce_scatter(self, tensor, ccl_manager, dim=3, axis=None, memory_config=None):
        """Reduce-scatter along mesh `axis`: sum across that axis's devices, scattering on `dim`.

        This is how **every block in this model closes** (sharded residual): a row-parallel matmul
        emits a partial sum over the full emb, and the reduce-scatter both completes the sum and
        lands the result as ``emb/tp`` — the residual's layout. There is deliberately no trailing
        all-gather; the decoder layer gathers once in front of the next norm instead.

        Caller should check whether communication is needed (tp > 1) before calling.
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        axis = self.tp_axis if axis is None else axis
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

    def allgather(self, tensor, ccl_manager, memory_config=None, axis=None, dim=3, linear=False):
        """All-gather along mesh `axis` on tensor `dim`. Used to rebuild full emb in front of a norm."""
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        axis = self.tp_axis if axis is None else axis
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
        rig = ", TP-SLICE" if self.is_tp_slice else ""
        return (
            f"MeshConfig({self.mesh_shape}, tp={self.tp}, sp={self.sp}, tp_axis={self.tp_axis}, "
            f"tp_devices={self.tp_devices}{rig})"
        )
