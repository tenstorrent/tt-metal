# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched decode layer layout (one 32-token tile per device; gpt-oss-20b at batch 32 on Blackhole 1x4, the envelope
of fused_decode.config.dense_moe_decode_supported).

The residual stream stays in L1 as BF16, width-sharded over RESIDUAL_CORES cores ([32, hidden / cores] shards), so the
two TP all-reduces of a layer are single ttnn.experimental.all_reduce_async ops and both RMSNorms run on the same shards
(sharded multi-core rms_norm) with no reshard in between. This is the layout the fused decode layer used for its one
token in row 0 of the same tile (fused_decode/, before its boundary op); with 32 users every row is a token.
"""

import ttnn

from .fused_decode.config import residual_memory_config, width_sharded_memory_config

# Fabric links per all-reduce (the measured setting of the fused decode layer's all_reduce_async).
DECODE_CCL_LINKS = 2


def sharded_norm_program_config(memory_config):
    """LayerNormShardedMultiCoreProgramConfig for a width-sharded [32, W] activation."""
    shard_spec = memory_config.shard_spec
    grid = shard_spec.grid.bounding_box().grid_size()
    block_w = shard_spec.shape[1] // ttnn.TILE_SIZE
    subblock_w = max(d for d in range(1, 5) if block_w % d == 0)
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid,
        subblock_w=subblock_w,
        block_h=shard_spec.shape[0] // ttnn.TILE_SIZE,
        block_w=block_w,
        inplace=False,
    )


class DecodeAllReduce:
    """One-op all-reduce of a [1, 1, 32, hidden] per-device partial sum across the TP axis.

    One persistent scratch buffer and global semaphore per call site ("attn", "moe"), shared by every layer: within a
    layer the two all-reduces alternate, and each is an all-rank synchronization, so no device reaches layer i+1's use of
    a slot before every device has finished layer i's."""

    def __init__(self, mesh_device, hidden_size, cluster_axis, topology=ttnn.Topology.Ring):
        self.mesh_device = mesh_device
        self.cluster_axis = cluster_axis
        self.topology = topology
        num_devices = mesh_device.shape[cluster_axis]
        self.memory_config = residual_memory_config(mesh_device, hidden_size)
        from .fused_decode.config import RESIDUAL_CORES

        scratch_memory_config = width_sharded_memory_config(mesh_device, hidden_size * num_devices, RESIDUAL_CORES)
        grid = mesh_device.compute_with_storage_grid_size()
        semaphore_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.slots = {
            name: (
                ttnn.empty(
                    [1, 1, 32, hidden_size * num_devices],
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=scratch_memory_config,
                ),
                ttnn.create_global_semaphore(mesh_device, semaphore_cores, 0),
            )
            for name in ("attn", "moe")
        }

    def __call__(self, partial, slot):
        """partial: [1, 1, 32, hidden] in any memory layout. Returns the BF16 sum in self.memory_config."""
        if partial.memory_config() != self.memory_config or partial.dtype != ttnn.bfloat8_b:
            moved = ttnn.to_memory_config(partial, self.memory_config, dtype=ttnn.bfloat8_b)
            partial.deallocate(True)
            partial = moved
        scratch, semaphore = self.slots[slot]
        out = ttnn.experimental.all_reduce_async(
            partial,
            scratch,
            cluster_axis=self.cluster_axis,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=semaphore,
            memory_config=self.memory_config,
            dtype=ttnn.bfloat16,
            topology=self.topology,
            num_links=DECODE_CCL_LINKS,
        )
        partial.deallocate(True)
        return out


def dram_width_sharded_memory_config(mesh_device, k, n):
    """DRAM width-sharded [k, n] weight: one [k, n / banks] shard per DRAM bank (tile-padded)."""
    banks = mesh_device.dram_grid_size().x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    shard_n = -(-n // (ttnn.TILE_SIZE * banks)) * ttnn.TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, shard_n), ttnn.ShardOrientation.ROW_MAJOR),
    )


class BatchedQKV:
    """QKV projection of the batched decode as a DRAM-sharded matmul (the weight streams from all DRAM banks; ~13 us
    against ~43 us for the auto-configured interleaved matmul at [32, 2880] x [2880, 1280]). The weight is resharded
    from the interleaved decode weight once, on device. The output is resharded to one head_dim-wide shard per core,
    the layout nlp_create_qkv_heads_decode reads."""

    CORES = 9  # divides K = 90 tiles; the measured best for this shape

    def __init__(self, mesh_device, wqkv, head_dim, tokens=32):
        k, n = wqkv.shape[-2], wqkv.shape[-1]
        tile = ttnn.TILE_SIZE
        grid = mesh_device.compute_with_storage_grid_size()
        self.weight = ttnn.to_memory_config(wqkv, dram_width_sharded_memory_config(mesh_device, k, n))
        cores = ttnn.num_cores_to_corerangeset(self.CORES, grid, row_wise=True)
        per_core_n = -(-n // (tile * self.CORES))
        self.in0_memory_config = ttnn.create_sharded_memory_config(
            (tokens, k // self.CORES),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.out_memory_config = ttnn.create_sharded_memory_config(
            (tokens, per_core_n * tile),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.heads_memory_config = ttnn.create_sharded_memory_config(
            (tokens, head_dim),
            ttnn.num_cores_to_corerangeset(n // head_dim, grid, row_wise=True),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_tiles = k // tile // self.CORES
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=max(d for d in range(1, min(k_tiles, 16) + 1) if k_tiles % d == 0),
            per_core_M=tokens // tile,
            per_core_N=per_core_n,
            fused_activation=None,
        )
        # The compute config the auto-configured matmul picks for BF16 x BF8 (an explicit program config would
        # otherwise default to LoFi).
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def __call__(self, x):
        x_sharded = ttnn.to_memory_config(x, self.in0_memory_config)
        out = ttnn.matmul(
            x_sharded,
            self.weight,
            program_config=self.program_config,
            memory_config=self.out_memory_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        x_sharded.deallocate(True)
        heads = ttnn.to_memory_config(out, self.heads_memory_config)
        out.deallocate(True)
        return heads


class BatchedOProj:
    """o_proj of the batched decode as a DRAM-sharded matmul on the unpadded weight (the decode weight carries 64 zero
    columns of CCL tile padding, which the one-op all-reduce does not need), BF8 out with the bias added, i.e. the
    all-reduce's input precision, so no typecast / slice follows. Weight and bias are cut from the decode tensors once,
    on device."""

    CORES = 8  # divides K = 32 tiles; the measured best for [32, 1024] x [1024, 2880]
    WORKERS = 2

    def __init__(self, mesh_device, o_proj, o_proj_bias, hidden, tokens=32):
        k = o_proj.shape[-2]
        tile = ttnn.TILE_SIZE
        grid = mesh_device.compute_with_storage_grid_size()
        w_shape = list(o_proj.shape)
        sliced = ttnn.slice(o_proj, [0] * len(w_shape), w_shape[:-1] + [hidden])
        self.weight = ttnn.to_memory_config(sliced, dram_width_sharded_memory_config(mesh_device, k, hidden))
        sliced.deallocate(True)
        b_shape = list(o_proj_bias.shape)
        self.bias = ttnn.slice(o_proj_bias, [0] * len(b_shape), b_shape[:-1] + [hidden])
        cores = ttnn.num_cores_to_corerangeset(self.CORES, grid, row_wise=True)
        per_core_n = -(-hidden // (tile * self.CORES))
        self.in0_memory_config = ttnn.create_sharded_memory_config(
            (tokens, k // self.CORES),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.out_memory_config = ttnn.create_sharded_memory_config(
            (tokens, per_core_n * tile),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_tiles = k // tile // self.CORES
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=k_tiles,
            per_core_M=tokens // tile,
            per_core_N=per_core_n,
            fused_activation=None,
            num_workers_per_dram_bank=self.WORKERS,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def __call__(self, heads):
        x = ttnn.to_memory_config(heads, self.in0_memory_config)
        out = ttnn.matmul(
            x,
            self.weight,
            program_config=self.program_config,
            memory_config=self.out_memory_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        x.deallocate(True)
        biased = ttnn.add(out, self.bias, memory_config=self.out_memory_config, dtype=ttnn.bfloat8_b)
        out.deallocate(True)
        return biased
