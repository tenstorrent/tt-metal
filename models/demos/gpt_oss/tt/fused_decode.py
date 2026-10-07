# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layouts and fused collectives shared by the fused decode layer (one token per user, TP across mesh columns).

Decode keeps the residual stream in L1, width-sharded over RESIDUAL_CORES cores ([32, hidden / cores] shards), so
that the two decoder collectives are single fused all-reduces (ttnn.experimental.all_reduce_async) and the two
RMSNorms run on the same shards (sharded multi-core rms_norm) with no reshard in between.
"""

import torch

import ttnn

# Largest core grid the fused decode uses (down projection 10x9, router/norm/all-reduce grids fit inside it).
FUSED_DECODE_MIN_GRID = (10, 9)


def fused_decode_layout_supported(
    is_blackhole, mesh_shape, tp, ep, num_experts, use_throughput_experts, tokens, grid=(11, 10)
):
    """Host-only form of fused_decode_supported (no device): the validated envelope of the fused decode layer
    (grid: the device compute-with-storage grid, x by y).

    The fused layer serves one token per device with TP over a single mesh row and no expert parallelism. Its
    core grids (residual / all-reduce 30 cores, QKV 40, o_proj 30, packed gate|up 48, down 90) are sized for the
    gpt-oss-20b shapes on the 11x10 Blackhole grid at TP=4, and its decode-only expert copies (+3.2 GB per device
    for 32 experts) were budgeted for that model. Every other layout keeps the original decode path."""
    return (
        is_blackhole
        and tuple(mesh_shape) == (1, 4)
        and tp == 4
        and ep == 1
        and num_experts == 32
        and not use_throughput_experts
        and tokens == 1
        and grid[0] >= FUSED_DECODE_MIN_GRID[0]
        and grid[1] >= FUSED_DECODE_MIN_GRID[1]
    )


def fused_decode_supported(mesh_device, mesh_config, hf_config, use_throughput_experts, tokens_per_device):
    """Whether the decoder layers of this model use the fused decode path (see fused_decode_layout_supported)."""
    if mesh_config is None:
        return False
    grid = mesh_device.compute_with_storage_grid_size()
    return fused_decode_layout_supported(
        ttnn.device.is_blackhole(mesh_device),
        tuple(mesh_device.shape),
        mesh_config.decode.tp,
        mesh_config.decode.ep,
        hf_config.num_local_experts,
        use_throughput_experts,
        tokens_per_device,
        (grid.x, grid.y),
    )


# 2880 = 90 tiles: 30 cores x 3 tiles. The fused all-reduce over 30 cores measured 8.4 us (BF8 in, BF16 out,
# 2 links) against 41 us for ttnn.all_reduce on the 1x4 Blackhole ring (probes/bench_allreduce.py).
RESIDUAL_CORES = 30
DECODE_CCL_LINKS = 2


def width_sharded_memory_config(mesh_device, width, cores, rows=32):
    """L1 width-sharded [rows, width / cores] shards on a rectangular grid (rectangular grids keep the sharded
    RMSNorm multicast contract simple)."""
    grid = mesh_device.compute_with_storage_grid_size()
    widths = [x for x in range(1, grid.x + 1) if cores % x == 0 and cores // x <= grid.y]
    assert widths, f"no rectangular grid of {cores} cores fits {grid}"
    gx = max(widths)
    assert width % (cores * ttnn.TILE_SIZE) == 0, f"width {width} does not split into {cores} tile-aligned shards"
    return ttnn.create_sharded_memory_config(
        (rows, width // cores),
        ttnn.CoreGrid(x=gx, y=cores // gx),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def residual_memory_config(mesh_device, hidden_size):
    return width_sharded_memory_config(mesh_device, hidden_size, RESIDUAL_CORES)


def sharded_norm_program_config(memory_config):
    """LayerNormShardedMultiCoreProgramConfig matching a width-sharded [32, W] activation."""
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
    """Fused single-op all-reduce of a [1, 1, 32, hidden] per-device partial sum across the TP axis.

    One persistent scratch buffer and global semaphore per call site ("attn", "moe"), shared by every layer:
    within a layer the attention and MoE all-reduces alternate, and each is an all-rank synchronization, so a
    device cannot reach layer i+1's use of a slot before every device has finished layer i's use of it.
    """

    def __init__(self, mesh_device, hidden_size, cluster_axis, topology=ttnn.Topology.Ring):
        self.mesh_device = mesh_device
        self.cluster_axis = cluster_axis
        self.topology = topology
        self.num_devices = mesh_device.shape[cluster_axis]
        self.memory_config = residual_memory_config(mesh_device, hidden_size)
        buffer_memory_config = width_sharded_memory_config(mesh_device, hidden_size * self.num_devices, RESIDUAL_CORES)
        grid = mesh_device.compute_with_storage_grid_size()
        semaphore_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.slots = {
            name: (
                ttnn.empty(
                    [1, 1, 32, hidden_size * self.num_devices],
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=buffer_memory_config,
                ),
                ttnn.create_global_semaphore(mesh_device, semaphore_cores, 0),
            )
            for name in ("attn", "moe")
        }

    def __call__(self, partial, slot):
        """partial: BF8 [1, 1, 32, hidden] in self.memory_config. Returns the BF16 sum in the same layout."""
        if partial.memory_config() != self.memory_config:
            partial = ttnn.to_memory_config(partial, self.memory_config)
        if partial.dtype != ttnn.bfloat8_b:
            partial = ttnn.typecast(partial, ttnn.bfloat8_b)
        scratch, semaphore = self.slots[slot]
        return ttnn.experimental.all_reduce_async(
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


def matmul_1d_program_config(output_memory_config, k, in0_block_w, grid=None):
    """1D multicast-in0 matmul config for a [32, K] activation (L1 interleaved) whose [32, N] output is written
    straight into `output_memory_config` (width-sharded, one rectangular grid). Fused bias/output layout instead of
    a separate bias add or reshard; `grid` overrides the compute grid for interleaved outputs."""
    k_tiles = k // ttnn.TILE_SIZE
    in0_block_w = max(d for d in range(1, in0_block_w + 1) if k_tiles % d == 0)
    if grid is None:
        shard_spec = output_memory_config.shard_spec
        grid = shard_spec.grid.bounding_box().grid_size()
        per_core_N = shard_spec.shape[1] // ttnn.TILE_SIZE
    else:
        grid, per_core_N = ttnn.CoreCoord(*grid[:2]), grid[2]
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_N,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def auto_matmul_compute_config(mesh_device, operands_low_precision):
    """The compute config ttnn.matmul picks when called without a program config: HiFi2 unless both operands are
    BFP8/BFP4 (then LoFi), no approximation, no FP32 accumulation, packer L1 accumulation. An explicit program
    config makes ttnn default to LoFi instead, so decode matmuls given program configs pass this to keep the
    fidelity of the unfused graph."""
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi if operands_low_precision else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def create_decode_gate_buffers(mesh_device, num_experts, width, tokens):
    """Constants and in-place output buffers of the decode router gate (ttnn.experimental.deepseek.moe.
    generalized_moe_gate): L1, one [32, 32] tile per token on one core each. One set serves every layer (owned by
    the CCL manager): the constants are identical across layers, and each layer copies its top-k out of the output
    buffers right after its gate, before the next layer's gate overwrites them. L1 buffers reserve their address
    range in every bank, so per-layer copies would hold ~200 KB of L1 per core for the whole model."""
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if isinstance(mesh_device, ttnn.MeshDevice) else None
    cores = ttnn.num_cores_to_corerangeset(tokens, mesh_device.compute_with_storage_grid_size(), row_wise=True)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )

    def to_device(t, dtype, layout):
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=mesh_device, memory_config=memory_config, mesh_mapper=replicate
        )

    def face(t, dtype):
        """[width] -> the gate's per-token [16, 16] face, transposed within the face."""
        return to_device(
            t.reshape(1, 16, 16).transpose(1, 2).contiguous().repeat(tokens, 1, 1), dtype, ttnn.TILE_LAYOUT
        )

    selection_bias = torch.full((width,), -1e9)
    selection_bias[:num_experts] = 0.0
    buffers = {
        "memory_config": memory_config,
        "selection_bias": face(selection_bias, ttnn.bfloat16),
        "expert_ids": face(torch.arange(width, dtype=torch.int32), ttnn.uint16),
        # Row-major so the top-k slices after the gate are plain row reads.
        "scores": to_device(torch.zeros(tokens, 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        "indices": to_device(torch.zeros(tokens, 32, 32, dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
    }
    return buffers
