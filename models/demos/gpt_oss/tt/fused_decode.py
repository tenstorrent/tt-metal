# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layouts, decode policy and fused collectives of the fused decode layer (one token per user, TP across mesh columns).

Decode keeps the residual stream in L1, width-sharded over RESIDUAL_CORES cores ([32, hidden / cores] shards): the two
RMSNorms run on those shards (sharded multi-core rms_norm) and the two all-reduces return into them. The projections
are the DRAM-streaming ops of experts/stream.py, which read the norm output row straight from these shards and write
their results straight into the next op's layout (Q/K/V heads, the all-reduce input, the routed ids / scores).
"""

import ttnn

# Smallest compute grid the fused decode layout was validated on (residual / all-reduce 10x3, SDPA 8x8, the stream ops'
# cores next to the 8 DRAM banks of the 11x10 Blackhole grid).
FUSED_DECODE_MIN_GRID = (10, 9)
# The streamed weight layouts split their columns over exactly this many DRAM banks (P150 Blackhole; a 7-bank part
# keeps the original decode path).
FUSED_DECODE_DRAM_BANKS = 8


def fused_decode_layout_supported(
    is_blackhole, mesh_shape, tp, ep, num_experts, use_throughput_experts, tokens, grid=(11, 10), dram_banks=8
):
    """Host-only form of fused_decode_supported (no device): the validated envelope of the fused decode layer
    (grid: the device compute-with-storage grid, x by y; dram_banks: the device's DRAM bank count).

    The fused layer serves one token per device with TP over a single mesh row and no expert parallelism. Its
    layouts (residual / all-reduce 30 cores, Q/K/V head buffers, the streamed weights' per-DRAM-bank column split
    and worker cores) are sized for the gpt-oss-20b shapes on the 11x10 Blackhole grid at TP=4, and its decode-only
    streamed weight copies (about +3.2 GB of DRAM per device) were budgeted for that model. Every other layout keeps
    the original decode path."""
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
        and dram_banks == FUSED_DECODE_DRAM_BANKS
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
        mesh_device.dram_grid_size().x,
    )


# Decode precision policy (decode-only weight copies; prefill keeps its own weights). The streamed matmuls run
# custom_mm at LoFi with FP32 accumulation. Real-weight accuracy gate evidence (work_log.md): BFP4 QKV fails the gate,
# BFP4 o_proj passes it but costs 7 points of top-1; BFP8 / BF16 router weights rank alike; experts are BFP4.
QKV_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
OPROJ_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
ROUTER_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b

# Streamed-op worker cores per DRAM bank (perf gate sweeps, work_log.md): dense projections are bank-bandwidth bound
# at 1; the routed gate|up (3 column pairs per bank) and down (12 columns per bank) gain from 3 and 2.
QKV_STREAM_READERS = 1
OPROJ_STREAM_READERS = 1
GATE_UP_STREAM_READERS = 3
DOWN_STREAM_READERS = 2

# Paged decode SDPA K chunk per layer type (probes/bench_sdpa.py, 8x8 grid): the 128-token sliding window is fastest
# at 128; full attention at 256 (11.8 vs 13.0 us at 200 tokens, 109 vs 155 us at 64k; 512 only wins past ~8k).
SDPA_DECODE_K_CHUNK_SLIDING = 128
SDPA_DECODE_K_CHUNK_FULL = 256


def sdpa_decode_program_config(sliding_window):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=0,
        k_chunk_size=SDPA_DECODE_K_CHUNK_SLIDING if sliding_window else SDPA_DECODE_K_CHUNK_FULL,
        exp_approx_mode=False,
    )


# Residual: 2880 = 90 tiles, 30 cores x 3 tiles. The decode all-reduce runs on the packed [32, 96] BF16 partial on
# one core (DecodeAllReduce): 4.1 us with 1 link vs 8.4 us for the 90-tile residual-layout payload (2 links). With a
# single-core payload it must use 1 link: at 2 links the second link gets no cores and its reader reads past its
# runtime args (watcher assert in all_reduce_async worker_reader.cpp).
RESIDUAL_CORES = 30
DECODE_CCL_LINKS = 1


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


def packed_partial_memory_config(mesh_device, hidden_size):
    """The packed [32, W] BF16 all-reduce payload (W = hidden / RESIDUAL_CORES; hidden value h at row h / W, column
    h % W, see experts/stream.py PackedResidualAdd), on one core."""
    return width_sharded_memory_config(mesh_device, hidden_size // RESIDUAL_CORES, 1)


class DecodeAllReduce:
    """Fused single-op all-reduce of the per-device decode partial sums across the TP axis, plus the residual add.

    The partial is the packed [1, 1, 32, W] BF16 tensor (packed_partial_memory_config): all_reduce_async time scales
    with the payload, and the [32, hidden] residual layout would carry 32x more tiles than the one token needs
    (probes: 4.1 us for the 3-tile packed BF16 payload vs 8.4 us for the 90-tile BF8 residual-layout payload).
    residual_add unpacks the sum into row 0 of the residual stream.

    One persistent scratch buffer and global semaphore per call site ("attn", "moe"), shared by every layer:
    within a layer the attention and MoE all-reduces alternate, and each is an all-rank synchronization, so a
    device cannot reach layer i+1's use of a slot before every device has finished layer i's use of it.
    """

    def __init__(self, mesh_device, hidden_size, cluster_axis, topology=ttnn.Topology.Ring):
        from .experts.stream import PackedResidualAdd

        self.mesh_device = mesh_device
        self.cluster_axis = cluster_axis
        self.topology = topology
        self.num_devices = mesh_device.shape[cluster_axis]
        self.memory_config = packed_partial_memory_config(mesh_device, hidden_size)
        packed_width = hidden_size // RESIDUAL_CORES
        buffer_memory_config = width_sharded_memory_config(mesh_device, packed_width * self.num_devices, 1)
        grid = mesh_device.compute_with_storage_grid_size()
        semaphore_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.slots = {
            name: (
                ttnn.empty(
                    [1, 1, 32, packed_width * self.num_devices],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=buffer_memory_config,
                ),
                ttnn.create_global_semaphore(mesh_device, semaphore_cores, 0),
            )
            for name in ("attn", "moe")
        }
        self.residual_add = PackedResidualAdd(mesh_device, residual_memory_config(mesh_device, hidden_size))

    def __call__(self, partial, slot):
        """partial: the packed BF16 [1, 1, 32, W] partial (self.memory_config). Returns the packed BF16 sum."""
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
