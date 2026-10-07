# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layouts and decode policy of the fused decode layer (one token per user, TP across mesh columns).

Between the projections, decode keeps the residual stream as a flat BF16 vector on one core, replicated on every TP
device (decode_boundary.py, which also defines the inter-layer residual contract): each layer boundary all-reduces
the row-parallel partial sums over the fabric, adds them to the residual and applies the next RMSNorm in one op. The
projections are the DRAM-streaming ops of experts/stream.py, which read the flat norm output in one read and write
their results straight into the next op's layout (Q/K/V heads, the flat partial sum, the routed ids / scores).
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
    layouts (the flat 3-page residual and boundary all-reduce of decode_boundary.py on a 4-device ring, the 30-core
    LM-head input, Q/K/V head buffers, the streamed weights' per-DRAM-bank column split and worker cores) are sized
    for the gpt-oss-20b shapes on the 11x10 Blackhole grid at TP=4, and its decode-only streamed weight copies
    (about +3.2 GB of DRAM per device) were budgeted for that model. Every other layout keeps the original decode
    path."""
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

# Decode LM head (decode_terminal.py): a DRAM-streaming copy of the LM-head weight, vocab split evenly over the TP
# devices, BFP8 x BF16 LoFi like the other dense projections. Readers per DRAM bank (probes/bench_terminal.py: 1 and 2
# reach ~504 GB/s, 4-5 are slower); weight columns buffered ahead while the fused final boundary runs.
LM_HEAD_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
LM_HEAD_STREAM_READERS = 1
LM_HEAD_STREAM_PREFETCH = 4
# Greedy decode sampler of the fused terminal path: "split_argmax" = argmax over the gathered per-device top-32
# candidates (kernels/terminal_pick.cpp); "sampling" = ttnn.sampling with k=1 (the seeded top-k / top-p draw).
DECODE_GREEDY_SAMPLER = "split_argmax"
# Sampler candidates of the fused terminal path: "fused" = per-core top-32 in the LM-head writers + merge + fabric
# exchange inside the LM-head op (decode_terminal.TerminalExchange); "ops" = ttnn.topk + merge op + two all_gathers.
DECODE_TERMINAL_EXCHANGE = "fused"

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


# Layer boundary all-reduce (decode_boundary.py): "fabric" = the boundary op's own fabric multicast of the flat partial
# sums; "ttnn" = ttnn.experimental.all_reduce_async on the flat partial, then the boundary's add + norm (the measured
# alternative, work_log.md).
DECODE_BOUNDARY_CCL = "fabric"
# Fused matmul + all-reduce send: the producing o_proj / MoE down stream op runs the boundary's fabric sender
# (decode_boundary.py DecodeBoundary.sending_program); the boundary op then only waits, adds and normalizes.
DECODE_BOUNDARY_FUSED_SEND = True
# Consumer fusion: each boundary runs inside the op that consumes its normed output (the next QKV stream, the router
# stream), whose weight readers stream ahead while it waits / computes (decode_boundary.py consumer_parts). Needs
# DECODE_BOUNDARY_FUSED_SEND with the fabric all-reduce.
DECODE_BOUNDARY_FUSE_CONSUMER = True
# The fused consumer's weight circular buffer holds all its columns (not 3), so its readers never stall on the boundary.
DECODE_BOUNDARY_PREFETCH_ALL = True
# Fabric links the fused send uses (2: one sender per link, the packets of the partial split between them).
DECODE_BOUNDARY_LINKS = 1

# The LM-head path after the last layer reads the normed hidden in row 0 of a [32, hidden] BF16 tensor, L1
# width-sharded on RESIDUAL_CORES cores (2880 = 90 tiles, 30 cores x 3 tiles).
RESIDUAL_CORES = 30


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
