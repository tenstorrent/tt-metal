# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layouts and decode policy of the fused decode layer (one token per user, TP across mesh columns).

Between the projections, decode keeps the residual stream as a flat BF16 vector on one core, replicated on every TP
device (fused_decode/boundary.py, which also defines the inter-layer residual contract): each layer boundary all-reduces
the row-parallel partial sums over the fabric, adds them to the residual and applies the next RMSNorm in one op. The
projections are the DRAM-streaming ops of fused_decode/stream.py, which read the flat norm output in one read and write
their results straight into the next op's layout (Q/K/V heads, the flat partial sum, the routed ids / scores).
"""

import os

import ttnn

# Smallest compute grid the fused decode layout was validated on (residual / all-reduce 10x3, SDPA 8x8, the stream ops'
# cores next to the 8 DRAM banks of the 11x10 Blackhole grid).
FUSED_DECODE_MIN_GRID = (10, 9)
# The streamed weight layouts split their columns over exactly this many DRAM banks (P150 Blackhole; a 7-bank part
# keeps the original decode path).
FUSED_DECODE_DRAM_BANKS = 8


def fused_decode_layout_supported(
    is_blackhole,
    mesh_shape,
    tp,
    ep,
    num_experts,
    use_throughput_experts,
    tokens,
    grid=(11, 10),
    dram_banks=8,
    system_devices=4,
):
    """Host-only form of fused_decode_supported (no device): the validated envelope of the fused decode layer
    (grid: the device compute-with-storage grid, x by y; dram_banks: the device's DRAM bank count; system_devices:
    the number of devices in the whole system).

    The fused layer serves one token per device with TP over a single mesh row and no expert parallelism. Its
    layouts (the flat 3-page residual and boundary all-reduce of fused_decode/boundary.py on a 4-device ring, the
    30-core LM-head input, Q/K/V head buffers, the streamed weights' per-DRAM-bank column split and worker cores) are
    sized for the gpt-oss-20b shapes on the 11x10 Blackhole grid at TP=4, and its decode-only streamed weight copies
    (about +3.2 GB of DRAM per device) were budgeted for that model. The boundary all-reduce also assumes the mesh is
    the whole system (a closed 4-device fabric ring), so a 1x4 slice of a larger system (e.g. one data-parallel
    submesh of a LoudBox or Galaxy) is excluded. Every other layout keeps the original decode path."""
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
        and system_devices == mesh_shape[0] * mesh_shape[1]
    )


# Batched decode MoE (experts/decode.py decode_forward_dense): with one 32-row tile of tokens per device nearly every
# expert is routed (an expert is idle with probability (28/32)^32 ~ 1.4%), so decode computes all experts densely with
# two DRAM-sharded matmuls over every expert's weights, side by side ([gate] and [up]: [hidden, E * I]; [down]:
# [E * I, hidden]), instead of the per-expert sparse matmuls. Each expert's TP shard is padded to whole tiles (720 ->
# 736 columns for gpt-oss-20b at TP=4, experts/dense_layout.py). DENSE_MOE_CORES worker cores width-shard the
# [32, E * 736] activation (16 tiles each), which is also the down matmul's input sharding; DENSE_MOE_HIDDEN_CORES
# width-shard the hidden dim (the gate|up input and the down output, 2 tiles each).
DENSE_MOE_TOKENS = 32
DENSE_MOE_CORES = 46
DENSE_MOE_HIDDEN_CORES = 45
# DRAM readers per bank of the DRAM-sharded matmuls; the weight shards are sized for it (dense_layout.dram_shard_tiles:
# gate / up 93 tiles per bank for 736 tiles, down 12 for 90).
DENSE_MOE_WORKERS_PER_BANK = 3
# Cores of the elementwise steps between gate|up and down (bias adds, SwiGLU, routing-weight scaling): twice the
# matmul cores (8 tiles each), the gate/up outputs resharded there and the scaled activation back.
DENSE_MOE_ELTWISE_CORES = 92


def dense_moe_decode_layout_supported(
    is_blackhole,
    mesh_shape,
    tp,
    ep,
    num_experts,
    use_throughput_experts,
    tokens,
    grid=(11, 10),
    dram_banks=8,
    system_devices=4,
):
    """Host-only form of dense_moe_decode_supported: the validated envelope of the batched decode MoE (gpt-oss-20b on
    Blackhole 1x4, TP=4, one 32-token tile per device, no expert parallelism)."""
    return (
        is_blackhole
        and tuple(mesh_shape) == (1, 4)
        and tp == 4
        and ep == 1
        and num_experts == 32
        and not use_throughput_experts
        and tokens == DENSE_MOE_TOKENS
        and grid[0] * grid[1] >= max(DENSE_MOE_CORES, DENSE_MOE_HIDDEN_CORES, DENSE_MOE_ELTWISE_CORES)
        and dram_banks == FUSED_DECODE_DRAM_BANKS
        and system_devices == mesh_shape[0] * mesh_shape[1]
    )


def dense_moe_decode_supported(mesh_device, mesh_config, hf_config, use_throughput_experts, tokens_per_device):
    """Whether this model's decode MoE uses the batched dense path (see dense_moe_decode_layout_supported)."""
    if mesh_config is None:
        return False
    grid = mesh_device.compute_with_storage_grid_size()
    return dense_moe_decode_layout_supported(
        ttnn.device.is_blackhole(mesh_device),
        tuple(mesh_device.shape),
        mesh_config.decode.tp,
        mesh_config.decode.ep,
        hf_config.num_local_experts,
        use_throughput_experts,
        tokens_per_device,
        (grid.x, grid.y),
        mesh_device.dram_grid_size().x,
        ttnn.get_num_devices(),
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
        ttnn.get_num_devices(),
    )


# Decode precision policy (decode-only weight copies; prefill keeps its own weights). The streamed matmuls run
# custom_mm at LoFi with FP32 accumulation. Measured against HF reference logits with real weights: BFP4 QKV fails the
# accuracy check, BFP4 o_proj passes it but costs 7 points of top-1; BFP8 / BF16 router weights rank alike; experts
# are BFP4.
QKV_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
OPROJ_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
ROUTER_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
# Routed expert weights of the decode MoE streams (ExpertGateUpStream / ExpertDownStream): bfloat4_b, or bfloat8_b for
# the layers in GPT_OSS_ROUTED_DECODE_BF8_LAYERS ("all", "" or a comma list of layer ids; default ""). bf8 holds the
# MXFP4 checkpoint values exactly, bf4 rounds ~20% of them (12.8% relative error on layer 10's down weights) and
# flattens the next-token distribution; a bf8 layer streams 1.9x the expert bytes.
ROUTED_DECODE_BF8_LAYERS_ENV = "GPT_OSS_ROUTED_DECODE_BF8_LAYERS"
# GPT_OSS_DECODE_EXPERTS_SPLIT holds the routed experts' MXFP4 weights exactly in the decode streams, as a BFP4 part
# plus a BFP2 part read by the compressed custom_mm (fused_decode/mxfp4_split.py; on in the gpt-oss-20b release
# settings, models/demos/gpt_oss/__init__.py). Overrides the bf8 expert switch for the streams it covers. Spec: "1" /
# "all" (gate|up and down), "gate_up" or "down", optionally ":<comma list of layer ids>" (default every layer); "" / "0"
# = off.
DECODE_EXPERTS_SPLIT_SPEC = os.environ.get("GPT_OSS_DECODE_EXPERTS_SPLIT", "0").strip()
DECODE_EXPERTS_SPLIT = DECODE_EXPERTS_SPLIT_SPEC not in ("", "0")


# GPT_OSS_DECODE_SPLIT_RANKS=k or a comma list of k per layer (with GPT_OSS_DECODE_EXPERTS_SPLIT): only each token's top-k
# routed experts read the exact split parts; the others read a BFP4 rounding (along K) stored after them. Unset: every
# routed expert is exact.
_SPLIT_RANKS_SPEC = os.environ.get("GPT_OSS_DECODE_SPLIT_RANKS", "").strip()
DECODE_SPLIT_RANKS = [int(v) for v in _SPLIT_RANKS_SPEC.split(",")] if _SPLIT_RANKS_SPEC else None


# GPT_OSS_DECODE_SPLIT_ROUND_ONLY=1: a split stream with no exact rank (split_r0) stores only its rounded BFP4 parts,
# columns contiguous like the BFP4 streams. Its [BFP4][BFP2][rounded BFP4] segments made every read skip the two exact
# parts it never reads (2.3x the bytes per column apart): the K-GPTQ base (k0) ran 0.7% slower than the package.
DECODE_SPLIT_ROUND_ONLY = os.environ.get("GPT_OSS_DECODE_SPLIT_ROUND_ONLY", "0") == "1"


# GPT_OSS_DECODE_SPLIT_NATIVE=1 (with GPT_OSS_DECODE_SPLIT_ROUND_ONLY): a round-only rank-0 stream runs as a plain
# bfloat4_b stream (the package's reader and custom_mm) of transposed tiles: its rounded part is already the package
# column layout (K tiles + a bias tile, 576-byte BFP4 tiles), only transposed with the bias terms at K 0 and 16.
DECODE_SPLIT_NATIVE = os.environ.get("GPT_OSS_DECODE_SPLIT_NATIVE", "0") == "1"


def decode_split_ranks(layer_idx):
    """The exact routed ranks k of a layer (GPT_OSS_DECODE_SPLIT_RANKS), or None (every routed expert exact)."""
    if DECODE_SPLIT_RANKS is None:
        return None
    if len(DECODE_SPLIT_RANKS) == 1:
        return DECODE_SPLIT_RANKS[0]
    assert layer_idx is not None, "a per-layer GPT_OSS_DECODE_SPLIT_RANKS needs the layer index"
    return DECODE_SPLIT_RANKS[int(layer_idx)]


def decode_split_dtype(layer_idx=None, stream=None):
    """The decode stream weight dtype of a layer's split stream ("gate_up" / "down"; both streams of a layer have the
    same exact ranks): "split" or "split_r<k>" (fused_decode/mxfp4_split.py)."""
    k = decode_split_ranks(layer_idx)
    return "split" if k is None else f"split_r{k}"


def decode_experts_split(layer_idx):
    """The decode streams ("gate_up", "down") of a layer held exactly by GPT_OSS_DECODE_EXPERTS_SPLIT."""
    if not DECODE_EXPERTS_SPLIT:
        return frozenset()
    which, _, layers = DECODE_EXPERTS_SPLIT_SPEC.partition(":")
    streams = {"1": ("gate_up", "down"), "all": ("gate_up", "down"), "gate_up": ("gate_up",), "down": ("down",)}[which]
    if layers and (layer_idx is None or int(layer_idx) not in {int(x) for x in layers.split(",")}):
        return frozenset()
    return frozenset(streams)


def routed_decode_bf8_layers():
    """Raw GPT_OSS_ROUTED_DECODE_BF8_LAYERS spec (recorded in the weight-cache marker)."""
    return os.getenv(ROUTED_DECODE_BF8_LAYERS_ENV, "").strip()


def routed_decode_weights_dtype(layer_idx):
    """Routed expert weight dtype of one layer's decode streams: bfloat8_b for the layers in the bf8 list."""
    spec = routed_decode_bf8_layers()
    if spec == "all":
        return ttnn.bfloat8_b
    layers = {int(v) for v in spec.split(",") if v.strip() != ""}
    return ttnn.bfloat8_b if layer_idx is not None and int(layer_idx) in layers else ttnn.bfloat4_b


# Streamed-op worker cores per DRAM bank (swept on the full model): dense projections are bank-bandwidth bound
# at 1; the routed gate|up (3 column pairs per bank) and down (12 columns per bank) gain from 3 and 2.
QKV_STREAM_READERS = 1
OPROJ_STREAM_READERS = 1
GATE_UP_STREAM_READERS = 3
DOWN_STREAM_READERS = 2

# The streamed QKV op also rotates Q / K and writes K / V into the paged cache (fused_decode/stream.py QKVRopeCache),
# replacing the fused-QK RoPE op and the fused K/V cache update (2 ops per layer); its weights are then stored in the
# K / V-first slot order (stream.py qkv_column_order). GPT_OSS_QKV_FUSED_ROPE_KV=0: the separate ops.
QKV_FUSED_ROPE_KV = os.environ.get("GPT_OSS_QKV_FUSED_ROPE_KV", "1") == "1"
# GPT_OSS_QKV_ROPE_EXACT=2 (gpt-oss-20b release settings): the fused QKV kernel copies y / cos / sin into DST exactly
# instead of loading them through LoFi custom_mm (which drops their last BF16 bit); y @ T still goes through LoFi
# (kernels/stream_qkv_rope_compute.cpp QKV_ROPE_EXACT_LITE). Unset: the LoFi loads.
QKV_ROPE_EXACT_LITE = os.environ.get("GPT_OSS_QKV_ROPE_EXACT") == "2"

# Decode LM head (fused_decode/terminal.py): a DRAM-streaming copy of the LM-head weight, vocab split evenly over the TP
# devices, BFP8 x BF16 LoFi like the other dense projections. Readers per DRAM bank (measured standalone: 1 and 2
# reach ~504 GB/s, 4-5 are slower); weight columns buffered ahead while the fused final boundary runs.
LM_HEAD_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
LM_HEAD_STREAM_READERS = 1
LM_HEAD_STREAM_PREFETCH = 4
# Greedy decode sampler of the fused terminal path: "split_argmax" = argmax over the gathered per-device top-32
# candidates (kernels/terminal_pick.cpp); "sampling" = ttnn.sampling with k=1 (the seeded top-k / top-p draw).
DECODE_GREEDY_SAMPLER = "split_argmax"
# Sampler candidates of the fused terminal path: "fused" = per-core top-32 in the LM-head writers + merge + fabric
# exchange inside the LM-head op (fused_decode.terminal.TerminalExchange); "ops" = ttnn.topk + merge op + two
# all_gathers.
DECODE_TERMINAL_EXCHANGE = "fused"

# Paged decode SDPA K chunk per layer type (measured standalone, 8x8 grid): the 128-token sliding window is fastest
# at 128; full attention at 256 (11.8 vs 13.0 us at 200 tokens, 109 vs 155 us at 64k; 512 only wins past ~8k).
SDPA_DECODE_K_CHUNK_SLIDING = 128
SDPA_DECODE_K_CHUNK_FULL = 256


# One-user decode (the fused decode layer), full attention: the K sequence of each of the device's 2 KV heads is split
# over up to SDPA_DECODE_B1_CORES_PER_HEAD cores of the whole 11x10 grid. The op's default cap (16 cores per head, so
# 32 of the 64 cores of an 8x8 grid) left the long-context SDPA compute-bound: at 131k positions 195 -> 133 us per
# layer (182 -> 267 GB/s of KV), at 200 positions 11.9 -> 13.0 us (measured standalone, per-device gpt-oss-20b shapes).
SDPA_DECODE_B1_GRID = (11, 10)
SDPA_DECODE_B1_CORES_PER_HEAD = 32


# GPT_OSS_DECODE_BOUNDARY_EARLY_SUM=1 (gpt-oss-20b release settings; unset = the previous order): the one-row boundary
# reader (kernels/boundary_reader.cpp BND_EARLY_SUM) publishes the residual and the all-reduce slots before the RMSNorm
# inputs, so the gamma DRAM read and the reduce-scaler fill overlap the slot sum instead of gating it (1.4-1.5 us of the
# 7.2 us router op and of the QKV op's boundary in a device profile, where the slots had long arrived). Same CB bytes:
# bit-identical.
DECODE_BOUNDARY_EARLY_SUM = os.environ.get("GPT_OSS_DECODE_BOUNDARY_EARLY_SUM") == "1"
# GPT_OSS_DECODE_ROUTER_FAST_TOPK=1 (gpt-oss-20b release settings; unset = 4 argmax passes): the one-row decode
# router's top-4 + softmax (kernels/router_topk.hpp topk_softmax_fast, in the router LinearStream writer) in one pass of
# register insertions over word reads of the logits; ~1.9 us of the 7.2 us router op ran after its matmul in a device
# profile. Same ids (ties to the lower id), weights and stage bytes: bit-identical.
DECODE_ROUTER_FAST_TOPK = os.environ.get("GPT_OSS_DECODE_ROUTER_FAST_TOPK") == "1"


def sdpa_decode_program_config(sliding_window, one_user=False):
    """one_user: the fused one-user decode layer (full attention takes the whole grid, see SDPA_DECODE_B1_*); otherwise
    the batched layout (one user per core of the 8x8 grid)."""
    if one_user and not sliding_window:
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*SDPA_DECODE_B1_GRID),
            q_chunk_size=0,
            k_chunk_size=SDPA_DECODE_K_CHUNK_FULL,
            exp_approx_mode=False,
            max_cores_per_head_batch=SDPA_DECODE_B1_CORES_PER_HEAD,
        )
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=0,
        k_chunk_size=SDPA_DECODE_K_CHUNK_SLIDING if sliding_window else SDPA_DECODE_K_CHUNK_FULL,
        exp_approx_mode=False,
    )


# Layer boundary all-reduce (fused_decode/boundary.py): "fabric" = the boundary op's own fabric multicast of the flat
# partial sums; "ttnn" = ttnn.experimental.all_reduce_async on the flat partial, then the boundary's add + norm (the
# measured alternative).
DECODE_BOUNDARY_CCL = "fabric"
# Fused matmul + all-reduce send: the producing o_proj / MoE down stream op runs the boundary's fabric sender
# (fused_decode/boundary.py DecodeBoundary.sending_program); the boundary op then only waits, adds and normalizes.
DECODE_BOUNDARY_FUSED_SEND = True
# Consumer fusion: each boundary runs inside the op that consumes its normed output (the next QKV stream, the router
# stream), whose weight readers stream ahead while it waits / computes (fused_decode/boundary.py consumer_parts). Needs
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
