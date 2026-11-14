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
    submesh of a LoudBox or Galaxy) is excluded.

    One chip (a 1x1 mesh, TP=1): no all-reduce (each boundary reads the producer's partial directly), 64 Q / 8 KV heads
    per device, the full vocab in one LM head, and the routed experts streamed from the ND-sharded expert tensors
    prefill reads (fused_decode/nd.py: the BFP8 experts leave no room for a decode-only copy).

    128 experts (gpt-oss-120b): 1x4 only, the routed experts streamed from the ND-sharded expert tensors prefill reads
    (nd_experts_on: decode-only copies next to them would not fit), the router's top-4 of 128 over 4 router cores.

    Every other layout keeps the original decode path; GPT_OSS_FUSED_DECODE=0 keeps it everywhere."""
    if os.environ.get("GPT_OSS_FUSED_DECODE", "1").strip().lower() in ("0", "false", "off"):
        return False
    one_chip = tuple(mesh_shape) == (1, 1) and tp == 1
    four_chips = tuple(mesh_shape) == (1, 4) and tp == 4 and system_devices == mesh_shape[0] * mesh_shape[1]
    return (
        is_blackhole
        and (one_chip or four_chips)
        and ep == 1
        and (num_experts == 32 or (num_experts == 128 and four_chips))
        and not use_throughput_experts
        and tokens == 1
        and grid[0] >= FUSED_DECODE_MIN_GRID[0]
        and grid[1] >= FUSED_DECODE_MIN_GRID[1]
        and dram_banks == FUSED_DECODE_DRAM_BANKS
    )


def nd_experts_on(tp, num_experts):
    """Whether the fused decode streams the routed experts from the ND-sharded expert tensors prefill also reads
    (fused_decode/nd.py) instead of decode-only copies: at TP=1 and for more than 32 experts (gpt-oss-120b at TP=4),
    where a second copy of the experts would not fit in DRAM."""
    return tp == 1 or num_experts > 32


def embed_vshard_on(tp, num_experts):
    """The embedding table vocab-sharded over the TP devices (fused_decode/embed.py) for more than 32 experts at TP > 1
    (gpt-oss-120b): each device holds vocab / tp rows + one zero row, 0.29 instead of 1.16 GB per device for the 201k x
    2880 BF16 table. Decode (fused_decode/inputs.py): each device writes its row (or zeros) as the first layer
    boundary's all-reduce partial; prefill: local ids, embedding, all-reduce. Exact: one nonzero term per element."""
    return tp > 1 and num_experts > 32


EXPERT_DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}


def _dtype_name(env, default):
    name = os.environ.get(env, "").strip().lower() or default
    if name not in EXPERT_DTYPES:
        raise ValueError(f"{env}={name!r}: expected one of {sorted(EXPERT_DTYPES)}")
    return name


# ND experts (nd_experts_on: one chip, gpt-oss-120b on 1x4), one copy shared by prefill and decode:
# GPT_OSS_ND_EXPERT_DTYPE = gate / up's dtype and GPT_OSS_ND_DOWN_DTYPE = down's (default: gate / up's), "bfp8" (the
# default: exact for the checkpoint's MXFP4 values) or "bfp4" (rounded as ../bfp4_quant.py expert_bfp4_rounding says).
# gpt-oss-120b on 1x4 (per device, usable 34.1 GB): BFP8 experts 31.1 GB, enough for up to ~32k tokens of context;
# BFP8 gate / up with BFP4 down 26.3 GB, for the longest contexts (128k); all BFP4 16.5 GB. GPT_OSS_ND_LAYER_DTYPES
# sets single layers' dtypes (bfp4_quant.nd_layer_dtypes). Each dtype's tensors have their own cache names; the
# weight-cache marker records the format (fused_expert_cache_tag).
ND_EXPERT_DTYPE_DEFAULT = "bfp8"
# Prefill expert chunk with BFP8 ND experts at TP > 1: the [1, E, chunk, I] intermediates of 128 experts must fit next
# to them (4096 tokens need ~1.5 GB).
ND_BFP8_PREFILL_CHUNK = 2048


def nd_expert_dtype_names():
    """(gate / up, down) dtype names of the ND experts."""
    gu = _dtype_name("GPT_OSS_ND_EXPERT_DTYPE", ND_EXPERT_DTYPE_DEFAULT)
    return gu, _dtype_name("GPT_OSS_ND_DOWN_DTYPE", gu)


def nd_expert_dtypes():
    gu, dn = nd_expert_dtype_names()
    return EXPERT_DTYPES[gu], EXPERT_DTYPES[dn]


def fused_expert_cache_tag(tp, num_experts):
    """The fused decode's expert tensors as the weight-cache marker records them: "nd_<gate / up>[_down<down>]" for the
    ND experts (plus "_layers<GPT_OSS_ND_LAYER_DTYPES>" when set), else the decode-only copies' dtype; then
    "_<rounding>" when BFP4 tensors are rounded other than by ttnn's packer (bfp4_quant.expert_bfp4_rounding)."""
    from ..bfp4_quant import LAYER_DTYPES_ENV, expert_bfp4_rounding

    if nd_experts_on(tp, num_experts):
        gu, dn = nd_expert_dtype_names()
        tag = f"nd_{gu}" + ("" if dn == gu else f"_down{dn}")
        layers = os.environ.get(LAYER_DTYPES_ENV, "").strip()
        if layers:
            tag += f"_layers{layers}"
        has_bfp4 = "bfp4" in (gu, dn) or "bfp4" in layers
        rounding = expert_bfp4_rounding(nd=True)
    else:
        tag = expert_decode_weight_dtype_name()
        has_bfp4 = True  # the prefill copies
        rounding = expert_bfp4_rounding(nd=False)
    return tag + (f"_{rounding}" if has_bfp4 and rounding != "max" else "")


# Batched decode MoE (experts/decode.py decode_forward_dense): with one 32-row tile of tokens per device nearly every
# expert is routed (an expert is idle with probability (28/32)^32 ~ 1.4%), so decode computes all experts densely with
# two DRAM-sharded matmuls over every expert's weights, side by side ([gate] and [up]: [hidden, E * I]; [down]:
# [E * I, hidden]), instead of the per-expert sparse matmuls. DENSE_MOE_CORES worker cores width-shard the [32, E * I]
# activation (16 tiles each for gpt-oss-20b at TP=4), which is also the down matmul's input sharding.
DENSE_MOE_TOKENS = 32
DENSE_MOE_CORES = 45
DENSE_MOE_WORKERS_PER_BANK = 3
# Cores of the elementwise steps between gate|up and down (bias adds, SwiGLU, routing-weight scaling): twice the
# matmul cores (8 tiles each), the gate/up outputs resharded there and the scaled activation back.
DENSE_MOE_ELTWISE_CORES = 90


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
        and grid[0] * grid[1] >= DENSE_MOE_CORES
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
# below.
QKV_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
OPROJ_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
ROUTER_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b

# Routed-expert decode-only copies (gpt-oss-20b on 1x4, streamed by ExpertGateUpStream / ExpertDownStream):
# GPT_OSS_FUSED_EXPERT_DTYPE = "bfp4" (default; K-grouped least-squared-error rounding, ../bfp4_quant.py) or "bfp8"
# (exact for the checkpoint's MXFP4 values, at 1.9x the streamed expert bytes). Each dtype's copies are cached under
# their own names.
ENV_FUSED_EXPERT_DTYPE = "GPT_OSS_FUSED_EXPERT_DTYPE"
EXPERT_DECODE_WEIGHT_DTYPE_DEFAULT = "bfp4"


def expert_decode_weight_dtype_name():
    """The routed-expert decode copies' dtype name: GPT_OSS_FUSED_EXPERT_DTYPE, else the default."""
    return _dtype_name(ENV_FUSED_EXPERT_DTYPE, EXPERT_DECODE_WEIGHT_DTYPE_DEFAULT)


def expert_decode_weight_dtype():
    """The ttnn dtype of the routed-expert decode copies (see expert_decode_weight_dtype_name)."""
    return EXPERT_DTYPES[expert_decode_weight_dtype_name()]


# Streamed-op worker cores per DRAM bank (swept on the full model): dense projections are bank-bandwidth bound
# at 1; the routed gate|up (3 column pairs per bank) and down (12 columns per bank) gain from 3 and 2.
QKV_STREAM_READERS = 1
OPROJ_STREAM_READERS = 1
GATE_UP_STREAM_READERS = 3
DOWN_STREAM_READERS = 2
# TP=1: 4x the QKV columns (160 tiles = 20 per bank) and o_proj K (128 tiles): 2 readers per bank (measured faster than
# 3-5: QKV 35 vs 41 us, o_proj 31 vs 53 us); the QKV consumer's weight ring is capped at TP1_QKV_PREFETCH columns (all 10
# columns of 91 BFP8 tiles would take 1 MB of L1).
TP1_QKV_STREAM_READERS = 2
TP1_OPROJ_STREAM_READERS = 2
TP1_QKV_PREFETCH = 6


def stream_readers(role, tp):
    """Weight readers per DRAM bank of the dense decode projections ("qkv", "o_proj") at this TP."""
    if tp == 1:
        return {"qkv": TP1_QKV_STREAM_READERS, "o_proj": TP1_OPROJ_STREAM_READERS}[role]
    return {"qkv": QKV_STREAM_READERS, "o_proj": OPROJ_STREAM_READERS}[role]


# The streamed QKV op also rotates Q / K and writes K / V into the paged cache (fused_decode/stream.py QKVRopeCache),
# replacing the fused-QK RoPE op and the fused K/V cache update (2 ops per layer); its weights are then stored in the
# K / V-first slot order (stream.py qkv_column_order). GPT_OSS_QKV_FUSED_ROPE_KV=0: the separate ops.
QKV_FUSED_ROPE_KV = os.environ.get("GPT_OSS_QKV_FUSED_ROPE_KV", "1") == "1"

# Decode LM head (fused_decode/terminal.py): a DRAM-streaming copy of the LM-head weight, vocab split evenly over the TP
# devices, BFP8 x BF16 LoFi like the other dense projections. Readers per DRAM bank (measured standalone: 1 and 2
# reach ~504 GB/s, 4-5 are slower); weight columns buffered ahead while the fused final boundary runs.
LM_HEAD_DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
LM_HEAD_STREAM_READERS = 1
LM_HEAD_STREAM_PREFETCH = 4
# Two-stage LM head at TP > 1 (fused_decode/terminal.py, kernels/stream_linear_writer.cpp LM_RESCORE): a BFP4 copy of
# the LM-head stream gives every output, then each LM-head core recomputes the columns holding its LM_RESCORE_COLUMNS
# best ids from the BFP8 weights; its sampler candidates (and those folded-logit columns) are the BFP8 values, so the
# greedy pick is the BFP8 head's. GPT_OSS_LM_RESCORE=0 (read when the model is built and per call): the BFP8 head only.
LM_RESCORE_COLUMNS = 6
# Greedy decode sampler of the fused terminal path: "split_argmax" = argmax over the gathered per-device top-32
# candidates (kernels/terminal_pick.cpp); "sampling" = ttnn.sampling with k=1 (the seeded top-k / top-p draw).
DECODE_GREEDY_SAMPLER = "split_argmax"
# Sampler candidates of the fused terminal path: "fused" = per-core top-32 in the LM-head writers + merge + fabric
# exchange inside the LM-head op (fused_decode.terminal.TerminalExchange); "ops" = ttnn.topk + merge op + two
# all_gathers.
DECODE_TERMINAL_EXCHANGE = "fused"


def _on(env):
    return os.environ.get(env, "1").strip().lower() not in ("0", "false", "off")


def lm_rescore_on():
    """GPT_OSS_LM_RESCORE (default on): the two-stage LM head (LM_RESCORE_COLUMNS)."""
    return _on("GPT_OSS_LM_RESCORE")


def fast_topk_on():
    """GPT_OSS_FAST_TOPK (default on, read per call): the router's top-4 of 32 + softmax in one pass
    (kernels/router_topk.hpp topk4_softmax_fast, the same output bits as the generic top-k)."""
    return _on("GPT_OSS_FAST_TOPK")


# Decode attention of the one-user layer, sliding-window and full: the split decode attention (fused_decode/
# split_attn.py) inside the decode QKV program, kv head g's 8 query heads over slice cores (flash decoding: each slice a
# contiguous range of the attended positions, the partial softmaxes merged on the kv head's leader in one step);
# GPT_OSS_SPLIT_ATTN=0 (read per call): the paged SDPA decode op instead. Program semaphore SDA_WAIT_SEM: the QKV
# writers signal the slices that Q and the position's K / V rows are written.
SDA_WAIT_SEM = 3
# Slices per kv head and fewest 32-position chunks per active slice, per layer type ("full_tp1": full attention with 8
# kv heads per device, one chip), capped by the cores the QKV program leaves free: the sliding window on 2 slices (its
# older chunks, read while QKV streams, and the newest chunk, read after QKV writes it); full attention on 32 slices at
# TP=4 (2 kv heads per device) and 5 at TP=1.
SDA_SLICES = {"sliding": (2, 1), "full": (32, 2), "full_tp1": (5, 2)}


def split_attn_on():
    """GPT_OSS_SPLIT_ATTN (default on, read per call): the split decode attention instead of the paged SDPA op."""
    return _on("GPT_OSS_SPLIT_ATTN")


def sda_slices(sliding, max_slices, n_kv=2):
    """(slices per kv head, min chunks per active slice) of the split decode attention for this layer type (n_kv: kv
    heads per device)."""
    slices, min_chunks = SDA_SLICES["sliding" if sliding else ("full_tp1" if n_kv >= 8 else "full")]
    return max(1, min(slices, max_slices)), min_chunks


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


# One-user decode at TP=1 (8 KV heads per device), on the whole 11x10 grid: (cores per KV head, K chunk) per layer type
# (the op splits each head's K sequence over its cores and reduces their partial results on one core per head), with
# approximate exp / math.
TP1_SDPA_SLIDING = (8, 128)
TP1_SDPA_FULL = (13, 256)


def sdpa_decode_compute_config(program_config, tp=None):
    """The fused decode SDPA's compute config: the attention program config's (HiFi4); approximate math at tp = 1."""
    config = program_config.get_compute_kernel_config()
    if tp == 1:
        config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, program_config.math_fidelity),
            math_approx_mode=True,
            fp32_dest_acc_en=program_config.fp32_dest_acc_en,
            packer_l1_acc=program_config.packer_l1_acc,
        )
    return config


def sdpa_decode_program_config(sliding_window, one_user=False, tp=None):
    """one_user: the fused one-user decode layer (full attention takes the whole grid, see SDPA_DECODE_B1_*); otherwise
    the batched layout (one user per core of the 8x8 grid). tp = 1: TP1_SDPA_*."""
    if one_user and tp == 1:
        cores, k_chunk = TP1_SDPA_SLIDING if sliding_window else TP1_SDPA_FULL
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*SDPA_DECODE_B1_GRID),
            q_chunk_size=0,
            k_chunk_size=k_chunk,
            exp_approx_mode=True,
            max_cores_per_head_batch=cores,
        )
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
