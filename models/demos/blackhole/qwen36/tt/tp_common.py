# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP helpers for Qwen3.5/3.6. Per-arch and per-mesh tuning is gated by the predicates below.

Used only when num_devices > 1. DRAM-sharded matmul cfgs, prefill progcfgs,
mesh shard/replicate, FP8 dequant, HF weight reorder for per-device sharding.
"""

import math
import os

import torch

import ttnn
from models.common.utility_functions import is_blackhole

# Hardware constants
TILE_SIZE = 32
DRAM_CORES = 8
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})


# fp32 destination tiles take two half-DST slots, so fp32_dest_acc_en halves the output-subblock budget.
DST_TILES = 8
DST_TILES_FP32_ACC = 4


# Compute kernel configs
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# Same fidelity, fp32 destination accumulation off. Only the GDN prefill in-projection uses this.
COMPUTE_HIFI2_NO_FP32_ACC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


# LoFi + no fp32 dest acc, for the two attention prefill matmuls on Wormhole.
COMPUTE_LOFI_NO_FP32_ACC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def sdpa_bf8_enabled(args):
    """QWEN_SDPA_BF8: bf8 Q/K/V and a bf8 paged KV cache. Default on for N300; the env var overrides either way."""
    env = os.environ.get("QWEN_SDPA_BF8")
    if env is not None:
        return env == "1"
    return not is_blackhole() and getattr(args, "device_name", None) == "N300"


def wh_9b_n300(args):
    """True only for Qwen3.5-9B on a Wormhole N300."""
    return not is_blackhole() and getattr(args, "dim", 0) <= 4096 and getattr(args, "device_name", None) == "N300"


def wh_t3k(args):
    """True only for an 8-chip Wormhole mesh (T3K)."""
    return not is_blackhole() and int(getattr(args, "num_devices", 1)) == 8


def wh_9b_n300_vision(args):
    """wh_9b_n300 for vision args. Key off text hidden size: args.dim is the vision width on both models."""
    if is_blackhole() or getattr(args, "device_name", None) != "N300":
        return False
    hf = getattr(args, "hf_config", None)
    text_cfg = getattr(hf, "text_config", None) if hf is not None else None
    text_dim = getattr(text_cfg, "hidden_size", None)
    if text_dim is None:
        # No text_config: refuse rather than enable the 9B-only path.
        return False
    return text_dim <= 4096


def rope_permuted_enabled(args):
    """Permuted RoPE. Off on T3K: it is incompatible with the unpadded KV reshard (paged_update_cache segfault)."""
    return wh_9b_n300(args) and getattr(args, "rope_head_dim", 0) < getattr(args, "head_dim", 0)


# Grid helpers
def prefill_grid_default():
    """BH P150: (8,10); WH: (8,8). y capped at 10 on BH (grid_x=10 breaks matmul)."""
    return (8, 10) if is_blackhole() else (8, 8)


# Max grid COLUMNS a tuned prefill config may use. A Blackhole galaxy reports a 12-wide worker
# grid, but harvested P150s expose only 11, so tuning to 12 would not port. 11 x 10 = 110 cores.
PREFILL_MAX_COLS_PORTABLE = 11

#   * widest_cols -- `_best_prefill_cols` ranks widths by (out_subblock_w, cols), subblock first.
_PREFILL_TUNING = {
    4: dict(widest_cols=False, in0_block_w_divisor=False, in0_block_w_cap=4),
    8: dict(widest_cols=True, in0_block_w_divisor=True, in0_block_w_cap=4),
}


def prefill_tuning(num_devices):
    """Prefill matmul tuning for this TP; unknown TP falls back to the frozen TP=4 values."""
    return _PREFILL_TUNING.get(num_devices, _PREFILL_TUNING[4])


def _roundup(a, b):
    return b * math.ceil(a / b)


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def _find_grid(n_tiles, target=32):
    max_r, max_c = 8, 8
    possible = [k for k in range(1, max_r * max_c + 1) if n_tiles % k == 0]
    possible.sort(key=lambda x: abs(x - target))
    for cores in possible:
        for rows in range(1, max_r + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_c:
                    return rows, cols
    raise ValueError(f"Cannot find grid for {n_tiles} tiles")


# DRAM-sharded config builders
def create_dram_sharded_mem_config(k, n):
    """WIDTH_SHARDED DRAM memory config for a weight matrix [k, n]."""
    padded_n = _roundup(n, TILE_SIZE * DRAM_CORES)
    shard_spec = ttnn.ShardSpec(
        DRAM_GRID,
        (k, padded_n // DRAM_CORES),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def create_dram_sharded_matmul_program_config(m, k, n, num_cores=None):
    """DRAM-sharded matmul program config (decode, small M)."""
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE

    if num_cores is None:
        rows, cols = _find_grid(k_tiles)
        num_cores = rows * cols

    k_tiles_per_core = k_tiles // num_cores
    if k_tiles_per_core == 0:
        k_tiles_per_core = k_tiles
        num_cores = 1
    in0_block_w = _find_largest_divisor(k_tiles_per_core)
    per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fused_activation=None,
    )


def create_matmul_1d_decode_progcfg(
    m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8, in0_block_w_cap=8
):
    """Explicit-grid 1D (mcast_in0) decode matmul progcfg on ~`num_cores` cores — small grids beat
    the ~80-core DRAM-sharded grid on the bandwidth-bound skinny decode matmuls. Weight must be interleaved.

    Grid is shaped WIDE-first (cols up to `grid_w`, the device worker-grid width — 11 on BH P150, 8 on
    WH): for a fixed core budget a wide-short grid shortens the in0 multicast column and beats a
    tall-narrow one (~2% on this matmul; see test_mlp_matmul_sweep wide1d_* vs forced1d_*). Default
    grid_w=8 preserves the legacy shaping for callers that don't pass the device width."""
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    # mcast_in0: every core streams the full K, so in0_block_w must divide the full k_tiles.
    # in0_block_w_cap bounds the K-block. Raise it only where that shape was checked.
    per_core_k = _find_largest_divisor(k_tiles, max_div=in0_block_w_cap)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    cap = 4 if fp32_acc else 8  # fp32_dest_acc caps subblock area at 4
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=per_core_k,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def matmul_1d_decode(x, weight, decode_1d_progcfg, compute_cfg, out_memory_config=ttnn.L1_MEMORY_CONFIG):
    """Small-grid 1D (mcast_in0) decode matmul on an interleaved weight; interleaves the K-sharded
    activation first. On Wormhole, compare memory_config by value so an aliased buffer is not deallocated."""
    if is_blackhole():
        x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.linear(
            x_il,
            weight,
            compute_kernel_config=compute_cfg,
            program_config=decode_1d_progcfg,
            memory_config=out_memory_config,
        )
        if x_il is not x:
            ttnn.deallocate(x_il)
        return out
    already_il = x.memory_config() == ttnn.L1_MEMORY_CONFIG
    x_il = x if already_il else ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    out = ttnn.linear(
        x_il,
        weight,
        compute_kernel_config=compute_cfg,
        program_config=decode_1d_progcfg,
        memory_config=out_memory_config,
    )
    if not already_il:
        ttnn.deallocate(x_il)
    return out


def create_activation_shard_config(k):
    """WIDTH_SHARDED L1 activation config for a [*, k] activation."""
    k_tiles = k // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    width_per_core = k // num_cores
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width_per_core),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def decode_ids_for_embed(token_ids):
    """Host-flatten decode ids to [1, B]. Do not reshape a padded row-major [B, 1] on device."""
    return token_ids.reshape(1, token_ids.numel())


def decode_embed(emb, tok, args):
    """Width-sharded decode embedding only when the index row is one full tile. Do not reshape [B, 1] on device."""
    mc = getattr(args, "emb_decode_memcfg", None)
    if mc is None:
        return _embed_narrow_batch(emb, tok) if wh_t3k(args) else emb(tok)
    last = int(tok.shape[-1])
    if last == 0 or last > TILE_SIZE or last % TILE_SIZE != 0:
        return _embed_narrow_batch(emb, tok) if wh_t3k(args) else emb(tok)
    return emb(tok, memory_config=mc)


# Split the single-core tilize only above this width; a narrow row loses to the extra launches.
_FAST_TILIZE_MIN_WIDTH = 256


def _embed_narrow_batch(emb, tok):
    """Pad a short embedding row to a tile so tilize can be multicore, then slice the logical rows back."""
    if getattr(emb, "embed_scale", None) is not None:
        return emb(tok)
    e = ttnn.embedding(tok, emb.weights, layout=ttnn.ROW_MAJOR_LAYOUT)
    shape = list(e.shape)
    rows = int(shape[-2]) if len(shape) >= 2 else 1
    if len(shape) < 2 or rows >= TILE_SIZE or int(shape[-1]) < _FAST_TILIZE_MIN_WIDTH:
        out = ttnn.to_layout(e, ttnn.TILE_LAYOUT)
        ttnn.deallocate(e)
        return out
    pad = [(0, 0)] * (len(shape) - 2) + [(0, TILE_SIZE - rows), (0, 0)]
    padded = ttnn.pad(e, pad, value=0.0)
    ttnn.deallocate(e)
    tiled = ttnn.tilize(padded, use_multicore=True)
    ttnn.deallocate(padded)
    out = ttnn.slice(tiled, [0] * len(shape), shape)
    ttnn.deallocate(tiled)
    return out


# 2D prefill matmul config
def _get_out_subblock_w(per_core_n, out_subblock_h, max_hw=DST_TILES_FP32_ACC):
    """Widest out_subblock_w that divides per_core_n within the DST budget."""
    for w in range(min(per_core_n, max_hw // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _full_grid_crs(grid):
    """Full-grid allowed_worker_cores for CCL-fused matmuls, which bypass ttnn::prim::matmul()'s normalize_program_config()."""
    gx, gy = grid
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def _safe_half_out_block_w(per_core_N, out_subblock_w):
    """Largest divisor of per_core_N, a multiple of out_subblock_w, that is at most half of per_core_N."""
    half = max(1, per_core_N // 2)
    best = out_subblock_w
    for w in range(out_subblock_w, half + 1, out_subblock_w):
        if per_core_N % w == 0:
            best = w
    return best


def _largest_divisor_le(n, cap):
    """Largest divisor of n that is <= cap (1 if none, since 1 always divides)."""
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


# Longest prefill M the full-grid MLP config below is used at.
PREFILL_FULL_GRID_MAX_M = 2048


def create_prefill_mlp_matmul_program_config_full_grid(
    m, k, n, grid_size=None, fused_activation=None, out_subblock_h=1
):
    """One-K-pass prefill MLP progcfg on the full grid (27B on Wormhole). out_subblock_h must divide per_core_M."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))
    out_subblock_h = _largest_divisor_le(per_core_M, out_subblock_h)
    k_tiles = math.ceil(k / TILE_SIZE)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=_largest_divisor_le(k_tiles, DST_TILES),
        out_subblock_h=out_subblock_h,
        # DST_TILES, not the fp32-acc ceiling: this path has fp32 dest acc off.
        out_subblock_w=_get_out_subblock_w(per_core_N, out_subblock_h, max_hw=DST_TILES),
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=per_core_N,  # one K pass
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


def create_prefill_matmul_program_config(
    m,
    k,
    n,
    grid_size=None,
    fused_activation=None,
    tuning=None,
    out_block_w=None,
    halve_out_block=False,
    max_subblock_hw=None,
):
    """2D prefill matmul progcfg (DRAM-interleaved).

    fused_activation in packer; sharded kernel rejects ttnn.linear(activation=...) with progcfg.
    out_block_w holds one N-slice of the per-core output. halve_out_block is for grids already at max cores.
    max_subblock_hw is the DST ceiling (4 with fp32 dest acc, 8 without)."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(
        per_core_N, out_subblock_h, max_hw=max_subblock_hw if max_subblock_hw is not None else DST_TILES_FP32_ACC
    )

    k_tiles = math.ceil(k / TILE_SIZE)
    cap = tuning["in0_block_w_cap"]
    if tuning["in0_block_w_divisor"]:
        # in0_block_w only has to divide k_tiles (no K tail in the 2D mcast kernel), so take the
        # largest legal block rather than scaling with grid width -- see _PREFILL_TUNING.
        in0_block_w = _find_largest_divisor(k_tiles, cap)
    else:
        in0_block_w = min(cap, max(1, k_tiles // grid_size[0]))

    if halve_out_block and out_block_w is None:
        out_block_w = _safe_half_out_block_w(per_core_N, out_subblock_w)

    kwargs = dict(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )
    if out_block_w is not None:
        kwargs["out_block_w"] = out_block_w
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**kwargs)


def create_prefill_kpass1_matmul_program_config(m, k, n, grid_size=None, fused_activation=None):
    """One-K-pass prefill progcfg. Requires fp32_dest_acc_en=False; an fp32-acc config overflows the CB."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))
    k_tiles = math.ceil(k / TILE_SIZE)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=min(4, max(1, k_tiles // grid_size[0])),
        out_subblock_h=1,
        out_subblock_w=_get_out_subblock_w(per_core_N, 1, max_hw=DST_TILES),
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=per_core_N,  # one K pass
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


def prefill_out_memory_config(seq_len, out_width, elem_bytes=2, budget=8 << 20):
    """L1 when the prefill output fits; DRAM on Wormhole when that output would clash with the matmul CBs."""
    if is_blackhole():
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG if seq_len * out_width * elem_bytes > budget else ttnn.L1_MEMORY_CONFIG


def _widest_prefill_cols(n, max_cols, subblock_slack=1):
    """Widest grid whose output subblock stays within `subblock_slack` of the best achievable.

    The TP=8 counterpart to `_best_prefill_cols`. More columns is usually a win at TP=8 (the halved
    per-device N leaves cores idle), but NOT when the extra width collapses the subblock: measured
    at S=2048, mlp_gate (N=2176 -> 68 tiles) goes cols 9 -> 11, per_core_N 8 -> 7, and 7 is prime so
    out_subblock_w drops 4 -> 1 -- a 2058us -> 2118us REGRESSION, i.e. the subblock-first ranking
    was right for that shape. Guarding on the subblock keeps the wide grid exactly where it pays:

        matmul     default        this rule       measured
        attn_wo    c10_bw2_sw4    c11_bw4_sw3     803.5 -> 718.7us
        gdn_out    c10_bw2_sw4    c11_bw4_sw3     802.3 -> 719.9us
        mlp_down   c10_bw4_sw4    c11_bw4_sw3    1787.4 -> 1724.9us
        mlp_gate   c9_bw4_sw4     c9_bw4_sw4     2058.1us (unchanged -- already optimal)
    """
    n_tiles = math.ceil(n / TILE_SIZE)
    sw = {cols: _get_out_subblock_w(math.ceil(n_tiles / cols), 1) for cols in range(1, max_cols + 1)}
    floor = max(sw.values()) - subblock_slack
    return max((cols for cols, w in sw.items() if w >= floor), default=1)


def _best_prefill_cols(n, max_cols):
    """Grid width (<=max_cols) maximizing the output subblock, tie-broken to more cores — avoids the
    1x1-subblock stall (e.g. gate/up N=4352 -> 7-wide -> 1x4) the default full width can force."""
    n_tiles = math.ceil(n / TILE_SIZE)
    best_cols, best_key = 1, None
    for cols in range(1, max_cols + 1):
        sw = _get_out_subblock_w(math.ceil(n_tiles / cols), 1)
        key = (sw, cols)  # prefer wider subblock, then more columns (more compute cores)
        if best_key is None or key > best_key:
            best_key, best_cols = key, cols
    return best_cols


def create_prefill_mlp_matmul_program_config(
    m, k, n, fused_activation=None, max_cols=None, tuning=None, halve_out_block=False, max_subblock_hw=None
):
    """FPU-tuned 2D prefill progcfg for MLP matmuls: picks the grid width that maximizes the output
    subblock (drives prefill FPU) instead of the default full width.

    max_cols caps the grid width. Default = prefill_grid_default()[0] (8). Pass the device worker-grid
    width (11 on BH P150) to let the subblock heuristic go wide -> the measured prefill winners
    (gate 9-wide, down/wo 10-wide, gdn_qkvz 11-wide; test_mlp_matmul_sweep_prefill). Fused AG/RS paths
    pin 8-wide separately and are unaffected.

    tuning: a `_PREFILL_TUNING` entry. With `widest_cols` (TP=8) the subblock-first width heuristic
    is replaced by "take the width, clamped to PREFILL_MAX_COLS_PORTABLE" -- measured device time at
    TP=8 falls monotonically with column count, so widest_cols does not trade cores for a wider subblock.
    halve_out_block is for grids already at max cores. max_subblock_hw does not change the grid width."""
    grid = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    limit = max_cols or grid[0]
    if tuning["widest_cols"]:
        # Cap the width at PREFILL_MAX_COLS_PORTABLE (harvested parts expose 11, not 12) and never
        # exceed the output tile count -- columns beyond it get per_core_N=1 with nothing to compute,
        # paying mcast cost for no work.
        cols = _widest_prefill_cols(n, max(1, min(limit, PREFILL_MAX_COLS_PORTABLE, math.ceil(n / TILE_SIZE))))
    else:
        cols = _best_prefill_cols(n, limit)
    return create_prefill_matmul_program_config(
        m,
        k,
        n,
        grid_size=(cols, grid[1]),
        fused_activation=fused_activation,
        tuning=tuning,
        halve_out_block=halve_out_block,
        max_subblock_hw=max_subblock_hw,
    )


# Mesh tensor helpers
def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
    """Torch weight [out,in] -> sharded mesh tensor. Transpose to [in,out]; dim=-1 column, dim=0 row.

    The bf16 cast + transpose runs as the as_tensor preprocess, so it only executes on a tensor-cache
    miss. On a hit the checkpoint tensor is never materialised (it may be a memory-mapped safetensor
    on a network mount, and reading 27B parameters through it is what pushed the CI weight load past
    its 1200 s pytest timeout)."""
    return ttnn.as_tensor(
        torch_tensor,
        preprocess=lambda t: t.to(torch.bfloat16).T.contiguous(),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        cache_file_name=cache_path,
    )


def agmm_k_block_size(k_local, default=8):
    """Largest power-of-2 K_block_size <= `default` that divides K_tiles/device (AGMM Ring has no tail).

    TP=4: 1280->40 tiles->8; TP=8: 640->20 tiles->4. Odd divisors (e.g. 5|20) are unsafe on Ring.
    """
    k_tiles = k_local // TILE_SIZE
    b = 1 << (min(default, max(1, k_tiles)).bit_length() - 1)
    while b > 1 and k_tiles % b:
        b //= 2
    return b


def agmm_gather_buffer(tt_ccl, x, cluster_axis=1):
    """Persistent gather buffer for all_gather_minimal_matmul_async on activation x [.,S,K/tp].

    TODO(#57458): once all_gather_minimal_matmul_async honours barrier_semaphore (an in-kernel receiver-ready
    handshake), drop this buffer and pass barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)
    instead; test_gdn_out_agmm_deterministic_under_device_skew must keep passing."""
    cache = tt_ccl.__dict__.setdefault("_agmm_gather_buffers", {})
    mesh = tt_ccl.mesh_device
    S, K_local = x.shape[-2], x.shape[-1]
    key = (S, K_local, cluster_axis, x.dtype)
    if key not in cache:
        shape = ttnn.Shape([1, 1, S, K_local * mesh.shape[cluster_axis]])
        pair = [
            ttnn.allocate_tensor_on_device(shape, x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(2)
        ]
        cache[key] = [pair, 0]
    pair, idx = cache[key]
    cache[key][1] = idx ^ 1
    return pair[idx]


def all_gather_matmul_prefill(
    x,
    weight,
    tt_ccl,
    compute_cfg,
    topology,
    grid=(7, 9),
    cluster_axis=1,
    fused_activation=None,
    out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    persistent_output_buffer=None,
):
    """Fused all-gather(dim=3) + column-parallel matmul for prefill (all_gather_minimal_matmul_async).

    x: K-sharded activation [.,S,K/tp]; weight: [K,N] col-sharded (K full). Gathers x to full K and
    matmuls in one op, replacing a separate all_gather + linear. fused_activation applied per tile
    before pack (non-parametrized op, e.g. ttnn.UnaryOpType.SILU). out_memory_config places the result
    (default DRAM; L1 keeps it resident for downstream slices). persistent_output_buffer: the gather buffer
    (agmm_gather_buffer); None lets the op allocate one per call."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    # AG-bound: 2 ethernet links parallelize the gather (P150x4 max; traced_8k TTFT win). grid.x must
    # = num_links*workers, and the 7-wide default (prime) forces 1 link -> widen to 8 (2 links, 4 workers).
    num_links = 2
    grid = (8, grid[1])
    workers = grid[0] // num_links
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=8,
        subblock_h=1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    out = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        fused_activation=fused_activation,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=True,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
        persistent_output_buffer=persistent_output_buffer,
    )[0]

    return out


def decode_ccl_tuning(args):
    """Decode GDN out-proj all-reduce tuning, or None for the tt_all_reduce defaults. T3K only."""
    return (1, 2) if wh_t3k(args) else None


def prefill_ccl_tuning():
    """Prefill (chunks_per_sync, num_workers_per_link). QWEN35_PREFILL_CCL overrides. Wormhole unless overridden."""
    _v = os.environ.get("QWEN35_PREFILL_CCL")
    if _v:
        _c, _w = (int(t) for t in _v.split(","))
        return _c, _w
    if is_blackhole():
        return 10, 2
    return 10, 4


def mlp_gateup_agmm_enabled(num_devices):
    """Fuse the ff_norm all-gather into the MLP gate/up matmul. Blackhole only; an 8-row grid overlaps NOC users."""
    return num_devices > 1 and is_blackhole()


def all_gather_swiglu_prefill(
    x, weight, tt_ccl, compute_cfg, topology, grid=(7, 9), cluster_axis=1, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """Fused all-gather + col-parallel gate/up matmul + SwiGLU for prefill (packing gate+up lets ff_norm's AG fuse in).

    x: K-sharded [.,S,K/tp]; weight: tile-pair-interleaved [gate|up] [K, 2N/tp]. Emits silu(gate)*up of width N/tp."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    num_links = 2
    grid = (8, grid[1])
    workers = grid[0] // num_links
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=16,
        subblock_h=1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    return ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=True,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
        fuse_swiglu=True,
    )[0]


def build_mmrs_decode_state(mesh_device, M, K_local, N, nd, dtype=ttnn.bfloat16):
    """Build (progcfg, intermediate_buffer, output_buffer) for a decode matmul_reduce_scatter out-proj.

    M = LOGICAL decode batch (max_batch_size) — the op returns the persistent buffer with its logical
    shape, so an oversized (tile-padded) M leaks into the residual stream. TILE layout pads M<32.
    dtype MUST match the out-proj input activation (bf16 for MLP/attn; FLOAT32 for GDN, which keeps
    fp32 for stability) — the op's default output dtype is the input's, and writing it into a
    mismatched buffer corrupts the output. Matmul on reduced grid (8,6); RS workers at offset (0,6).
    interm [1,1,M,N], out [1,1,M,N/nd]."""
    cg = (8, 6)
    per_core_N = max(1, math.ceil(N / TILE_SIZE / cg[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=cg,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // cg[0])),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / cg[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(cg),
    )
    mk = lambda w: ttnn.from_torch(
        torch.zeros(1, 1, M, w),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return pc, mk(N), mk(N // nd)


def matmul_reduce_scatter_decode(
    x, weight, tt_ccl, interm_buf, out_buf, progcfg, compute_cfg, topology, rs_offset=(0, 6)
):
    """Fused row-parallel matmul + reduce-scatter(dim=3) for decode (matmul_reduce_scatter_async).

    x: K-sharded [.,M,K_local]; weight: [K_local,N] K-sharded. Matmul runs on progcfg's (reduced)
    grid; RS workers land at rs_offset (disjoint rows) to avoid the collision that deadlocks a
    full-grid fused CCL. Persistent buffers are caller-owned. Returns [.,M,N/nd] (fractured, DRAM)."""
    _, rs_out = ttnn.experimental.matmul_reduce_scatter_async(
        x,
        weight,
        persistent_intermediate_buffer=interm_buf,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=1,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=progcfg,
        compute_kernel_config=compute_cfg,
    )
    # rs_out IS the persistent output buffer; clone so the caller can deallocate its copy while the
    # persistent buffer survives for the next token (else layer.py's deallocate frees it -> corruption).
    return ttnn.clone(rs_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype):
    """Lazily allocate (and cache on tt_ccl) shared persistent buffers for the prefill fused out-proj.

    Prefill M (=chunk seq, e.g. 2048) makes per-layer buffers huge (fp32 [1,1,2048,5120]≈42MB × 64
    layers = infeasible). Prefill runs layers sequentially and each op's output is cloned before the
    next layer reuses the buffer, so ONE shared set per (M,N,nd,dtype) is safe. Allocated during the
    pre-capture warmup forward (eager), reused inside the trace. Keyed so variable M/dtype coexist."""
    cache = getattr(tt_ccl, "_qwen36_mmrs_prefill_bufs", None)
    if cache is None:
        cache = {}
        tt_ccl._qwen36_mmrs_prefill_bufs = cache
    key = (M, N, nd, str(dtype))
    if key not in cache:
        mesh = tt_ccl.mesh_device
        mk = lambda w: ttnn.from_torch(
            torch.zeros(1, 1, M, w),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cache[key] = (mk(N), mk(N // nd))
    return cache[key]


def matmul_reduce_scatter_prefill(x, weight, tt_ccl, compute_cfg, topology, nd, dtype, grid=(8, 8), rs_offset=(0, 8)):
    """Fused row-parallel out-proj matmul + reduce-scatter for PREFILL (matmul_reduce_scatter_async).

    Unlike decode (M=1, where the 2D matmul collapses to ~8 cores and this loses), at prefill M>>1 the
    2D matmul fills the grid, so overlapping the RS with the matmul is a WIN (biggest for the fp32
    GDN-out with its large RS). grid=(8,8): matmul rows 0-7, RS workers rows 8-9. x: K-sharded
    [.,M,K_local]; weight [K_local,N]. Returns [1,1,M,N/nd].
    Blackhole only. The Wormhole port hangs and wedges ethernet; do not enable it by config."""
    M, K_local = x.shape[-2], x.shape[-1]
    N = weight.shape[-1]
    interm, out_buf = _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype)
    x4 = ttnn.reshape(x, (1, 1, M, K_local))
    # RS-bound: 2 ethernet links parallelize the fp32 cross-device reduce (P150x4 max; traced_8k win).
    # grid (8,8) leaves rows 8-9 for the 2 RS worker rows.
    num_links = 2
    per_core_N = max(1, math.ceil(N / TILE_SIZE / grid[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // grid[0])),
        out_subblock_h=1,
        # Keep 1x1: op242 is RS-bound and this op is pipelined to overlap the matmul with the RS.
        # Widening the subblock desyncs that overlap and measured net-negative on traced_8k TTFT.
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / grid[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(grid),
    )
    _, rs = ttnn.experimental.matmul_reduce_scatter_async(
        x4,
        weight,
        persistent_intermediate_buffer=interm,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=num_links,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        compute_kernel_config=compute_cfg,
    )
    return ttnn.clone(rs, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def sharded_decode_matmul(
    x,
    weight,
    compute_cfg,
    decode_progcfg,
    act_shard_cfg,
    prefill_progcfg_fn,
    prefill_k,
    decode_out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    prefill_compute_cfg=None,
    prefill_out_dtype=None,
):
    """DRAM-WIDTH_SHARDED weight matmul; branches on M (decode vs prefill).

    Decode (M<=32): L1-sharded act + DRAM-sharded kernel. Prefill: 2D matmul.
    Gate on x.shape[-2] (seq/M), not x.shape[1] (Z=1 in both modes). Decode result placement is
    `decode_out_memory_config` (default DRAM-interleaved; pass L1 to keep the small decode
    activation resident). Prefill result is always DRAM-interleaved.
    prefill_compute_cfg must match the prefill progcfg (one K pass needs fp32 dest acc off).
    prefill_out_dtype pins the prefill result; ttnn.linear otherwise inherits in0 dtype."""
    seq = x.shape[-2]
    if seq <= TILE_SIZE:
        # Reshard act to L1 if needed; skip dealloc when x already sharded (GDN reuses x).
        already_sharded = x.memory_config() == act_shard_cfg
        x_sh = x if already_sharded else ttnn.to_memory_config(x, act_shard_cfg)
        out = ttnn.linear(
            x_sh,
            weight,
            compute_kernel_config=compute_cfg,
            program_config=decode_progcfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        if not already_sharded:
            ttnn.deallocate(x_sh)
        return ttnn.to_memory_config(out, decode_out_memory_config)
    pc = prefill_progcfg_fn(seq, prefill_k, weight.shape[-1])
    return ttnn.linear(
        x,
        weight,
        compute_kernel_config=prefill_compute_cfg or compute_cfg,
        program_config=pc,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **({"dtype": prefill_out_dtype} if prefill_out_dtype is not None else {}),
    )


def replicate(torch_tensor, mesh, cache_path, dtype=ttnn.bfloat16):
    """Small tensor (norm/bias) -> replicated on every device."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def shard_small(torch_tensor, mesh, cache_path, dim=-1, dtype=ttnn.bfloat16):
    """Small per-head tensor (conv taps, A_log, dt_bias) -> sharded."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def replicate_kv_weight(weight, n_kv_heads, tp, head_dim):
    """Replicate KV weight so each device gets >=1 head. No-op when tp <= n_kv_heads."""
    if tp <= n_kv_heads:
        return weight
    chunks = weight.reshape(n_kv_heads, head_dim, -1)
    parts = []
    for d in range(tp):
        kv_idx = (d * n_kv_heads) // tp
        parts.append(chunks[kv_idx])
    return torch.cat(parts, dim=0).reshape(tp * head_dim, -1)


# FP8 dequantization
def dequant_fp8_block(weight_fp8, scale_inv, block_size=128):
    """Dequantize a block-wise FP8 weight tensor to bfloat16."""
    out_f, in_f = weight_fp8.shape
    weight_bf16 = weight_fp8.to(torch.bfloat16).reshape(out_f // block_size, block_size, in_f // block_size, block_size)
    weight_bf16 = weight_bf16 * scale_inv[:, None, :, None].to(torch.bfloat16)
    return weight_bf16.reshape(out_f, in_f)


# Weight-prep (reorder HF weights for per-device sharding)
def prepare_attn_qkv(q_w, k_w, v_w, qg_per, kv_per, tp):
    """Fuse attn q+gate/k/v for column-parallel shard: each device gets [qg_d|k_d|v_d].

    q_w: [n_heads*head_dim*2, in]; k_w/v_w: [n_kv_heads*head_dim, in].
    qg_per/kv_per: per-device out block sizes."""
    parts = []
    for d in range(tp):
        parts.append(q_w[d * qg_per : (d + 1) * qg_per, :])
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
    return torch.cat(parts, dim=0)


def prepare_attn_qkv_deint(q_w, k_w, v_w, nh_local, hd, kv_per, tp):
    """Like prepare_attn_qkv but de-interleaves [q,g] per head -> [all_q|all_gate|k|v] per device.

    Avoids prefill relayout in _make_heads (column perm only; numerically identical).
    q_w: [nh_total*hd*2, in]; nh_local/kv_per: per-device block sizes."""
    hd2 = hd * 2
    parts = []
    for d in range(tp):
        base = d * nh_local * hd2
        q_rows = [q_w[base + h * hd2 : base + h * hd2 + hd, :] for h in range(nh_local)]
        g_rows = [q_w[base + h * hd2 + hd : base + h * hd2 + hd2, :] for h in range(nh_local)]
        # Per-device layout [all_q | k | v | all_gate]: q/k/v contiguous so _make_heads* can hand
        # the fused q|k|v block straight to nlp_create_qkv_heads (no re-concat); gate trails, applied
        # post-SDPA. (Column perm only; numerically identical to [q|gate|k|v].)
        parts.append(torch.cat(q_rows, dim=0))  # all_q
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(torch.cat(g_rows, dim=0))  # all_gate (last)
    return torch.cat(parts, dim=0)


def prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp):
    """Interleave GDN Q/K/V heads for row-parallel shard (contiguous q/k/v block per device).

    qkv_w: [key_dim*2 + value_dim, hidden]."""
    q_part = qkv_w[:key_dim, :]
    k_part = qkv_w[key_dim : 2 * key_dim, :]
    v_part = qkv_w[2 * key_dim :, :]

    q_per = nk // tp
    v_per = nv // tp
    shards = []
    for s in range(tp):
        q_s = q_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        k_s = k_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        v_s = v_part[s * v_per * dv : (s + 1) * v_per * dv, :]
        shards.append(torch.cat([q_s, k_s, v_s], dim=0))
    return torch.cat(shards, dim=0)


def tuned_vocab_all_gather(
    input_tensor, mesh_device, tt_ccl, dim, topology, num_workers_per_link, chunks_per_sync, dtype=ttnn.bfloat16
):
    """Vocab all-gather with tunable workers. dtype must match the logits; the drafter gathers fp32."""
    if list(mesh_device.shape) == [1, 1]:
        return input_tensor
    num_links = tt_ccl.get_num_links(None)
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
    if input_tensor.dtype != dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, dtype)
    gathered = ttnn.experimental.all_gather_async(
        input_tensor,
        persistent_output_buffer=None,
        dim=dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=num_links,
        topology=topology,
        memory_config=None,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=2,
        subdevice_id=None,
    )
    input_tensor.deallocate(True)
    return gathered


def fc_decode_program_config(mesh_device, k, n):
    """1-tile-tall decode matmul config for k >> n, or None when the shape does not fit."""
    kt, nt = k // 32, n // 32
    if k % 32 or n % 32 or kt == 0 or nt == 0:
        return None
    grid = mesh_device.compute_with_storage_grid_size()
    per_core_n = 2 if nt % 2 == 0 else 1
    cores = nt // per_core_n
    if cores > grid.x * grid.y:
        return None
    in0_block_w = max((d for d in range(1, 33) if kt % d == 0), default=1)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True)
        .bounding_box()
        .grid_size(),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def tiny_all_gather(input_tensor, mesh_device, tt_ccl, dim, topology, dtype):
    """All-gather a one-element-per-device tensor. Deallocates input_tensor."""
    num_links = tt_ccl.get_num_links(None)
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
    if input_tensor.dtype != dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, dtype)
    gathered = ttnn.experimental.all_gather_async(
        input_tensor,
        persistent_output_buffer=None,
        dim=dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=num_links,
        topology=topology,
        memory_config=None,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=1,
        num_workers_per_link=1,
        num_buffers_per_channel=2,
        subdevice_id=None,
    )
    input_tensor.deallocate(True)
    return gathered


def vocab_shard_offsets(mesh_device, num_devices, shard_width):
    """Replicated offsets that turn shard-local argmaxes into global vocab ids."""
    off = ttnn.Tensor(
        [float(d * shard_width) for d in range(num_devices)],
        [1, 1, 1, num_devices],
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    return ttnn.to_layout(ttnn.to_device(off, mesh_device), ttnn.TILE_LAYOUT)


def greedy_pick(logits, mesh_device, tt_ccl, topology, shard_offsets=None, vocab_size=None):
    """Greedy argmax of one logit row. With shard_offsets, reduce per shard and combine scalars; ties take the lowest global id."""
    if shard_offsets is None:
        u = ttnn.untilize(logits, use_multicore=True)
        out = ttnn.argmax(u, dim=-1, keepdim=False)
        ttnn.deallocate(u)
        return out

    num_devices = int(shard_offsets.shape[-1])
    shard_width = int(logits.shape[-1])
    # Evenly fractured vocab only; a partial last shard would need a per-device valid width.
    assert (
        shard_width * num_devices == vocab_size
    ), f"shard argmax needs an evenly fractured head: {shard_width} x {num_devices} != {vocab_size}"

    vmax = ttnn.max(logits, dim=-1, keepdim=True)
    u = ttnn.untilize(logits, use_multicore=True)
    idx = ttnn.argmax(u, dim=-1, keepdim=True)
    ttnn.deallocate(u)
    idxf = ttnn.typecast(idx, ttnn.float32)  # fp32 holds a vocab index exactly
    ttnn.deallocate(idx)
    idxf = ttnn.to_layout(ttnn.reshape(idxf, (1, 1, 1, 1)), ttnn.TILE_LAYOUT)

    kw = dict(mesh_device=mesh_device, tt_ccl=tt_ccl, dim=3, topology=topology, dtype=ttnn.float32)
    v8 = tiny_all_gather(vmax, **kw)
    i8 = tiny_all_gather(idxf, **kw)

    g8 = ttnn.add(i8, shard_offsets)
    best = ttnn.max(v8, dim=-1, keepdim=True)
    win = ttnn.eq(v8, best)
    # vocab_size is past every valid id, so min cannot return it unless nothing won.
    cand = ttnn.where(win, g8, float(vocab_size))
    tok = ttnn.min(cand, dim=-1, keepdim=True)
    for t in (v8, i8, g8, best, win, cand):
        ttnn.deallocate(t)
    out = ttnn.reshape(ttnn.typecast(ttnn.to_layout(tok, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32), (1, 1, 1))
    ttnn.deallocate(tok)
    return out


def prepare_conv_taps(conv_w, key_dim, nk, dk, nv, dv, kernel_size, tp):
    """Split fused conv1d into kernel taps, reordered to match prepare_gdn_qkv grouping."""
    cw = conv_w.float()
    q_per = nk // tp
    v_per = nv // tp
    taps = []
    for j in range(kernel_size):
        tap = cw[:, 0, j]
        q_tap = tap[:key_dim]
        k_tap = tap[key_dim : 2 * key_dim]
        v_tap = tap[2 * key_dim :]
        shards = []
        for s in range(tp):
            q_s = q_tap[s * q_per * dk : (s + 1) * q_per * dk]
            k_s = k_tap[s * q_per * dk : (s + 1) * q_per * dk]
            v_s = v_tap[s * v_per * dv : (s + 1) * v_per * dv]
            shards.append(torch.cat([q_s, k_s, v_s]))
        taps.append(torch.cat(shards))
    return taps
