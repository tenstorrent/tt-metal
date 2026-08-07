# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP helpers for Qwen3.5/3.6 on Blackhole (9B single-device + 27B TP=4 / TP=8).

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


# Compute kernel configs
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


# Grid helpers
def prefill_grid_default():
    """BH P150: (8,10); WH: (8,8). y capped at 10 on BH (grid_x=10 breaks matmul)."""
    return (8, 10) if is_blackhole() else (8, 8)


# Max grid COLUMNS a tuned prefill config may use. A Blackhole galaxy reports a 12-wide worker
# grid, but harvested P150s expose only 11, so tuning to 12 would not port. 11 x 10 = 110 cores.
PREFILL_MAX_COLS_PORTABLE = 11

# Why TP=8 wants different values (measured at S=2048, 27B, 1x8 Ring):
#   * widest_cols -- `_best_prefill_cols` ranks candidate widths by (out_subblock_w, cols), i.e.
#     subblock first. At TP=8 the halved N makes wide grids yield a small per_core_N and hence a
#     narrow subblock, so that ranking retreats to fewer columns and leaves cores idle. Measured
#     device time is monotonically decreasing in column count instead: attn_wo went 1944us @ 60
#     cores -> 700us @ 110, and mlp_gate 2943us @ 60 -> 1935us @ 110. So take the width.
#   * in0_block_w_divisor -- `min(cap, k_tiles // grid_x)` is a function of the per-device K, which
#     halves. attn_wo/gdn_out go k_tiles 48 -> 24 and `24 // 11 = 2`, but in0_block_w only has to
#     DIVIDE k_tiles, so a larger block is legal and much faster (attn_wo @ 11 cols, from the sweep:
#     bw2 786us, bw4 719us, bw6 700us, bw8 705us).
#
# in0_block_w_cap is L1-BOUND, NOT just a legality bound. in0_block_w sizes the in0 circular
# buffer, and `_wo_proj` / the MLP prefill arm write their OUTPUT to L1 (attention/tp.py:246,
# mlp.py:284) -- so the CBs and a resident L1 output tensor compete for the same 1536 KB. Measured
# on the real model: cap=8 overflows and test_model_tp_long_prefill dies with
#   "Statically allocated circular buffers in program N clash with L1 buffers on core range
#    [0-0 - 10-8]. L1 buffer allocated at 1314560 and static circular buffer region ends at 1372032"
# from attention/tp.py:241. A standalone per-op sweep CANNOT see this: in isolation the only L1
# tenant is the op under test, so it reports a win that the full model has no room for. Any future
# raise of this cap must be validated by test_model_tp_long_prefill, not by the sweep alone.
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


def create_matmul_1d_decode_progcfg(m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8):
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
    per_core_k = _find_largest_divisor(k_tiles)
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
    activation first since mcast_in0 needs the full K per core. See test_mlp_matmul_sweep."""
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


# 2D prefill matmul config
def _get_out_subblock_w(per_core_n, out_subblock_h):
    for w in range(min(per_core_n, 4 // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _full_grid_crs(grid):
    """Full-grid allowed_worker_cores for CCL-fused matmuls, which bypass ttnn::prim::matmul()'s normalize_program_config()."""
    gx, gy = grid
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def _safe_half_out_block_w(per_core_N, out_subblock_w):
    """Largest divisor of per_core_N, itself a multiple of out_subblock_w, that is <= per_core_N // 2.

    Halves (at least) the output/intermediate CB footprint vs. the default out_block_w=per_core_N.
    Falls back to out_subblock_w (always a valid divisor of per_core_N by construction) if no smaller
    multiple divides evenly."""
    half = max(1, per_core_N // 2)
    best = out_subblock_w
    for w in range(out_subblock_w, half + 1, out_subblock_w):
        if per_core_N % w == 0:
            best = w
    return best


def prefill_kpass_width(n, grid_size=None, max_waste=0.20):
    """Smallest width >= n whose TILE count minimises the prefill matmul's K-pass count.

    A 2D prefill matmul walks its per-core output in ceil(per_core_N / out_block_w) N-blocks and
    re-traverses K once per block -- re-reading a DRAM-resident in0 each time. MEASURED on N300
    (M=2048, K=4096, HiFi2 BF16 x BFP8, 8x8 grid) the cost tracks that pass count almost exactly and
    is NOT explained by the output subblock:

        N=6176 (193 tiles, prime) per_core_N=25 out_block_w=5   5 passes  2631us
        N=6528 (204 tiles)        per_core_N=26 out_block_w=2  13 passes  4194us  +59%
        N=6912 (216 tiles)        per_core_N=27 out_block_w=9   3 passes  1906us  -28%
        N=7168 (224 tiles)        per_core_N=28 out_block_w=4   7 passes  2938us  +12%

    The trap is that out_block_w must be a MULTIPLE of out_subblock_w, so _get_out_subblock_w's greedy
    "widest subblock" can force a narrow block and MORE passes -- N=7168 gets the ideal 1x4 subblock
    and is still slower than N=6912's 1x3. A prime tile count (193) is worst of all: its only divisors
    are 1/5/25, and 25 (one pass) overflows L1.

    So: pad the width until the tile count factors well. Candidates are multiples of grid_size[0] (so
    per_core_N divides exactly and every column of the grid is used), within max_waste extra tiles.
    Scored by (passes, waste) -- fewest K passes, then least padding. Returns n unchanged when nothing
    beats it, so callers can use it unconditionally."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    cols = grid_size[0]
    n_tiles = math.ceil(n / TILE_SIZE)

    def passes_for(tiles):
        per_core_N = math.ceil(tiles / cols)
        sub_w = _get_out_subblock_w(per_core_N, 1)
        blk_w = _safe_half_out_block_w(per_core_N, sub_w)
        return math.ceil(per_core_N / blk_w)

    best = (passes_for(n_tiles), 0, n_tiles)
    for tiles in range(_roundup(n_tiles, cols), int(n_tiles * (1 + max_waste)) + 1, cols):
        cand = (passes_for(tiles), tiles - n_tiles, tiles)
        if cand < best:
            best = cand
    return best[2] * TILE_SIZE


def create_prefill_matmul_program_config(
    m, k, n, grid_size=None, fused_activation=None, tuning=None, out_block_w=None, halve_out_block=False
):
    """2D prefill matmul progcfg (DRAM-interleaved).

    fused_activation in packer; sharded kernel rejects ttnn.linear(activation=...) with progcfg.
    tuning: a `_PREFILL_TUNING` entry (see `prefill_tuning`); None = the frozen TP=4 behavior.
    out_block_w: when set (< per_core_N), the output/intermediate CB only needs to hold one
    out_block_w-wide slice of the per-core output at a time instead of the full per_core_N width —
    same lever already used by build_mmrs_decode_state/matmul_reduce_scatter_prefill below.
    halve_out_block: auto-derive a safe halved out_block_w (see _safe_half_out_block_w) instead of
    passing one explicitly. Needed on grids that are already at their physical max (WH tops out at
    8x8=64 cores vs BH's 8x10=80, with a smaller per-core L1 budget besides), where per_core_M/N can't
    be shrunk further by adding more cores."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)

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


def prefill_out_memory_config(seq_len, out_width, elem_bytes=2, budget=8 << 20):
    """L1 for prefill matmul outputs that fit; DRAM for the big ones.

    Several prefill projections (MLP down-proj, attention wo, ...) were tuned to emit into L1 — a
    real win measured on Blackhole, whose L1 is larger. Those outputs are [seq_len, out_width] and
    grow with the prefill chunk (16MB+ at seq_len=2048, out_width=4096, bf16). On Wormhole that
    leaves so little L1 free that the very matmul producing them can no longer place its own
    statically-allocated circular buffers, and the op dies with "clash with L1 buffers" — CBs are
    L1-only, so the output is what has to move. Blackhole keeps the tuned L1 path unchanged."""
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
    m, k, n, fused_activation=None, max_cols=None, tuning=None, halve_out_block=False
):
    """FPU-tuned 2D prefill progcfg for MLP matmuls: picks the grid width that maximizes the output
    subblock (drives prefill FPU) instead of the default full width.

    max_cols caps the grid width. Default = prefill_grid_default()[0] (8). Pass the device worker-grid
    width (11 on BH P150) to let the subblock heuristic go wide -> the measured prefill winners
    (gate 9-wide, down/wo 10-wide, gdn_qkvz 11-wide; test_mlp_matmul_sweep_prefill). Fused AG/RS paths
    pin 8-wide separately and are unaffected.

    tuning: a `_PREFILL_TUNING` entry. With `widest_cols` (TP=8) the subblock-first width heuristic
    is replaced by "take the width, clamped to PREFILL_MAX_COLS_PORTABLE" -- measured device time at
    TP=8 falls monotonically with column count, so trading cores for a wider subblock loses.
    halve_out_block: see create_prefill_matmul_program_config — pass on grids already at their
    physical core-count max (WH) where the full per_core_N-wide output/intermediate CB overflows L1."""
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
    )


# Mesh tensor helpers
def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
    """Torch weight [out,in] -> sharded mesh tensor. Transpose to [in,out]; dim=-1 column, dim=0 row."""
    w = torch_tensor.to(torch.bfloat16).T.contiguous()
    return ttnn.as_tensor(
        w,
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
):
    """Fused all-gather(dim=3) + column-parallel matmul for prefill (all_gather_minimal_matmul_async).

    x: K-sharded activation [.,S,K/tp]; weight: [K,N] col-sharded (K full). Gathers x to full K and
    matmuls in one op, replacing a separate all_gather + linear. fused_activation applied per tile
    before pack (non-parametrized op, e.g. ttnn.UnaryOpType.SILU). out_memory_config places the result
    (default DRAM; L1 keeps it resident for downstream slices)."""
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
    )[0]

    return out


def prefill_ccl_tuning():
    """(chunks_per_sync, num_workers_per_link) for the PREFILL collectives.

    tt_all_reduce() takes these as arguments defaulting to 10 / 2, and qwen36 never passed them;
    DistributedNorm hardcodes the same 10 / 2 for every non-decode call. So unlike their decode
    counterparts (which get per-op configs from model_config) the prefill all-gathers and
    reduce-scatters have never been tuned at all. All four run on 5 cores at 6.8-8.1 GB/s against a
    ~12.5 GB/s Wormhole link -- 55-65% efficiency, i.e. real headroom.

    Used by BOTH prefill collectives: the reduce-scatters (tt_all_reduce call sites in gdn/tp.py and
    mlp.py) and the all-gathers (via tt/prefill_norm_tuned.py).

    MEASURED on N300 at seq 2048, device times straight out of tt-perf-report:

      ALL-GATHER -- the reliable win. Per-op, 2 runs each:
        wpl=2 (upstream)  1,245 / 1,242 us
        wpl=4             1,012 / 1,012 us      <- used
        wpl=8             1,015 / 1,016 us
        => ~-460us per layer across the two gathers, and repeatable to ~10us.

      REDUCE-SCATTER -- smaller and noisy. Layer RS total, 3 runs each:
        wpl=2 (upstream)  2,234 / 2,478 / 2,076   mean 2,263, spread 402
        wpl=4             1,995 / 2,063 / 2,037   mean 2,032, spread  68
        => ~-230us on the mean, and it tightens the spread ~6x, which matters more for tail latency
           than the mean does. Individual RS ops still range 963-1,277us run to run, so do not read
           much into any single profile.

      wpl=8 is indistinguishable from 4 once both collectives are tuned (CCL mean 4,313 vs 4,278 us
      over 2-3 runs each), so 4 stays -- it is the smaller departure from the upstream default.
      chunks_per_sync made no measurable difference anywhere and stays at 10.

    QWEN35_PREFILL_CCL="cps,wpl" overrides, for re-sweeping."""
    _v = os.environ.get("QWEN35_PREFILL_CCL")
    if _v:
        _c, _w = (int(t) for t in _v.split(","))
        return _c, _w
    return 10, 4


def mlp_gateup_agmm_enabled(num_devices):
    """Fuse the ff_norm all-gather into the MLP gate/up matmul (prefill). TP-only (needs the gather).

    BH-only: all_gather_swiglu_prefill's grid assumes BH's taller (9-10 row) compute grid; WH tops
    out at 8 rows, so this fusion is unvalidated there. Falls back to the unfused AG + matmul path on WH.

    MEASURED on N300 (2026-08): the row count is NOT the only blocker, so do not just clamp the grid.
    With grid height forced to 8, all_gather_minimal_matmul_async's program factory builds its in0/in1
    sender+receiver core ranges from grid_size.y-1/-2/-3; at y=8 those overlap so two data-movement
    kernels land on one core needing both NOCs, and program creation dies with
    "TT_FATAL ... local_noc0_in_use and local_noc1_in_use" (tt_metal.cpp:152). Reproduced at num_links
    1 AND 2, so it is not a link-count artifact — enabling WH needs a C++ change to that op's core/NOC
    assignment for 8-row grids. Worth doing: the two all-gathers this would hide are 1,239us + 1,242us
    of a 21,669us single-layer GDN prefill at seq 2048 (5 cores each, fully exposed)."""
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
    [.,M,K_local]; weight [K_local,N]. Returns [1,1,M,N/nd] (cloned; shared buffer survives).

    BH-ONLY. Porting this to Wormhole was tried and reverted (N300, 2026-08). It looks like the one CCL
    fusion that should port, because this op takes an explicit reduce_scatter_core_grid_offset, so WH's
    8-row grid is expressible as matmul (8,6) on rows 0-5 + RS workers at (0,6) on rows 6-7 — the exact
    disjoint split build_mmrs_decode_state already uses for the DECODE out-proj. It HANGS anyway: the
    op never returns and the hang wedges the ethernet cores, after which the next device open fails
    with "Timed out waiting for ETH heartbeat ... Stuck at 0xaabb0001". Recovery is
    `tt-topology -l mesh` — note this host's layout is MESH; the tool's default is linear and flashing
    that breaks device discovery entirely.

    Reproduced twice, so it is not the link count: first at num_links=2, then at num_links=1 after
    finding that an N300 (1,2) submesh reports get_num_links() == 1 on every axis while the value below
    is hardcoded to 2 (a P150x4 number). Same symptom both times. Likely the same class of defect as
    the all-gather fusion (see mlp_gateup_agmm_enabled) — the fused-CCL program factories assume a
    taller grid than WH has, and a decode config that works at M=1 does not carry to a prefill M that
    fills the matmul grid. Needs C++ investigation, not a config change. Cost of leaving it off: the
    two reduce-scatters are ~1,060us + ~995us of an 18,644us single-layer GDN prefill at seq 2048."""
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
):
    """DRAM-WIDTH_SHARDED weight matmul; branches on M (decode vs prefill).

    Decode (M<=32): L1-sharded act + DRAM-sharded kernel. Prefill: 2D matmul.
    Gate on x.shape[-2] (seq/M), not x.shape[1] (Z=1 in both modes). Decode result placement is
    `decode_out_memory_config` (default DRAM-interleaved; pass L1 to keep the small decode
    activation resident). Prefill result is always DRAM-interleaved."""
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
        x, weight, compute_kernel_config=compute_cfg, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
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
