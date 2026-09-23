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

# SP-prefill experiment (QWEN36_SP_PROJ_LOFI=1, sequence_parallel + num_devices==1 only): LoFi on
# the GDN/attention prefill in-proj/out-proj matmuls. Same as COMPUTE_HIFI2 otherwise.
COMPUTE_LOFI_PROJ = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
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
# TP=1 (sequence_parallel, num_devices==1) values are from a one-die sweep of the actual MLP
# prefill shapes (K=2048/6144, N=2048/5120/6144/8224) on the model's own compute config: 2026-09-20
# one-die sweep, scratchpad/mm/results.md. Unlike TP=4/8, the full K never splits across devices, so
# k_tiles (64 or 192) is always divisible by 8 -- in0_block_w_cap=8 is legal and faster than the TP=4
# cap=4. `rows=8` pins the grid height to the 8 rows that per_core_M=4 (32 M-tiles / 8) actually uses,
# instead of leaving 2 of the default 10 rows idle. `prefer_square_subblock` lets
# `create_prefill_mlp_matmul_program_config` try a (2,2) output subblock (legal: fp32_dest_acc_en
# caps subblock area at 4, and both 2,2 factors are area-4) when the plain wide-column search would
# otherwise land on a smaller subblock -- see `_square_prefill_cols`.
_PREFILL_TUNING = {
    1: dict(widest_cols=False, in0_block_w_divisor=True, in0_block_w_cap=8, rows=8, prefer_square_subblock=True),
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


def create_prefill_matmul_program_config(
    m, k, n, grid_size=None, fused_activation=None, tuning=None, out_subblock=None
):
    """2D prefill matmul progcfg (DRAM-interleaved).

    fused_activation in packer; sharded kernel rejects ttnn.linear(activation=...) with progcfg.
    tuning: a `_PREFILL_TUNING` entry (see `prefill_tuning`); None = the frozen TP=4 behavior.
    out_subblock: optional (h, w) override (see `_square_prefill_cols`); None = the default (1, w)
    search below."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))

    if out_subblock is not None:
        out_subblock_h, out_subblock_w = out_subblock
    else:
        out_subblock_h = 1
        out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)

    k_tiles = math.ceil(k / TILE_SIZE)
    cap = tuning["in0_block_w_cap"]
    if tuning["in0_block_w_divisor"]:
        # in0_block_w only has to divide k_tiles (no K tail in the 2D mcast kernel), so take the
        # largest legal block rather than scaling with grid width -- see _PREFILL_TUNING.
        in0_block_w = _find_largest_divisor(k_tiles, cap)
    else:
        # The 2D mcast matmul kernel asserts Kt % in0_block_w == 0, but k_tiles // grid_x need not
        # divide k_tiles -- at TP=2 (k_tiles 32, grid_x 9) it yields 3 and the op dies with
        #   matmul_device_operation.cpp:499  Kt (32) must be divisible by in0_block_w (3)
        # which is why TP=2 could not run at all (SP=16 x TP=2, SP=4 x TP=2). Snap DOWN to the
        # largest divisor of k_tiles, which is always legal and never larger than the tuned value.
        in0_block_w = _find_largest_divisor(k_tiles, min(cap, max(1, k_tiles // grid_size[0])))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
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


def _square_prefill_cols(m, n, limit, rows):
    """TP=1 counterpart to `_best_prefill_cols`/`_widest_prefill_cols` (see `_PREFILL_TUNING[1]`).

    Goes straight to the widest portable grid and checks whether a (2,2) output subblock reaches the
    fp32_dest_acc_en area-4 cap when the plain (1, w<=4) search cannot (e.g. per_core_N=18 -> w=3 with
    h=1, but 18 is even so (2,2) reaches area 4). Requires per_core_M and per_core_N both even. When
    neither the plain nor the square subblock reach area 4, there's no win from going wide, so this
    falls back to `_best_prefill_cols`'s narrower, subblock-maximizing search (e.g. N=5120 -> per_core_N
    15 at 11 cols is odd, but 16 at 10 cols is a clean (1,4)).

    Returns (cols, out_subblock | None); None means "let the caller's default (1, w) search decide"."""
    n_tiles = math.ceil(n / TILE_SIZE)
    m_tiles = math.ceil(m / TILE_SIZE)
    per_core_M = max(1, math.ceil(m_tiles / rows))
    cols = max(1, min(limit, PREFILL_MAX_COLS_PORTABLE, n_tiles))
    per_core_N = max(1, math.ceil(n_tiles / cols))
    if _get_out_subblock_w(per_core_N, 1) == 4:
        return cols, None
    if per_core_M % 2 == 0 and per_core_N % 2 == 0:
        return cols, (2, 2)
    return _best_prefill_cols(n, limit), None


def create_prefill_mlp_matmul_program_config(m, k, n, fused_activation=None, max_cols=None, tuning=None):
    """FPU-tuned 2D prefill progcfg for MLP matmuls: picks the grid width that maximizes the output
    subblock (drives prefill FPU) instead of the default full width.

    max_cols caps the grid width. Default = prefill_grid_default()[0] (8). Pass the device worker-grid
    width (11 on BH P150) to let the subblock heuristic go wide -> the measured prefill winners
    (gate 9-wide, down/wo 10-wide, gdn_qkvz 11-wide; test_mlp_matmul_sweep_prefill). Fused AG/RS paths
    pin 8-wide separately and are unaffected.

    tuning: a `_PREFILL_TUNING` entry. With `widest_cols` (TP=8) the subblock-first width heuristic
    is replaced by "take the width, clamped to PREFILL_MAX_COLS_PORTABLE" -- measured device time at
    TP=8 falls monotonically with column count, so trading cores for a wider subblock loses. With
    `prefer_square_subblock` (TP=1) cols/out_subblock instead come from `_square_prefill_cols`.
    `rows`, if present, overrides the grid height (default prefill_grid_default()[1])."""
    grid = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    limit = max_cols or grid[0]
    rows = tuning.get("rows", grid[1])
    out_subblock = None
    if tuning.get("prefer_square_subblock"):
        cols, out_subblock = _square_prefill_cols(m, n, limit, rows)
    elif tuning["widest_cols"]:
        # Cap the width at PREFILL_MAX_COLS_PORTABLE (harvested parts expose 11, not 12) and never
        # exceed the output tile count -- columns beyond it get per_core_N=1 with nothing to compute,
        # paying mcast cost for no work.
        cols = _widest_prefill_cols(n, max(1, min(limit, PREFILL_MAX_COLS_PORTABLE, math.ceil(n / TILE_SIZE))))
    else:
        cols = _best_prefill_cols(n, limit)
    return create_prefill_matmul_program_config(
        m, k, n, grid_size=(cols, rows), fused_activation=fused_activation, tuning=tuning, out_subblock=out_subblock
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


def ccl_cast(x):
    """QWEN36_CCL_BF8=1: narrow a row-parallel partial to bfloat8_b before the reduce-scatter.

    tt_all_reduce's `dtype` argument only takes effect on the TG all-reduce branch; the
    1-D-mesh branch feeds reduce_scatter_minimal_async the tensor as-is, so the wire format is
    whatever the matmul produced (bf16). The reduce-scatters are ~13% of device time and ~25%
    of the critical path at SP=4 x TP=8, and they sit at the 2-link ethernet roofline, so bytes
    are the only lever. bf8 halves them.

    Precision caveat: this narrows PARTIAL SUMS, which is exactly where the GDN out-proj is
    known to be sensitive -- tp_common's fused path keeps fp32 there because bf16 took PCC to
    ~0.69. Measure PCC, do not assume.
    """
    if os.environ.get("QWEN36_CCL_BF8") != "1":
        return x
    return ttnn.typecast(x, ttnn.bfloat8_b) if x.dtype != ttnn.bfloat8_b else x


def agmm_disabled():
    """QWEN36_NO_AGMM=1 turns off every all_gather_minimal_matmul_async fusion.

    Needed to run under FABRIC_2D: that op's dm_in0_sender kernel is written against the
    linear (1D) fabric API and fails to compile under 2D ("template argument deduction/
    substitution failed" from tt_metal/fabric/hw/inc/linear/api.h). Dropping the fusion falls
    back to a separate all_gather + matmul, which was measured e2e-neutral at TP=8
    (51.15 ms with the fusion vs 51.15 without), so this buys 2D routing at no known cost.
    """
    # A replicated residual has nothing to gather, so every AGMM fusion is not just useless
    # but wrong there (it would double-gather). Fold that into this one predicate.
    return os.environ.get("QWEN36_NO_AGMM") == "1" or repl_residual_enabled()


def mlp_gateup_agmm_enabled(num_devices):
    """Fuse the ff_norm all-gather into the MLP gate/up matmul (prefill). TP-only (needs the gather)."""
    return num_devices > 1 and not agmm_disabled()


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


def mmrs_prefill_supported(device, nd, rs_offset=(0, 8)):
    """Whether the fused prefill out-proj matmul_reduce_scatter can be PLACED on this device.

    The op selects 36 reduce-scatter worker cores (3 full rows on a 12-wide Blackhole grid) and
    places them at rs_offset, above the matmul's own (8,8) grid. A 10-row Blackhole leaves only
    2 rows there, so placement fails at every TP on this SKU:

        ccl_common.cpp:562 Core grid offset 0-8 pushed 12 of the 36 selected worker cores
        (first: 0-10) off the worker grid; kernels cannot be placed there.

    Ring size is NOT the discriminator -- the same 36 cores are selected at nd=4 and nd=8.
    Falling back to the unfused row-parallel matmul + tt_all_reduce is correct and works.
    """
    rows_free = max(0, device.compute_with_storage_grid_size().y - rs_offset[1])
    return rows_free >= 3


def matmul_reduce_scatter_prefill(x, weight, tt_ccl, compute_cfg, topology, nd, dtype, grid=(8, 8), rs_offset=(0, 8)):
    """Fused row-parallel out-proj matmul + reduce-scatter for PREFILL (matmul_reduce_scatter_async).

    Unlike decode (M=1, where the 2D matmul collapses to ~8 cores and this loses), at prefill M>>1 the
    2D matmul fills the grid, so overlapping the RS with the matmul is a WIN (biggest for the fp32
    GDN-out with its large RS). grid=(8,8): matmul rows 0-7, RS workers rows 8-9. x: K-sharded
    [.,M,K_local]; weight [K_local,N]. Returns [1,1,M,N/nd] (cloned; shared buffer survives)."""
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


def repl_residual_enabled():
    """QWEN36_REPL_RESIDUAL=1: keep the residual stream REPLICATED instead of hidden-fractured.

    Today each half-layer pays three collectives: reduce_scatter (leaving the residual fractured
    on the hidden dim) -> norm stats all_gather -> norm output all_gather. The fracture is what
    forces the DistributedNorm, and the norm's output gather is what re-materialises the full
    hidden dim that the next column-parallel matmul needs anyway.

    Measured on 8 chips, 1024x2048 bf16, traced and in isolation:
        reduce_scatter 117us + stats all_gather 48us + output all_gather 123us = 288us
        fused ttnn.all_reduce                                                  = 223us
    and collectives are a near-FIXED ~120us each regardless of bytes (an 8x larger message costs
    the same), so cutting the COUNT from 3 to 1 is the lever, not cutting the size.

    Replicating costs activation memory, which is why the fractured layout exists -- but this is
    a 2B model on 189 MB of L1 per chip, so that tradeoff does not bind here.
    """
    return os.environ.get("QWEN36_REPL_RESIDUAL") == "1"


def residual_all_reduce(x, mesh_device, tt_ccl, cluster_axis, dim, topology, memory_config=None, num_links=None):
    """Row-parallel partial -> summed activation, as either a fused all-reduce or today's RS.

    With repl_residual_enabled() this returns a REPLICATED full-hidden tensor (one collective);
    otherwise it falls back to tt_all_reduce, which reduce-scatters and returns a tensor
    fractured on `dim`. Callers must not assume a shape: the two differ by design, and
    is_distributed_norm() is gated on the same predicate so the norms match the layout.
    """
    from models.tt_transformers.tt.ccl import tt_all_reduce

    if not repl_residual_enabled():
        return tt_all_reduce(
            x,
            mesh_device,
            tt_ccl,
            cluster_axis=cluster_axis,
            dim=dim,
            topology=topology,
            memory_config=memory_config,
        )
    if num_links is None:
        num_links = tt_ccl.get_num_links(cluster_axis)
    # Axis convention differs between the two ops and getting it wrong is silent. tt_all_reduce
    # IGNORES cluster_axis on a flat mesh (it has a `1 in mesh.shape` branch that reduces across
    # the whole line), which is why every caller here passes cluster_axis=0 even on a (1,N)
    # submesh -- where axis 0 has extent 1. ttnn.all_reduce does NOT ignore it and would reduce
    # over a single device, i.e. return the unreduced partial. Pass None on a flat mesh, which is
    # its documented "reduce across the line" form.
    flat = 1 in list(mesh_device.shape)
    return ttnn.all_reduce(x, cluster_axis=None if flat else cluster_axis, num_links=num_links, topology=topology)


def atupe_opts_enabled():
    """QWEN36_ATUPE_OPTS=0 turns off every optimisation ported from atupe/qwen35-sp-prefill.

    Aniruddha's round-4 changes (KDA fused conv, SDPA q=128/k=256 on 8x8, L1-resident eltwise
    intermediates) were all gated on ``self.mesh.get_num_devices() == 1`` -- his branch is the
    4-die QuietBox config, SP=4 x TP=1. On our SP=4 x TP=8 Galaxy each span's submesh is
    ``MeshShape(1, 8)``, so as written every one of those gates is False and the code is inert.
    Nothing in the optimisations themselves is TP=1-specific (the conv taps are already
    per-device sharded by shard_small, and the intermediates only get SMALLER at TP>1), so this
    predicate replaces the device-count test. **Default OFF until the SP=4 x TP=8 A/B verifies
    it** (same discipline as QWEN36_REPL_RESIDUAL); unset is byte-identical to before the port,
    which is what makes the A/B clean. The A/B run sets ``=1`` explicitly. The per-feature opt-outs he added
    (QWEN36_SP_KDA_CONV, QWEN36_SP_L1_RES, QWEN36_SP_SDPA_LEGACY) still work underneath it.
    """
    return os.environ.get("QWEN36_ATUPE_OPTS", "0") == "1"


def kda_channel_chunk(channels):
    """channel_chunk_size for qkv_causal_conv1d_silu at this device's Q+K+V width.

    The op validates: chunk % 32 == 0, chunk <= channels, and channels % chunk == 0 (it must
    divide exactly). Aniruddha's probe used 512 at C=6144 (tp=1, 12 chunks). At tp=8 the 2B's
    per-device width is 768 and 512 does not divide it. Default = the largest tile-aligned
    divisor <= 512, which gives 384 at C=768 and reproduces his 512 at C=6144.
    QWEN36_SP_KDA_CHUNK overrides for sweeps; it is validated against the same three rules so a
    bad value fails here with a readable message rather than as a TT_FATAL inside the op.
    """
    ov = os.environ.get("QWEN36_SP_KDA_CHUNK")
    if ov:
        c = int(ov)
        assert (
            c > 0 and c % 32 == 0 and c <= channels and channels % c == 0
        ), f"QWEN36_SP_KDA_CHUNK={c} invalid for C={channels}: need %32==0, <=C, and C%chunk==0"
        return c
    start = (min(512, channels) // 32) * 32
    for c in range(start, 31, -32):
        if channels % c == 0:
            return c
    return 32
