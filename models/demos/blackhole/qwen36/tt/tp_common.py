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


def create_prefill_matmul_program_config(m, k, n, grid_size=None, fused_activation=None, tuning=None):
    """2D prefill matmul progcfg (DRAM-interleaved).

    fused_activation in packer; sharded kernel rejects ttnn.linear(activation=...) with progcfg.
    tuning: a `_PREFILL_TUNING` entry (see `prefill_tuning`); None = the frozen TP=4 behavior."""
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


def create_prefill_mlp_matmul_program_config(m, k, n, fused_activation=None, max_cols=None, tuning=None):
    """FPU-tuned 2D prefill progcfg for MLP matmuls: picks the grid width that maximizes the output
    subblock (drives prefill FPU) instead of the default full width.

    max_cols caps the grid width. Default = prefill_grid_default()[0] (8). Pass the device worker-grid
    width (11 on BH P150) to let the subblock heuristic go wide -> the measured prefill winners
    (gate 9-wide, down/wo 10-wide, gdn_qkvz 11-wide; test_mlp_matmul_sweep_prefill). Fused AG/RS paths
    pin 8-wide separately and are unaffected.

    tuning: a `_PREFILL_TUNING` entry. With `widest_cols` (TP=8) the subblock-first width heuristic
    is replaced by "take the width, clamped to PREFILL_MAX_COLS_PORTABLE" -- measured device time at
    TP=8 falls monotonically with column count, so trading cores for a wider subblock loses."""
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
        m, k, n, grid_size=(cols, grid[1]), fused_activation=fused_activation, tuning=tuning
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


def agmm_subblock_h(m_block):
    """QWEN36_AGMM_SUBH=2: 2x4 output subblocks (8 dest tiles) for the fused all-gather matmuls with dst_full_sync_en (fp32 dest).
    EXPERIMENT ONLY — measured 2026-09-07: the op accepts it but the kernel produces garbage (PCC 0). Keep unset."""
    try:
        sh = int(os.environ.get("QWEN36_AGMM_SUBH", "1") or 1)
    except ValueError:
        sh = 1
    return sh if (sh > 1 and m_block % sh == 0) else 1


def agmm_compute_cfg(compute_cfg, role="in"):
    """QWEN36_AGMM_FIDELITY (default unset = caller's config, HiFi2): "lofi" -> LoFi for every fused all-gather matmul
    routed through all_gather_matmul_prefill; "lofi_out" -> LoFi only for role="out" (the GDN out-projection). The op is
    FPU-bound at HiFi2 (2 passes); LoFi is 1 pass. Numerics change -> gate on the 8-layer PCC/KL and the repo test."""
    mode = os.environ.get("QWEN36_AGMM_FIDELITY", "")
    lofi = mode == "lofi" or (mode == "lofi_out" and role == "out")
    full_sync = agmm_subblock_h(4) > 1  # QWEN36_AGMM_SUBH=2 -> 8 fp32 dest tiles
    if lofi or full_sync:
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi if lofi else compute_cfg.math_fidelity,
            math_approx_mode=compute_cfg.math_approx_mode,
            fp32_dest_acc_en=compute_cfg.fp32_dest_acc_en,
            packer_l1_acc=compute_cfg.packer_l1_acc,
            dst_full_sync_en=full_sync,
        )
    return compute_cfg


_AGMM_PERSISTENT = {}
_AGMM_BARRIER_TENSORS = {}


def _agmm_persistent_mode():
    return os.environ.get("QWEN36_AGMM_PERSISTENT", "0") == "1"


def _agmm_barrier_mode():
    """QWEN36_AGMM_BARRIER: 0/unset off; 1 = Python-side entry barrier (a 1-tile all_gather_async with barrier_semaphore
    before every fused all-gather matmul); 2 = pass barrier_semaphore INTO all_gather_minimal_matmul_async (in-kernel
    entry barrier; needs the factory/kernel support from profiles/agmm_inkernel_barrier.patch, otherwise it is a no-op).
    """
    try:
        return int(os.environ.get("QWEN36_AGMM_BARRIER", "0") or 0)
    except ValueError:
        return 0


def _agmm_barrier_kwargs(tt_ccl, cluster_axis):
    """Extra kwargs for the fused all-gather matmul call in barrier mode 2."""
    if _agmm_barrier_mode() == 2:
        return {"barrier_semaphore": tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)}
    return {}


def _agmm_persistent_intermediate(x4, tt_ccl, out_memory_config):
    """Caller-owned all-gather intermediate for all_gather_minimal_matmul_async: [.., S, K_local * ring] in the op's
    output dtype (bf16 here) / TILE / out_memory_config (matches compute_output_specs slot 0)."""
    ring = tt_ccl.mesh_device.get_num_devices()
    shape = (x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3] * ring)  # x4 is always [1, 1, S, K_local]
    # Always DRAM: a process-lifetime L1 buffer steals L1 from every later kernel (an L1 one clashed with the
    # chunk_gdn_prep CBs); the op only needs a TILE buffer of the right shape/dtype for the gather target.
    key = (id(tt_ccl), tt_ccl.mesh_device.id(), shape)  # per CCL manager AND mesh device (tests reopen devices)
    t = _AGMM_PERSISTENT.get(key)
    if t is None:
        t = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_ccl.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(tt_ccl.mesh_device),
        )
        _AGMM_PERSISTENT[key] = t
    return t


def _agmm_pre_barrier(x4, tt_ccl, topology, cluster_axis):
    """Cross-device entry barrier before a fused all-gather matmul: a 1-tile all_gather_async with a barrier
    semaphore (every device waits until all peers have entered, i.e. finished the preceding ops)."""
    key = (id(tt_ccl), tt_ccl.mesh_device.id(), cluster_axis)  # per CCL manager AND mesh device (tests reopen devices)
    t = _AGMM_BARRIER_TENSORS.get(key)
    if t is None:
        t = ttnn.from_torch(
            torch.zeros((1, 1, TILE_SIZE, TILE_SIZE), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tt_ccl.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(tt_ccl.mesh_device),
        )
        _AGMM_BARRIER_TENSORS[key] = t
    g = ttnn.experimental.all_gather_async(
        t,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=1,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
    )
    ttnn.deallocate(g)


_AG_SEM_POOL = {}


def ag_semaphores(tt_ccl, cluster_axis):
    """Global-semaphore pair for the fused all-gather ops.

    Default (QWEN36_CCL_AG_SEM_POOL unset/0): TT_CCL's rotation, which hands the same pair to every 2nd call.
    QWEN36_CCL_AG_SEM_POOL=N: a dedicated pool of N fixed-role pairs cycled round-robin, so a pair is reused
    only every N-th call. Determinism probe for the traced multi-layer prefill: the op resets its two
    semaphores at kernel end, so a straggling increment from a slower device lands on a pair that the next
    call already reuses. Created lazily on the first (pre-capture warm) call; global semaphores persist.
    """
    try:
        n = int(os.environ.get("QWEN36_CCL_AG_SEM_POOL", "0") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        return tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis)
    key = (id(tt_ccl), tt_ccl.mesh_device.id(), cluster_axis)  # per CCL manager AND mesh device (tests reopen devices)
    pool = _AG_SEM_POOL.get(key)
    if pool is None:
        pool = {
            "pairs": [
                [ttnn.create_global_semaphore(tt_ccl.mesh_device, tt_ccl.sub_device_crs, 0) for _ in range(2)]
                for _ in range(n)
            ],
            "idx": 0,
        }
        _AG_SEM_POOL[key] = pool
    pair = pool["pairs"][pool["idx"]]
    pool["idx"] = (pool["idx"] + 1) % n
    return list(pair)


def _agmm_layout(grid, n_tiles, default_n_block, num_links=2, m_rows=None):
    """(grid, force_transpose, workers_per_link, n_block) for the fused all-gather matmuls.

    Default (QWEN36_AGMM_LAYOUT unset / "t8x9"): transposed 8x9 grid, 2 links x 4 workers, N over the 9 rows.
    "nt11x8": untransposed 11x8 grid — in0 (gather) axis = 8 rows = 2 links x 4 workers exactly as before, M 8
    tiles/core unpadded, N over 11 columns (GDN 129 tiles -> 12/core, 2% padding vs 11%), the 4 fabric muxes stay on
    the free device row 9; 88 compute cores instead of 72. Measured standalone: GDN in-proj 587 -> 457 us, attention
    in-proj 603 -> 450 us, PCC identical (see profiles/REPORT.md, Task 2)."""
    layout = os.environ.get("QWEN36_AGMM_LAYOUT", "t8x9")
    # The op auto-transposes when M > N (the untransposed orientation needs N >= M), so narrow-N out-projections
    # (N_local 1280 < S 2048) keep the default layout.
    if layout == "nt11x8" and (m_rows is None or n_tiles * TILE_SIZE > m_rows):
        grid = (11, 8)
        workers = grid[1] // num_links  # in0 axis is grid.y when not transposed
        per_core = max(1, math.ceil(n_tiles / grid[0]))
        n_block = ((per_core + 3) // 4) * 4 if default_n_block % 4 == 0 else per_core
        return grid, False, workers, n_block
    grid = (8, grid[1])
    return grid, True, grid[0] // num_links, default_n_block


def proj_chunks_mode():
    """QWEN36_GDN_PROJ_CHUNKS: split the fused in-projection AGMM output into per-consumer tensors.

    0 / unset (DEFAULT) → off: the op writes one wide tensor and the model slices it.
    1 → on, chunk outputs in DRAM (one memory config serves every chunk; DRAM is what the widest
        consumer, the GDN `z` gate, requires during chunk-prefill).
    2 → on, chunk outputs in L1 (perf A/B only — at S=2048 the GDN `z` chunk is 6 MB of L1 and
        clashes with the chunk_gdn_prep/scan CBs, exactly what the sliced path avoids).
    """
    try:
        return int(os.environ.get("QWEN36_GDN_PROJ_CHUNKS", "0") or 0)
    except ValueError:
        return 0


def proj_chunks_memcfg(mode):
    """Output memory config for the chunked in-projection (one config covers ALL chunks)."""
    return ttnn.L1_MEMORY_CONFIG if mode >= 2 else ttnn.DRAM_MEMORY_CONFIG


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
    chunk_sizes=None,
    role="in",
):
    """Fused all-gather(dim=3) + column-parallel matmul for prefill (all_gather_minimal_matmul_async).

    x: K-sharded activation [.,S,K/tp]; weight: [K,N] col-sharded (K full). Gathers x to full K and
    matmuls in one op, replacing a separate all_gather + linear. fused_activation applied per tile
    before pack (non-parametrized op, e.g. ttnn.UnaryOpType.SILU). out_memory_config places the result
    (default DRAM; L1 keeps it resident for downstream slices).

    chunk_sizes: per-consumer output widths in ELEMENTS. When given, the op writes one output tensor
    per width instead of a single [.,S,N] tensor, and a list is returned in that order — this replaces
    the caller's post-projection ttnn.slice chain. Every width must be a multiple of TILE_SIZE and they
    must sum to the weight's N (device op validation:
    ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/
    all_gather_minimal_matmul_async_device_operation.cpp:172-205). The matmul blocking depends only on
    the tile count, so chunked and sliced outputs are bit-identical."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    # AG-bound: 2 ethernet links parallelize the gather (P150x4 max; traced_8k TTFT win). grid.x must
    # = num_links*workers, and the 7-wide default (prime) forces 1 link -> widen to 8 (2 links, 4 workers).
    num_links = 2
    grid = (8, grid[1])
    # Narrow-N out-projections (N_local=1280 -> 5 tiles/core) get an N block no wider than the per-core N and a
    # subblock that divides it (the op handles partial N blocks either way; measured 401 vs 408 us, kept for clarity).
    n_tiles = math.ceil(weight.shape[-1] / TILE_SIZE)
    n_tiles_per_core = max(1, math.ceil(n_tiles / grid[0]))
    n_block = min(8, n_tiles_per_core)
    grid, force_transpose, workers, n_block = _agmm_layout(grid, n_tiles, n_block, num_links, m_rows=S)
    sub_w = max(d for d in (4, 2, 1) if n_block % d == 0)
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=n_block,
        subblock_h=agmm_subblock_h(4),
        subblock_w=sub_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    _chunks = list(chunk_sizes) if chunk_sizes else []
    compute_cfg = agmm_compute_cfg(compute_cfg, role)
    # Determinism probes (default off). The op documents that EITHER persistent_output_buffer OR barrier_semaphore is
    # required (the factory implements neither barrier, so only the persistent gather buffer is available): without one,
    # a device that runs ahead writes its next call's shard into a peer's gather buffer address before the peer has
    # allocated it (the peer may still be reading another tensor there). QWEN36_AGMM_PERSISTENT=1 keeps one gather
    # intermediate per (shape, dtype, memcfg) alive for the process (created on the pre-capture warm call).
    _persist = _agmm_persistent_intermediate(x4, tt_ccl, out_memory_config) if _agmm_persistent_mode() else None
    if _agmm_barrier_mode() == 1:
        _agmm_pre_barrier(x4, tt_ccl, topology, cluster_axis)
    out = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        persistent_output_buffer=_persist,
        **_agmm_barrier_kwargs(tt_ccl, cluster_axis),
        config=cfg,
        fused_activation=fused_activation,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=ag_semaphores(tt_ccl, cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=force_transpose,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
        chunks=len(_chunks) if _chunks else 1,
        chunk_sizes=_chunks,
    )
    # The op strips the gather intermediates, so the return is exactly the chunk outputs
    # (all_gather_minimal_matmul_async_device_operation.cpp:524-528).
    return out if _chunks else out[0]


def all_gather_then_matmul_prefill(x, weight, tt_ccl, compute_cfg, topology, cluster_axis=1, max_cols=11):
    """Un-fused alternative to all_gather_matmul_prefill for narrow-N out-projections: plain
    all_gather_async (2 links) of the K-sharded activation, then the tuned 2D mcast matmul on the full
    grid with the column-sharded weight. fp32 accumulation inside the matmul; output [1,1,S,N_local]."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    xg = ttnn.experimental.all_gather_async(
        x4,
        dim=3,
        multi_device_global_semaphore=ag_semaphores(tt_ccl, cluster_axis),
        num_links=2,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pc = create_prefill_mlp_matmul_program_config(S, xg.shape[-1], weight.shape[-1], max_cols=max_cols)
    out = ttnn.linear(
        xg, weight, compute_kernel_config=compute_cfg, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(xg)
    return out


def mlp_gateup_agmm_enabled(num_devices):
    """Fuse the ff_norm all-gather into the MLP gate/up matmul (prefill). TP-only (needs the gather)."""
    return num_devices > 1


def all_gather_swiglu_prefill(
    x, weight, tt_ccl, compute_cfg, topology, grid=(7, 9), cluster_axis=1, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """Fused all-gather + col-parallel gate/up matmul + SwiGLU for prefill (packing gate+up lets ff_norm's AG fuse in).

    x: K-sharded [.,S,K/tp]; weight: tile-pair-interleaved [gate|up] [K, 2N/tp]. Emits silu(gate)*up of width N/tp."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    num_links = 2
    grid = (8, grid[1])
    # gate|up: N_block counts interleaved tiles (pairs); keep 16 unless the layout override / env sets one.
    grid, force_transpose, workers, _ = _agmm_layout(
        grid, math.ceil(weight.shape[-1] / TILE_SIZE), 16, num_links, m_rows=S
    )
    # 11x8 layout: N (272 interleaved gate|up tiles) splits 25/core; one 28-wide N block (7% padding) needs M_block 4 to
    # fit the fp32 intermediate in L1 (M8/N28 overflows). Measured: M8/N16 751 us -> M4/N28 673 us (-10%); M4/N32 697.
    if grid == (11, 8):
        m_block = int(os.environ.get("QWEN36_AGMM_MLP_MBLOCK", "4"))
        n_block = int(os.environ.get("QWEN36_AGMM_MLP_NBLOCK", "28"))
    else:
        m_block = int(os.environ.get("QWEN36_AGMM_MLP_MBLOCK", "8"))
        n_block = int(os.environ.get("QWEN36_AGMM_MLP_NBLOCK", "16"))
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=n_block,
        subblock_h=agmm_subblock_h(m_block),
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    compute_cfg = agmm_compute_cfg(compute_cfg, "mlp")
    # Same entry-barrier guard as all_gather_matmul_prefill (mode 1: Python pre-barrier op; mode 2: in-kernel).
    if _agmm_barrier_mode() == 1:
        _agmm_pre_barrier(x4, tt_ccl, topology, cluster_axis)
    return ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        **_agmm_barrier_kwargs(tt_ccl, cluster_axis),
        config=cfg,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=ag_semaphores(tt_ccl, cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=force_transpose,
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
        dtype=dtype,  # matmul output dtype matches the persistent buffers
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
