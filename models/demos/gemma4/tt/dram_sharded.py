# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM-width-sharded matmul helpers for Gemma4 tensor-parallel decode.

Decode is weight-read-bound (M<=32, one activation tile), so spreading each
per-device weight shard across all DRAM banks and running the DRAM-sharded
matmul kernel cuts the per-token weight-read time. Prefill (M>32) reuses the
same width-sharded weight through a 2D matmul program config.

Decode program / activation grids follow tt_transformers + mlp_1d
(``find_grid_k_n``, ``per_core_N = ceil(n/(tile*cores))``). Prefill 2D progcfg
helpers are adapted from the Qwen3.6 Blackhole TP path (tp_common.py).
"""

import math
import os

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole

TILE_SIZE = 32
# P150 Blackhole DRAM bank count. Wormhole meshes differ — can_dram_shard is
# BH-only so this constant is never applied on WH (wrong bank count → garbage).
# Override with GEMMA4_DRAM_CORES only when also rebuilding .ws weight caches —
# a mismatched bank count reuses stale WIDTH_SHARDED bins and produces garbage.
# Matches tt_transformers: WH forces 8; BH uses device dram_grid_size.x (7/8).
DRAM_CORES = max(1, int(os.environ.get("GEMMA4_DRAM_CORES", "8")))
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})
# BH QuietBox / P150 usable L1 for statically allocated CBs.
_L1_MAX_BYTES = 1_572_864
_L1_HEADROOM_BYTES = 64_000
# Optional upper bound on decode in0_block_w (tt_transformers / mlp_1d default
# to find_largest_divisor(..., 8)). Prefer L1-aware shrinking below over a hard
# empirical cap — set GEMMA4_DECODE_IN0_BLOCK_W_MAX only for sweeps.
_DECODE_IN0_BLOCK_W_MAX = max(1, int(os.environ.get("GEMMA4_DECODE_IN0_BLOCK_W_MAX", "8")))


def _roundup(a, b):
    return b * math.ceil(a / b)


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def _find_grid(n_tiles, target=32):
    """Pick a core count dividing n_tiles closest to `target`, factored into <=8x8."""
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


def _find_grid_k_n(k_tiles, n_tiles, max_rows=8, max_cols=8):
    """Core grid that evenly divides both K and N tile counts.

    Same contract as ``tt_transformers.ModelArgs.find_grid_k_n`` /
    ``models.common.modules.mlp.mlp_1d._find_grid_k_n``. A K-only grid with
    ``per_core_N = n_tiles // num_cores`` silently truncates N when
    ``n_tiles % num_cores != 0`` — tt_transformers documents this as bad PCC
    (``dram_shard_grid_width`` comment). Prefer the largest feasible core count.
    """
    max_cores = max_rows * max_cols
    possible = [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0]
    possible.sort(reverse=True)
    for cores in possible:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise ValueError(f"Cannot find grid for K={k_tiles}, N={n_tiles} tiles")


def _padded_n_tiles(n):
    """N tiles after DRAM-bank width padding (weight shard layout)."""
    return _roundup(n, TILE_SIZE * DRAM_CORES) // TILE_SIZE


def _decode_core_grid(k, n):
    """Compute-core grid for decode DRAM-sharded matmul + matching act shard."""
    k_tiles = k // TILE_SIZE
    n_tiles = _padded_n_tiles(n)
    rows, cols = _find_grid_k_n(k_tiles, n_tiles)
    return rows, cols, rows * cols


def prefill_grid_default():
    """BH P150: (8,10); WH: (8,8). y capped at 10 on BH (grid_x=10 breaks matmul)."""
    return (8, 10) if is_blackhole() else (8, 8)


def prefill_max_cols_default(mesh_device=None):
    """Max grid width for FPU-tuned prefill progcfg.

    Safe default is ``prefill_grid_default()[0]`` (8). On BH, ``grid_x>=10`` can
    garble the regular 2D matmul (Qwen notes the same); auto-using the full
    worker-grid width (11 on P150) cut 128k TTFT (~73s→~55s) but destroyed
    generation quality on Gemma4-31B. Keep the ``_best_prefill_cols`` search
    inside the safe band (e.g. gate_up 7-wide / out_subblock_w=4).

    Override with ``GEMMA4_PREFILL_MAX_COLS`` for sweeps (9 stays coherent on
    31B/P150x8; 11 is faster but incorrect). ``mesh_device`` is API parity only.
    """
    env = os.environ.get("GEMMA4_PREFILL_MAX_COLS")
    if env is not None:
        return max(1, int(env))
    return prefill_grid_default()[0]


# Prefill activation-row cutoff. The 2D matmul's circular buffers scale with
# per_core_M (= ceil(M/TILE/grid_y)), so a single-shot matmul at long context
# (M = seq_len, up to 256k) overflows L1. Following tt_transformers, we reshape
# [1, 1, M, K] -> [1, M/cutoff, cutoff, K] and run ONE batched matmul sized to
# ``cutoff`` rows (the extra batch dim is iterated by the kernel, reusing CBs).
# This keeps per_core_M tiny AND avoids the memory blow-up of a chunk+concat
# (which would need source chunks + a full-size destination simultaneously).
_PREFILL_CUTOFF = 512 if is_blackhole() else 1024
# The tuned o_proj path is shape-specific: it helps 31B but regresses 12B.
_OPROJ_TUNED = os.environ.get("GEMMA4_OPROJ_TUNED", "0") != "0"
# Fallback per-call row cap for the (rare) M not divisible by the cutoff.
_PREFILL_M_CHUNK = prefill_grid_default()[1] * 8 * TILE_SIZE


def in_prefill_l1_matmul_band(m: int) -> bool:
    return TILE_SIZE < int(m) <= _PREFILL_CUTOFF


def prefill_matmul_lofi_enabled(m: int) -> bool:
    """LoFi tall-prefill matmuls are opt-in after long-context regressions."""
    enabled = os.environ.get("GEMMA4_PREFILL_MATMUL_LOFI", "0").lower() in ("1", "true", "yes")
    return enabled and int(m) > _PREFILL_CUTOFF


def prefill_lofi_ckc():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _prefill_hifi2_ckc():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _prefill_hifi4_ckc():
    """Short-prefill tuned paths must not silently inherit LoFi."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


_L1_FALLBACK_SHAPES: set[tuple[int, int, int]] = set()


def matmul_rows(x):
    rows = 1
    for i in range(len(x.shape) - 1):
        rows *= int(x.shape[i])
    return rows


def linear_l1_safe(x, weight, *, program_config=None, memory_config=None, compute_kernel_config=None):
    """Use a tuned config when it fits, caching auto fallback on L1 overflow."""
    if program_config is None:
        return ttnn.linear(x, weight, memory_config=memory_config, compute_kernel_config=compute_kernel_config)

    key = (matmul_rows(x), int(x.shape[-1]), int(weight.shape[-1]))
    if key not in _L1_FALLBACK_SHAPES:
        try:
            return ttnn.linear(
                x,
                weight,
                program_config=program_config,
                memory_config=memory_config,
                compute_kernel_config=compute_kernel_config,
            )
        except RuntimeError as error:
            if "circular buffer" not in str(error).lower():
                raise
            _L1_FALLBACK_SHAPES.add(key)
            logger.warning(f"Gemma4 tuned matmul {key} exceeded L1; using ttnn auto for this shape")
    return ttnn.linear(x, weight, memory_config=memory_config, compute_kernel_config=compute_kernel_config)


def should_prefill_long_2d(m: int) -> bool:
    enabled = os.environ.get("GEMMA4_PREFILL_LONG_2D", "1").lower() not in ("0", "false", "no")
    return enabled and int(m) > _PREFILL_CUTOFF and int(m) % _PREFILL_CUTOFF == 0


def weight_memcfg(k, n):
    """WIDTH_SHARDED DRAM memory config for a per-device weight shard [k, n]."""
    padded_n = _roundup(n, TILE_SIZE * DRAM_CORES)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_GRID, (k, padded_n // DRAM_CORES), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _tile_size_bytes(dtype=None):
    """Approximate single-tile footprint for L1 CB budgeting."""
    if dtype in (ttnn.bfloat8_b, getattr(ttnn, "bfloat4_b", None)):
        return 1088
    return 2048  # bfloat16 / unknown — conservative


def _estimate_decode_l1_bytes_for_in0(n_tiles, in0_block_w, dtype=None):
    """Rough static-CB estimate for the DRAM-sharded decode kernel (in1-dominated)."""
    tile_aligned = _roundup(_tile_size_bytes(dtype), 64)
    # in1 triple-buffer × padded-N/DRAM_CORES × in0_block_w (factory layout).
    in1 = math.ceil(n_tiles / DRAM_CORES) * in0_block_w * 3 * tile_aligned
    # in0 / out / interm / reshard overhead (order-of-magnitude pad).
    return in1 + 200_000


def _decode_in0_block_w(k, n, num_cores, dtype=None):
    """Largest in0_block_w that divides K/core and fits the L1 CB budget.

    Mirrors ``tt_transformers.dram_matmul_config`` /
    ``mlp_1d._dram_matmul_config`` (``find_largest_divisor(k/(tile*cores))``),
    then shrinks when the in1 triple-buffer estimate would overflow L1 — the
    generic replacement for a hard-coded in0 cap of 2.
    """
    k_tiles_per_core = max(1, (k // TILE_SIZE) // num_cores)
    n_tiles = _padded_n_tiles(n)
    budget = _L1_MAX_BYTES - _L1_HEADROOM_BYTES
    in0 = _find_largest_divisor(k_tiles_per_core, max_div=_DECODE_IN0_BLOCK_W_MAX)
    while in0 > 1 and _estimate_decode_l1_bytes_for_in0(n_tiles, in0, dtype) > budget:
        in0 = _find_largest_divisor(k_tiles_per_core, max_div=in0 - 1)
    return in0


def decode_progcfg(m, k, n, dtype=None):
    """DRAM-sharded matmul program config for decode (small M).

    Matches tt_transformers / mlp_1d:
      * core grid divides both K and N tiles (``find_grid_k_n``)
      * ``per_core_N = ceil(n / (tile * num_cores))`` — never floor-truncate
      * ``in0_block_w`` from K/core, L1-shrunk when needed
    """
    _rows, _cols, num_cores = _decode_core_grid(k, n)
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_decode_in0_block_w(k, n, num_cores, dtype=dtype),
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=math.ceil(n / (TILE_SIZE * num_cores)),
        fused_activation=None,
    )


def decode_1d_matmul_config(mesh_device, k, n, m=TILE_SIZE):
    """Tuned narrow-N decode config; wide shapes retain ttnn auto."""
    if os.environ.get("GEMMA4_QKV_DECODE_PROGCFG", "1").lower() in ("0", "false", "no"):
        return None
    if k % TILE_SIZE or n % TILE_SIZE or m > TILE_SIZE:
        return None
    grid = mesh_device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    k_tiles, n_tiles = k // TILE_SIZE, n // TILE_SIZE
    if n_tiles >= 2 * grid_cores:
        return None
    cap = min(grid_cores, n_tiles // 2)
    cores = next((c for c in range(cap, 0, -1) if n_tiles % c == 0), 0)
    if cores < 2:
        return None
    rows = next((y for y in range(1, grid.y + 1) if cores % y == 0 and cores // y <= grid.x), None)
    if rows is None:
        return None
    per_core_n = n_tiles // cores
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cores // rows, rows),
        in0_block_w=_find_largest_divisor(k_tiles, max_div=4),
        out_subblock_h=1,
        out_subblock_w=_find_largest_divisor(per_core_n, max_div=4),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    return program_config, compute_kernel_config


def activation_memcfg(k, n):
    """WIDTH_SHARDED L1 activation config matching ``decode_progcfg``'s core grid."""
    rows, cols, num_cores = _decode_core_grid(k, n)
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, k // num_cores),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _estimate_decode_l1_bytes(k, n, dtype=None):
    """L1 estimate using the same grid/in0 selection as ``decode_progcfg``."""
    try:
        _rows, _cols, num_cores = _decode_core_grid(k, n)
    except ValueError:
        return float("inf")
    in0 = _decode_in0_block_w(k, n, num_cores, dtype=dtype)
    return _estimate_decode_l1_bytes_for_in0(_padded_n_tiles(n), in0, dtype)


def can_dram_shard(k, n, dtype=None):
    """True if a [k, n] weight shard is safe for the DRAM-sharded decode path.

    Blackhole-only: ``DRAM_CORES`` matches P150; Wormhole bank counts differ and
    produce garbage (CI PCC ~0). Requires a compute grid that evenly divides
    both K and N tiles (same as tt_transformers) and an in0_block_w that fits
    L1 after shrinking.
    """
    if not is_blackhole():
        return False
    if k % TILE_SIZE != 0 or n <= 0:
        return False
    try:
        rows, cols, num_cores = _decode_core_grid(k, n)
    except ValueError:
        return False
    # Activation width-shard needs k evenly split across the core grid.
    if (k // TILE_SIZE) % num_cores != 0 or (k // num_cores) % TILE_SIZE != 0:
        return False
    # N must also land evenly so per_core_N * num_cores covers the padded shard.
    if _padded_n_tiles(n) % num_cores != 0:
        return False
    if _estimate_decode_l1_bytes(k, n, dtype) > _L1_MAX_BYTES - _L1_HEADROOM_BYTES:
        return False
    return True


def _get_out_subblock_w(per_core_n, out_subblock_h):
    for w in range(min(per_core_n, 4 // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _best_prefill_cols(n, max_cols):
    """Grid width (<=max_cols) maximizing the output subblock, tie-broken to more cores.

    Avoids the 1x1-subblock stall the default full-width grid can force on wide N
    (ported from Qwen3.6 ``tp_common._best_prefill_cols`` / PR #48861).
    """
    n_tiles = math.ceil(n / TILE_SIZE)
    best_cols, best_key = 1, None
    for cols in range(1, max_cols + 1):
        sw = _get_out_subblock_w(math.ceil(n_tiles / cols), 1)
        key = (sw, cols)  # prefer wider subblock, then more columns
        if best_key is None or key > best_key:
            best_key, best_cols = key, cols
    return best_cols


def prefill_progcfg(m, k, n, grid_size=None, max_cols=None, fused_activation=None):
    """FPU-tuned 2D matmul program config for prefill on a DRAM-width-sharded weight.

    When ``grid_size`` is omitted, picks the grid width that maximizes
    ``out_subblock_w`` (drives prefill FPU) instead of always using the full
    ``prefill_grid_default()`` width. ``max_cols`` caps that search (pass the
    device worker-grid width, 11 on BH P150, for the measured wide-grid winners).
    """
    if grid_size is None:
        base = prefill_grid_default()
        cols = _best_prefill_cols(n, max_cols if max_cols is not None else base[0])
        grid_size = (cols, base[1])
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))
    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)
    k_tiles = math.ceil(k / TILE_SIZE)
    # Kernel requires Kt % in0_block_w == 0. Prefer ~k_tiles/cols capped at 4,
    # then snap down to a divisor (26B down_proj K=288 → Kt=9; 9//2=4 is invalid).
    candidate = min(4, max(1, k_tiles // max(1, grid_size[0])))
    in0_block_w = _find_largest_divisor(k_tiles, max_div=candidate)
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


def prefill_linear_above_cutoff(x, weight, *, out_memory_config=None):
    """Reshape tall matmuls so their program's circular buffers stay cutoff-sized."""
    out_mc = out_memory_config if out_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    x_shape = [int(x.shape[i]) for i in range(len(x.shape))]
    orig_leading = x_shape[:-1]
    n_in = x_shape[-1]
    m = matmul_rows(x)
    n_out = int(weight.shape[-1])
    flat = [1, 1, m, n_in]
    x_work = x if x_shape == flat else ttnn.reshape(x, flat)

    def restore(out):
        wanted = (*orig_leading, int(out.shape[-1]))
        actual = tuple(int(out.shape[i]) for i in range(len(out.shape)))
        return out if actual == wanted else ttnn.reshape(out, wanted)

    if not should_prefill_long_2d(m):
        return restore(ttnn.linear(x_work, weight, memory_config=out_mc))

    batch = m // _PREFILL_CUTOFF
    reshaped = ttnn.reshape(x_work, (1, batch, _PREFILL_CUTOFF, n_in))
    program_config = prefill_progcfg(_PREFILL_CUTOFF, n_in, n_out)
    compute_kernel_config = prefill_lofi_ckc() if prefill_matmul_lofi_enabled(m) else _prefill_hifi2_ckc()
    output = linear_l1_safe(
        reshaped,
        weight,
        program_config=program_config,
        memory_config=out_mc,
        compute_kernel_config=compute_kernel_config,
    )
    return restore(ttnn.reshape(output, (1, 1, m, int(output.shape[-1]))))


def interleaved_prefill_config(m, k, n):
    """Shape-gated QKV prefill config for an interleaved weight."""
    if not in_prefill_l1_matmul_band(m):
        return None, None, None
    return prefill_progcfg(m, k, n), ttnn.DRAM_MEMORY_CONFIG, _prefill_hifi4_ckc()


def _out_subblock_hw(per_core_n, per_core_m):
    best = (1, 1)
    for height in range(1, min(per_core_m, 4) + 1):
        if per_core_m % height:
            continue
        for width in range(1, min(per_core_n, 4 // height) + 1):
            if per_core_n % width == 0 and height * width > best[0] * best[1]:
                best = (height, width)
    return best


def _factor_1d_grid(cores, grid_x, grid_y):
    cols = min(grid_x, cores)
    while cols > 1 and cores % cols:
        cols -= 1
    rows = cores // cols
    return (cols, rows) if 1 <= rows <= grid_y else None


def _pick_1d_cores(n_tiles, grid_x, grid_y, prefer=42):
    candidates = [
        cores
        for cores in range(8, grid_x * grid_y + 1)
        if n_tiles % cores == 0 and _factor_1d_grid(cores, grid_x, grid_y) is not None
    ]
    if not candidates:
        return None
    if prefer in candidates:
        return prefer
    return max(candidates, key=lambda cores: (-abs(cores - prefer), cores))


def prefill_progcfg_1d(m, k, n, cores=None, in0_block_w=None, grid_size=None, fuse_batch=False):
    if grid_size is None:
        grid_size = prefill_grid_default()
    grid_x, grid_y = grid_size
    m_tiles, k_tiles, n_tiles = math.ceil(m / TILE_SIZE), math.ceil(k / TILE_SIZE), math.ceil(n / TILE_SIZE)
    cores = cores or _pick_1d_cores(n_tiles, grid_x, grid_y)
    if cores is None or n_tiles % cores:
        return None
    factored = _factor_1d_grid(cores, grid_x, grid_y)
    if factored is None:
        return None
    cols, rows = factored
    in0_block_w = in0_block_w or _find_largest_divisor(k_tiles, max_div=4)
    if k_tiles % in0_block_w:
        return None
    per_core_n = n_tiles // cores
    out_h, out_w = _out_subblock_hw(per_core_n, m_tiles)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=out_h,
        out_subblock_w=out_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=fuse_batch,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet(set()),
        num_global_cb_receivers=0,
        untilize_out=False,
    )


def _interleaved_mlp_prefill_config(m, k, n):
    env = os.environ.get("GEMMA4_PREFILL_1D_MLP", "1").lower()
    if env in ("0", "false", "no"):
        return None, None, None
    if not in_prefill_l1_matmul_band(m):
        return None, None, None
    # Default covers the 12B batch-1 TTFT band (M<=128). 31B hung mid-decode
    # after short-prefill 1D (K>=5376); opt in with GEMMA4_PREFILL_1D_MLP=all.
    if env not in ("all", "full") and (int(m) > 128 or int(k) >= 5376):
        return None, None, None
    # Swept for TP-sharded widths (31B TP=8 → n=5376). Full-width TP=1
    # fused gate+up (n≈43k) overflows Wormhole L1 CBs and falls back dirty.
    if int(n) > 8192:
        return None, None, None
    program_config = prefill_progcfg_1d(m, k, n)
    if program_config is None:
        return None, None, None
    output_bytes = int(m) * int(n) * 2
    out_memcfg = ttnn.L1_MEMORY_CONFIG if output_bytes <= 4 * 1024 * 1024 else ttnn.DRAM_MEMORY_CONFIG
    return program_config, out_memcfg, _prefill_hifi4_ckc()


def interleaved_gate_up_prefill_config(m, k, n):
    return _interleaved_mlp_prefill_config(m, k, n)


def interleaved_down_proj_prefill_config(m, k, n):
    return _interleaved_mlp_prefill_config(m, k, n)


def _program_grid(program_config):
    grid = program_config.compute_with_storage_grid_size
    return (int(grid.x), int(grid.y)) if hasattr(grid, "x") else (int(grid[0]), int(grid[1]))


def _out_shard_matches_program(out_memcfg, program_config):
    if out_memcfg is None or not out_memcfg.is_sharded():
        return True
    spec = out_memcfg.shard_spec
    if spec is None:
        return False
    box = spec.grid.bounding_box().grid_size()
    grid_x, grid_y = _program_grid(program_config)
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    return (
        int(box.x) <= grid_x
        and int(box.y) <= grid_y
        and shard_h == program_config.per_core_M * TILE_SIZE
        and shard_w == program_config.per_core_N * TILE_SIZE
    )


def l1_block_sharded_memcfg(rows, cols, grid=None):
    grid_x, grid_y = grid or prefill_grid_default()
    row_tiles, col_tiles = math.ceil(rows / TILE_SIZE), math.ceil(cols / TILE_SIZE)
    shard_rows = [value for value in range(1, grid_y + 1) if row_tiles % value == 0]
    shard_cols = [value for value in range(1, grid_x + 1) if col_tiles % value == 0]
    if not shard_rows or not shard_cols:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.create_sharded_memory_config(
        shape=(rows, cols),
        core_grid=ttnn.CoreGrid(x=max(shard_cols), y=max(shard_rows)),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def interleaved_o_proj_prefill_config(m, k, n, grid=None):
    """Shape-gated o_proj tuning; deliberately opt-in per safety follow-up."""
    if not _OPROJ_TUNED or not in_prefill_l1_matmul_band(m):
        return None, None, None
    grid = grid or prefill_grid_default()
    program_config = prefill_progcfg(m, k, n, grid_size=grid)
    out_memcfg = l1_block_sharded_memcfg(m, n, grid=grid)
    if not _out_shard_matches_program(out_memcfg, program_config):
        return None, None, None
    return program_config, out_memcfg, _prefill_hifi2_ckc()


def lm_head_decode_config(mesh_device, m, k, n):
    """Tuned last-token LM head with safe HiFi3 + fp32 destination accumulation."""
    if max(1, math.ceil(m / TILE_SIZE)) > 1 or n > 64 * 1024:
        return None, None, None
    grid = mesh_device.compute_with_storage_grid_size()
    program_config = prefill_progcfg_1d(
        m,
        k,
        n,
        cores=grid.x * grid.y,
        in0_block_w=1,
        grid_size=(grid.x, grid.y),
    )
    if program_config is None:
        return None, None, None
    mode = os.environ.get("GEMMA4_LM_HEAD_FIDELITY", "hifi3_destacc").lower()
    if mode == "hifi4":
        fidelity, dest_acc = ttnn.MathFidelity.HiFi4, False
    elif mode == "hifi4_destacc":
        fidelity, dest_acc = ttnn.MathFidelity.HiFi4, True
    else:
        fidelity, dest_acc = ttnn.MathFidelity.HiFi3, True
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=dest_acc,
        packer_l1_acc=True,
    )
    return program_config, ttnn.L1_MEMORY_CONFIG, compute_kernel_config


class DramShardedLinear:
    """A single DRAM-width-sharded weight served for both decode and prefill.

    Decode (M<=32): width-shard the activation to L1, run the DRAM-sharded
    kernel, return a DRAM-interleaved result. Prefill (M>32): plain matmul with
    an FPU-tuned 2D program config (auto-selection overflows L1 for these shapes).
    """

    def __init__(self, weight_torch, mesh_device, mesh_mapper, k, n, dtype, cache_file_name):
        self.k = k
        self.n = n
        self._dtype = dtype
        self._prefill_max_cols = prefill_max_cols_default(mesh_device)
        self.weight = ttnn.as_tensor(
            weight_torch,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            cache_file_name=cache_file_name,
            memory_config=weight_memcfg(k, n),
        )
        # Act shard grid must match decode_progcfg (K and N), not a K-only grid.
        self._act_memcfg = activation_memcfg(k, n)
        self._decode_pc = decode_progcfg(TILE_SIZE, k, n, dtype=dtype)

    def _prefill_pc(self, m):
        return prefill_progcfg(m, self.k, self.n, max_cols=self._prefill_max_cols)

    def __call__(self, x, compute_kernel_config=None, out_memory_config=None):
        out_mc = out_memory_config if out_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        # Prefill with batch>1 reshapes activations to [B, 1, S, K] (see
        # DecoderLayer). Row count for the matmul is the product of all leading
        # dims — not just shape[-2], which would make the cutoff reshape
        # (1, S/cutoff, cutoff, K) disagree with volume B*S*K.
        # ttnn Shape only supports integer indexing (no slices).
        x_shape = [int(x.shape[i]) for i in range(len(x.shape))]
        orig_leading = x_shape[:-1]
        n_in = x_shape[-1]
        M = 1
        for d in orig_leading:
            M *= d
        flat_shape = [1, 1, M, n_in]
        x_work = x if x_shape == flat_shape else ttnn.reshape(x, flat_shape)

        def _restore(out):
            out_leading = [int(out.shape[i]) for i in range(len(out.shape) - 1)]
            if out_leading == orig_leading:
                return out
            return ttnn.reshape(out, (*orig_leading, int(out.shape[-1])))

        if M <= TILE_SIZE:
            # Decode DRAM-sharded kernel + activation memcfg are tiled for
            # M=TILE_SIZE (32). Packed-verify / small-batch paths pass M=B*P
            # (e.g. 4, 16) — pad to one tile, run, then slice back so callers
            # keep the logical [..., M, N] volume (avoids reshape volume mismatch).
            pad = TILE_SIZE - M
            x_run = x_work
            if pad:
                x_run = ttnn.pad(x_work, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            x_sh = ttnn.to_memory_config(x_run, self._act_memcfg)
            if pad:
                x_run.deallocate(True)
            out = ttnn.linear(
                x_sh,
                self.weight,
                program_config=self._decode_pc,
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            x_sh.deallocate(True)
            out = ttnn.to_memory_config(out, out_mc)
            if pad:
                out_pad = out
                out = ttnn.slice(
                    out_pad,
                    [0, 0, 0, 0],
                    [out_pad.shape[0], out_pad.shape[1], M, out_pad.shape[3]],
                )
                out_pad.deallocate(True)
            return _restore(out)

        # Prefill on the width-sharded weight via the 2D matmul kernel.
        if M <= _PREFILL_CUTOFF:
            pc = self._prefill_pc(M)
            out = ttnn.linear(
                x_work,
                self.weight,
                program_config=pc,
                compute_kernel_config=compute_kernel_config,
                memory_config=out_mc,
            )
            return _restore(out)

        if M % _PREFILL_CUTOFF == 0:
            # Reshape M into (batch, cutoff) so per_core_M is sized to the cutoff
            # (tiny CBs) and the batch dim is iterated by the kernel. Both
            # reshapes are metadata-only (cutoff is tile-aligned). Single matmul,
            # single-size output — no concat, no memory doubling.
            batch = M // _PREFILL_CUTOFF
            x_r = ttnn.reshape(x_work, (1, batch, _PREFILL_CUTOFF, n_in))
            pc = self._prefill_pc(_PREFILL_CUTOFF)
            out_r = ttnn.linear(
                x_r, self.weight, program_config=pc, compute_kernel_config=compute_kernel_config, memory_config=out_mc
            )
            out = ttnn.reshape(out_r, (1, 1, M, out_r.shape[-1]))
            return _restore(out)

        # Fallback for M not divisible by the cutoff (rare; small M in practice):
        # chunk + concat. Only reached for shapes that don't hit the long-context
        # memory pressure, so the concat's transient extra buffer is affordable.
        outs = []
        for start in range(0, M, _PREFILL_M_CHUNK):
            end = min(start + _PREFILL_M_CHUNK, M)
            x_c = ttnn.slice(x_work, [0, 0, start, 0], [1, 1, end, self.k])
            pc = self._prefill_pc(end - start)
            outs.append(
                ttnn.linear(
                    x_c,
                    self.weight,
                    program_config=pc,
                    compute_kernel_config=compute_kernel_config,
                    memory_config=out_mc,
                )
            )
            x_c.deallocate(True)
        out = ttnn.concat(outs, dim=-2, memory_config=out_mc)
        for o in outs:
            o.deallocate(True)
        return _restore(out)


def decode_in0_l1_enabled() -> bool:
    """Un-shard decode matmul in0 into L1 rather than DRAM. Default ON."""
    return os.environ.get("GEMMA4_DECODE_IN0_L1", "1").lower() not in ("0", "false", "no")


def width_shard_core_count(memcfg):
    if memcfg is None or not memcfg.is_sharded() or memcfg.shard_spec is None:
        return None
    box = memcfg.shard_spec.grid.bounding_box().grid_size()
    return int(box.x) * int(box.y)


def width_shard_matches_1d_progcfg(memcfg, program_config) -> bool:
    """True when a width-sharded in0's core grid equals the 1D matmul grid."""
    if memcfg is None or program_config is None or not memcfg.is_sharded():
        return False
    spec = memcfg.shard_spec
    if spec is None:
        return False
    box = spec.grid.bounding_box().grid_size()
    pc_x, pc_y = _program_grid(program_config)
    return int(box.x) == pc_x and int(box.y) == pc_y


def prefill_progcfg_1d_for_width_sharded_in0(m, k, n, in0_memcfg, grid_size=None):
    """1D progcfg whose core grid matches ``in0_memcfg``, or ``None`` if impossible.

    Prefers the sharded-in0 core count (LN island) over the interleaved sweep
    winner. ``fuse_batch=True`` is required when in0 is sharded. ``in0_block_w``
    must divide per-core K tiles, not full ``kt``.
    """
    cores = width_shard_core_count(in0_memcfg)
    if cores is None:
        return None
    kt = math.ceil(k / TILE_SIZE)
    if kt % cores:
        return None
    in0_block_w = _find_largest_divisor(kt // cores, max_div=4)
    program_config = prefill_progcfg_1d(
        m, k, n, cores=cores, grid_size=grid_size, fuse_batch=True, in0_block_w=in0_block_w
    )
    if program_config is None or not width_shard_matches_1d_progcfg(in0_memcfg, program_config):
        return None
    return program_config
