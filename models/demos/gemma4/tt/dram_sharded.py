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
from models.common.tensor_utils import with_dram_sharded_cores_per_bank
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.mesh_orientation import decode_flag

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
# GEMMA4_DRAM_SHARD_ROW_VIEW=1: decode rows below one tile (batch < 32) take the matmul through metadata reshapes
# of the row tile instead of a Pad program before and a Slice program after (see DramShardedLinear.__call__).
# GEMMA4_DRAM_SHARD_DECODE_1D=1: decode through the 1D-multicast matmul (tt/decode_matmul.py, every worker core one
# N tile, K streamed in GEMMA4_DECODE_MM_IN0_BLOCK_W blocks) on an interleaved on-device copy of the width-sharded
# weight, instead of the DRAM-sharded kernel (whose core grid must divide both K and N tiles: 8 cores for the
# dense 31B / 12B wqkv and o_proj) with its activation reshard before and after: 3 programs -> 1 per matmul. The
# copy is made at load from the warm ``.ws`` cache (no second cache file); prefill keeps its 2D config.
# Both are decode fusion flags (tt/mesh_orientation.py): the env var when set, else the build profile's default.


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

    Same contract as ``tt_transformers.ModelArgs.find_grid_k_n``. A K-only grid with
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
# The slab must fill the grid's rows: a 512-row slab is 16 row tiles, which the Blackhole (cols, 10) grid serves with
# per_core_M = 2 on 8 of its 10 rows (the 0522 profile: every prefill matmul on 56 / 64 cores, at the floor of those
# cores). Blackhole: grid rows x 8 row tiles (2560 rows; the 8 tiles per core fit the CBs); a prompt the slab does not
# divide (2048, 8192, ...) takes the remainder path below in chunks of the same size, each filling all rows.
# Wormhole keeps 1024 rows on its (8,8) grid (32 tiles = 4 per core on all 8 rows). GEMMA4_PREFILL_CUTOFF overrides.
_PREFILL_CUTOFF = int(
    os.environ.get("GEMMA4_PREFILL_CUTOFF") or (prefill_grid_default()[1] * 8 * TILE_SIZE if is_blackhole() else 1024)
)
# Leftover rows after the full slabs are processed in chunks of this size (default 1024: the pre-slab configs).
_PREFILL_REMAINDER_CHUNK = int(os.environ.get("GEMMA4_PREFILL_REMAINDER_CHUNK") or min(1024, _PREFILL_CUTOFF))
# The 2D matmul's static CB region must leave this much L1 free (serving interleaves prefills with decode-time L1
# residents; the unit tests keep a 33 KB buffer at the top of L1): the slab shrinks per weight until it fits. 0 = no cap.
# 64 KB (measured 2026-10-24): the 4096-token unit tests pass on both models, the 31B keeps its 2560-row slab (bfp8 MLP
# weights) and the 12B's 2240-row slab still takes a 2K prompt as one matmul; 128 KB cost the 31B +1.0 % at 2K.
_PREFILL_L1_HEADROOM = int(os.environ.get("GEMMA4_PREFILL_L1_HEADROOM_KB", "64")) * 1024
# CB region start + the kernel's small CBs, measured on Blackhole: program 1650 (M 8, N 48, k-block 4, bf16 weights)
# reported 1815936 B = 112000 + in0 + in1 + out below.
_PREFILL_CB_BASE = 112000
_L1_BYTES = 1572864 if is_blackhole() else 1499136


def prefill_cb_bytes(per_core_M, per_core_N, in0_block_w, weight_tile_bytes):
    """Static CB bytes per core of the 2D prefill matmul: in0 and in1 double-buffered, one bf16 output block."""
    return (
        _PREFILL_CB_BASE
        + 2 * per_core_M * in0_block_w * 2048
        + 2 * in0_block_w * per_core_N * weight_tile_bytes
        + per_core_M * per_core_N * 2048
    )


def weight_tile_bytes(dtype) -> int:
    return {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}.get(dtype, 2048)


def prefill_slab_rows(k, n, dtype, max_cols=None, grid=None, headroom=None, l1_bytes=None):
    """Rows of one prefill slab for a weight of this shape / dtype: the largest per_core_M <= 8 (x grid rows x 32)
    whose static CBs leave ``headroom`` bytes of L1 (GEMMA4_PREFILL_L1_HEADROOM_KB; 0 = the full 8-tile slab).
    Calibrated examples (Blackhole): at 128 KB 31B gate_up bfp8 2240 rows, the same weight in bf16 (the unit tests)
    1280, 12B gate_up bf16 1920; at the 64 KB default the 31B keeps 2560 and the 12B takes 2240; narrow weights 2560."""
    grid = grid or prefill_grid_default()
    headroom = _PREFILL_L1_HEADROOM if headroom is None else headroom
    rows = grid[1] * TILE_SIZE
    if headroom <= 0:
        return 8 * rows
    cols = _best_prefill_cols(n, max_cols if max_cols is not None else grid[0])
    per_core_N = max(1, math.ceil(n / TILE_SIZE / cols))
    k_tiles = math.ceil(k / TILE_SIZE)
    in0_block_w = _find_largest_divisor(k_tiles, max_div=min(4, max(1, k_tiles // max(1, cols))))
    budget = (l1_bytes or _L1_BYTES) - headroom
    for m in range(8, 0, -1):
        if prefill_cb_bytes(m, per_core_N, in0_block_w, weight_tile_bytes(dtype)) <= budget:
            return m * rows
    return rows


# Fallback per-call row cap for the (rare) M not divisible by the cutoff.
_PREFILL_M_CHUNK = prefill_grid_default()[1] * 8 * TILE_SIZE


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
    # A block may span several activation shards, so the multicast block count is no longer tied
    # to the shard grid. Widen past k_tiles_per_core while the in1 triple-buffer still fits; never
    # return less than the width above, or a wide-N projection's block count gets worse.
    k_tiles = max(1, k // TILE_SIZE)
    for bw in range(min(16, k_tiles), in0, -1):
        if k_tiles % bw:
            continue
        if k_tiles_per_core % bw and bw % k_tiles_per_core:
            continue
        if _estimate_decode_l1_bytes_for_in0(n_tiles, bw, dtype) <= budget:
            return bw
    return in0


def decode_progcfg(m, k, n, dtype=None):
    """DRAM-sharded matmul program config for decode (small M).

    Matches tt_transformers / mlp_1d:
      * core grid divides both K and N tiles (``find_grid_k_n``)
      * ``per_core_N = ceil(n / (tile * num_cores))`` — never floor-truncate
      * ``in0_block_w`` from K/core, L1-shrunk when needed
    """
    _rows, _cols, num_cores = _decode_core_grid(k, n)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_decode_in0_block_w(k, n, num_cores, dtype=dtype),
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=math.ceil(n / (TILE_SIZE * num_cores)),
        fused_activation=None,
    )
    return with_dram_sharded_cores_per_bank(program_config, m, k, n, num_cores)


def activation_memcfg(k, n):
    """WIDTH_SHARDED L1 activation config matching ``decode_progcfg``'s core grid."""
    grid_rows, cols, num_cores = _decode_core_grid(k, n)
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, k // num_cores),
        core_grid=ttnn.CoreGrid(x=cols, y=grid_rows),
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
    # A short prefill (4 row tiles at 128 tokens) occupies only M/32 of the grid rows: the 2D kernel maps M over the
    # rows, transpose_mcast is rejected for a sharded weight and the DRAM-sharded kernel takes one row tile only
    # (measured 2026-10-24), so the small-M grid is not improvable here without a kernel or weight-layout change.
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


def prefill_row_chunks(m, slab=None, remainder=None):
    """Row ranges of a prefill matmul whose M is not a multiple of the slab: full slabs first, then the leftover in
    chunks of at most ``remainder`` rows (the pre-slab configs; see ``__call__``)."""
    slab = slab or _PREFILL_M_CHUNK
    remainder = remainder or _PREFILL_REMAINDER_CHUNK
    bounds, start = [], 0
    while start < m:
        step = slab if m - start >= slab else remainder
        bounds.append((start, min(start + step, m)))
        start += step
    return bounds


class DramShardedLinear:
    """A single DRAM-width-sharded weight served for both decode and prefill.

    Decode (M<=32): width-shard the activation to L1, run the DRAM-sharded
    kernel, return a DRAM-interleaved result. Prefill (M>32): plain matmul with
    an FPU-tuned 2D program config (auto-selection overflows L1 for these shapes).
    """

    def __init__(self, weight_torch, mesh_device, mesh_mapper, k, n, dtype, cache_file_name, pipeline_row=None):
        """``pipeline_row``: ``(workers per DRAM bank, in0_block_w)`` from ``decode_matmul.pipeline_row_for`` puts the
        decode kernel on its multi-core pipeline (readers per bank reading the activation from its L1 shards; both
        decode entry points, the prefill config untouched); None = the single-reader kernel."""
        self.k = k
        self.n = n
        self._dtype = dtype
        self._prefill_max_cols = prefill_max_cols_default(mesh_device)
        # the slab this weight's CBs allow (GEMMA4_PREFILL_CUTOFF still pins it)
        self._slab_rows = (
            _PREFILL_CUTOFF
            if os.environ.get("GEMMA4_PREFILL_CUTOFF")
            else min(_PREFILL_CUTOFF, prefill_slab_rows(k, n, dtype, max_cols=self._prefill_max_cols))
        )
        if self._slab_rows < _PREFILL_CUTOFF:  # the rule's decision, once per weight (the knob arms grep this)
            logger.info(
                f"prefill slab {self._slab_rows} rows for k={k} n={n} dtype={dtype} (headroom {_PREFILL_L1_HEADROOM // 1024} KB)"
            )
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
        self.pipeline_row = None
        if pipeline_row is not None:
            workers, block_w = int(pipeline_row[0]), int(pipeline_row[1])
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block_w,
                per_core_M=self._decode_pc.per_core_M,
                per_core_N=self._decode_pc.per_core_N,
                fused_activation=None,
            )
            pc.num_workers_per_dram_bank = workers
            pc.use_multicore_pipeline = True
            self._decode_pc = pc
            self.pipeline_row = (workers, block_w)
        # The pipeline splits a column's K over the bank's workers and, without fp32 Dest accumulation, adds the
        # partials in bf16 (16-bit Dest groups + a Float16_b L1 accumulator + the bf16 partial exchange): on real
        # activations that cost the 12B 4 / 385 top-1 tokens. With fp32 Dest accumulation the whole K of a column
        # pass stays in fp32 Dest (no L1 partials), more accurate than the single-reader kernel; the caller's own
        # compute config, when given, still wins. GEMMA4_DRAM_SHARD_PIPELINE=l1acc keeps the packer-L1 form.
        self._pipeline_ckc = None
        if pipeline_row is not None:
            from models.demos.gemma4.tt.decode_matmul import pipeline_fp32_acc

            if pipeline_fp32_acc():
                self._pipeline_ckc = ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(),
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                )
        self._decode_1d = None
        if decode_flag("GEMMA4_DRAM_SHARD_DECODE_1D") == "1":
            from models.demos.gemma4.tt.decode_matmul import decode_1d_progcfg, decode_compute_kernel_config

            sharded = self.weight
            self.weight = ttnn.to_memory_config(sharded, ttnn.DRAM_MEMORY_CONFIG)
            sharded.deallocate(True)
            n_padded = int(self.weight.shape[-1])
            self._decode_1d = (
                decode_1d_progcfg(mesh_device, k, n_padded),
                decode_compute_kernel_config(mesh_device),
            )

    def _prefill_pc(self, m):
        return prefill_progcfg(m, self.k, self.n, max_cols=self._prefill_max_cols)

    def _decode_ckc(self, compute_kernel_config):
        """The decode kernel's compute config: the caller's, else the pipelined row's fp32-Dest config, else None."""
        return compute_kernel_config if compute_kernel_config is not None else self._pipeline_ckc

    @property
    def act_memory_config(self):
        """The L1 width-sharded layout the decode kernel reads its activation in (one row tile)."""
        return self._act_memcfg

    def accepts_sharded_rows(self) -> bool:
        """True when a producer may hand this matmul its decode activation already in ``act_memory_config``
        (the DRAM-sharded kernel; the 1D decode path reads interleaved rows)."""
        return self._decode_1d is None

    def _decode_sharded_rows(self, x_sh, out_mc, compute_kernel_config, m_rows, keep_sharded=False):
        """Decode on an activation that is already width-sharded in ``act_memory_config`` (no reshard program)."""
        out = ttnn.linear(
            x_sh,
            self.weight,
            program_config=self._decode_pc,
            compute_kernel_config=self._decode_ckc(compute_kernel_config),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        x_sh.deallocate(True)
        if keep_sharded:
            return out
        out = ttnn.to_memory_config(out, out_mc)
        if m_rows < TILE_SIZE:
            n_out = int(out.shape[-1])
            out = ttnn.reshape(out, ttnn.Shape([1, 1, m_rows, n_out]), ttnn.Shape([1, 1, TILE_SIZE, n_out]))
        return out

    def __call__(self, x, compute_kernel_config=None, out_memory_config=None, keep_sharded=False, rows=None):
        """``keep_sharded``: a decode (one row tile) result is returned in the kernel's L1 width-sharded output
        layout (no sharded -> interleaved copy; the consumer reads the shards). ``rows``: the logical row count of
        the result when the input is a width-sharded one-row-tile activation (its logical rows are the tile)."""
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
        if x.is_sharded() and M <= TILE_SIZE and self._decode_1d is None and x.memory_config() == self._act_memcfg:
            return self._decode_sharded_rows(
                x, out_mc, compute_kernel_config, M if rows is None else int(rows), keep_sharded=keep_sharded
            )
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
            # GEMMA4_DRAM_SHARD_ROW_VIEW=1: a TILE activation of M < 32 logical rows already occupies one row tile
            # (padded_shape rows == 32) and the decode kernel is row-independent, so the tile is presented as 32
            # logical rows (a metadata reshape) and the result as M rows again, instead of a Pad program before and
            # a Slice program after the matmul (2 of the 5 programs per decode matmul at batch 1).
            row_view = (
                pad > 0
                and decode_flag("GEMMA4_DRAM_SHARD_ROW_VIEW") == "1"
                and x_work.layout == ttnn.TILE_LAYOUT
                and int(x_work.padded_shape[-2]) == TILE_SIZE
                and len(x_work.shape) == 4
            )
            if row_view:
                full = ttnn.Shape([1, 1, TILE_SIZE, n_in])
                x_run = ttnn.reshape(x_work, full, full)
            elif pad:
                x_run = ttnn.pad(x_work, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            if self._decode_1d is not None:
                # interleaved weight, 1D multicast config (no activation reshard)
                pc, ckc = self._decode_1d
                out = ttnn.linear(
                    x_run,
                    self.weight,
                    program_config=pc,
                    compute_kernel_config=compute_kernel_config if compute_kernel_config is not None else ckc,
                    memory_config=out_mc,
                )
                if pad and not row_view:
                    x_run.deallocate(True)
            else:
                x_sh = ttnn.to_memory_config(x_run, self._act_memcfg)
                if pad and not row_view:
                    x_run.deallocate(True)
                out = ttnn.linear(
                    x_sh,
                    self.weight,
                    program_config=self._decode_pc,
                    compute_kernel_config=self._decode_ckc(compute_kernel_config),
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                x_sh.deallocate(True)
                if keep_sharded:
                    return out  # the kernel's L1 width-sharded form (one row tile; the consumer reads the shards)
                out = ttnn.to_memory_config(out, out_mc)
            if row_view:
                n_out = int(out.shape[-1])
                out = ttnn.reshape(out, ttnn.Shape([1, 1, M, n_out]), ttnn.Shape([1, 1, TILE_SIZE, n_out]))
            elif pad:
                out_pad = out
                out = ttnn.slice(
                    out_pad,
                    [0, 0, 0, 0],
                    [out_pad.shape[0], out_pad.shape[1], M, out_pad.shape[3]],
                )
                out_pad.deallocate(True)
            return _restore(out)

        # Prefill on the width-sharded weight via the 2D matmul kernel.
        if os.environ.get("GEMMA4_DEBUG_L1_DUMP") == "1" and not getattr(self, "_l1_dumped", False):
            self._l1_dumped = True  # diagnostics: the L1 residents a program's CBs must stay below
            ttnn.dump_device_memory_state(x_work.device(), f"prefill_matmul_k{self.k}_n{self.n}_M{M}")
        slab = self._slab_rows
        if M <= slab:
            pc = self._prefill_pc(M)
            out = ttnn.linear(
                x_work,
                self.weight,
                program_config=pc,
                compute_kernel_config=compute_kernel_config,
                memory_config=out_mc,
            )
            return _restore(out)

        if M % slab == 0:
            # Reshape M into (batch, slab) so per_core_M is sized to the slab
            # (tiny CBs) and the batch dim is iterated by the kernel. Both
            # reshapes are metadata-only (the slab is tile-aligned). Single matmul,
            # single-size output — no concat, no memory doubling.
            batch = M // slab
            x_r = ttnn.reshape(x_work, (1, batch, slab, n_in))
            pc = self._prefill_pc(slab)
            out_r = ttnn.linear(
                x_r, self.weight, program_config=pc, compute_kernel_config=compute_kernel_config, memory_config=out_mc
            )
            out = ttnn.reshape(out_r, (1, 1, M, out_r.shape[-1]))
            return _restore(out)

        # M not divisible by the slab: full slabs first, then the leftover in chunks of at most
        # _PREFILL_REMAINDER_CHUNK rows (the pre-slab 1024-row configs, per_core_M <= 4): an odd leftover such as
        # 4096 = 2560 + 1536 sized its own 1536-row config (per_core_M 5) whose CBs overflow L1 on the 31B's gate_up
        # (7x10 grid, 1815936 B) and clash with live L1 buffers on the 12B (CI 2026-10-24, test_layer_forward /
        # test_shared_mlp at 4096); 8192 = 3 x 2560 + 512 is unchanged. chunk + concat; only reached for shapes
        # that don't hit the long-context memory pressure, so the concat's transient extra buffer is affordable.
        outs = []
        for start, end in prefill_row_chunks(M, slab=slab, remainder=min(_PREFILL_REMAINDER_CHUNK, slab)):
            # a slice over the whole extent is a view of x_work (freeing it would free the caller's activation)
            whole = start == 0 and end == M
            x_c = x_work if whole else ttnn.slice(x_work, [0, 0, start, 0], [1, 1, end, self.k])
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
            if not whole:
                x_c.deallocate(True)
        out = ttnn.concat(outs, dim=-2, memory_config=out_mc)
        for o in outs:
            o.deallocate(True)
        return _restore(out)
