# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM-width-sharded matmul helpers for Gemma4 tensor-parallel decode.

Decode is weight-read-bound (M<=32, one activation tile), so spreading each
per-device weight shard across all DRAM banks and running the DRAM-sharded
matmul kernel cuts the per-token weight-read time. Prefill (M>32) reuses the
same width-sharded weight through a 2D matmul program config.

Ported/adapted from the Qwen3.6 Blackhole TP path (tp_common.py).
"""

import math
import os

import ttnn
from models.common.utility_functions import is_blackhole

TILE_SIZE = 32
# P150 Blackhole DRAM bank count. Wormhole meshes differ — can_dram_shard is
# BH-only so this constant is never applied on WH (wrong bank count → garbage).
DRAM_CORES = 8
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})
# BH QuietBox / P150 usable L1 for statically allocated CBs.
_L1_MAX_BYTES = 1_572_864
_L1_HEADROOM_BYTES = 64_000
# Cap decode in0_block_w: unbounded divisors (e.g. 6) blow L1 on 31B gate_up bf16.
_DECODE_IN0_BLOCK_W_MAX = 2


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


def decode_progcfg(m, k, n):
    """DRAM-sharded matmul program config for decode (small M)."""
    k_tiles = math.ceil(k / TILE_SIZE)
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    k_tiles_per_core = k_tiles // num_cores
    if k_tiles_per_core == 0:
        k_tiles_per_core = k_tiles
        num_cores = 1
    in0_block_w = _find_largest_divisor(k_tiles_per_core, max_div=_DECODE_IN0_BLOCK_W_MAX)
    per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=per_core_N,
        fused_activation=None,
    )


def activation_memcfg(k):
    """WIDTH_SHARDED L1 activation config for a [*, k] activation."""
    k_tiles = k // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, k // num_cores),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _tile_size_bytes(dtype=None):
    """Approximate single-tile footprint for L1 CB budgeting."""
    if dtype in (ttnn.bfloat8_b, getattr(ttnn, "bfloat4_b", None)):
        return 1088
    return 2048  # bfloat16 / unknown — conservative


def _estimate_decode_l1_bytes(k, n, dtype=None):
    """Rough static-CB estimate for the DRAM-sharded decode kernel (in1-dominated)."""
    k_tiles = k // TILE_SIZE
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    k_tiles_per_core = max(1, k_tiles // num_cores)
    in0_block_w = _find_largest_divisor(k_tiles_per_core, max_div=_DECODE_IN0_BLOCK_W_MAX)
    tile_aligned = _roundup(_tile_size_bytes(dtype), 64)
    # in1 triple-buffer × padded-N/DRAM_CORES × in0_block_w (factory layout).
    in1 = math.ceil(n_tiles / DRAM_CORES) * in0_block_w * 3 * tile_aligned
    # in0 / out / interm / reshard overhead (order-of-magnitude pad).
    return in1 + 200_000


def can_dram_shard(k, n, dtype=None):
    """True if a [k, n] weight shard is safe for the DRAM-sharded decode path.

    Blackhole-only: ``DRAM_CORES`` matches P150; Wormhole bank counts differ and
    produce garbage (CI PCC ~0). Also rejects shapes that would overflow L1 CBs
    (e.g. 31B fused gate_up @ TP=4 with bf16).
    """
    if not is_blackhole():
        return False
    if k % TILE_SIZE != 0 or n <= 0:
        return False
    try:
        rows, cols = _find_grid(k // TILE_SIZE)
    except ValueError:
        return False
    num_cores = rows * cols
    # Activation width-shard needs k evenly split across the core grid.
    if (k // TILE_SIZE) % num_cores != 0 or (k // num_cores) % TILE_SIZE != 0:
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


def l1_block_sharded_memcfg(rows, cols, grid=None):
    """L1 BLOCK_SHARDED memory config for a 2D activation/output ``(rows, cols)``.

    Matches ``sweeps/sweep_common.act_memcfg(..., "l1_block_sharded")``: split
    row-tiles over y and col-tiles over x, taking the largest divisors that fit
    the worker grid. Used for the fused-QKV prefill matmul output — the measured
    winner in ``test_qkv_matmul_sweep`` (M=128 K=5376 N=2048 → CoreGrid 8x4).
    """
    if grid is None:
        grid = prefill_grid_default()  # (x, y) = (cols, rows) of worker grid
    grid_x, grid_y = grid
    row_tiles = math.ceil(rows / TILE_SIZE)
    col_tiles = math.ceil(cols / TILE_SIZE)
    ys = [y for y in range(1, grid_y + 1) if row_tiles % y == 0]
    xs = [x for x in range(1, grid_x + 1) if col_tiles % x == 0]
    if not ys or not xs:
        # Shape cannot block-shard evenly — fall back to interleaved L1.
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.create_sharded_memory_config(
        shape=(rows, cols),
        core_grid=ttnn.CoreGrid(x=max(xs), y=max(ys)),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def interleaved_prefill_config(m, k, n):
    """``(program_config, out_memory_config, compute_kernel_config)`` for a prefill
    matmul on a DRAM-*interleaved* weight, or all-``None`` to keep ttnn's auto
    selection (byte-identical to passing nothing).

    Off Blackhole ``can_dram_shard`` is always False, so the projections never
    reach ``DramShardedLinear`` and ttnn auto-selects with no program config at
    all. For the fused QKV shape (M=128 K=5376 N=2048 at TP=8) it picks
    ``per_core_N=1``, which forces ``out_subblock_w=1`` and starves the FPU —
    39.9% DRAM and 10.9% FPU in the 1x8 profile, saturating neither.
    ``prefill_progcfg`` returns 8x8 / ``in0_block_w=4`` / ``out_subblock 1x4``
    for that shape, measured ~1.35x faster at identical PCC by
    ``models/demos/gemma4/sweeps/test_qkv_matmul_sweep.py``.

    Bounded by ``_PREFILL_CUTOFF`` for the same reason ``DramShardedLinear``
    chunks there: the 2D kernel's CBs scale with ``per_core_M``, so pinning this
    config at long context would blow L1. Above the cutoff (and for decode,
    M<=32) we return ``(None, None)`` and the caller behaves exactly as before.

    Output is L1 *block-sharded* (sweep overall winner for QKV: ~1.40x vs auto),
    not interleaved L1. Callers that need interleaved/DRAM for
    ``nlp_create_qkv_heads`` / reshape must ``sharded_to_interleaved`` after the
    matmul. Deliberately *not* included: a width-sharded weight measured slightly
    faster again, but the same weight tensor also serves decode, and decode's
    auto-selected matmul rejects a width-sharded in1 with a circular-buffer
    TT_THROW. That variant needs a second weight copy, so it is not free and is
    left out.

    Math fidelity is pinned to HiFi2 — *not* cosmetic. Supplying a program config
    changes what ttnn's default compute-kernel-config selection picks: a profile
    of this path with the config but no explicit fidelity came back as LoFi where
    the auto-selected baseline was HiFi2. That silently trades accuracy for part
    of the speedup and still clears a 0.99 PCC gate, so it does not show up as a
    test failure. Pin it, and the config change is a pure scheduling change.
    """
    if not TILE_SIZE < m <= _PREFILL_CUTOFF:
        return None, None, None
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    return prefill_progcfg(m, k, n), l1_block_sharded_memcfg(m, n), compute_kernel_config


def _out_subblock_hw(per_core_n, per_core_m):
    """Largest ``(h, w)`` with ``h*w <= 4``, ``w|per_core_n``, ``h|per_core_m``."""
    best = (1, 1)
    for h in range(1, min(per_core_m, 4) + 1):
        if per_core_m % h:
            continue
        for w in range(1, min(per_core_n, 4 // h) + 1):
            if per_core_n % w == 0 and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def _factor_1d_grid(cores, grid_x, grid_y):
    """``(cols, rows)`` packing ``cores`` into ``grid_x x grid_y``, or ``None``."""
    cols = min(grid_x, cores)
    while cols > 1 and cores % cols:
        cols -= 1
    rows = cores // cols
    if rows < 1 or rows > grid_y:
        return None
    return cols, rows


def _pick_1d_cores(n_tiles, grid_x, grid_y, prefer=42):
    """Core count dividing ``n_tiles`` that fits the worker grid; prefer ``prefer``.

    ``prefer=42`` is the measured gate+up winner on WH 8x8 (Nt=168 → 42 cores).
    """
    max_cores = grid_x * grid_y
    candidates = [
        c for c in range(8, max_cores + 1) if n_tiles % c == 0 and _factor_1d_grid(c, grid_x, grid_y) is not None
    ]
    if not candidates:
        return None
    if prefer in candidates:
        return prefer
    return max(candidates, key=lambda c: (-abs(c - prefer), c))


def prefill_progcfg_1d(m, k, n, cores=None, in0_block_w=None, grid_size=None):
    """1D-multicast prefill program config (``MatmulMultiCoreReuseMultiCast1D``).

    Sweep family for the fused gate+up shape: every core holds all of M and a
    slice of N. Measured winner for M=128 K=5376 N=5376 (31B TP=8) is
    ``1d_c42_bw4`` — see ``test_gate_up_matmul_sweep``. Returns ``None`` when
    no valid core count divides N-tiles into the worker grid.
    """
    if grid_size is None:
        grid_size = prefill_grid_default()
    grid_x, grid_y = grid_size
    mt = math.ceil(m / TILE_SIZE)
    kt = math.ceil(k / TILE_SIZE)
    nt = math.ceil(n / TILE_SIZE)
    if cores is None:
        cores = _pick_1d_cores(nt, grid_x, grid_y, prefer=42)
    if cores is None or nt % cores:
        return None
    factored = _factor_1d_grid(cores, grid_x, grid_y)
    if factored is None:
        return None
    cols, rows = factored
    if in0_block_w is None:
        in0_block_w = _find_largest_divisor(kt, max_div=4)
    if kt % in0_block_w:
        return None
    per_core_n = nt // cores
    out_h, out_w = _out_subblock_hw(per_core_n, mt)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=out_h,
        out_subblock_w=out_w,
        per_core_M=mt,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet(set()),
        num_global_cb_receivers=0,
        untilize_out=False,
    )


def interleaved_gate_up_prefill_config(m, k, n):
    """``(program_config, out_memory_config, compute_kernel_config)`` for fused
    gate+up on a DRAM-*interleaved* weight, or all-``None`` for ttnn auto.

    Off Blackhole ``can_dram_shard`` is False, so ``SharedMLP.gate_up_proj`` is a
    bare ``ttnn.linear``. ``test_gate_up_matmul_sweep`` ranks the overall winner
    for M=128 K=5376 N=5376 at TP=8 as ``1d_c42_bw4`` + L1-interleaved in0/out +
    HiFi2 / bfp8 (~1.27x vs shipped auto). ``test_gate_up_output_slice_cost``
    confirmed L1-interleaved out is consumable by the GeGLU ``ttnn.slice`` split
    (and faster as a matmul+slice group than DRAM out).

    Same ``_PREFILL_CUTOFF`` band as ``interleaved_prefill_config``: residency
    measured ``in0+out`` L1 interleaved up to ISL 1024 on WH; above that (and for
    decode ``M<=32``) return ``None`` and keep the prior auto path. Callers that
    want the full win should also move in0 to L1 interleaved when it is still in
    DRAM (rms_norm leaves it there by default).
    """
    if not TILE_SIZE < m <= _PREFILL_CUTOFF:
        return None, None, None
    program_config = prefill_progcfg_1d(m, k, n)
    if program_config is None:
        return None, None, None
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    return program_config, ttnn.L1_MEMORY_CONFIG, compute_kernel_config


def matmul_rows(x):
    """Row count a matmul sees for ``x``: the product of all but the last dim.

    Batched prefill hands in ``[B, 1, S, K]``, so ``shape[-2]`` alone would
    undercount by a factor of B (same reasoning as ``DramShardedLinear.__call__``).
    """
    rows = 1
    for i in range(len(x.shape) - 1):
        rows *= int(x.shape[i])
    return rows


class DramShardedLinear:
    """A single DRAM-width-sharded weight served for both decode and prefill.

    Decode (M<=32): width-shard the activation to L1, run the DRAM-sharded
    kernel, return a DRAM-interleaved result. Prefill (M>32): plain matmul with
    an FPU-tuned 2D program config (auto-selection overflows L1 for these shapes).
    """

    def __init__(self, weight_torch, mesh_device, mesh_mapper, k, n, dtype, cache_file_name):
        self.k = k
        self.n = n
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
        self._act_memcfg = activation_memcfg(k)
        self._decode_pc = decode_progcfg(TILE_SIZE, k, n)

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
