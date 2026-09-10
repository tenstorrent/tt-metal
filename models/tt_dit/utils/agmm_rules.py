# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Rule-based layout and blocking for `all_gather_minimal_matmul_async` (the "v3 rules").

Fitted against ~4,600 block-sweep measurements on 4x8 Blackhole Galaxy (three H3 AGMM
projection shapes, K in {5376, 7168}, N in {1344, 5376, 7168}, M from 320 to 15,356) and
validated blind on two rounds of 20 stratified-random tile-unaligned Ms each: 86/89 blind
shapes within 5% of the swept optimum (round 1: 28/29, round 2: 58/60), ~1% mean penalty.
Sweep driver: `models/tt_dit/utils/sweep_mm_block_sizes.py`; per-shape results table in the
"AGMM Block Geometry" write-up.

The rules, in the order they are applied:

Layout (`pick_layout`)
  score(layout) = ceil(M_tiles / M_axis_cores) * ceil(N_tiles / N_axis_cores)
  - M > N: the op auto-transposes (`transpose_core_grid = M > N`, no override exists), so the
    only legal worker grid keeps the bottom device row free for the fabric muxes: (x, y-1).
  - Tiny M with large N (M_tiles <= 20, N_tiles >= 100): the ring gather dominates and hides
    compute imbalance; the full-width (x, y-1) grid wins even when its score is worse.
  - Otherwise strict argmin over the two non-transposed grids -- (x-1, y) (muxes in the free
    column) vs (x, y-1) (muxes in the free row) -- with a tie-break to full-width (x, y-1)
    when within 2% (15% for fused-SwiGLU shapes, whose narrow-grid N split consistently
    underperforms its score).

Blocking (`pick_blocks`): fewest, most balanced blocks per core that fit L1.
  - K_block: the largest divisor of K_tiles_per_device in [6, 8] (a hard op constraint is
    that K_block divides K_tiles_per_device; 6-8 amortizes reload without starving L1).
  - N_block: split count j ascending; within j the largest even n <= min(per_core_N, 16)
    whose last block is at least n/2 (a 16+3-style degenerate tail loses 20-35%). Odd n only
    when nothing even qualifies, and never for fused SwiGLU (gate/up tile pairs interleave
    along N; a block must not split a pair).
  - M_block: split count i ascending (a single block only when per_core_M <= 6);
    m = ceil(per_core_M / i), with odd non-divisors rounded up to even when the split count
    is preserved and total padding stays <= 1.25x. Accept m <= 10 that fits L1 and meets the
    demand -- at least 6, relaxed to 3 when one N block covers all of per_core_N (the
    L1-tight full-strip case). A skinny m under a fat n costs up to 37% on tall shapes.
  - Subblock: 2x2 when both blocks are even, else the largest h*w <= 4 dividing both
    (fp32 dest accumulation halves the DEST register file to 4 tiles).

Everything here is pure Python with no ttnn/torch imports so it can be unit-tested host-only;
`utils/matmul.py` wraps the result into a `MinimalMatmulConfig`.
"""

from __future__ import annotations

TILE = 32

# Block-size candidate bounds, matching the sweep that produced the fit.
MN_BLOCK_MIN, MN_BLOCK_MAX = 2, 16
K_BLOCK_MIN = 2

# L1 circular-buffer budget used to pre-filter block combos (KB). Blackhole usable L1 is
# ~1464 KB; the margin covers kernel/firmware overhead. The rules are only validated on
# Blackhole -- callers on other archs should stay on their legacy config path.
BLACKHOLE_L1_BUDGET_KB = 1400


def estimate_l1_kb(m_blk: int, k_blk: int, n_blk: int, use_addcmul: bool = False) -> int:
    """Estimate L1 circular-buffer footprint in KB for a block config.

    Mirrors minimal_matmul_program_factory.cpp CB allocation:
      c_0 (in0):    2 * M * K tiles  (double-buffered, bf16 = 2 KB/tile)
      c_1 (in1):    2 * K * N tiles  (double-buffered, bf16)
      c_2 (out):    2 * M * N tiles  (double-buffered, bf16)
      c_3 (interm): M * N tiles      (single-buffered, f32 = 4 KB/tile)
      c_4 (bias):   N tiles          (single-buffered, bf16)
    Fused addcmul adds: M * N tiles (ternary_a, bf16) + N tiles (ternary_c, bf16).
    """
    bf16_kb = 2
    f32_kb = 4
    kb = (
        2 * m_blk * k_blk * bf16_kb
        + 2 * k_blk * n_blk * bf16_kb
        + 2 * m_blk * n_blk * bf16_kb
        + m_blk * n_blk * f32_kb
        + n_blk * bf16_kb
    )
    if use_addcmul:
        kb += m_blk * n_blk * bf16_kb + n_blk * bf16_kb
    return kb


def get_k_block_candidates(k_per_device_tiles: int) -> list[int]:
    """Divisors of K_tiles_per_device (hard op constraint), floored at K_BLOCK_MIN."""
    return sorted(d for d in range(K_BLOCK_MIN, k_per_device_tiles + 1) if k_per_device_tiles % d == 0)


def get_mn_block_candidates(per_core_tiles: int) -> list[int]:
    """Evens in [MN_BLOCK_MIN, MN_BLOCK_MAX] union divisors of the per-core tile count."""
    evens = set(range(MN_BLOCK_MIN, MN_BLOCK_MAX + 1, 2))
    divisors = set(d for d in range(MN_BLOCK_MIN, MN_BLOCK_MAX + 1) if per_core_tiles % d == 0)
    return sorted(evens | divisors)


def pick_subblock(m_block: int, n_block: int, fp32_dest: bool = True) -> tuple[int, int]:
    """Best (sub_h, sub_w) with sub_h | m_block, sub_w | n_block, sub_h * sub_w <= DEST capacity.

    With fp32 dest the DEST register file holds 4 tiles; 2x2 is preferred among h*w == 4
    candidates (better math-LLK tile reuse than 4x1 / 1x4).
    """
    max_dest_volume = 4 if fp32_dest else 8
    if fp32_dest and m_block % 2 == 0 and n_block % 2 == 0:
        return (2, 2)
    best, best_product = (1, 1), 1
    for h in range(1, min(m_block, max_dest_volume) + 1):
        if m_block % h:
            continue
        for w in range(1, min(n_block, max_dest_volume) + 1):
            if n_block % w:
                continue
            if h * w <= max_dest_volume and h * w > best_product:
                best, best_product = (h, w), h * w
    return best


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def pick_layout(
    M: int, K: int, N: int, full_grid: tuple[int, int] = (12, 10), fuse_swiglu: bool = False
) -> tuple[tuple[int, int], bool]:
    """Pick the AGMM worker grid for the shape. Returns ((grid_x, grid_y), transposed).

    `full_grid` is the device's full worker grid; the op needs either the last column
    (non-transposed muxes) or the last row (transposed muxes, and the full-width
    non-transposed variant) kept free.
    """
    gx, gy = full_grid
    m_tiles, n_tiles = _ceil(M, TILE), _ceil(N, TILE)
    full_width = (gx, gy - 1)  # muxes on the free bottom row
    narrow = (gx - 1, gy)  # muxes in the free last column
    if M > N:
        return full_width, True  # the op auto-transposes; M runs over grid x
    if m_tiles <= 20 and n_tiles >= 100:
        return full_width, False  # gather-latency regime
    score_narrow = _ceil(m_tiles, narrow[1]) * _ceil(n_tiles, narrow[0])
    score_wide = _ceil(m_tiles, full_width[1]) * _ceil(n_tiles, full_width[0])
    tie = 1.15 if fuse_swiglu else 1.02
    if score_wide <= tie * score_narrow:
        return full_width, False
    return narrow, False


def _n_candidates(per_core_n: int, fuse_swiglu: bool) -> list[int]:
    out = []
    for j in range(1, per_core_n + 1):
        group = [
            n
            for n in range(2, min(per_core_n, MN_BLOCK_MAX) + 1)
            if _ceil(per_core_n, n) == j
            and (j == 1 or per_core_n - (j - 1) * n >= _ceil(n, 2))
            and not (fuse_swiglu and n % 2)
        ]
        group.sort(key=lambda n: (n % 2, -n))  # evens first (descending), then odds
        out.extend(group)
    return out


def _m_search(per_core_m, k_blk, n_blk, m_univ, demand, use_addcmul, l1_budget_kb):
    if per_core_m <= 2:
        return 2 if estimate_l1_kb(2, k_blk, n_blk, use_addcmul) <= l1_budget_kb else None
    i = 1 if per_core_m <= 6 else 2
    while i <= per_core_m:
        m = max(2, _ceil(per_core_m, i))
        if (
            m % 2
            and per_core_m % m != 0
            and m + 1 <= MN_BLOCK_MAX
            and _ceil(per_core_m, m + 1) == i
            and _ceil(per_core_m, m + 1) * (m + 1) <= 1.25 * per_core_m
        ):
            m += 1
        if m <= 10 and m in m_univ and estimate_l1_kb(m, k_blk, n_blk, use_addcmul) <= l1_budget_kb:
            return m if m >= demand else None
        if m not in m_univ or m > 10:
            for d in sorted((d for d in m_univ if per_core_m % d == 0 and 2 <= d <= 10), reverse=True):
                if d >= max(demand, 6) and estimate_l1_kb(d, k_blk, n_blk, use_addcmul) <= l1_budget_kb:
                    return d
        i += 1
    return None


def pick_blocks(
    per_core_m: int,
    per_core_n: int,
    k_per_device_tiles: int,
    fuse_swiglu: bool = False,
    use_addcmul: bool = False,
    l1_budget_kb: int = BLACKHOLE_L1_BUDGET_KB,
) -> tuple[int, int, int, int, int] | None:
    """Pick (M_block, K_block, N_block, sub_h, sub_w) for the per-core work, or None."""
    k_cands = [d for d in get_k_block_candidates(k_per_device_tiles) if 6 <= d <= 8]
    if not k_cands:
        return None  # pathological K; let the caller keep its legacy fallback
    k_blk = max(k_cands)
    m_univ = set(get_mn_block_candidates(per_core_m))
    n_univ = set(get_mn_block_candidates(per_core_n))
    n_cands = [n for n in _n_candidates(per_core_n, fuse_swiglu) if n in n_univ]
    for relax in (False, True):
        for n_blk in n_cands:
            demand = 2 if relax else min(3 if n_blk >= per_core_n else 6, per_core_m)
            m_blk = _m_search(per_core_m, k_blk, n_blk, m_univ, demand, use_addcmul, l1_budget_kb)
            if m_blk:
                sub_h, sub_w = pick_subblock(m_blk, n_blk)
                return (m_blk, k_blk, n_blk, sub_h, sub_w)
    return None


def pick_v3(
    M: int,
    K: int,
    N: int,
    cluster_size: int,
    full_grid: tuple[int, int] = (12, 10),
    fuse_swiglu: bool = False,
    use_addcmul: bool = False,
    l1_budget_kb: int = BLACKHOLE_L1_BUDGET_KB,
) -> dict | None:
    """Resolve layout + blocking for an AGMM shape. Returns None when the rules don't apply
    (K not tile-aligned, K_tiles not divisible by the ring size, or no K_block in [6, 8]).

    Returns {"core_grid": (x, y), "transposed": bool, "blocks": (m, k, n), "subblock": (h, w)}.
    """
    if K % TILE:
        return None
    k_tiles = K // TILE
    if k_tiles % cluster_size:
        return None
    (gx, gy), transposed = pick_layout(M, K, N, full_grid, fuse_swiglu)
    m_tiles, n_tiles = _ceil(M, TILE), _ceil(N, TILE)
    per_core_m = _ceil(m_tiles, gx if transposed else gy)
    per_core_n = _ceil(n_tiles, gy if transposed else gx)
    blocks = pick_blocks(per_core_m, per_core_n, k_tiles // cluster_size, fuse_swiglu, use_addcmul, l1_budget_kb)
    if blocks is None:
        return None
    m_blk, k_blk, n_blk, sub_h, sub_w = blocks
    return {
        "core_grid": (gx, gy),
        "transposed": transposed,
        "blocks": (m_blk, k_blk, n_blk),
        "subblock": (sub_h, sub_w),
    }
