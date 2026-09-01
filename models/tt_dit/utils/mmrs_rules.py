# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Rule-based blocking for `minimal_matmul_strided_reduce_scatter_async` (the "v2.3 rules").

Fitted against the MMRS block-sweep campaign on 4x8 Blackhole Galaxy (five ff2 shape families:
H3 3584x5376, Wan2.2 3456x5120, LTX 4096x4096, Flux 2304x6144 and 3072x6144; M from ~1,000 to
~15,600) and validated blind on pre-registered predictions: v2 scored 35/40 on its confirmation
round, and the v2.3 short-N amendment below scored a further blind round of stratified Ms
straddling its branch boundary. Retro-scored across every measured shape: 77/79 within 5% of the
swept optimum, ~1.6% mean penalty. Sweep driver: `models/tt_dit/utils/sweep_mm_block_sizes.py`
(use case `mmrs`); per-shape results in the "MMRS Block Geometry" write-up.

The rules:

Grid: always 12x8 for the matmul, with the reduce-scatter on the two rows above. 12x8 won every
  full three-grid comparison (by 2-13% over 12x7, 5-25% over 12x9, the 12x9 deficit growing with
  M): the optimum trades matmul cores against RS bandwidth and is interior.

Per-core tiles are orientation-aware: the minimal matmul transposes its worker grid only when
  M > N, so pcM = ceil(M_tiles / 12), pcN = ceil(N_tiles / 8) when M > N, and the 8/12 roles swap
  otherwise. (Fitting without this produced fake "family conflicts".)

M_block = min(6, ceil(pcM / 2)), rounded up to even (floor 2) -- balanced halves, always >= 2
  blocks per core so the windowed L1 handoff engages.

K_block = the smallest divisor of K_tiles keeping the K-chunk count (K_tiles / K_block) <= 28.
  K is already per-device (row-parallel), so there is no ring-divisibility constraint, unlike
  AGMM; too many chunks costs sync overhead, too-large K_block starves pipelining. This is what
  separates LTX (128 K-tiles -> 8) from Wan/H3 (-> 4) and Flux-2304 (72 -> 3).

N_block = 8, except in the short-N regime (N_tiles <= 128, which separates LTX 4096x4096 from
  Wan/H3/Flux at 160/168/192 without naming the family): there N_block = 6 while pcM < 24, and
  N_block = 16 (one block covering the core's width) above, walking K_block down its divisor
  ladder until the combo fits the L1 budget -- at K_block = 8 the wide-N CBs are L1-illegal.

Subblock: 2x2 when both blocks are even, else the largest h*w <= 4 dividing both (fp32 dest
  accumulation halves the DEST register file to 4 tiles).

Everything here is pure Python with no ttnn/torch imports so it can be unit-tested host-only;
`utils/matmul.py` wraps the result into a `FusedMMRSConfig`.
"""

from __future__ import annotations

TILE = 32

# The matmul grid the rules are fitted for, inside a 12x10 Blackhole worker grid; the
# reduce-scatter occupies the (12x10 - 12x8) zone above it.
MM_GRID = (12, 8)
FULL_GRID = (12, 10)

# Largest acceptable K-chunk count (K_tiles / K_block) before ring-sync overhead dominates.
MAX_K_CHUNKS = 28

# K_block candidate bounds, matching the sweep that produced the fit.
K_BLOCK_MIN, K_BLOCK_MAX = 2, 16

# Boundary of the short-N regime's wide-N branch, in per-core M tiles. Swept shapes at
# pcM = 22/23 prefer N_block 6 and pcM = 25 prefers wide N; the crossover region is flat
# (the two are within ~1.5% of each other there), so the exact threshold is not knife-edge.
WIDE_N_PCM = 24

# L1 circular-buffer budget used to gate the wide-N branch (KB). Blackhole usable L1 is
# ~1464 KB; the margin covers kernel/firmware overhead.
BLACKHOLE_L1_BUDGET_KB = 1400


def estimate_l1_kb(m_blk: int, k_blk: int, n_blk: int) -> int:
    """Estimate L1 circular-buffer footprint in KB for a block config (fused-addcmul epilogue).

    Mirrors minimal_matmul_program_factory.cpp CB allocation:
      c_0 (in0):    2 * M * K tiles  (double-buffered, bf16 = 2 KB/tile)
      c_1 (in1):    2 * K * N tiles  (double-buffered, bf16)
      c_2 (out):    2 * M * N tiles  (double-buffered, bf16)
      c_3 (interm): M * N tiles      (single-buffered, f32 = 4 KB/tile)
      c_4 (bias):   N tiles          (single-buffered, bf16)
    plus the fused addcmul's M * N tiles (ternary_a, bf16) + N tiles (ternary_c, bf16).
    """
    bf16_kb = 2
    f32_kb = 4
    return (
        2 * m_blk * k_blk * bf16_kb
        + 2 * k_blk * n_blk * bf16_kb
        + 2 * m_blk * n_blk * bf16_kb
        + m_blk * n_blk * f32_kb
        + n_blk * bf16_kb
        + m_blk * n_blk * bf16_kb
        + n_blk * bf16_kb
    )


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _k_divisors(k_tiles: int) -> list[int]:
    return sorted(d for d in range(K_BLOCK_MIN, K_BLOCK_MAX + 1) if k_tiles % d == 0)


def pick_subblock(m_block: int, n_block: int) -> tuple[int, int]:
    """Best (sub_h, sub_w) with sub_h | m_block, sub_w | n_block, sub_h * sub_w <= 4 (fp32 dest)."""
    if m_block % 2 == 0 and n_block % 2 == 0:
        return (2, 2)
    best, best_product = (1, 1), 1
    for h in range(1, min(m_block, 4) + 1):
        if m_block % h:
            continue
        for w in range(1, min(n_block, 4) + 1):
            if n_block % w:
                continue
            if h * w <= 4 and h * w > best_product:
                best, best_product = (h, w), h * w
    return best


def pick_v23(
    M: int,
    K: int,
    N: int,
    full_grid: tuple[int, int] = FULL_GRID,
    l1_budget_kb: int = BLACKHOLE_L1_BUDGET_KB,
) -> dict | None:
    """Resolve the fused-MMRS blocking for a shape. Returns None when the rules don't apply
    (K or N not tile-aligned, K_tiles with no divisor in [2, 16], or a device grid other than
    the 12x10 the fit was swept on).

    Returns {"mm_grid": (12, 8), "blocks": (m, k, n), "subblock": (h, w)}.
    """
    if full_grid != FULL_GRID:
        return None
    if K % TILE or N % TILE:
        return None
    m_tiles, k_tiles, n_tiles = _ceil(M, TILE), K // TILE, N // TILE

    gx, gy = MM_GRID
    if M > N:
        pc_m = _ceil(m_tiles, gx)
    else:
        pc_m = _ceil(m_tiles, gy)

    m_blk = max(2, min(6, _ceil(pc_m, 2)))
    if m_blk % 2:
        m_blk = min(6, m_blk + 1)

    k_divs = _k_divisors(k_tiles)
    if not k_divs:
        return None  # pathological K; let the caller keep its legacy fallback
    k_blk = next((d for d in k_divs if k_tiles // d <= MAX_K_CHUNKS), k_divs[-1])

    if n_tiles <= 128:
        # Short-N regime (LTX-class shapes).
        if pc_m < WIDE_N_PCM:
            n_blk = 6
        else:
            n_blk = 16
            while estimate_l1_kb(m_blk, k_blk, n_blk) > l1_budget_kb:
                lower = [d for d in k_divs if d < k_blk]
                if not lower:
                    break
                k_blk = lower[-1]
    else:
        n_blk = 8

    sub_h, sub_w = pick_subblock(m_blk, n_blk)
    return {"mm_grid": MM_GRID, "blocks": (m_blk, k_blk, n_blk), "subblock": (sub_h, sub_w)}
