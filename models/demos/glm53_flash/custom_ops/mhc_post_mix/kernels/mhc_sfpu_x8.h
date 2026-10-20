// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Hand-written SFPU routines for the mHC post-block (compute kernel, TRISC_MATH only), on fp32 dest tiles.
// Dest geometry (measured on Blackhole with fp32 dest + unpack-to-dest, _logs/optmhc_probe_v2c.log; it differs
// from the "columns 0-7 / 8-15" wording in tt-llk ckernel_sfpu_binary_bcast.h): a tile is 64 address units,
// faces at +0/+16/+32/+48 (upper-left, upper-right, lower-left, lower-right); within a face the 4-row band b
// starts at +4b, and one SFPLOAD/SFPSTORE at +4b moves the 32 datums (rows 4b..4b+3) x (EVEN face columns
// 0,2,..,14), at +4b+2 the ODD face columns 1,3,..,15; lane i*8 + c = row 4b+i, column 2c (+1). An 8-lane
// subvector is therefore one row's 8 same-parity columns, which is exactly what SFPSHFT2's subvector rotate
// and the lane mask of the LLK column-broadcast work on.
//
//   "lane layout" of a coefficient tile: the 4 column groups a 32x32 tile offers to slot loads are
//   m = 0: even columns of the left face, 1: odd columns of the left face, 2: even of the right face, 3: odd
//   of the right face (`group_off(m)`); group m holds coefficient m of the row in all its columns, so a slot
//   load gives every lane the row's coefficient -- the per-row scale a stream mix needs, no lane shuffling.
//
//   expand_comb / expand_post  raw Sinkhorn tiles (coefficient k in column k) -> lane-layout tiles, once per
//                              token-tile: per band and coefficient, load the slot of column k's parity, keep
//                              lane k/2 (mask), rotate-and-add 1/2/4 lanes (subvector ROR1), store the broadcast
//                              slot at its group. Two chains interleave.
//   stream_mix                 out (+)= sum_k coef_k[row] * x_k, 5 SFPMADs per slot, accumulators in LREGs.
//
// Every operation is an exact fp32 multiply by 1.0/0.0, an add of 0.0, a lane rotate, or an fp32 FMA, so the
// results are bit-exact (expansion) / within fp32 rounding (mix). Dependent SFPU instructions are kept >= 2
// issue slots apart (2-cycle result latency); the comments mark the producers.

#pragma once

#include "api/compute/sfpu_binary_bcast.h"

#ifdef TRISC_MATH
namespace mhc_sfpu {

using namespace ckernel;

constexpr uint32_t DT = 64;     // dest address units per fp32 tile
constexpr uint32_t RIGHT = 16;  // the right face of a half (columns 16-31)
constexpr uint32_t ODD = 2;     // the odd columns of a face (the even ones are at +0)
constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
constexpr uint32_t ROR1 = sfpu::SFPSHFT2_MOD1_SUBVEC_SHFLROR1;
constexpr uint32_t L0 = p_sfpu::LREG0, L1 = p_sfpu::LREG1, L2 = p_sfpu::LREG2, L3 = p_sfpu::LREG3;
constexpr uint32_t L4 = p_sfpu::LREG4, L5 = p_sfpu::LREG5, L6 = p_sfpu::LREG6, L7 = p_sfpu::LREG7;
constexpr uint32_t C0 = p_sfpu::LCONST_0, C1 = p_sfpu::LCONST_1;

// address of the (rows 4b..4b+3, even columns of the left face) slot of band b, relative to the tile
FORCE_INLINE uint32_t band_off(uint32_t b) { return (b < 4 ? 0u : 32u) + 4u * (b & 3u); }
// column group m of a band (face m >> 1, column parity m & 1), relative to its first slot
FORCE_INLINE uint32_t group_off(uint32_t m) { return (m >> 1) * RIGHT + (m & 1) * ODD; }
// slot (parity) and lane of raw coefficient column k (k < 16: left face) within a band
FORCE_INLINE uint32_t coef_slot(uint32_t k) { return (k & 1) * ODD; }
FORCE_INLINE uint32_t coef_lane(uint32_t k) { return k >> 1; }

// LREG4+k <- 1.0 in lane lanes[k] of every 8-lane subvector, 0.0 elsewhere (k < count <= 4). Clobbers LREG2.
FORCE_INLINE void make_lane_masks(const uint32_t* lanes, uint32_t count) {
    sfpu::_build_lane_mask_col0_();  // LREG6 <- the lane-0 mask
    for (uint32_t k = 0; k < count; ++k) {
        if (L4 + k != L6) {
            TT_SFPMOV(0, L6, L4 + k, 0);
        }
    }
    TTI_SFPNOP;
    for (uint32_t k = 0; k < count; ++k) {
        for (uint32_t r = 0; r < lanes[k]; ++r) {  // ROR1 moves lane c to lane c+1
            TT_SFPSHFT2(0, L4 + k, L4 + k, ROR1);
            TTI_SFPNOP;
        }
    }
}

// dstA <- lane(maskA) of srcA broadcast over its 8-lane subvectors; the same for B. The two chains interleave
// (A in LREG0/LREG2, B in LREG1/LREG3) so every consumer sits two slots behind its producer.
FORCE_INLINE void bcast_pair(
    uint32_t srcA, uint32_t maskA, uint32_t dstA, uint32_t srcB, uint32_t maskB, uint32_t dstB) {
    TT_SFPLOAD(L0, IM, ADDR_MOD_7, srcA);
    TT_SFPLOAD(L1, IM, ADDR_MOD_7, srcB);
    TT_SFPMUL(L0, maskA, C0, L0, 0);  // keep lane k
    TT_SFPMUL(L1, maskB, C0, L1, 0);
    TTI_SFPSHFT2(0, L0, L2, ROR1);  // stage 1: lanes {k, k+1}
    TTI_SFPSHFT2(0, L1, L3, ROR1);
    TTI_SFPADD(L0, C1, L2, L0, 0);
    TTI_SFPADD(L1, C1, L3, L1, 0);
    TTI_SFPSHFT2(0, L0, L2, ROR1);  // stage 2: 4 lanes
    TTI_SFPSHFT2(0, L1, L3, ROR1);
    TTI_SFPSHFT2(0, L2, L2, ROR1);
    TTI_SFPSHFT2(0, L3, L3, ROR1);
    TTI_SFPADD(L0, C1, L2, L0, 0);
    TTI_SFPADD(L1, C1, L3, L1, 0);
    TTI_SFPSHFT2(0, L0, L2, ROR1);  // stage 3: all 8 lanes
    TTI_SFPSHFT2(0, L1, L3, ROR1);
    TTI_SFPSHFT2(0, L2, L2, ROR1);
    TTI_SFPSHFT2(0, L3, L3, ROR1);
    TTI_SFPSHFT2(0, L2, L2, ROR1);
    TTI_SFPSHFT2(0, L3, L3, ROR1);
    TTI_SFPSHFT2(0, L2, L2, ROR1);
    TTI_SFPSHFT2(0, L3, L3, ROR1);
    TTI_SFPADD(L0, C1, L2, L0, 0);
    TTI_SFPADD(L1, C1, L3, L1, 0);
    TT_SFPSTORE(L0, IM, ADDR_MOD_7, dstA);
    TT_SFPSTORE(L1, IM, ADDR_MOD_7, dstB);
}

// raw comb tile `src` (coefficient i*n+j in column i*n+j) -> lane-layout tile `dst` of stream j: group i holds
// comb[:, i*n+j], i < n (n even, n <= 4: all coefficients sit in the left face).
FORCE_INLINE void expand_comb(uint32_t src, uint32_t dst, uint32_t /*unused*/, uint32_t n, uint32_t j) {
    uint32_t lanes[4];
    for (uint32_t i = 0; i < n; ++i) {
        lanes[i] = coef_lane(i * n + j);
    }
    make_lane_masks(lanes, n);
    for (uint32_t b = 0; b < 8; ++b) {
        const uint32_t s = src * DT + band_off(b);
        const uint32_t d = dst * DT + band_off(b);
        for (uint32_t i = 0; i + 1 < n; i += 2) {
            bcast_pair(
                s + coef_slot(i * n + j),
                L4 + i,
                d + group_off(i),
                s + coef_slot((i + 1) * n + j),
                L5 + i,
                d + group_off(i + 1));
        }
    }
}

// raw post tile `src` (coefficient m in column m) -> lane-layout tile `dst`: group m holds post[:, m], m < n.
FORCE_INLINE void expand_post(uint32_t src, uint32_t dst, uint32_t /*unused*/, uint32_t n) {
    uint32_t lanes[4];
    for (uint32_t m = 0; m < n; ++m) {
        lanes[m] = coef_lane(m);
    }
    make_lane_masks(lanes, n);
    for (uint32_t b = 0; b < 8; ++b) {
        const uint32_t s = src * DT + band_off(b);
        const uint32_t d = dst * DT + band_off(b);
        for (uint32_t m = 0; m + 1 < n; m += 2) {
            bcast_pair(s + coef_slot(m), L4 + m, d + group_off(m), s + coef_slot(m + 1), L5 + m, d + group_off(m + 1));
        }
    }
}

// out (+)= sum_{k<n} comb[group k] * X_k + post[group j] * Y over the whole tile (lane-layout coefficient
// tiles); X_k is dest tile k.
// `accumulate` starts from the current out (collapse: the streams sum into one tile), `scale` multiplies the
// result by the fp32 `scale_bits` (collapse: 1/n). LREG1/3/5/4 accumulate the band's 4 slots, LREG7 holds the
// coefficient slot, LREG0/LREG2 alternate as the data registers, LREG6 the scale.
FORCE_INLINE void stream_mix(
    uint32_t out,
    uint32_t comb,
    uint32_t /*unused*/,
    uint32_t post,
    uint32_t y,
    uint32_t n,
    uint32_t j,
    bool accumulate,
    bool scale,
    uint32_t scale_bits) {
    if (scale) {
        TT_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_UPPER, scale_bits >> 16);
        TT_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_LOWER, scale_bits & 0xFFFFu);
    }
    for (uint32_t b = 0; b < 8; ++b) {
        const uint32_t off = band_off(b);
        const uint32_t o = out * DT + off;
        if (accumulate) {
            TT_SFPLOAD(L1, IM, ADDR_MOD_7, o);
            TT_SFPLOAD(L3, IM, ADDR_MOD_7, o + ODD);
            TT_SFPLOAD(L4, IM, ADDR_MOD_7, o + RIGHT);
            TT_SFPLOAD(L5, IM, ADDR_MOD_7, o + RIGHT + ODD);
        }
        for (uint32_t k = 0; k <= n; ++k) {
            const bool first = (k == 0) && !accumulate;
            const uint32_t x = (k < n ? k : y) * DT + off;
            const uint32_t c = (k < n) ? comb * DT + off + group_off(k) : post * DT + off + group_off(j);
            TT_SFPLOAD(L7, IM, ADDR_MOD_7, c);        // coefficient slot
            TT_SFPLOAD(L0, IM, ADDR_MOD_7, x);        // left face, even columns
            TT_SFPLOAD(L2, IM, ADDR_MOD_7, x + ODD);  // left face, odd columns
            if (first) {
                TTI_SFPMUL(L0, L7, C0, L1, 0);
            } else {
                TTI_SFPMAD(L0, L7, L1, L1, 0);
            }
            TT_SFPLOAD(L0, IM, ADDR_MOD_7, x + RIGHT);  // right face, even columns
            if (first) {
                TTI_SFPMUL(L2, L7, C0, L3, 0);
            } else {
                TTI_SFPMAD(L2, L7, L3, L3, 0);
            }
            TT_SFPLOAD(L2, IM, ADDR_MOD_7, x + RIGHT + ODD);  // right face, odd columns
            if (first) {
                TTI_SFPMUL(L0, L7, C0, L4, 0);
                TTI_SFPMUL(L2, L7, C0, L5, 0);
            } else {
                TTI_SFPMAD(L0, L7, L4, L4, 0);
                TTI_SFPMAD(L2, L7, L5, L5, 0);
            }
        }
        if (scale) {
            TTI_SFPMUL(L1, L6, C0, L1, 0);
            TTI_SFPMUL(L3, L6, C0, L3, 0);
            TTI_SFPMUL(L4, L6, C0, L4, 0);
            TTI_SFPMUL(L5, L6, C0, L5, 0);
        }
        TT_SFPSTORE(L1, IM, ADDR_MOD_7, o);
        TT_SFPSTORE(L3, IM, ADDR_MOD_7, o + ODD);
        TT_SFPSTORE(L4, IM, ADDR_MOD_7, o + RIGHT);
        TT_SFPSTORE(L5, IM, ADDR_MOD_7, o + RIGHT + ODD);
    }
}

// ---------------------------------------------------------------------------------------------------------
// Pre-block routines (the fused pre-block custom op; same geometry and register conventions as above).

// out = sum_{k<n} coef[group k] * X_k over the whole tile (lane-layout coefficient tile `coef`, X_k = dest
// tile k): the pre-mix h = sum_i pre_i x_i. LREG7 coefficient, LREG0/LREG2 data, LREG1/3/4/5 accumulators.
FORCE_INLINE void pre_mix(uint32_t out, uint32_t coef, uint32_t /*unused*/, uint32_t n) {
    for (uint32_t b = 0; b < 8; ++b) {
        const uint32_t off = band_off(b);
        const uint32_t o = out * DT + off;
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t x = k * DT + off;
            const uint32_t c = coef * DT + off + group_off(k);
            TT_SFPLOAD(L7, IM, ADDR_MOD_7, c);
            TT_SFPLOAD(L0, IM, ADDR_MOD_7, x);
            TT_SFPLOAD(L2, IM, ADDR_MOD_7, x + ODD);
            if (k == 0) {
                TTI_SFPMUL(L0, L7, C0, L1, 0);
            } else {
                TTI_SFPMAD(L0, L7, L1, L1, 0);
            }
            TT_SFPLOAD(L0, IM, ADDR_MOD_7, x + RIGHT);
            if (k == 0) {
                TTI_SFPMUL(L2, L7, C0, L3, 0);
            } else {
                TTI_SFPMAD(L2, L7, L3, L3, 0);
            }
            TT_SFPLOAD(L2, IM, ADDR_MOD_7, x + RIGHT + ODD);
            if (k == 0) {
                TTI_SFPMUL(L0, L7, C0, L4, 0);
                TTI_SFPMUL(L2, L7, C0, L5, 0);
            } else {
                TTI_SFPMAD(L0, L7, L4, L4, 0);
                TTI_SFPMAD(L2, L7, L5, L5, 0);
            }
        }
        TT_SFPSTORE(L1, IM, ADDR_MOD_7, o);
        TT_SFPSTORE(L3, IM, ADDR_MOD_7, o + ODD);
        TT_SFPSTORE(L4, IM, ADDR_MOD_7, o + RIGHT);
        TT_SFPSTORE(L5, IM, ADDR_MOD_7, o + RIGHT + ODD);
    }
}

// LREG1 <- per-row sum over the 4 slots of band `off` of tiles first..first+count-1, of value*value (sum of
// squares) -- every lane of a row's 8-lane subvectors ends up holding the row's total over all 32*count
// columns (3 subvector rotate-adds). Uses LREG0/LREG2 (data), LREG3 (rotate scratch).
FORCE_INLINE void band_rowsum_sq(uint32_t first, uint32_t count, uint32_t off) {
    TTI_SFPMOV(0, C0, L1, 0);  // accumulator = 0
    for (uint32_t k = first; k < first + count; ++k) {
        const uint32_t x = k * DT + off;
        TT_SFPLOAD(L0, IM, ADDR_MOD_7, x);
        TT_SFPLOAD(L2, IM, ADDR_MOD_7, x + ODD);
        TTI_SFPMAD(L0, L0, L1, L1, 0);
        TT_SFPLOAD(L0, IM, ADDR_MOD_7, x + RIGHT);
        TTI_SFPMAD(L2, L2, L1, L1, 0);
        TT_SFPLOAD(L2, IM, ADDR_MOD_7, x + RIGHT + ODD);
        TTI_SFPMAD(L0, L0, L1, L1, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(L2, L2, L1, L1, 0);
        TTI_SFPNOP;
    }
    // subvector tree: lanes {c, c+1}, {c..c+3}, all 8 (cyclic, so every lane gets the full sum)
    TTI_SFPSHFT2(0, L1, L3, ROR1);
    TTI_SFPNOP;
    TTI_SFPADD(L1, C1, L3, L1, 0);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, L1, L3, ROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, L3, L3, ROR1);
    TTI_SFPNOP;
    TTI_SFPADD(L1, C1, L3, L1, 0);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, L1, L3, ROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, L3, L3, ROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, L3, L3, ROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, L3, L3, ROR1);
    TTI_SFPNOP;
    TTI_SFPADD(L1, C1, L3, L1, 0);
    TTI_SFPNOP;
}

// out[row][column (group, lane)] += sum over tiles first..first+count-1 of the row's sum of squares (the one
// column that carries the statistic; the rest of `out` is untouched). Mask for `lane` built into LREG4.
FORCE_INLINE void rowsum_sq_into(
    uint32_t out, uint32_t first, uint32_t /*unused*/, uint32_t count, uint32_t group, uint32_t lane) {
    const uint32_t lanes[1] = {lane};
    make_lane_masks(lanes, 1);
    for (uint32_t b = 0; b < 8; ++b) {
        const uint32_t off = band_off(b);
        band_rowsum_sq(first, count, off);
        const uint32_t o = out * DT + off + group_off(group);
        TT_SFPLOAD(L0, IM, ADDR_MOD_7, o);
        TTI_SFPMUL(L1, L4, C0, L1, 0);  // keep the statistic's lane only
        TTI_SFPNOP;
        TTI_SFPADD(L0, C1, L1, L0, 0);
        TTI_SFPNOP;
        TT_SFPSTORE(L0, IM, ADDR_MOD_7, o);
    }
}

// out[row][*] = the row's sum of squares over tiles first..first+count-1 (every column of `out`).
FORCE_INLINE void rowsum_sq_fill(uint32_t out, uint32_t first, uint32_t /*unused*/, uint32_t count) {
    for (uint32_t b = 0; b < 8; ++b) {
        const uint32_t off = band_off(b);
        band_rowsum_sq(first, count, off);
        const uint32_t o = out * DT + off;
        TT_SFPSTORE(L1, IM, ADDR_MOD_7, o);
        TT_SFPSTORE(L1, IM, ADDR_MOD_7, o + ODD);
        TT_SFPSTORE(L1, IM, ADDR_MOD_7, o + RIGHT);
        TT_SFPSTORE(L1, IM, ADDR_MOD_7, o + RIGHT + ODD);
    }
}

// target[row][*] *= src[row][column (group, lane)]: scale every row of `target` by one column of `src`.
FORCE_INLINE void scale_rows_from_lane(
    uint32_t target, uint32_t src, uint32_t /*unused*/, uint32_t group, uint32_t lane) {
    const uint32_t lanes[1] = {lane};
    make_lane_masks(lanes, 1);
    for (uint32_t b = 0; b < 8; ++b) {
        const uint32_t off = band_off(b);
        // broadcast the column across the subvectors (single chain)
        TT_SFPLOAD(L0, IM, ADDR_MOD_7, src * DT + off + group_off(group));
        TTI_SFPNOP;
        TTI_SFPMUL(L0, L4, C0, L0, 0);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L0, L2, ROR1);
        TTI_SFPNOP;
        TTI_SFPADD(L0, C1, L2, L0, 0);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L0, L2, ROR1);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L2, L2, ROR1);
        TTI_SFPNOP;
        TTI_SFPADD(L0, C1, L2, L0, 0);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L0, L2, ROR1);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L2, L2, ROR1);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L2, L2, ROR1);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L2, L2, ROR1);
        TTI_SFPNOP;
        TTI_SFPADD(L0, C1, L2, L0, 0);
        const uint32_t t = target * DT + off;
        TT_SFPLOAD(L1, IM, ADDR_MOD_7, t);
        TT_SFPLOAD(L3, IM, ADDR_MOD_7, t + ODD);
        TTI_SFPMUL(L1, L0, C0, L1, 0);
        TT_SFPLOAD(L5, IM, ADDR_MOD_7, t + RIGHT);
        TTI_SFPMUL(L3, L0, C0, L3, 0);
        TT_SFPLOAD(L6, IM, ADDR_MOD_7, t + RIGHT + ODD);
        TTI_SFPMUL(L5, L0, C0, L5, 0);
        TT_SFPSTORE(L1, IM, ADDR_MOD_7, t);
        TTI_SFPMUL(L6, L0, C0, L6, 0);
        TT_SFPSTORE(L3, IM, ADDR_MOD_7, t + ODD);
        TT_SFPSTORE(L5, IM, ADDR_MOD_7, t + RIGHT);
        TT_SFPSTORE(L6, IM, ADDR_MOD_7, t + RIGHT + ODD);
    }
}

}  // namespace mhc_sfpu
#endif  // TRISC_MATH
