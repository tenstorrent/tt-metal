// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Batch-invariant fp32 row SUM of one DST tile, for the reductions whose tile rows are tokens (decode slots).
//
// The library's sfpu_reduce<SUM, Float32, REDUCE_ROW> (ckernel_sfpu_reduce.h perform_reduce_row_face_pair) adds the
// four 4-row groups of a face pair with DIFFERENT fp32 add trees. Per row, with p_c the partial of SFPU column c (the
// row's values at tile columns 2c, 2c + 1, 16 + 2c, 17 + 2c: f0e, f0o, f1e, f1o):
//   rows 0-3   (A): p_c = f0e + (f0o + (f1e + f1o))   columns ((p0 + p7) + (p6 + p5)) + ((p4 + p3) + (p2 + p1))
//   rows 4-7   (B): p_c = f0e + (f0o + (f1e + f1o))   columns ((p1 + p0) + (p7 + p6)) + ((p5 + p4) + (p3 + p2))
//   rows 8-11  (C): p_c = (f0e + f0o) + (f1e + f1o)   columns as A
//   rows 12-15 (D): p_c = (f0e + f0o) + (f1e + f1o)   columns as B
// (the same for rows 16-31), so a row's sum depends on the 4-row group it sits in: identical requests in slots 0-3,
// 4-7, 8-11 and 12-15 round differently and decode differently.
//
// glm53_row_sum_f32 adds EVERY 4-row group with group A's tree, instruction for instruction: the same SFPLOADs, the
// library's recorded vertical window (init_reduce_sum_avg's float [0, 6): A1 B1 A2 B2 A3 B3) issued inline, then a
// rotate-by-1 / 2 / 4 butterfly whose column-0 lane performs A's three horizontal SFPADDs with the same operands in the
// same roles (VA = the lane's own partial, VC = the rotated one). Rows 0-3 and 16-19 are therefore bitwise the
// library's result, and every other row equals what the same values give in rows 0-3: a row's sum depends on its data
// only. Two groups are interleaved (LREG0-3 / LREG4-7) exactly like the library's horizontal_reduce, so every
// SFPSHFT2 is at least one instruction behind the SFPADD it reads (the auto-stall does not cover that pair).
// Cost: 7 rotates per 4-row group (28 per 16 rows) instead of the library's 11 per 16 rows.
//
// Contract (as the library's row reduce): each row's sum lands in column 0 of that row; the other even columns of
// faces 0 / 2 hold other (differently ordered) sums of the same row; odd columns and faces 1 / 3 are untouched.
// Every caller reads column 0 only. One 32x32 fp32 tile; init: sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>()
// (the state the library call needed; this body only relies on its ADDR_MOD_7). fp32 DST (FP32 load / store mode).
#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

namespace glm53_rs {
constexpr std::uint32_t FACE_ROWS = 16;   // dest rows per face (face 1 = face 0 + 16)
constexpr std::uint32_t ODD_COLUMNS = 2;  // SFPLOAD / SFPSTORE address offset of a face's odd columns
constexpr std::uint32_t FP32 = static_cast<std::uint32_t>(InstrModLoadStore::FP32);

// LREG1 = ror<shift>(LREG0), LREG5 = ror<shift>(LREG4) (lane c <- lane c - shift of its 8-lane sub-row)
template <std::uint32_t shift>
inline void rotate() {
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
#pragma GCC unroll 4
    for (std::uint32_t i = 1; i < shift; i++) {
        TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    }
}

// LREG0 = LREG0 * 1.0 + LREG1, LREG4 = LREG4 * 1.0 + LREG5
inline void fold() {
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);
}

// the 4-row groups at dest rows G0 and G1 of one face pair -> each row's sum in column 0 of that row
template <std::uint32_t G0, std::uint32_t G1>
inline void group_pair() {
    // G0 -> LREG0-3, G1 -> LREG4-7: left face even / odd columns, right face even / odd columns
    TTI_SFPLOAD(p_sfpu::LREG0, FP32, ADDR_MOD_7, G0);
    TTI_SFPLOAD(p_sfpu::LREG1, FP32, ADDR_MOD_7, G0 + ODD_COLUMNS);
    TTI_SFPLOAD(p_sfpu::LREG2, FP32, ADDR_MOD_7, G0 + FACE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG3, FP32, ADDR_MOD_7, G0 + FACE_ROWS + ODD_COLUMNS);
    TTI_SFPLOAD(p_sfpu::LREG4, FP32, ADDR_MOD_7, G1);
    TTI_SFPLOAD(p_sfpu::LREG5, FP32, ADDR_MOD_7, G1 + ODD_COLUMNS);
    TTI_SFPLOAD(p_sfpu::LREG6, FP32, ADDR_MOD_7, G1 + FACE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG7, FP32, ADDR_MOD_7, G1 + FACE_ROWS + ODD_COLUMNS);
    // vertical: p = f0e + (f0o + (f1e + f1o)), the library's recorded float window [0, 6) for both groups
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // A1
    TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, 0);  // B1
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG1, 0);  // A2
    TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG5, 0);  // B2
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // A3
    TTI_SFPADD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, 0);  // B3
    // horizontal, column 0: (p0 + p7) then + (p6 + p5), then + ((p4 + p3) + (p2 + p1)) -- group A's tree
    rotate<1>();
    fold();
    rotate<2>();
    fold();
    rotate<4>();
    fold();
    TTI_SFPSTORE(p_sfpu::LREG0, FP32, ADDR_MOD_7, G0);
    TTI_SFPSTORE(p_sfpu::LREG4, FP32, ADDR_MOD_7, G1);
}

// the whole tile: face pair 0 + 1 (tile rows 0-15: groups at dest rows 0, 4, 8, 12), face pair 2 + 3 (rows 16-31)
inline void glm53_row_sum_f32_tile() {
    glm53_rs::group_pair<0, 4>();
    glm53_rs::group_pair<8, 12>();
    glm53_rs::group_pair<2 * FACE_ROWS, 2 * FACE_ROWS + 4>();
    glm53_rs::group_pair<2 * FACE_ROWS + 8, 2 * FACE_ROWS + 12>();
}
}  // namespace glm53_rs

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH

// column 0 of every row of DST tile idst <- the fp32 sum of that row (the same add tree for every row; see above).
// Replaces sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(idst) after the same init.
ALWI void glm53_row_sum_f32(uint32_t idst) {
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, glm53_rs::glm53_row_sum_f32_tile, idst, VectorMode::RC_custom)));
}
