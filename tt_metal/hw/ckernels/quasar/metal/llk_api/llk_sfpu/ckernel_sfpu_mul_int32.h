// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// SFPMUL24 multiplies the low 23 bits of each operand; mod1 bit 0 selects which half of the
// 46-bit product reaches the destination.
constexpr std::uint32_t MUL_INT32_MUL24_LOW23 = 0x0;   // product bits [22:0]
constexpr std::uint32_t MUL_INT32_MUL24_HIGH23 = 0x1;  // product bits [45:23]
// SFPSHFT mod1: bit 0 takes the amount from imm12 (negative shifts right), bit 1 makes a right
// shift arithmetic, and bit 2 (with bit 0) takes the value from lreg_c instead of lreg_dest.
constexpr std::uint32_t MUL_INT32_SHFT_IMM_FROM_C = 0x5;
constexpr std::uint32_t MUL_INT32_SHFT_ARITH_IMM_FROM_C = 0x7;
constexpr std::uint32_t MUL_INT32_SHFT_RIGHT_23 = (-23) & 0xFFF;
constexpr std::uint32_t MUL_INT32_SHFT_RIGHT_31 = (-31) & 0xFFF;
// SFPSETSGN mod 1 takes the new sign from imm12_math[0].
constexpr std::uint32_t MUL_INT32_SETSGN_FROM_IMM = 0x1;
constexpr std::uint32_t MUL_INT32_SIGN_POSITIVE = 0x0;
// SFPABS mod1 bit 0 clear selects the two's-complement integer absolute value.
constexpr std::uint32_t MUL_INT32_ABS_INT = 0x0;
// SFPIADD mod1 bit 1 negates lreg_dest, so the result is lreg_c - lreg_dest.
constexpr std::uint32_t MUL_INT32_IADD_C_MINUS_DST = 0x2 | p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;

// Int32 multiply ported from the BH DISABLE_SFPLOADMACRO path.
// Uses SFPMUL24 (24-bit partial products) + shifts to produce a full 32-bit
// result.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void _mul_int32_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in0_offset + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG2, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in1_offset + (d << 1));

        // Dest layout depends on how operands reached dest:
        //   UNP_DEST / Int32 L1 with 2's-comp tiles → 2's-comp Int32
        //   copy_tile Int8 + fp32_dest_acc FPU → sign-mag Int32 (SIGN_MAGNITUDE_FORMAT=true)
        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
            TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
        }

        TTI_SFPSHFT((-23) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG1, 5);  // lreg1 = lreg0 >> 23
        TTI_SFPSHFT((-23) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG3, 5);  // lreg3 = lreg2 >> 23

        TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG4, 0);  // low(a_lo * b_lo)
        TTI_SFPMUL24(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG5, 1);  // high(a_lo * b_lo)
        TTI_SFPMUL24(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG6, 0);  // low(a_hi * b_lo)
        TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG7, 0);  // low(a_lo * b_hi)

        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);  // lreg5 += lreg6
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);  // lreg5 += lreg7
        TTI_SFPSHFT(23, p_sfpu::LREG5, p_sfpu::LREG5, 5);                                          // lreg5 <<= 23
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);  // lreg4 += lreg5

        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
        }

        TT_SFPSTORE(p_sfpu::LREG4, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, out_offset + (d << 1));
    }
}

// Fixed-point multiply: out = sign * ((|in0| * |in1|) >> 23), with in1 a Q0.23 requant scale.
// It keeps the middle slice of the 46-bit product, so it is not _mul_int32_ followed by a shift.
// Exact while at most one operand reaches 2^23 and each cross product below fits in 23 bits, which
// holds whenever |out| < 2^23:
//   (|a| >> 23) * (|b| & 0x7FFFFF) < 2^23  and  (|a| & 0x7FFFFF) * (|b| >> 23) < 2^23.
// A zero result is +0 in either encoding. Two's-complement INT32_MIN has no 31-bit magnitude and
// is not supported.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void _mul_int32_sm_shift23_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

    constexpr std::uint32_t A_RAW = p_sfpu::LREG0;  // becomes the result sign word after the XOR
    constexpr std::uint32_t A_MAG = p_sfpu::LREG1;
    constexpr std::uint32_t B_RAW = p_sfpu::LREG2;
    constexpr std::uint32_t B_MAG = p_sfpu::LREG3;
    constexpr std::uint32_t ACC = p_sfpu::LREG4;
    constexpr std::uint32_t TMP = p_sfpu::LREG5;
    constexpr std::uint32_t HI = p_sfpu::LREG6;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(A_RAW, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in0_offset + (d << 1));
        TT_SFPLOAD(B_RAW, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, in1_offset + (d << 1));

        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPSETSGN(MUL_INT32_SIGN_POSITIVE, A_RAW, A_MAG, MUL_INT32_SETSGN_FROM_IMM);
            TTI_SFPSETSGN(MUL_INT32_SIGN_POSITIVE, B_RAW, B_MAG, MUL_INT32_SETSGN_FROM_IMM);
        } else {
            TTI_SFPABS(A_RAW, A_MAG, MUL_INT32_ABS_INT);
            TTI_SFPABS(B_RAW, B_MAG, MUL_INT32_ABS_INT);
        }
        // Bit 31 is the sign in both encodings, so this is the result sign either way.
        TTI_SFPXOR(B_RAW, A_RAW);

        TTI_SFPMUL24(B_MAG, A_MAG, ACC, MUL_INT32_MUL24_HIGH23);  // high23(a_lo * b_lo)

        TTI_SFPSHFT(MUL_INT32_SHFT_RIGHT_23, A_MAG, HI, MUL_INT32_SHFT_IMM_FROM_C);
        TTI_SFPMUL24(HI, B_MAG, TMP, MUL_INT32_MUL24_LOW23);  // low23(a_hi * b_lo)
        TTI_SFPIADD(0, TMP, ACC, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);

        TTI_SFPSHFT(MUL_INT32_SHFT_RIGHT_23, B_MAG, HI, MUL_INT32_SHFT_IMM_FROM_C);
        TTI_SFPMUL24(A_MAG, HI, TMP, MUL_INT32_MUL24_LOW23);  // low23(a_lo * b_hi)
        TTI_SFPIADD(0, TMP, ACC, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC);

        // Negate in two's complement as (ACC ^ m) - m, m = sign ? -1 : 0. A zero magnitude stays
        // +0; setting the sign bit and casting would hand SFPCAST a -0.
        TTI_SFPSHFT(MUL_INT32_SHFT_RIGHT_31, A_RAW, TMP, MUL_INT32_SHFT_ARITH_IMM_FROM_C);
        TTI_SFPXOR(TMP, ACC);
        TTI_SFPIADD(0, ACC, TMP, MUL_INT32_IADD_C_MINUS_DST);

        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(TMP, TMP, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
        }
        TT_SFPSTORE(TMP, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, out_offset + (d << 1));
    }
}

}  // namespace sfpu
}  // namespace ckernel
