// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Shared LREG layout across all three quant ops:
//   LREG0 = operand A load / running accumulator / result before store
//   LREG1 = operand B (fp32 scale)
//   LREG2 = zero-point (fp32), loaded once by the matching init
// LREG3-7 stay free. Each init must run before the first call to its kernel.
//
// SIGN_MAGNITUDE_FORMAT gates the SM<->2's-comp casts: the default (false) path
// treats int32 dest content as 2's-complement (UNP_DEST / Int32-L1), while
// STOCH_RND emits sign-magnitude (SMAG32). Inputs/outputs are therefore cast on
// the !SIGN_MAGNITUDE_FORMAT path; the true path stores STOCH_RND output as-is.

// ---- quant: out_int32 = round_to_int8(A_fp32 * B_fp32 + zero_point) ----

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _quant_int32_init_(const std::uint32_t zero_point /* fp32 bits */) {
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, zero_point & 0xFFFF);  // zp low half
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, zero_point >> 16);     // zp high half -> LREG2 = fp32 zp
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _quant_int32_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_in0 + (d << 1));  // operand A (fp32)
        TT_SFPLOAD(
            p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_in1 + (d << 1));  // operand B (fp32 scale)

        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);  // LREG0 = A*B + zp

        // fp32 -> signed int8 (SMAG32 container), round-nearest-even; (1<<3) selects the imm8 descale slot (0 = no
        // descale)
        TTI_SFP_STOCH_RND(
            p_sfpu::sfp_stochrnd_rnd_mod::NearEven,
            0 /* imm8 descale */,
            0 /* lreg_b (unused on fp32 path) */,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            (1 << 3) | p_sfpu::sfp_stochrnd_mod::FP32_TO_INT8);

        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);  // sign-mag -> 2's-comp for INT32 store
        }

        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_out + (d << 1));  // store int32
    }
}

// ---- requant: out_int32 = round_to_int8(int32_to_fp32(A) * B_fp32 + zero_point) ----

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _requant_int32_init_(const std::uint32_t zero_point /* fp32 bits */) {
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, zero_point & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, zero_point >> 16);  // LREG2 = fp32 zp
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _requant_int32_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_in0 + (d << 1));  // operand A (int32)
        TT_SFPLOAD(
            p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_in1 + (d << 1));  // operand B (fp32 scale)

        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(
                p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);  // 2's-comp -> sign-mag input
        }
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1: int32(sign-mag) -> fp32 RNE */);

        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);  // LREG0 = A_fp*B + zp

        TTI_SFP_STOCH_RND(
            p_sfpu::sfp_stochrnd_rnd_mod::NearEven,
            0 /* imm8 descale */,
            0 /* lreg_b (unused on fp32 path) */,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            (1 << 3) | p_sfpu::sfp_stochrnd_mod::FP32_TO_INT8);

        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(
                p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);  // sign-mag -> 2's-comp output
        }

        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_out + (d << 1));  // store int32
    }
}

// ---- dequant: out_fp32 = (int32_to_fp32(A) - zero_point) * B_fp32 ----
// init negates zp so the body computes (A + (-zp)) * B.

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _dequant_int32_init_(const std::uint32_t neg_zero_point /* fp32 bits of -zp */) {
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, neg_zero_point & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, neg_zero_point >> 16);  // LREG2 = fp32 (-zp)
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _dequant_int32_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_in0 + (d << 1));  // operand A (int32)
        TT_SFPLOAD(
            p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_in1 + (d << 1));  // operand B (fp32 scale)

        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(
                p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);  // 2's-comp -> sign-mag input
        }
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1: int32(sign-mag) -> fp32 RNE */);

        // LCONST_1 = 1.0 -> A*1 + (-zp) = A - zp ; then LCONST_0 = 0.0 ignored as +C -> (A - zp) * B
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* mod1 */);

        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_out + (d << 1));  // store fp32
    }
}

}  // namespace ckernel::sfpu
