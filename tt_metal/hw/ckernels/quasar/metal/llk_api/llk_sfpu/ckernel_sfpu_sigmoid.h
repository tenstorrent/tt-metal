// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// Two sigmoid implementations behind three names: TTI SFPNONLINEAR path (`calculate_sigmoid` /
// `_calculate_sigmoid_` / `_calculate_sigmoid_regs_`, also what `_calculate_silu_` composes with)
// and the SFPI body `_sfpu_sigmoid_` (used by `clamped_silu_glu` / ttnn swiglu_sfpu). Not bit-equivalent.

// Calculates SIGMOID for results already in registers, used as a helper function.
// Input of sigmoid is expected to already be in src_reg, and the result will be stored in dest_reg.
// work_reg is an intermediate register used for calculations.
inline void _calculate_sigmoid_regs_(
    const std::uint32_t src_reg, const std::uint32_t work_reg, const std::uint32_t dest_reg) {
    // ALthough SFPMUL/SFPADD are 2 cycle instructions, we don't need a TTI_NOP
    // because the hardware implicitly stalls if the next instruction depends on results
    TTI_SFPMOV(src_reg, work_reg, 1);  // Copy src_reg to work_reg and invert sign bit (take negative of input)
    TTI_SFPNONLINEAR(
        work_reg,
        dest_reg,
        p_sfpnonlinear::EXP_MODE);  // Read value from work_reg, take exponential, load back into dest_reg
    TTI_SFPADD(
        p_sfpu::LCONST_1,
        dest_reg,
        p_sfpu::LCONST_1,
        work_reg,
        0);  // Add 1 to dest_reg, store in work_reg, takes 2 cycles
    TTI_SFPNONLINEAR(
        work_reg,
        dest_reg,
        p_sfpnonlinear::RECIP_MODE);  // Read value from work_reg, approximate recip, load back into dest_reg
}

// Row pairs per software-pipelined block: Dest is addressed by SFPLOAD/SFPSTORE immediates inside a
// block and advanced by a single INCRWC at its end (8 pairs = one 16-row face).
constexpr int SIGMOID_BLOCK_ITERATIONS = 8;

/**
 * @brief One row pair of a software-pipelined sigmoid block.
 *
 * Computes sigmoid of the pair in X, whose negated input the previous step already left in W. The
 * next pair is loaded and negated in between, so each SFPNONLINEAR is followed by an ALU instruction
 * rather than an SFPLOAD/SFPSTORE. On perf_eltwise_unary_sfpu_quasar this order measured 211 cycles
 * per tile, against 239 for the same pipeline with SFPSTORE right after the reciprocal.
 *
 * @tparam X: LREG holding this pair's input; receives the result.
 * @tparam W: Work LREG for this pair, already holding -X.
 * @tparam NEXT_X: LREG the next pair is loaded into (the other register set).
 * @tparam NEXT_W: Work LREG of the next pair; receives -NEXT_X.
 * @tparam OFFSET: Dest address of this pair relative to the Dest counter.
 * @tparam HAS_NEXT: Load and negate the next pair (false for the last pair of a block).
 */
template <
    std::uint32_t X,
    std::uint32_t W,
    std::uint32_t NEXT_X,
    std::uint32_t NEXT_W,
    std::uint32_t OFFSET,
    bool HAS_NEXT>
inline void _calculate_sigmoid_pipelined_pair_() {
    TTI_SFPNONLINEAR(W, X, p_sfpnonlinear::EXP_MODE);
    TTI_SFPADD(p_sfpu::LCONST_1, X, p_sfpu::LCONST_1, W, 0 /*instr_mod1*/);
    if constexpr (HAS_NEXT) {
        TTI_SFPLOAD(
            NEXT_X,
            p_sfpu::sfpmem::DEFAULT,
            ADDR_MOD_7,
            0 /*done*/,
            OFFSET + ckernel::math::SFP_ROWS /*dest_reg_addr*/);
    }
    TTI_SFPNONLINEAR(W, X, p_sfpnonlinear::RECIP_MODE);
    if constexpr (HAS_NEXT) {
        TTI_SFPMOV(NEXT_X, NEXT_W, 1 /*instr_mod1: invert sign*/);
    }
    TTI_SFPSTORE(X, 0 /*instr_mod0*/, ADDR_MOD_7, 0 /*done*/, OFFSET /*dest_reg_addr*/);
}

/** @brief Pairs I..N-1 of a block, alternating register sets {LREG0,LREG1} and {LREG2,LREG3}. */
template <int I, int N>
inline void _calculate_sigmoid_block_pairs_() {
    if constexpr (I < N) {
        constexpr bool EVEN = (I % 2) == 0;
        _calculate_sigmoid_pipelined_pair_<
            EVEN ? p_sfpu::LREG0 : p_sfpu::LREG2,
            EVEN ? p_sfpu::LREG1 : p_sfpu::LREG3,
            EVEN ? p_sfpu::LREG2 : p_sfpu::LREG0,
            EVEN ? p_sfpu::LREG3 : p_sfpu::LREG1,
            I * ckernel::math::SFP_ROWS,
            (I + 1) < N>();
        _calculate_sigmoid_block_pairs_<I + 1, N>();
    }
}

/** @brief N row pairs at the Dest counter, then advance the counter past them. */
template <int N>
inline void _calculate_sigmoid_block_() {
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, 0 /*dest_reg_addr*/);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG1, 1 /*instr_mod1: invert sign*/);
    _calculate_sigmoid_block_pairs_<0, N>();
    ckernel::math::_incr_counters_<0x0, 0x0, N * ckernel::math::SFP_ROWS, 0x0>();
}

/**
 * @brief Sigmoid over ITERATIONS row pairs starting at the Dest counter (TTI SFPNONLINEAR path).
 *
 * Software-pipelined: pairs alternate between two LREG sets so the next pair's SFPLOAD/SFPMOV fill
 * the slots after each SFPNONLINEAR, and Dest is stepped once per block instead of once per pair.
 * Advances the Dest counter by ITERATIONS * SFP_ROWS, like the per-pair loop it replaces.
 *
 * @tparam ITERATIONS: Number of row pairs (SFP_ROWS rows each) to process.
 * @note Uses LREG0-LREG3 and ADDR_MOD_7 (all-zero, programmed by the SFPU init).
 */
template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_sigmoid_() {
    constexpr int BLOCK = ITERATIONS < SIGMOID_BLOCK_ITERATIONS ? ITERATIONS : SIGMOID_BLOCK_ITERATIONS;
    for (int b = 0; b < ITERATIONS / BLOCK; b++) {
        _calculate_sigmoid_block_<BLOCK>();
    }
    if constexpr ((ITERATIONS % BLOCK) != 0) {
        _calculate_sigmoid_block_<ITERATIONS % BLOCK>();
    }
}

// SFPI sigmoid body used by clamped_silu_glu and ttnn's minimal_matmul swiglu_sfpu.
// See the file-top comment for how this relates to calculate_sigmoid / _calculate_sigmoid_.
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);
    } else {
        result = _sfpu_reciprocal_<1>(denominator);
    }

    return result;
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    if constexpr (!APPROXIMATION_MODE) {
        _init_reciprocal_<false>();
    } else {
        sigmoid_appx_init();
    }
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_sigmoid() {
    _calculate_sigmoid_<ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
