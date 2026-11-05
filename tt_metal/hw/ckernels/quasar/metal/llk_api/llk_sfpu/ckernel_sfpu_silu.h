// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// silu(x) = x * sigmoid(x) = x * 1 / (1 + exp(-x)). Per row pair that is a chain of six dependent
// instructions:
//   SFPLOAD x -> SFPMOV t = -x -> SFPNONLINEAR d = exp(t) -> SFPADD t = d + 1
//             -> SFPNONLINEAR d = 1 / t -> SFPMUL t = x * d -> SFPSTORE t
// On Quasar, an SFPMUL or an SFPLOAD/SFPSTORE issued in the cycle right after an SFPNONLINEAR waits
// one cycle, and so does an SFPSTORE right after the SFPMUL that produced its value. A dependent
// SFPADD or SFPMOV right after SFPNONLINEAR does not. Done one pair at a time, the chain loses a cycle
// per pair to these waits. The block below overlaps three pairs so every one of those slots holds an
// instruction of another pair.

// Row pairs per block: inside a block Dest is addressed with SFPLOAD/SFPSTORE immediates, and the
// Dest counter is stepped once at the end of the block (8 pairs = one 16-row face).
constexpr int SILU_BLOCK_ITERATIONS = 8;

/** @brief LREGs of row pair I: input x, temporary t, nonlinear result d. Even and odd pairs alternate sets. */
template <int I>
struct SiluRegs {
    static constexpr std::uint32_t X = (I % 2 == 0) ? p_sfpu::LREG0 : p_sfpu::LREG3;
    static constexpr std::uint32_t T = (I % 2 == 0) ? p_sfpu::LREG1 : p_sfpu::LREG4;
    static constexpr std::uint32_t D = (I % 2 == 0) ? p_sfpu::LREG2 : p_sfpu::LREG5;
    static constexpr std::uint32_t OFFSET = I * ckernel::math::SFP_ROWS;  // Dest address relative to the counter
};

/**
 * @brief Steady-state step I of a silu block of N row pairs.
 *
 * Finishes pair I, runs the middle of pair I+1 and starts pair I+2, in this order:
 *   SFPMUL(I)  SFPNONLINEAR exp(I+1)  SFPADD(I+1)  SFPLOAD(I+2)  SFPSTORE(I)  SFPNONLINEAR recip(I+1)  SFPMOV(I+2)
 * exp is followed by its own SFPADD and recip by the next pair's SFPMOV; neither waits. SFPSTORE(I)
 * sits three instructions after the SFPMUL that produced its value. SFPADD and the SFPNONLINEAR that
 * reads its result are never adjacent. Steps near the block end skip the work of pairs that do not
 * exist; the last pair of a block still waits twice (recip -> SFPMUL -> SFPSTORE).
 */
template <int I, int N>
inline void _calculate_silu_block_steps_() {
    if constexpr (I < N) {
        using P = SiluRegs<I>;
        using P1 = SiluRegs<I + 1>;
        using P2 = SiluRegs<I + 2>;
        constexpr bool HAS_P1 = I + 1 < N;
        constexpr bool HAS_P2 = I + 2 < N;

        TTI_SFPMUL(P::X, P::D, p_sfpu::LCONST_0, P::T, 0 /*instr_mod1*/);  // t = x * sigmoid(x)
        if constexpr (HAS_P1) {
            TTI_SFPNONLINEAR(P1::T, P1::D, p_sfpnonlinear::EXP_MODE);                        // d = exp(-x)
            TTI_SFPADD(p_sfpu::LCONST_1, P1::D, p_sfpu::LCONST_1, P1::T, 0 /*instr_mod1*/);  // t = d + 1
        }
        if constexpr (HAS_P2) {
            TTI_SFPLOAD(P2::X, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, P2::OFFSET);
        }
        TTI_SFPSTORE(P::T, 0 /*instr_mod0*/, ADDR_MOD_7, 0 /*done*/, P::OFFSET);
        if constexpr (HAS_P1) {
            TTI_SFPNONLINEAR(P1::T, P1::D, p_sfpnonlinear::RECIP_MODE);  // d = 1 / t
        }
        if constexpr (HAS_P2) {
            TTI_SFPMOV(P2::X, P2::T, 1 /*instr_mod1: invert sign*/);  // t = -x
        }
        _calculate_silu_block_steps_<I + 1, N>();
    }
}

/** @brief Silu of N row pairs at the Dest counter, then advance the counter past them. */
template <int N>
inline void _calculate_silu_block_() {
    using P0 = SiluRegs<0>;
    using P1 = SiluRegs<1>;
    // Prologue: pair 0 up to its reciprocal, pair 1 loaded and negated.
    TTI_SFPLOAD(P0::X, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, P0::OFFSET);
    TTI_SFPMOV(P0::X, P0::T, 1 /*instr_mod1: invert sign*/);
    TTI_SFPNONLINEAR(P0::T, P0::D, p_sfpnonlinear::EXP_MODE);
    TTI_SFPADD(p_sfpu::LCONST_1, P0::D, p_sfpu::LCONST_1, P0::T, 0 /*instr_mod1*/);
    if constexpr (N > 1) {
        TTI_SFPLOAD(P1::X, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, P1::OFFSET);
    }
    TTI_SFPNONLINEAR(P0::T, P0::D, p_sfpnonlinear::RECIP_MODE);
    if constexpr (N > 1) {
        TTI_SFPMOV(P1::X, P1::T, 1 /*instr_mod1: invert sign*/);
    }
    _calculate_silu_block_steps_<0, N>();
    ckernel::math::_incr_counters_<0x0, 0x0, N * ckernel::math::SFP_ROWS, 0x0>();
}

/**
 * @brief Silu over ITERATIONS row pairs starting at the Dest counter.
 *
 * Software-pipelined (see @ref _calculate_silu_block_steps_). Advances the Dest counter by
 * ITERATIONS * SFP_ROWS, as the one-pair-at-a-time loop it replaces did.
 *
 * @tparam ITERATIONS: Number of row pairs (SFP_ROWS rows each) to process.
 * @note Uses LREG0-LREG5 and ADDR_MOD_7 (all-zero, programmed by the SFPU init).
 */
template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_silu_() {
    constexpr int BLOCK = ITERATIONS < SILU_BLOCK_ITERATIONS ? ITERATIONS : SILU_BLOCK_ITERATIONS;
    for (int b = 0; b < ITERATIONS / BLOCK; b++) {
        _calculate_silu_block_<BLOCK>();
    }
    if constexpr ((ITERATIONS % BLOCK) != 0) {
        _calculate_silu_block_<ITERATIONS % BLOCK>();
    }
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_silu() {
    _calculate_silu_<ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
