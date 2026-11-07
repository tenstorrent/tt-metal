// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_srcs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

/** @brief Square math policy, shared by Dest and SrcS. */
struct SquareMath {
    using value_type = sfpi::vFloat;
    sfpi_inline static sfpi::vFloat apply(sfpi::vFloat x) { return x * x; }
};

/**
 * @brief Configure the SFPU address mode used by the square op.
 *
 * Programs ADDR_MOD_6 with a dest increment of 2 (one SFPU pass writes 2 rows on Quasar).
 *
 * @note Call this before @ref calculate_square to set up the address mode it relies on.
 */
inline void init_square() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
}

// Row pairs per block: inside a block Dest is addressed with SFPLOAD/SFPSTORE immediates, and the
// Dest counter is stepped once at the end of the block (8 pairs = one 16-row face).
constexpr int SQUARE_BLOCK_ITERATIONS = 8;

/**
 * @brief Pairs I..N-1 of a square block, software-pipelined over two LREGs.
 *
 * Pair I was loaded into X and squared by the previous step. This step loads pair I+1 into the other
 * register, stores pair I, then squares pair I+1. So no SFPSTORE is ever directly followed by an
 * SFPLOAD (the hardware leaves one empty issue cycle between them), and each SFPSTORE is one
 * instruction away from the SFPMUL that produced its value.
 *
 * @tparam I: Index of the pair to store, values = <0..N-1>
 * @tparam N: Pairs in this block
 * @tparam X: LREG holding pair I squared, values = <LREG0/LREG1>
 */
template <int I, int N, std::uint32_t X>
inline void _calculate_square_block_pairs_() {
    if constexpr (I < N) {
        constexpr std::uint32_t NEXT = (X == p_sfpu::LREG0) ? p_sfpu::LREG1 : p_sfpu::LREG0;
        constexpr std::uint32_t OFFSET = I * ckernel::math::SFP_ROWS;
        if constexpr (I + 1 < N) {
            TTI_SFPLOAD(
                NEXT,
                p_sfpu::sfpmem::DEFAULT,
                ADDR_MOD_7,
                0 /*done*/,
                OFFSET + ckernel::math::SFP_ROWS /*dest_reg_addr*/);
        }
        TTI_SFPSTORE(X, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, OFFSET /*dest_reg_addr*/);
        if constexpr (I + 1 < N) {
            TTI_SFPMUL(NEXT, NEXT, p_sfpu::LCONST_0, NEXT, 0 /*instr_mod1*/);
        }
        _calculate_square_block_pairs_<I + 1, N, NEXT>();
    }
}

/** @brief Square N row pairs at the Dest counter, then advance the counter past them. */
template <int N>
inline void _calculate_square_block_() {
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, 0 /*dest_reg_addr*/);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /*instr_mod1*/);
    _calculate_square_block_pairs_<0, N, p_sfpu::LREG0>();
    ckernel::math::_incr_counters_<0x0, 0x0, N * ckernel::math::SFP_ROWS, 0x0>();
}

/**
 * @brief Square a Dest span of ITERATIONS row pairs in place.
 *
 * Software-pipelined (see @ref _calculate_square_block_pairs_). Advances the Dest counter by
 * ITERATIONS * SFP_ROWS, as the one-pair-at-a-time loop it replaces did.
 *
 * @tparam ITERATIONS: Number of row pairs (SFP_ROWS rows each) to process.
 * @note Uses LREG0-LREG1 and ADDR_MOD_7 (all-zero, programmed by the SFPU init). @ref init_square
 *       still programs ADDR_MOD_6 for callers that expect it.
 */
template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_square() {
    constexpr int BLOCK = ITERATIONS < SQUARE_BLOCK_ITERATIONS ? ITERATIONS : SQUARE_BLOCK_ITERATIONS;
    for (int b = 0; b < ITERATIONS / BLOCK; b++) {
        _calculate_square_block_<BLOCK>();
    }
    if constexpr ((ITERATIONS % BLOCK) != 0) {
        _calculate_square_block_<ITERATIONS % BLOCK>();
    }
}

/** @brief SrcS square op type; Sfpi only. */
template <sfpi::DataLayout LAYOUT, SfpuImpl IMPL = SfpuImpl::Sfpi>
using SquareSrcs = SrcsUnary<SquareMath, LAYOUT, resolve_sfpu_impl<IMPL, false /*HAS_LOADMACRO*/>()>;

// dispatch_sfpu_srcs_op relies on naming a float op for I32 without instantiating it; keep that true.
static_assert(!srcs_op_supports<SquareSrcs, sfpi::DataLayout::I32>, "float SrcS ops must stay nameable for I32");

}  // namespace sfpu
}  // namespace ckernel
