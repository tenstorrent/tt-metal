// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
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

/**
 * @brief Square a Dest span in place.
 *
 * @note Call @ref init_square before this.
 */
template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_square() {
    using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
    using Output = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat, ADDR_MOD_6>>;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_unary_operands<SquareMath, 1 /*ITERATIONS*/>(Input{}, Output{});
    }
}

/** @brief SrcS square op type; Sfpi only. */
template <sfpi::DataLayout LAYOUT, SfpuImpl IMPL = SfpuImpl::Sfpi>
using SquareSrcs = SrcsUnary<SquareMath, LAYOUT, resolve_sfpu_impl<IMPL, false /*HAS_LOADMACRO*/>()>;

// dispatch_sfpu_srcs_op relies on naming a float op for I32 without instantiating it; keep that true.
static_assert(!srcs_op_supports<SquareSrcs, sfpi::DataLayout::I32>, "float SrcS ops must stay nameable for I32");

}  // namespace sfpu
}  // namespace ckernel
