// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_sfpu_srcs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

/// Square math policy, shared by Dest and SrcS.
struct SquareMath {
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

/// Square a Dest span in place. Call init_square first.
template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_square() {
    using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
    using Output = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat, ADDR_MOD_6>>;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_unary_operands<SquareMath, 1>(Input{}, Output{});
    }
}

/// SrcS square op type, Sfpi only.
template <sfpi::DataLayout LAYOUT, SfpuIssue ISSUE = SfpuIssue::Sfpi>
using SquareSrcs = SrcsUnary<SquareMath, LAYOUT, resolve_sfpu_issue<ISSUE, false /*HAS_LOADMACRO*/>()>;

}  // namespace sfpu
}  // namespace ckernel
