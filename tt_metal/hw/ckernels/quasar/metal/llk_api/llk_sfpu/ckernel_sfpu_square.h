// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_ops.h"
#include "ckernel_sfpu_srcs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

/// Math policy for square: x * x. Shared by the Dest and SrcS paths via @ref calculate_unary_operands.
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

/**
 * @brief Square a Dest span in place: dest = x * x (one face with default ITERATIONS).
 *
 * @tparam ITERATIONS: Number of SFPU passes (each covers 2 rows).
 * @note Call @ref init_square before this to program the address mode it depends on.
 */
template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_square() {
    using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
    using Output = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat, ADDR_MOD_6>>;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_unary_operands<SquareMath, 1>(Input{}, Output{});
    }
}

/**
 * @brief SrcS square op type (init() / run(), see @ref SfpuSrcsUnaryOp): x * x over one slice per call.
 *
 * Sfpi only: square has no SFPLOADMACRO version, so requesting LoadMacro fails to compile.
 *
 * @tparam LAYOUT: Load and store layout, values = <F16a/F16b/F32>; unpack destination and pack
 *         source formats must match.
 * @tparam ISSUE: Issue mechanism, values = <Sfpi>.
 */
template <sfpi::DataLayout LAYOUT, SfpuIssue ISSUE = SfpuIssue::Sfpi>
using SquareSrcs = SrcsUnary<SquareMath, LAYOUT, resolve_sfpu_issue<ISSUE, false /*HAS_LOADMACRO*/>()>;

}  // namespace sfpu
}  // namespace ckernel
