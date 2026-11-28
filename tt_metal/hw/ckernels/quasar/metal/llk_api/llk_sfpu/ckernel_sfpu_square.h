// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_sfpu_dest.h"
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

/** @brief Math of op::Square on Dest and SrcS. */
template <sfpi::DataLayout LAYOUT, IntEncoding INT_ENCODING>
struct math_for<op::Square, LAYOUT, INT_ENCODING> {
    using type = SquareMath;
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
 * @tparam ITERATIONS: number of SFPI rows to process.
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format; Default lets the HW derive it from the ALU config.
 * @note Call @ref init_square before this.
 */
template <int ITERATIONS = SFPU_ITERATIONS, sfpi::DataLayout LAYOUT = sfpi::DataLayout::Default>
inline void calculate_square() {
    using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<LAYOUT, sfpi::vFloat>>;
    using Output = SfpuOperand<SfpuReg::Dest, SfpiFormat<LAYOUT, sfpi::vFloat, ADDR_MOD_6>>;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_operands<SquareMath, 1 /*ITERATIONS*/>(math_setup<SquareMath>(), Output{}, Input{});
    }
}

/** @brief Dest square op type: the tuned loop, keeps the ADDR_MOD_6 store of @ref calculate_square. */
template <sfpi::DataLayout LAYOUT, int ITERATIONS, trisc::DstTileShape TILE_SHAPE>
struct SfpuDestOp<SquareMath, LAYOUT, ITERATIONS, TILE_SHAPE>
    : SfpuDestOpBase<SfpuDestOp<SquareMath, LAYOUT, ITERATIONS, TILE_SHAPE>, 1 /*ARITY*/> {
    static_assert(
        std::is_same_v<sfpi::vFloat, layout_value_t<LAYOUT>>, "this SFPU op does not take tiles of this Dest layout");

    static constexpr trisc::DstTileShape tile_shape = TILE_SHAPE;

    static void init_op() { init_square(); }
    static void calculate() { calculate_square<ITERATIONS, LAYOUT>(); }
};

}  // namespace sfpu
}  // namespace ckernel
