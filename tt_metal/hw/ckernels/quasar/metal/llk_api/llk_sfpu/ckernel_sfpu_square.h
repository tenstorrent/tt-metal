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

/**
 * @brief Square on independently located floating-point operands: output = input * input.
 *
 * Advances explicit indices only; the caller owns setup and synchronization. Input/output
 * ranges must coincide or be disjoint.
 */
template <int ITERATIONS, class Input, class Output>
sfpi_inline void calculate_square_operands(const Input& input, const Output& output) {
    static_assert(ITERATIONS > 0, "Square requires at least one SFPI access");
    static_assert(
        std::is_same_v<typename Input::value_type, sfpi::vFloat> &&
            std::is_same_v<typename Output::value_type, sfpi::vFloat>,
        "Square requires floating-point operands");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat value = input.load(d);
        output.store(d, value * value);
    }
}

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
        // The store advances Dest; keep explicit operand indices fixed at zero.
        calculate_square_operands<1>(Input{}, Output{});
    }
}

// Reference adapter for the unified Dest/SrcS API; exercised by test_isolate_sfpu_square_quasar.
/**
 * @brief Square over one SrcS slice (slots per @ref SrcsLayout).
 *
 * @tparam LAYOUT: Load and store layout, values = <F16a/F16b/F32>; unpack destination and pack
 *         source formats must match.
 * @note The caller runs unpack/pack and clears the SrcS valids after this call, as
 *       llk_sfpu_srcs_unary does.
 */
template <sfpi::DataLayout LAYOUT>
sfpi_inline void calculate_square_srcs() {
    using Layout = SrcsLayout<LAYOUT>;
    using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, sfpi::vFloat>>;
    calculate_square_operands<Layout::ops>(Operand{Layout::in0}, Operand{Layout::out});
}

}  // namespace sfpu
}  // namespace ckernel
