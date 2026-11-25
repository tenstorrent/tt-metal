// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

/**
 * @brief Programs ADDR_MOD_6 = {dest.incr = 2}, one SFPU row pair per store of @ref _calculate_sfpu_isinf_isnan_.
 *
 * @note ADDR_MOD_6 is shared: call this after any other init that programs it.
 */
inline void isinf_isnan_init() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
}

/**
 * @brief Writes 1.0 where the predicate holds and 0.0 elsewhere, over ITERATIONS Dest row pairs.
 *
 * Each predicate compares the fp32 bits against one bound: |v| == +inf (isinf), |v| > +inf (isnan),
 * |v| <= max finite (isfinite), v == +inf / -inf (isposinf / isneginf).
 *
 * @tparam operation: One of isinf, isposinf, isneginf, isnan, isfinite.
 * @tparam APPROXIMATION_MODE: Unused; the predicates are exact.
 * @tparam ITERATIONS: Number of Dest row pairs to process.
 * @note Call @ref isinf_isnan_init first; the store of 1.0 advances Dest through ADDR_MOD_6.
 */
template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    static_assert(
        operation == SfpuType::isinf || operation == SfpuType::isposinf || operation == SfpuType::isneginf ||
            operation == SfpuType::isnan || operation == SfpuType::isfinite,
        "_calculate_sfpu_isinf_isnan_ supports only isinf, isposinf, isneginf, isnan and isfinite");

    constexpr std::int32_t pos_inf_bits = 0x7F800000;
    constexpr std::int32_t neg_inf_bits = static_cast<std::int32_t>(0xFF800000u);
    constexpr std::int32_t max_finite_bits = 0x7F7FFFFF;
    constexpr std::int32_t bound = operation == SfpuType::isneginf   ? neg_inf_bits
                                   : operation == SfpuType::isfinite ? max_finite_bits
                                                                     : pos_inf_bits;
    const sfpi::vInt bound_v = bound;

    // SFPU microcode
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        const sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;

        if constexpr (operation == SfpuType::isposinf || operation == SfpuType::isneginf) {
            v_if(sfpi::as<sfpi::vInt>(in) == bound_v) { sfpi::dst_reg[0].mode(ADDR_MOD_6) = 1.0f; }
            v_endif;
        } else {
            const sfpi::vInt mag = sfpi::as<sfpi::vInt>(sfpi::setsgn(in, 0 /*sgn*/));
            if constexpr (operation == SfpuType::isinf) {
                v_if(mag == bound_v) { sfpi::dst_reg[0].mode(ADDR_MOD_6) = 1.0f; }
                v_endif;
            } else if constexpr (operation == SfpuType::isnan) {
                // Integer order: the NaNs are the bit patterns above +inf.
                v_if(bound_v < mag) { sfpi::dst_reg[0].mode(ADDR_MOD_6) = 1.0f; }
                v_endif;
            } else {
                v_if(bound_v >= mag) { sfpi::dst_reg[0].mode(ADDR_MOD_6) = 1.0f; }
                v_endif;
            }
        }
    }
}

} // namespace ckernel::sfpu
