// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Whether the predicated SFPSTORE of each row advances Dest itself, through ADDR_MOD_6 =
// {dest.incr = SFP_DESTREG_STRIDE} from isinf_isnan_init (the store names it ADDR_MOD_2 under the SFPU
// addr-mod base), or the body increments Dest with a TTINCRWC per row. A throttled matmul
// (MM_THROTTLE > 0) programs ADDR_MOD_6 for its own MOP, and the fused-activation matmul / conv
// kernels run matmul_block between the activation's init and its body, so in a kernel built with
// MM_THROTTLE neither the init nor the body touches ADDR_MOD_6.
#if defined(MM_THROTTLE) && MM_THROTTLE > 0
constexpr bool ISINF_ISNAN_STORE_INCR = false;
#else
constexpr bool ISINF_ISNAN_STORE_INCR = true;
#endif

/**
 * @brief Programs the state _calculate_sfpu_isinf_isnan_ reads: ADDR_MOD_6 and the RWC counters.
 *
 * ADDR_MOD_6 = {dest.incr = SFP_DESTREG_STRIDE}, one SFPU row per predicated store; a kernel built with
 * MM_THROTTLE leaves ADDR_MOD_6 alone (see ISINF_ISNAN_STORE_INCR).
 *
 * @note ADDR_MOD_6 is shared: call this after any init that programs it differently, and re-run that
 *       op's init before its body if it runs after this one.
 */
inline void isinf_isnan_init() {
    if constexpr (ISINF_ISNAN_STORE_INCR) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = sfpi::SFP_DESTREG_STRIDE}}.set(
            ADDR_MOD_6);
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void isfinite_init() { isinf_isnan_init(); }

inline void isinf_init() { isinf_isnan_init(); }

inline void isnan_init() { isinf_isnan_init(); }

inline void isneginf_init() { isinf_isnan_init(); }

inline void isposinf_init() { isinf_isnan_init(); }

// 1.0 on the lanes the enclosing v_if enables; outside an MM_THROTTLE kernel the store also advances
// Dest by one SFPU row (ADDR_MOD_2, see ISINF_ISNAN_STORE_INCR).
sfpi_inline void _isinf_isnan_store_true_() {
    if constexpr (ISINF_ISNAN_STORE_INCR) {
        sfpi::dst_reg[0].mode(ADDR_MOD_2) = 1.0f;
    } else {
        sfpi::dst_reg[0] = 1.0f;
    }
}

/**
 * @brief Writes 1.0 where the predicate holds and 0.0 elsewhere, over ITERATIONS Dest rows.
 *
 * Each predicate is an interval test on the raw fp32 bits, read as a signed integer:
 *   isinf    <=> |v| == pos_inf_bits       isposinf <=> v == pos_inf_bits
 *   isnan    <=> |v| >  pos_inf_bits       isneginf <=> v == neg_inf_bits
 *   isfinite <=> |v| <= max_finite_bits
 * which is equivalent to the usual exponent/mantissa field test for each class. The bound is
 * subtracted in one SFPIADD; for the magnitude tests it cannot overflow, and for the equality tests
 * only an all-bits-zero difference matches. The default 0.0 is stored first and 1.0 is stored over it
 * on the lanes that satisfy the predicate.
 *
 * @tparam operation: One of isinf, isposinf, isneginf, isnan, isfinite.
 * @tparam APPROXIMATION_MODE: Unused; the predicates are exact.
 * @tparam ITERATIONS: Number of Dest rows to process.
 * @note Call @ref isinf_isnan_init first; outside an MM_THROTTLE kernel it programs the ADDR_MOD_6 the
 *       store advances Dest through (see ISINF_ISNAN_STORE_INCR).
 */
template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_() {
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
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;

        if constexpr (operation == SfpuType::isposinf || operation == SfpuType::isneginf) {
            v_if(bound_v - sfpi::as<sfpi::vInt>(in) == 0) { _isinf_isnan_store_true_(); }
            v_endif;
        } else {
            sfpi::vInt mag = sfpi::as<sfpi::vInt>(sfpi::setsgn(in, 0 /*sgn*/));
            if constexpr (operation == SfpuType::isinf) {
                v_if(bound_v - mag == 0) { _isinf_isnan_store_true_(); }
                v_endif;
            } else if constexpr (operation == SfpuType::isnan) {
                v_if(bound_v < mag) { _isinf_isnan_store_true_(); }
                v_endif;
            } else {
                v_if(bound_v >= mag) { _isinf_isnan_store_true_(); }
                v_endif;
            }
        }

        if constexpr (!ISINF_ISNAN_STORE_INCR) {
            sfpi::dst_reg++;
        }
    }
}

}  // namespace ckernel::sfpu
