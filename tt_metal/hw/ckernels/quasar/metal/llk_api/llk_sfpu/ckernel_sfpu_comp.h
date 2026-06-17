// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief sfpi container type for an *integer* dst_reg[0] format (vInt→SMAG32, vSMag16→SMAG16,
 * vUInt16→UINT16).
 *
 * @note Floats (all widths share @c vFloat) and the 8-bit formats (no sfpi dst_reg conversion; raw
 *       SFPLOAD path) are handled in @ref zero_comp_traits, not here.
 */
template <DataFormat FMT>
struct dst_container {
    using type = sfpi::vInt;
};  // Int32
template <>
struct dst_container<DataFormat::Int16> {
    using type = sfpi::vSMag16;
};
template <>
struct dst_container<DataFormat::UInt16> {
    using type = sfpi::vUInt16;
};

/**
 * @brief Per-DataFormat load/store and 0/1 encoding for the comparison-to-zero result.
 *
 * @c load reads the element as raw @c vInt bits for the shared predicate (see @ref _zero_comp_pred_);
 * @c store writes @c zero/@c one back in FMT's native encoding. The 16/32-bit and float formats use
 * the sfpi @c dst_reg[0] container; the 8-bit formats use a raw explicit-mode SFPLOAD/SFPSTORE.
 *
 * @tparam FMT: SFPU DataFormat (sfpu_math). Int32 / Int16 / Int8 signed, UInt16 / UInt8 unsigned, or
 *         any IEEE float width (Float32/Float16/Float16_b all share the float path).
 *
 * @todo Once sfpi adds an 8-bit dst_reg[0] conversion (vSMag8/vUInt8), add Int8→vSMag8 /
 *       UInt8→vUInt8 to dst_container and drop the is_raw8 raw-SFPLOAD branches so 8-bit follows the
 *       same container path as the other formats.
 */
template <DataFormat FMT>
struct zero_comp_traits {
    static constexpr bool is_float = (FMT == DataFormat::Float32);
    static constexpr bool is_raw8 = (FMT == DataFormat::Int8 || FMT == DataFormat::UInt8);

    // Result encoding: float formats emit 0.0f/1.0f, every integer format emits integer 0/1.
    using result_t = std::conditional_t<is_float, sfpi::vFloat, sfpi::vInt>;

    // dst_reg[0] container carrying FMT's width/encoding (unused on the 8-bit raw path): vFloat for
    // every float width, else the per-format integer container.
    using container_t = std::conditional_t<is_float, sfpi::vFloat, typename dst_container<FMT>::type>;

    // sfpmem mode for the raw 8-bit SFPLOAD/SFPSTORE path.
    static constexpr std::uint32_t sfpmem8 =
        (FMT == DataFormat::UInt8) ? ckernel::p_sfpu::sfpmem::UINT8 : ckernel::p_sfpu::sfpmem::INT8;

    static inline __attribute__((always_inline)) sfpi::vInt load() {
        if constexpr (is_raw8) {
            return sfpi::vInt(__builtin_rvtt_sfpload(0, sfpmem8, sfpi::SFPLOAD_ADDR_MODE_NOINC));
        } else {
            container_t c = sfpi::dst_reg[0];
            return sfpi::as<sfpi::vInt>(c);
        }
    }
    // result_t(0)/result_t(1): vInt(0/1), or vFloat(0/1) which the vFloat(float) ctor folds to 0.0f/1.0f.
    static inline __attribute__((always_inline)) result_t zero() { return result_t(0); }
    static inline __attribute__((always_inline)) result_t one() { return result_t(1); }
    static inline __attribute__((always_inline)) void store(result_t r) {
        if constexpr (is_raw8) {
            __builtin_rvtt_sfpstore(r.get(), 0, sfpmem8, ckernel::ADDR_MOD_6);
        } else {
            sfpi::dst_reg[0].mode<>(ckernel::ADDR_MOD_6) = sfpi::as<container_t>(r);
        }
    }
};

/**
 * @brief Lanes of @c v that receive the written (non-default) result for COMP_MODE, tested against zero.
 *
 * Two tests, both reading identically across every format (Quasar loads signed integers as
 * sign-magnitude and floats as IEEE, both with the sign in bit 31): the magnitude @c setsgn(v,0)
 * (sign bit cleared, so ±0 -> 0) and the sign via a native @c v<0 / @c v>=0 compare (the bit-31
 * test). Clearing the sign makes the magnitude-zero test hold for both +0 (0x00000000) and -0
 * (0x80000000), so -0 counts as zero, matching IEEE (-0.0 == 0); "negative" is @c v<0 &&
 * magnitude-nonzero, excluding -0.
 *
 * gtez/ltez return their strict complement (ltz/gtz) predicate; the loop defaults those lanes' result
 * to 1 and writes 0 here (see @ref _zero_comp_writes_zero_).
 *
 * @note Do NOT use @c abs(vInt) for the magnitude — that is two's-complement abs and leaves
 *       sign-magnitude -0 (0x80000000) unchanged (nonzero), breaking eqz(-0). @c setsgn(v,0) is the
 *       correct, format-agnostic primitive.
 *
 * @tparam COMP_MODE: Comparison-to-zero mode, values =
 *         <equal_zero/not_equal_zero/less_than_zero/greater_than_zero/greater_than_equal_zero/less_than_equal_zero>
 */
template <SfpuType COMP_MODE>
inline __attribute__((always_inline)) sfpi::vBool _zero_comp_pred_(sfpi::vInt v) {
    static_assert(
        COMP_MODE == SfpuType::equal_zero || COMP_MODE == SfpuType::not_equal_zero ||
            COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::greater_than_zero ||
            COMP_MODE == SfpuType::less_than_equal_zero || COMP_MODE == SfpuType::greater_than_equal_zero,
        "_zero_comp_pred_: COMP_MODE must be one of the six comparison-to-zero SfpuType modes "
        "(equal_zero/not_equal_zero/less_than_zero/greater_than_zero/less_than_equal_zero/greater_than_equal_zero)");
    const sfpi::vInt mag = sfpi::setsgn(v, 0);  // sign bit cleared -> magnitude (±0 -> 0)
    if constexpr (COMP_MODE == SfpuType::equal_zero) {
        return mag == 0;  // ±0
    } else if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
        return mag != 0;
    } else if constexpr (COMP_MODE == SfpuType::less_than_zero) {
        return (v < 0) && (mag != 0);  // sign set and nonzero -> excludes -0.0
    } else if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
        return (v >= 0) && (mag != 0);  // sign clear and nonzero
    } else if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
        return (v < 0) && (mag != 0);   // strict-negative (ltz) lanes; loop defaults 1 and writes 0 here -> gtez
    } else {                            // less_than_equal_zero
        return (v >= 0) && (mag != 0);  // strict-positive (gtz) lanes; loop defaults 1 and writes 0 here -> ltez
    }
}

/**
 * @brief Whether COMP_MODE's loop defaults the result to 1 and writes 0 on its predicate (vs the
 * default 0 / write 1).
 *
 * gtez/ltez are the lane-wise complement of ltz/gtz, so @ref _zero_comp_pred_ returns their strict
 * (ltz/gtz) predicate and the loop inverts the default+write — costing one fewer SFPU op than an
 * explicit OR-combine. ±0 (magnitude 0) is in neither strict set, so it keeps the default 1 (true)
 * for both, matching IEEE.
 */
template <SfpuType COMP_MODE>
inline constexpr bool _zero_comp_writes_zero_() {
    return COMP_MODE == SfpuType::greater_than_equal_zero || COMP_MODE == SfpuType::less_than_equal_zero;
}

/**
 * @brief Program the dest-increment addr mod shared by every comparison-to-zero instantiation.
 *
 * @c ADDR_MOD_6 (dest.incr=2) lets the body's SFPSTORE advance the dest counter, so
 * @ref _calculate_zero_comp_ needs no dst_reg++. It is HW state shared across every COMP_MODE/FMT
 * instantiation, so programming it once here (rather than per @ref _calculate_zero_comp_ call) keeps
 * it out of the per-tile loop body.
 *
 * @note Call once after @ref _llk_math_eltwise_sfpu_init_ and before @ref _calculate_zero_comp_.
 */
inline void _init_zero_comp_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
}

/**
 * @brief Element-wise comparison-to-zero over a tile, written as 1/0 booleans.
 *
 * Defaults each result lane and writes the opposite into the lanes matching COMP_MODE (eqz/nez/ltz/gtz
 * default 0 and write 1; gtez/ltez default 1 and write 0 — see @ref _zero_comp_writes_zero_), in FMT's
 * native encoding (see @ref zero_comp_traits). The body's SFPSTORE rides @c ADDR_MOD_6 (dest.incr=2,
 * programmed in @ref _init_zero_comp_) to advance the dest counter, so the loop needs no dst_reg++.
 *
 * @tparam APPROXIMATION_MODE: Unused (no approx path); retained for dispatcher signature symmetry.
 * @tparam FMT: SFPU DataFormat (sfpu_math): Int32/Int16/Int8/UInt16/UInt8, or Float32 for any float
 *         width — the caller must pass Float32 for Float16/Float16_b too, whose DEFAULT sfpmem mode
 *         resolves the actual width from the dest format config. Anything else is a compile error.
 * @tparam COMP_MODE: Comparison-to-zero mode.
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32×16 face).
 * @note Requires @ref _init_zero_comp_ to have programmed @c ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, DataFormat FMT, SfpuType COMP_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_zero_comp_() {
    constexpr bool is_int_fmt = FMT == DataFormat::Int32 || FMT == DataFormat::Int16 || FMT == DataFormat::Int8 ||
                                FMT == DataFormat::UInt16 || FMT == DataFormat::UInt8;
    constexpr bool is_float_fmt =
        FMT == DataFormat::Float32 || FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b;
    static_assert(
        is_int_fmt || is_float_fmt,
        "_calculate_zero_comp_: unsupported FMT (expected an integer format or an IEEE float width)");

    using traits = zero_comp_traits<FMT>;

    // gtez/ltez default to 1 and write 0 on their (strict-sign) predicate; the other modes default
    // to 0 and write 1 (see @ref _zero_comp_writes_zero_).
    constexpr bool writes_zero = _zero_comp_writes_zero_<COMP_MODE>();

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt bits = traits::load();

        typename traits::result_t result = writes_zero ? traits::one() : traits::zero();
        v_if(_zero_comp_pred_<COMP_MODE>(bits)) { result = writes_zero ? traits::zero() : traits::one(); }
        v_endif;

        traits::store(result);
    }
}

}  // namespace sfpu
}  // namespace ckernel
