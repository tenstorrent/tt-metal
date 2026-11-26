// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
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

    // Result encoding: float formats emit 0.0f/1.0f, every integer format emits integer 0/1.
    using result_t = std::conditional_t<is_float, sfpi::vFloat, sfpi::vInt>;

    // dst_reg[0] container carrying FMT's width/encoding (unused on the 8-bit raw path): vFloat for
    // every float width, else the per-format integer container.
    using container_t = std::conditional_t<is_float, sfpi::vFloat, typename dst_container<FMT>::type>;

    // Load and store avoids converting SM8 to/from 2's complement
    static inline __attribute__((always_inline)) sfpi::vInt load() {
        if constexpr (FMT == DataFormat::UInt8) {
            return sfpi::vInt(sfpi::dst_reg[0].mode<sfpi::DataLayout::U8>());
        } else if constexpr (FMT == DataFormat::Int8) {
            // The signed 8-bit format is sign magnitude.  Just pun it
            // to vInt.
            return sfpi::as<sfpi::vInt>(sfpi::vSMag(sfpi::dst_reg[0].mode<sfpi::DataLayout::SM8>()));
        } else {
            container_t c = sfpi::dst_reg[0];
            return sfpi::as<sfpi::vInt>(c);
        }
    }
    // INCR selects the addr mod: ADDR_MOD_6 (dest.incr=2) advances Dest, sfpi's no-increment mode
    // leaves it on the row so a second store can overwrite it.
    template <bool INCR = true>
    static inline __attribute__((always_inline)) void store(result_t r) {
        constexpr int addr_mod =
            INCR ? static_cast<int>(ckernel::ADDR_MOD_6) : static_cast<int>(sfpi::SFPSTORE_ADDR_MODE_NOINC);
        if constexpr (FMT == DataFormat::UInt8) {
            sfpi::dst_reg[0].mode<sfpi::DataLayout::U8>(addr_mod) = r;
        } else if constexpr (FMT == DataFormat::Int8) {
            // The signed 8-bit format is sign magnitude.  Just pun
            // the vInt.
            sfpi::dst_reg[0].mode<sfpi::DataLayout::SM8>(addr_mod) = sfpi::as<sfpi::vSMag>(r);
        } else {
            sfpi::dst_reg[0].mode<>(addr_mod) = sfpi::as<container_t>(r);
        }
    }

    // result_t(0)/result_t(1): vInt(0/1), or vFloat(0/1) which the vFloat(float) ctor folds to 0.0f/1.0f.
    static inline __attribute__((always_inline)) result_t zero() { return result_t(0); }
    static inline __attribute__((always_inline)) result_t one() { return result_t(1); }
};

/**
 * @brief Whether COMP_MODE's predicate is the strict-negative one (ltz, and gtez as its complement).
 */
template <SfpuType COMP_MODE>
inline constexpr bool _zero_comp_tests_negative_() {
    return COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::greater_than_equal_zero;
}

/**
 * @brief Lanes of @c v that receive the written (non-default) result for COMP_MODE, tested against zero.
 *
 * Every format has its sign in bit 31, so zero is the two words +0 and -0. The sign modes are integer
 * compares of the raw word, where -0 (0x80000000) is INT32_MIN, below every negative element.
 *
 * gtez/ltez return their strict complement (ltz/gtz) predicate; the loop defaults those lanes' result
 * to 1 and writes 0 here (see @ref _zero_comp_writes_zero_).
 *
 * @note Do NOT use @c abs(vInt) for the magnitude — that is two's-complement abs and leaves
 *       sign-magnitude -0 (0x80000000) unchanged (nonzero), breaking eqz(-0). Clearing the sign bit
 *       (SFPSETSGN with sign=0) is the correct, format-agnostic primitive.
 * @note The compares must stay integer ones (@c vInt operands): a float compare would misread an Int32
 *       whose magnitude has a NaN bit pattern. A float NaN is therefore signed like any other element;
 *       tt-metal#55273 tracks returning 0 for it.
 *
 * @tparam COMP_MODE: Comparison-to-zero mode, values =
 *         <equal_zero/not_equal_zero/less_than_zero/greater_than_zero/greater_than_equal_zero/less_than_equal_zero>
 * @param v: The element's raw word.
 * @param neg_zero: The word 0x80000000 in every lane; read only by the strict-negative modes
 *        (see @ref _zero_comp_tests_negative_).
 */
template <SfpuType COMP_MODE>
inline __attribute__((always_inline)) sfpi::vBool _zero_comp_pred_(sfpi::vInt v, [[maybe_unused]] sfpi::vInt neg_zero) {
    static_assert(
        COMP_MODE == SfpuType::equal_zero || COMP_MODE == SfpuType::not_equal_zero ||
            COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::greater_than_zero ||
            COMP_MODE == SfpuType::less_than_equal_zero || COMP_MODE == SfpuType::greater_than_equal_zero,
        "_zero_comp_pred_: COMP_MODE must be one of the six comparison-to-zero SfpuType modes "
        "(equal_zero/not_equal_zero/less_than_zero/greater_than_zero/less_than_equal_zero/greater_than_equal_zero)");

    if constexpr (COMP_MODE == SfpuType::equal_zero || COMP_MODE == SfpuType::not_equal_zero) {
        // Clear bit 31 (sign) -> magnitude (±0 -> 0) in a single SFPSETSGN; the vInt<->vSMag
        // reinterprets are free (no instruction).
        const sfpi::vSMag mag = sfpi::setsgn(sfpi::as<sfpi::vSMag>(v), 0);
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            return mag == 0;  // ±0
        } else {
            return mag != 0;
        }
    } else if constexpr (_zero_comp_tests_negative_<COMP_MODE>()) {
        // ltz lanes; for gtez the loop defaults 1 and writes 0 here.
        return (v < 0) && (v > neg_zero);
    } else {
        // gtz lanes; for ltez the loop defaults 1 and writes 0 here.
        return v > 0;
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
 * @ref calculate_zero_comp needs no dst_reg++. It is HW state shared across every COMP_MODE/FMT
 * instantiation, so programming it once here (rather than per @ref calculate_zero_comp call) keeps
 * it out of the per-tile loop body.
 *
 * @note Call once after @ref _llk_math_eltwise_sfpu_init_ and before @ref calculate_zero_comp.
 */
inline void init_zero_comp() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
}

/**
 * @brief Element-wise comparison-to-zero over a tile, written as 1/0 booleans.
 *
 * Defaults each result lane and writes the opposite into the lanes matching COMP_MODE (eqz/nez/ltz/gtz
 * default 0 and write 1; gtez/ltez default 1 and write 0 — see @ref _zero_comp_writes_zero_), in FMT's
 * native encoding (see @ref zero_comp_traits). The default is stored to Dest and the other value
 * stored over it on the predicate's lanes. The predicated SFPSTORE rides @c ADDR_MOD_6 (dest.incr=2,
 * programmed in @ref init_zero_comp) to advance the dest counter, so the loop needs no dst_reg++.
 *
 * @tparam APPROXIMATION_MODE: Unused (no approx path); retained for dispatcher signature symmetry.
 * @tparam FMT: SFPU DataFormat (sfpu_math): Int32/Int16/Int8/UInt16/UInt8, or Float32 for any float
 *         width — the caller must pass Float32 for Float16/Float16_b too, whose DEFAULT sfpmem mode
 *         resolves the actual width from the dest format config. Anything else is a compile error.
 * @tparam COMP_MODE: Comparison-to-zero mode.
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32×16 face).
 * @note Requires @ref init_zero_comp to have programmed @c ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, DataFormat FMT, SfpuType COMP_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_zero_comp() {
    constexpr bool is_int_fmt = FMT == DataFormat::Int32 || FMT == DataFormat::Int16 || FMT == DataFormat::Int8 ||
                                FMT == DataFormat::UInt16 || FMT == DataFormat::UInt8;
    constexpr bool is_float_fmt =
        FMT == DataFormat::Float32 || FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b;
    static_assert(
        is_int_fmt || is_float_fmt,
        "calculate_zero_comp: unsupported FMT (expected an integer format or an IEEE float width)");

    using traits = zero_comp_traits<FMT>;

    // gtez/ltez default to 1 and write 0 on their (strict-sign) predicate; the other modes default
    // to 0 and write 1 (see @ref _zero_comp_writes_zero_).
    constexpr bool writes_zero = _zero_comp_writes_zero_<COMP_MODE>();

    const typename traits::result_t default_result = writes_zero ? traits::one() : traits::zero();
    const typename traits::result_t taken_result = writes_zero ? traits::zero() : traits::one();

    // -0 as a raw word, one SFPLOADI per call; the modes that do not read it leave it at 0, which
    // emits nothing.
    const sfpi::vInt neg_zero =
        _zero_comp_tests_negative_<COMP_MODE>() ? sfpi::as<sfpi::vInt>(sfpi::vFloat(-0.0f)) : sfpi::vInt(0);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vInt bits = traits::load();

        // TODO(tt-metal#67405): gtz/lez can store sfpi::mask(bits > 0) & one once ttsim models the
        // mask form of SFPGT/SFPLE.
        traits::template store<false /*INCR*/>(default_result);
        v_if(_zero_comp_pred_<COMP_MODE>(bits, neg_zero)) { traits::store(taken_result); }
        v_endif;
    }
}

// Int32 in Dest is two's complement: mixed signs decide by sign alone (no x - s overflow), and equality is
// two orderings because sfpi's vInt == compares sign-magnitude, where INT_MIN reads as -0.

// (x OP s) ? 1 : 0 for a scalar s, OP one of unary_eq/ne/gt/ge/lt/le; x and s two's complement.
template <SfpuType COMP_MODE>
inline __attribute__((always_inline)) sfpi::vInt _int32_scalar_comp_(sfpi::vInt x, int scalar) {
    static_assert(
        COMP_MODE == SfpuType::unary_eq || COMP_MODE == SfpuType::unary_ne || COMP_MODE == SfpuType::unary_gt ||
            COMP_MODE == SfpuType::unary_ge || COMP_MODE == SfpuType::unary_lt || COMP_MODE == SfpuType::unary_le,
        "_int32_scalar_comp_: COMP_MODE must be one of the six unary_* scalar compares");
    const sfpi::vInt s = scalar;
    sfpi::vInt result = 0;
    if constexpr (COMP_MODE == SfpuType::unary_eq) {
        v_if(x <= s && x >= s) { result = 1; }
        v_endif;
    } else if constexpr (COMP_MODE == SfpuType::unary_ne) {
        v_if(x < s || x > s) { result = 1; }
        v_endif;
    } else if constexpr (COMP_MODE == SfpuType::unary_gt) {
        v_if(x >= 0 && s < 0) { result = 1; }
        v_elseif(x < 0 && s >= 0) { result = 0; }
        v_elseif(x > s) { result = 1; }
        v_endif;
    } else if constexpr (COMP_MODE == SfpuType::unary_ge) {
        v_if(x >= 0 && s < 0) { result = 1; }
        v_elseif(x < 0 && s >= 0) { result = 0; }
        v_elseif(x >= s) { result = 1; }
        v_endif;
    } else if constexpr (COMP_MODE == SfpuType::unary_lt) {
        v_if(x >= 0 && s < 0) { result = 0; }
        v_elseif(x < 0 && s >= 0) { result = 1; }
        v_elseif(x < s) { result = 1; }
        v_endif;
    } else {  // unary_le
        v_if(x >= 0 && s < 0) { result = 0; }
        v_elseif(x < 0 && s >= 0) { result = 1; }
        v_elseif(x <= s) { result = 1; }
        v_endif;
    }
    return result;
}

/**
 * @brief Int32 compare against a scalar over a tile, written as 1/0 Int32: unary_{eq,ne,gt,ge,lt,le}_tile_int32.
 *
 * @tparam APPROXIMATION_MODE: Unused; retained for dispatcher signature symmetry.
 * @tparam COMP_MODE: One of SfpuType::unary_eq/ne/gt/ge/lt/le.
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32x16 face).
 * @param scalar: Two's-complement Int32 comparand.
 */
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_unary_int(int scalar) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vInt x = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = _int32_scalar_comp_<COMP_MODE>(x, scalar);
        sfpi::dst_reg++;
    }
}

// Same kernel under the tt-llk entry-point name the gt/ge/lt/le compute API calls on every arch.
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_comp_unary_int_(int scalar) {
    calculate_comp_unary_int<APPROXIMATION_MODE, COMP_MODE, ITERATIONS>(scalar);
}

/**
 * @brief Int32 comparison-to-zero over a tile, written as 1/0 Int32: {eqz,nez,gtz,gez,ltz,lez}_tile_int32.
 *
 * The compute API calls this with the zero-compare SfpuType modes on every arch; on Quasar it is the
 * two's-complement counterpart of @ref calculate_zero_comp, whose integer path reads sign-magnitude.
 *
 * @tparam APPROXIMATION_MODE: Unused; retained for dispatcher signature symmetry.
 * @tparam COMP_MODE: One of the six comparison-to-zero SfpuType modes.
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32x16 face).
 */
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_int() {
    static_assert(
        COMP_MODE == SfpuType::equal_zero || COMP_MODE == SfpuType::not_equal_zero ||
            COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::greater_than_zero ||
            COMP_MODE == SfpuType::less_than_equal_zero || COMP_MODE == SfpuType::greater_than_equal_zero,
        "calculate_comp_int: COMP_MODE must be one of the six comparison-to-zero SfpuType modes");
    constexpr SfpuType scalar_mode = COMP_MODE == SfpuType::equal_zero             ? SfpuType::unary_eq
                                     : COMP_MODE == SfpuType::not_equal_zero       ? SfpuType::unary_ne
                                     : COMP_MODE == SfpuType::less_than_zero       ? SfpuType::unary_lt
                                     : COMP_MODE == SfpuType::greater_than_zero    ? SfpuType::unary_gt
                                     : COMP_MODE == SfpuType::less_than_equal_zero ? SfpuType::unary_le
                                                                                   : SfpuType::unary_ge;
    calculate_comp_unary_int<APPROXIMATION_MODE, scalar_mode, ITERATIONS>(0);
}

// Quasar has no UInt32 data format, so the UInt32 unary comparisons (unary_*_tile_uint32) reject the call here
// for the same reason as the Int32 ones above.
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_unary_uint32([[maybe_unused]] const std::uint32_t scalar) {
    LLK_ASSERT(false, "UInt32 unary comparisons (unary_*_tile_uint32) are not supported on Quasar");
}

}  // namespace sfpu
}  // namespace ckernel
