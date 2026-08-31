// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_trisc_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Compute atan2(y, x) for one SFPU vector, resolved into the correct quadrant.
 *
 * Reduces to a minimax polynomial in a = min(|x|,|y|) / max(|x|,|y|) on [0, 1], then folds
 * the quadrant back in from the operand signs. IEEE corner cases (|x| == |y|, both zero,
 * negative-signed x including -0.0, and NaN in either operand) are handled explicitly.
 *
 * Dest width and APPROXIMATION_MODE control independent pieces:
 *   - Polynomial: 7-term when @p is_fp32_dest_acc_en, otherwise 3-term. Unaffected by
 *     APPROXIMATION_MODE.
 *   - Store rounding: round-to-nearest bf16 only when Dest is 16-bit. An FP32 Dest
 *     keeps the full-width result even when APPROXIMATION_MODE is true.
 *   - Reciprocal for @c a: LUT seed only (0 Newton steps) when APPROXIMATION_MODE is
 *     true or Dest is 16-bit; LUT seed plus two Newton-Raphson steps only for FP32 Dest
 *     with APPROXIMATION_MODE false.
 *
 * @tparam APPROXIMATION_MODE Selects the reciprocal used to form @c a. Does not select
 *         the polynomial or the bf16 store round — those follow Dest width only.
 * @tparam is_fp32_dest_acc_en 32-bit Dest: 7-term polynomial, no bf16 store round, and
 *         (when APPROXIMATION_MODE is false) a Newton-refined reciprocal.
 * @param y: Numerator operand; its sign is copied onto the result.
 * @param x: Denominator operand; a negative sign reflects the result into the
 *        second/third quadrant.
 * @note Call @ref calculate_sfpu_atan2_init with matching template args first. Init
 *       programs the reciprocal path that matches the combination above (Newton
 *       constant only for the FP32 precise reciprocal).
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_atan2_(sfpi::vFloat y, sfpi::vFloat x) {
    // Reciprocal only: LUT seed is bf16-accurate; two Newton steps are used solely on
    // the precise FP32-Dest path. Polynomial degree and store rounding stay on Dest width.
    constexpr int recip_iterations = (APPROXIMATION_MODE || !is_fp32_dest_acc_en) ? 0 : 2;

    sfpi::vFloat r;
    sfpi::vFloat q;
    sfpi::vFloat s;

    // setsgn (not abs) clears the sign bit, so a ±NaN operand still reaches max — the NaN
    // special case at the end depends on that.
    auto [min, max] = sfpi::min_max(sfpi::setsgn(x, 0 /* sgn */), sfpi::setsgn(y, 0 /* sgn */));

    // a = min(|x|, |y|) / max(|x|, |y|), i.e. a is on [0, 1].
    sfpi::vFloat a = min * _sfpu_reciprocal_<recip_iterations>(max);

    // Minimax atan(a): Dest width only — 7-term on FP32 Dest, 3-term otherwise.
    // APPROXIMATION_MODE does not select this path.

    if constexpr (is_fp32_dest_acc_en) {
        q = 0x1.01cp-8f;
        s = a * a;
        sfpi::vFloat c6 = -0x1.4bcp-6f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c6.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c5 = 0x1.93p-5f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c5.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c4 = -0x1.48cp-4f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c4.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c3 = 0x1.bd4p-4f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c3.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c2 = -0x1.24p-3f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c2.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c1 = 0x1.99938ap-3f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c0 = -0x1.555558p-2f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    } else {
        q = -0x1.de8p-5f;
        s = a * a;
        sfpi::vFloat c1 = 0x1.668p-3f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c0 = -0x1.54p-2f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    sfpi::vFloat half_pi = 0x1.921fb6p+0f;
    sfpi::vFloat t = q * s;
    sfpi::vFloat x_abs = sfpi::setsgn(x, 0 /* sgn */);
    r = t * a + a;

    // Special cases:

    v_if(sfpi::as<sfpi::vInt>(min) >= sfpi::as<sfpi::vInt>(x_abs)) {
        // if |y| ≥ |x| then r = π/2 - r
        r = half_pi - r;
        v_if(sfpi::as<sfpi::vInt>(min) >= sfpi::as<sfpi::vInt>(max)) {
            // if |x| = |y| (including both infinite), then r = π/4
            r = sfpi::addexp(half_pi, -1 /* exp */);
            v_if(min == 0.0f) {
                // if both zero, then r = ±0
                // SFPI note: the later v_if(x < 0.0f) behaves like a signbit check, so r=-0
                // is handled by that path.
                r = 0.0f;
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;

    // if sign of x is negative (including x=-0), then r = π - r
    v_if(x < 0.0f) {
        sfpi::vFloat pi = sfpi::addexp(half_pi, 1 /* exp */);
        r = pi - r;
    }
    v_endif;

    // 16-bit Dest only. FP32 Dest keeps the full-width result, including APPROXIMATION_MODE.
    if constexpr (!is_fp32_dest_acc_en) {
        r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
    }

    r = sfpi::copysgn(r, y);

    // As integers, every NaN bit pattern sorts above +inf; min_max propagated a NaN operand
    // into max (see above), so this one compare catches NaN in either operand.
    sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
    v_if(sfpi::as<sfpi::vInt>(infinity) < sfpi::as<sfpi::vInt>(max)) { r = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;

    return r;
}

/**
 * @brief Apply atan2 over two Dest operand tiles into a result tile.
 *
 * @tparam APPROXIMATION_MODE Forwarded to @ref _sfpu_atan2_; selects only the
 *         reciprocal (LUT-only vs Newton), not the polynomial or store rounding.
 * @tparam ITERATIONS: Number of SFPU loop iterations over the Dest tile.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode; must match the init.
 *         Selects the 7-term vs 3-term polynomial and whether the result is
 *         rounded to bf16.
 * @tparam TILE_SHAPE: Destination tile shape used to derive the operand stride.
 * @param dst_index_in0: Dest tile index of the y operand.
 * @param dst_index_in1: Dest tile index of the x operand.
 * @param dst_index_out: Dest tile index the result is written to; may alias either operand.
 * @note Call @ref calculate_sfpu_atan2_init with the same APPROXIMATION_MODE /
 *       is_fp32_dest_acc_en before this function.
 */
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool is_fp32_dest_acc_en = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_sfpu_atan2(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_atan2_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

/**
 * @brief Initialisation hook for atan2; programs the reciprocal used to form @c a.
 *
 * @tparam APPROXIMATION_MODE Must match @ref calculate_sfpu_atan2. true (or a 16-bit
 *         Dest) selects the LUT-only reciprocal init; Newton-Raphson constants are
 *         programmed only for FP32 Dest with APPROXIMATION_MODE false.
 * @tparam is_fp32_dest_acc_en Must match @ref calculate_sfpu_atan2.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_atan2_init() {
    _init_reciprocal_<APPROXIMATION_MODE || !is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
