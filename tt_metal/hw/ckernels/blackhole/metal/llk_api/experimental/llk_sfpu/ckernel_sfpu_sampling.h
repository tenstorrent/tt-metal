// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// One 32x32 DEST tile spans 32 sfpi dst_reg slots.
constexpr std::uint32_t dst_tile_size_sfpi = 32;

// The "first column" helpers walk 4 SFPU slots at stride 2, covering DEST rows 0-15 of face 0.
constexpr int ITERATIONS_FIRST_COLUMN = 4;

// Slot advance per iteration: +1 would land on the odd-column slot of the same rows, which the
// column-0 callers never read, so the walk steps over it.
constexpr int FIRST_COLUMN_SLOT_STRIDE = 2;

// SfpuType has no generic float add/sub/mul (only the *_int32/*_uint32 variants), so the column-0
// binary helpers dispatch on this local tag instead.
enum class SamplingBinaryOp { add, sub, mul };

/**
 * @brief Compute 1/in for one SFPU slot, selecting the legacy-compatible or the sign-correct variant.
 *
 * Returns by value on purpose. The caller copy-initializes from this prvalue, which is the same
 * construct as the original single-expression init, so C++17 guaranteed elision keeps `out` directly
 * initialized by the reciprocal call. Selecting the variant with a local `vFloat out;` + assignment
 * would instead route through vVal::operator= (__builtin_rvtt_sfpassign_lv) and change the emitted
 * SFPU sequence on the legacy path -- which must stay bit-identical for blaze.
 *
 * @tparam legacy_compat: Use blaze's bit-identical reciprocal, values = <true/false>
 * @param in: Value to invert.
 * @note Callers must pass in > 0. The two branches disagree on sign: _reciprocal_compat_ opens with
 *       setsgn(in, 1) and so returns the magnitude |1/in| (legacy_compat = true gives +0.25 for -4.0),
 *       while legacy_compat = false is sign-correct. Every caller today feeds a softmax partition
 *       function or a cumulative probability, both strictly positive. The legacy path must stay
 *       bit-identical for blaze, so the divergence is documented rather than fixed.
 * @note Call @ref sampling_recip_init with the matching legacy_compat before this function; the
 *       legacy_compat = false path reads vConstFloatPrgm0 as its Newton-Raphson constant.
 */
template <bool legacy_compat, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat sampling_recip_value(sfpi::vFloat in) {
    if constexpr (legacy_compat) {
        return ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);
    } else if constexpr (APPROX) {
        return ckernel::sfpu::sfpu_reciprocal_iter<0>(in);
    } else if constexpr (is_fp32_dest_acc_en) {
        return ckernel::sfpu::sfpu_reciprocal_iter<2>(in);
    } else {
        return ckernel::sfpu::sfpu_reciprocal_iter<1>(in);
    }
}

/**
 * @brief Program the SFPU constants the sampling reciprocal needs.
 *
 * @tparam legacy_compat: Must match the calculate_sampling_recip_scalar call it precedes.
 * @note Call before @ref calculate_sampling_recip_scalar. The legacy_compat = false path calls
 *       sfpu_reciprocal_iter, which reads sfpi::vConstFloatPrgm0 (LREG12) as its Newton-Raphson
 *       constant; only sfpu_reciprocal_init<false> writes the 2.0f it expects. recip_init /
 *       recip_tile_init do not. Without this, a kernel that ran e.g. exp_tile_init earlier leaves
 *       1.442695f there and every Newton step is silently wrong -- no assert, no build error.
 *       The legacy_compat = true path carries its own constants and needs no setup, so this is a
 *       no-op there.
 */
template <bool legacy_compat = true>
inline void sampling_recip_init() {
    if constexpr (!legacy_compat) {
        sfpu_reciprocal_init<APPROX>();
    }
}

/**
 * @brief Replace one SFPU slot (DEST rows 0-3 of face 0) with its reciprocal, in place.
 *
 * The public entry point for the sampling reciprocal; @ref sampling_recip_value is the leaf that
 * picks the variant. On a 16-bit DEST outside APPROX it converts to bf16 with round-to-nearest
 * first, so the store does not truncate.
 *
 * @tparam legacy_compat: Use blaze's bit-identical reciprocal, values = <true/false>
 * @note Callers must pass values > 0: with legacy_compat = true the result is the magnitude
 *       |1/in| rather than 1/in -- see @ref sampling_recip_value for why that divergence stands.
 * @note Call @ref sampling_recip_init with the same legacy_compat before this function; the
 *       legacy_compat = false path reads vConstFloatPrgm0 as its Newton-Raphson constant.
 */
template <bool legacy_compat, bool is_fp32_dest_acc_en>
inline void calculate_sampling_recip_scalar() {
    sfpi::vFloat in = sfpi::dst_reg[0];
    sfpi::vFloat out = sampling_recip_value<legacy_compat, is_fp32_dest_acc_en>(in);
    if constexpr (!(is_fp32_dest_acc_en || APPROX)) {
        out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[0] = out;
}

/**
 * @brief Clamp one SFPU slot (DEST rows 0-3 of face 0) to an upper bound, in place.
 *
 * Lanes at or below the bound are left untouched, so a value that is already in range keeps its
 * exact bits.
 *
 * @param param: Upper bound as a raw fp32 bit pattern (decoded by Converter::as_float).
 */
inline void calculate_sampling_clamp_max_scalar(const std::uint32_t param) {
    const sfpi::vFloat max_val = ckernel::sfpu::Converter::as_float(param);
    sfpi::vFloat in = sfpi::dst_reg[0];
    v_if(in > max_val) { sfpi::dst_reg[0] = max_val; }
    v_endif;
}

/**
 * @brief Apply a column-0 elementwise float comparison across DEST rows 0-15 of face 0.
 *
 * Writes 1.0f where the comparison holds and 0.0f where it does not, so the result is a keep-mask
 * the binary helpers can multiply through. Used by the top-P mask (exclusive_CDF < top_p).
 *
 * @tparam OP: Comparison to apply, values = <le/lt/ge>
 * @param dst_index_in0: DEST tile index of the left operand.
 * @param dst_index_in1: DEST tile index of the right operand.
 * @param dst_index_out: DEST tile index the mask is written to.
 */
template <SfpuType OP>
inline void calculate_sampling_binary_comp_first_column(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        OP == SfpuType::le || OP == SfpuType::lt || OP == SfpuType::ge,
        "sampling_binary_comp_first_column supports le/lt/ge only");

    for (int d = 0; d < ITERATIONS_FIRST_COLUMN; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = sfpi::vConst0;

        if constexpr (OP == SfpuType::le) {
            v_if(in0 <= in1) { result = sfpi::vConst1; }
            v_endif;
        } else if constexpr (OP == SfpuType::lt) {
            v_if(in0 < in1) { result = sfpi::vConst1; }
            v_endif;
        } else {
            v_if(in0 >= in1) { result = sfpi::vConst1; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg += FIRST_COLUMN_SLOT_STRIDE;
    }
}

/**
 * @brief Scale column 0 of DEST rows 0-15 of face 0 by a scalar, in place.
 *
 * @param param: Multiplier as a raw fp32 bit pattern (decoded by Converter::as_float).
 */
inline void calculate_sampling_mul_unary_scalar_first_column(const std::uint32_t param) {
    const sfpi::vFloat parameter = ckernel::sfpu::Converter::as_float(param);

    for (int d = 0; d < ITERATIONS_FIRST_COLUMN; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = val * parameter;
        sfpi::dst_reg += FIRST_COLUMN_SLOT_STRIDE;
    }
}

/**
 * @brief Apply a column-0 elementwise float binary op across DEST rows 0-15 of face 0.
 *
 * Used by the top-P mask: exclusive_CDF = CDF - probs (sub), then masked = probs * keep (mul).
 *
 * @tparam OP: Operation to apply, values = <add/sub/mul>
 * @param dst_index_in0: DEST tile index of the first operand.
 * @param dst_index_in1: DEST tile index of the second operand.
 * @param dst_index_out: DEST tile index the result is written to.
 */
template <SamplingBinaryOp OP>
inline void calculate_sampling_binary_first_column(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    for (int d = 0; d < ITERATIONS_FIRST_COLUMN; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        if constexpr (OP == SamplingBinaryOp::add) {
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in0 + in1;
        } else if constexpr (OP == SamplingBinaryOp::sub) {
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in0 - in1;
        } else {
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in0 * in1;
        }

        sfpi::dst_reg += FIRST_COLUMN_SLOT_STRIDE;
    }
}

}  // namespace ckernel::sfpu
