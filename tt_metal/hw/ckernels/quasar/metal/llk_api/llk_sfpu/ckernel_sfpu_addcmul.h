// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Per-lane fused multiply-add: @c out = a + value * b * c.
 *
 * Each lane costs one SFPMUL (@c value * b) feeding one SFPMAD (@c (value*b) * c + a).
 *
 * @tparam APPROXIMATION_MODE: Unused for @c addcmul; kept for API parity with other
 *         SFPU kernels.
 * @tparam is_fp32_dest_acc_en: Whether Dest is 32-bit. When false the fp32 result is
 *         rounded down to the Dest width before the store.
 * @tparam DST_FORMAT: Dest data format, values = <Float32/Float16_b/Float16>. Selects
 *         the 16-bit rounding target; ignored when @c is_fp32_dest_acc_en is true.
 * @tparam ITERATIONS: Inner SFPU row-pair count per face. Defaults to 8 for the
 *         standard 16-row face. The outer per-face loop and section base setup are
 *         owned by @c _llk_math_eltwise_ternary_sfpu_params_.
 * @tparam TILE_SHAPE: Destination tile shape used to calculate operand offsets.
 *
 * @param dst_index_in0: DEST tile index holding @c a, the addend.
 * @param dst_index_in1: DEST tile index holding @c b, the first multiplicand.
 * @param dst_index_in2: DEST tile index holding @c c, the second multiplicand.
 * @param dst_index_out: DEST tile index that receives the result; may alias an input.
 * @param value: Scalar multiplier as a raw fp32 bit pattern, broadcast to every lane.
 * @note Call @ref _llk_math_eltwise_ternary_sfpu_init_ with @c SfpuType::addcmul before
 *       this function.
 */
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    DataFormat DST_FORMAT,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_addcmul(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2,
    const std::uint32_t dst_index_out,
    const std::uint32_t value) {
    static_assert(
        DST_FORMAT == DataFormat::Float32 || DST_FORMAT == DataFormat::Float16_b || DST_FORMAT == DataFormat::Float16,
        "calculate_addcmul() is lanewise FP32 math; only Float32, Float16_b and Float16 Dest formats are supported.");

    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);

    const std::uint32_t off_in0 = dst_index_in0 * dst_tile_size_sfpi;
    const std::uint32_t off_in1 = dst_index_in1 * dst_tile_size_sfpi;
    const std::uint32_t off_in2 = dst_index_in2 * dst_tile_size_sfpi;
    const std::uint32_t off_out = dst_index_out * dst_tile_size_sfpi;

    // Loop-invariant, so it is decoded once and stays in one LREG.
    const sfpi::vFloat value_float = Converter::as_float(value);

    // Full unroll hands the scheduler eight independent MUL->MAD chains to interleave,
    // covering the MAD latency shadow without hand-placed NOPs.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[off_in0];
        sfpi::vFloat in1 = sfpi::dst_reg[off_in1];
        sfpi::vFloat in2 = sfpi::dst_reg[off_in2];

        sfpi::vFloat prod = value_float * in1;
        sfpi::vFloat result = prod * in2 + in0;

        if constexpr (!is_fp32_dest_acc_en) {
            // SFPSTORE truncates, so round to nearest before narrowing to a 16-bit Dest.
            if constexpr (DST_FORMAT == DataFormat::Float16) {
                result = sfpi::convert<sfpi::vFloat16a>(result, sfpi::RoundMode::Nearest);
            } else {
                result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            }
        }

        sfpi::dst_reg[off_out] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
