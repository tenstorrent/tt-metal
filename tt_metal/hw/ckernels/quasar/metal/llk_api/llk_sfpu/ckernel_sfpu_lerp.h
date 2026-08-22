// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_binary.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

/**
 * @brief Per-lane linear interpolation: @c out = start + weight * (end - start).
 *
 * On a 16-bit DEST the fp32 result is narrowed with @ref float32_to_bf16_rne before the
 * store, because SFPSTORE truncates instead of rounding. That narrowing is bf16-specific,
 * so the static_assert admits Float32 and Float16_b only.
 *
 * @tparam APPROXIMATION_MODE: unused, preserved to match the BH metal signature
 * @tparam is_fp32_dest_acc_en: set when DEST is 32-bit; skips the bf16 narrowing
 * @tparam data_format: DEST register format, values = <Float32/Float16_b>
 * @tparam ITERATIONS: number of sfpi row-pairs to process (one call per face)
 * @tparam TILE_SHAPE: destination tile shape used to calculate operand offsets
 * @param dst_index_in0: DEST tile index holding the interpolation start point
 * @param dst_index_in1: DEST tile index holding the interpolation end point
 * @param dst_index_in2: DEST tile index holding the per-lane weight
 * @param dst_index_out: DEST tile index that receives the result
 * @note Run @c SFPU_TERNARY_INIT(lerp) before this function, and drive it through
 *       @c SFPU_TERNARY_CALL so the per-face loop and section base setup are owned by
 *       @ref _llk_math_eltwise_ternary_sfpu_params_.
 */
template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    DataFormat data_format,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_lerp(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2,
    const std::uint32_t dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat start = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat end = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat weight = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];
        sfpi::vFloat result = start + weight * (end - start);
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
