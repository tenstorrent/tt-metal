// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_addcmul(
    const std::uint32_t dst_index_in0,  // input_a
    const std::uint32_t dst_index_in1,  // input_b
    const std::uint32_t dst_index_in2,  // input_c
    const std::uint32_t dst_index_out,  // output
    const std::uint32_t value) {        // scalar value to multiply with input_b
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Bfp8_b,
        "Unsupported data format for calculate_addcmul(). Only Float32, Float16_b (BFloat16), and Bfp8_b (BFloat8B) "
        "are allowed.");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    // addcmul = input_a + ((value * input_b) * input_c)
    const sfpi::vFloat value_float = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat in2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];
        sfpi::vFloat result = in0 + (value_float * in1) * in2;
        if constexpr (!is_fp32_dest_acc_en) {
            // Round-to-nearest-even fp32->bf16, matching the previous
            // TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN) implementation.
            result = float32_to_bf16_rne(result);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
