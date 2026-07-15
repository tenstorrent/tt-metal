// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_defs.h"
#include "sfpi.h"
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
    static_assert(ITERATIONS % 2 == 0, "calculate_addcmul() processes dest rows in interleaved pairs.");

    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    const sfpi::vFloat value_float = Converter::as_float(value);

    const std::uint32_t off_in0 = dst_index_in0 * dst_tile_size_sfpi;
    const std::uint32_t off_in1 = dst_index_in1 * dst_tile_size_sfpi;
    const std::uint32_t off_in2 = dst_index_in2 * dst_tile_size_sfpi;
    const std::uint32_t off_out = dst_index_out * dst_tile_size_sfpi;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        // Two independent rows (A at +0, B at +1) so the scheduler can overlap
        // their dependent MUL->MAD->round chains and hide the 2-cycle latency.
        sfpi::vFloat a_in0 = sfpi::dst_reg[off_in0];
        sfpi::vFloat a_in1 = sfpi::dst_reg[off_in1];
        sfpi::vFloat a_in2 = sfpi::dst_reg[off_in2];
        sfpi::vFloat b_in0 = sfpi::dst_reg[off_in0 + 1];
        sfpi::vFloat b_in1 = sfpi::dst_reg[off_in1 + 1];
        sfpi::vFloat b_in2 = sfpi::dst_reg[off_in2 + 1];

        // Sequence both products before both MADs so the scheduler can issue
        // MUL_a, MUL_b back-to-back (each hiding the other's latency), then
        // MAD_a, MAD_b, keeping the pipeline NOP-free.
        sfpi::vFloat a_prod = value_float * a_in1;
        sfpi::vFloat b_prod = value_float * b_in1;
        sfpi::vFloat a_res = a_prod * a_in2 + a_in0;
        sfpi::vFloat b_res = b_prod * b_in2 + b_in0;

        if constexpr (!is_fp32_dest_acc_en) {
            sfpi::dst_reg[off_out] = sfpi::convert<sfpi::vFloat16b>(a_res, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[off_out + 1] = sfpi::convert<sfpi::vFloat16b>(b_res, sfpi::RoundMode::Nearest);
        } else {
            sfpi::dst_reg[off_out] = a_res;
            sfpi::dst_reg[off_out + 1] = b_res;
        }
        sfpi::dst_reg += 2;
    }
}
}  // namespace ckernel::sfpu
