// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "sfpi.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_addcdiv(
    const uint dst_index_in0,  // input_a
    const uint dst_index_in1,  // input_b
    const uint dst_index_in2,  // input_c
    const uint dst_index_out,  // output
    const uint value) {        // scalar value to multiply with input_b
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_addcdiv(). Supported data formats are: Float32, Float16_b");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    const sfpi::vFloat value_float = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat in2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];
        sfpi::vFloat result = in0 + (in1 * value_float * _sfpu_reciprocal_<2>(in2));
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_addcdiv() {
    // calculate_addcdiv always runs the Newton-refined _sfpu_reciprocal_<2>, so the Newton
    // constant is needed in every mode, not just the non-approximate one.
    _init_reciprocal_</*APPROXIMATION_MODE=*/false>();
}

// Op class for elementwise addcdiv: out = in0 + value * in1 / in2.
// Same name and leading template parameters as on Wormhole/Blackhole.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en = false,
    DataFormat data_format = DataFormat::Invalid,
    int ITERATIONS = 8,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct Addcdiv : SfpuTernaryOp<Addcdiv<APPROXIMATION_MODE, is_fp32_dest_acc_en, data_format, ITERATIONS, SLOT>, SLOT> {
    static_assert(SLOT == trisc::DstTileShape::Tile32x32, "the kernel assumes 32x32 Dest tiles");
    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_in2,
        const std::uint32_t dst_index_out,
        const std::uint32_t value) {
        calculate_addcdiv<APPROXIMATION_MODE, is_fp32_dest_acc_en, data_format, ITERATIONS>(
            dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, value);
    }

    static inline __attribute__((always_inline)) void init_op() { init_addcdiv<APPROXIMATION_MODE>(); }
};

}  // namespace ckernel::sfpu
