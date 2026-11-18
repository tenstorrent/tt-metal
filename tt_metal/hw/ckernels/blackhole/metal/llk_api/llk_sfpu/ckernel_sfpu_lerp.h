// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void lerp_init() {
    // calculate_lerp's store advances dest by one sfpi row (two dest counter steps).
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_5);
}

// fp32 -> bf16 round-to-nearest-even for a value stored straight to a 16-bit Dest: the low 16 bits
// are left as is, so the result must not be read back or kept in an fp32 Dest.
sfpi_inline sfpi::vFloat _lerp_bf16_rne_for_16b_store_(
    const sfpi::vFloat in, const sfpi::vUInt bias, const sfpi::vUInt one) {
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);
    const sfpi::vUInt lsb = (bits >> 16) & one;
    bits = bits + bias + lsb;
    return sfpi::as<sfpi::vFloat>(bits);
}

// Requires lerp_init() (programs ADDR_MOD_5).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_lerp(
    const uint dst_index_in0,  // input (start)
    const uint dst_index_in1,  // end
    const uint dst_index_in2,  // weight
    const uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    const auto off_in0 = dst_index_in0 * dst_tile_size_sfpi;
    const auto off_in1 = dst_index_in1 * dst_tile_size_sfpi;
    const auto off_in2 = dst_index_in2 * dst_tile_size_sfpi;
    const auto off_out = dst_index_out * dst_tile_size_sfpi;
    const sfpi::vFloat half = 0.5f;
    // RNE constants, loaded once per call. Unassigned unless rounding, which also gates their use.
    sfpi::vUInt rne_bias;
    sfpi::vUInt rne_one;
    if constexpr (!is_fp32_dest_acc_en) {
        rne_bias = 0x7fffU;
        rne_one = 1;
    }
    auto finish = [&](const sfpi::vFloat half_res) {
        sfpi::vFloat res = half_res * 2.0f;
        if constexpr (!is_fp32_dest_acc_en) {
            res = _lerp_bf16_rne_for_16b_store_(res, rne_bias, rne_one);
        }
        return res;
    };
    // lerp: out = input + weight * (end - input), evaluated at half scale as
    // 2 * (input / 2 + weight * (end / 2 - input / 2)): for weight in [0, 1] no intermediate can
    // overflow, and the power-of-two scalings are exact above the bottom binade, so the result is
    // the same as the direct form wherever that one is finite.
    //
    // One row per iteration, rotated: row d is formed while row d - 1 is scaled back and stored,
    // so no instruction reads the result of the SFPMAD right before it.
    sfpi::vFloat half_res;
    {
        const sfpi::vFloat in0 = sfpi::dst_reg[off_in0];
        const sfpi::vFloat in0_half = in0 * 0.5f;
        const sfpi::vFloat in1 = sfpi::dst_reg[off_in1];
        const sfpi::vFloat in2 = sfpi::dst_reg[off_in2];
        const sfpi::vFloat diff_half = in1 * half - in0_half;
        half_res = in2 * diff_half + in0_half;
    }
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        const sfpi::vFloat in0 = sfpi::dst_reg[off_in0 + 1];
        const sfpi::vFloat in0_half = in0 * 0.5f;
        const sfpi::vFloat in1 = sfpi::dst_reg[off_in1 + 1];
        const sfpi::vFloat prev = finish(half_res);
        const sfpi::vFloat in2 = sfpi::dst_reg[off_in2 + 1];
        const sfpi::vFloat diff_half = in1 * half - in0_half;
        sfpi::dst_reg[off_out].mode(ADDR_MOD_5) = prev;
        half_res = in2 * diff_half + in0_half;
    }
    sfpi::dst_reg[off_out].mode(ADDR_MOD_5) = finish(half_res);
}

}  // namespace ckernel::sfpu
