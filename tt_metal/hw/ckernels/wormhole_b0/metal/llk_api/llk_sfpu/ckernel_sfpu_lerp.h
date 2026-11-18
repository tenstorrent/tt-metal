// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "llk_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void lerp_init() {
    // calculate_lerp's second store walks dest through physical addr_mod slot 6, advancing by
    // the two rows the iteration just wrote (one sfpi row is two dest counter steps), so the
    // loop body needs no separate increment. The ternary SFPU dispatch sets addr_mod_base, so
    // the store names ADDR_MOD_2, which maps to physical slot 6.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_lerp(
    const std::uint32_t dst_index_in0,  // input (start)
    const std::uint32_t dst_index_in1,  // end
    const std::uint32_t dst_index_in2,  // weight
    const std::uint32_t dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    static_assert(ITERATIONS % 2 == 0, "two rows per iteration");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    const std::uint32_t off_in0 = dst_index_in0 * dst_tile_size_sfpi;
    const std::uint32_t off_in1 = dst_index_in1 * dst_tile_size_sfpi;
    const std::uint32_t off_in2 = dst_index_in2 * dst_tile_size_sfpi;
    const std::uint32_t off_out = dst_index_out * dst_tile_size_sfpi;
    // RNE constants bound once per face, not re-materialised on every row.
    // Unassigned unless round_to_bf16_rne, which also gates their use.
    constexpr bool round_to_bf16_rne = !is_fp32_dest_acc_en;
    sfpi::vUInt rne_bias;
    sfpi::vUInt rne_mask;
    if constexpr (round_to_bf16_rne) {
        rne_bias = kBf16RneBias;
        rne_mask = kBf16RneMask;
    }
    // lerp: out = (1 - weight) * input + weight * end
    // Convex form: no (end - input) difference, so finite inputs cannot overflow for weight in [0, 1],
    // and weight == 0 / weight == 1 return input / end bit-exact.
    //
    // Two rows per iteration with both rows' first products formed before either row's final SFPMAD,
    // so the two rows' dependent SFPMADs can interleave. Each row is ordered to need three LREGs
    // (the product and 1 - weight can replace end and weight), leaving room for the RNE constants.
    // The second store advances dest (see lerp_init).
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat a_in1 = sfpi::dst_reg[off_in1];
        sfpi::vFloat a_in2 = sfpi::dst_reg[off_in2];
        sfpi::vFloat a_prod = a_in2 * a_in1;
        sfpi::vFloat a_rem = 1.0f - a_in2;
        sfpi::vFloat b_in1 = sfpi::dst_reg[off_in1 + 1];
        sfpi::vFloat b_in2 = sfpi::dst_reg[off_in2 + 1];
        sfpi::vFloat b_prod = b_in2 * b_in1;
        sfpi::vFloat b_rem = 1.0f - b_in2;
        sfpi::vFloat a_in0 = sfpi::dst_reg[off_in0];
        sfpi::vFloat b_in0 = sfpi::dst_reg[off_in0 + 1];
        sfpi::vFloat a_res = a_rem * a_in0 + a_prod;
        sfpi::vFloat b_res = b_rem * b_in0 + b_prod;
        if constexpr (round_to_bf16_rne) {
            a_res = float32_to_bf16_rne(a_res, rne_bias, rne_mask);
            b_res = float32_to_bf16_rne(b_res, rne_bias, rne_mask);
        }
        sfpi::dst_reg[off_out] = a_res;
        sfpi::dst_reg[off_out + 1].mode(ADDR_MOD_2) = b_res;
    }
}

}  // namespace ckernel::sfpu
