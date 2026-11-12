// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    const uint dst_index_in0,  // input (start)
    const uint dst_index_in1,  // end
    const uint dst_index_in2,  // weight
    const uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    static_assert(ITERATIONS % 2 == 0, "two rows per iteration");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    const uint off_in0 = dst_index_in0 * dst_tile_size_sfpi;
    const uint off_in1 = dst_index_in1 * dst_tile_size_sfpi;
    const uint off_in2 = dst_index_in2 * dst_tile_size_sfpi;
    const uint off_out = dst_index_out * dst_tile_size_sfpi;
    // lerp: out = input + weight * (end - input)
    //
    // Two rows per iteration, both differences computed before either row's second SFPMAD: the
    // compiler then interleaves the two rows' dependent SFPMADs, and the SFPNOP each of a one-row
    // body's two dependent SFPMADs costs (write latency that nothing else can fill) disappears.
    // Identical instructions on identical operands, only the issue order changes, so the results
    // are bit-identical to the one-row body. The second store advances dest (see lerp_init).
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat a_in0 = sfpi::dst_reg[off_in0];
        sfpi::vFloat a_in1 = sfpi::dst_reg[off_in1];
        sfpi::vFloat a_in2 = sfpi::dst_reg[off_in2];
        sfpi::vFloat b_in0 = sfpi::dst_reg[off_in0 + 1];
        sfpi::vFloat b_in1 = sfpi::dst_reg[off_in1 + 1];
        sfpi::vFloat b_in2 = sfpi::dst_reg[off_in2 + 1];
        sfpi::vFloat a_diff = a_in1 - a_in0;
        sfpi::vFloat b_diff = b_in1 - b_in0;
        sfpi::vFloat a_res = a_in0 + a_in2 * a_diff;
        sfpi::vFloat b_res = b_in0 + b_in2 * b_diff;
        if constexpr (!is_fp32_dest_acc_en) {
            a_res = float32_to_bf16_rne(a_res);
            b_res = float32_to_bf16_rne(b_res);
        }
        sfpi::dst_reg[off_out] = a_res;
        sfpi::dst_reg[off_out + 1].mode(ADDR_MOD_2) = b_res;
    }
}

}  // namespace ckernel::sfpu
