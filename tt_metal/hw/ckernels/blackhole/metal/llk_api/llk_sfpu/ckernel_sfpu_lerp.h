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
    // RNE constants, loaded once per call. Unassigned unless rounding, which also gates their use.
    sfpi::vUInt rne_bias;
    sfpi::vUInt rne_one;
    if constexpr (!is_fp32_dest_acc_en) {
        rne_bias = 0x7fffU;
        rne_one = 1;
    }
    // lerp: out = input + weight * (end - input); lanes where end - input is not finite use
    // (input - weight * input) + weight * end. No instruction reads the SFPMAD right before it.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat in0 = sfpi::dst_reg[off_in0];
        const sfpi::vFloat in1 = sfpi::dst_reg[off_in1];
        const sfpi::vFloat diff = in1 - in0;
        const sfpi::vFloat in2 = sfpi::dst_reg[off_in2];
        const sfpi::vFloat rem = in0 - in2 * in0;
        const sfpi::vFloat diff_nan = diff - diff;  // 0 for finite diff, NaN otherwise
        sfpi::vFloat res = in2 * in1 + rem;
        v_if(diff_nan == 0.0f) { res = in0 + in2 * diff; }
        v_endif;
        if constexpr (!is_fp32_dest_acc_en) {
            res = _lerp_bf16_rne_for_16b_store_(res, rne_bias, rne_one);
        }
        sfpi::dst_reg[off_out].mode(ADDR_MOD_5) = res;
    }
}

}  // namespace ckernel::sfpu
