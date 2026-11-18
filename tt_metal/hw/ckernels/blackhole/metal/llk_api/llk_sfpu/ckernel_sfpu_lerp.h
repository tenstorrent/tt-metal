// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel_addrmod.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void lerp_init() {
    // calculate_lerp's second store advances dest by the two rows an iteration writes
    // (one sfpi row is two dest counter steps), so the loop body needs no separate increment.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}

// fp32 -> bf16 round-to-nearest-even ("bits + 0x7fff + lsb") for a value that goes straight to a
// 16-bit Dest store. The low 16 bits are not cleared: the 16-bit store keeps only the high half,
// so the result must not stay in an fp32 Dest or be read back before the store.
// bias (0x7fff) and one (1) are bound by the caller, which keeps them in LREGs across rows.
sfpi_inline sfpi::vFloat _lerp_bf16_rne_for_16b_store_(
    const sfpi::vFloat in, const sfpi::vUInt bias, const sfpi::vUInt one) {
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);
    const sfpi::vUInt lsb = (bits >> 16) & one;
    bits = bits + bias + lsb;
    return sfpi::as<sfpi::vFloat>(bits);
}

// Requires lerp_init() (programs ADDR_MOD_6).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_lerp(
    const std::uint32_t dst_index_in0,  // input (start)
    const std::uint32_t dst_index_in1,  // end
    const std::uint32_t dst_index_in2,  // weight
    const std::uint32_t dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");
    static_assert(ITERATIONS % 2 == 0, "calculate_lerp() processes dest rows in pairs.");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
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
    // lerp: out = (input - weight * input) + weight * end
    // Convex form: no (end - input) difference, so finite inputs cannot overflow for weight in [0, 1],
    // and weight == 0 / weight == 1 return input / end bit-exact. (1 - weight) * input is formed by
    // one SFPMAD as input - weight * input, so 1 - weight is never rounded on its own.
    //
    // Two rows (a, b) per iteration, ordered so that no instruction reads the result of the SFPMAD
    // right before it. Each row needs three LREGs (rem replaces input), which leaves two for the
    // RNE constants.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        const sfpi::vFloat a_in0 = sfpi::dst_reg[off_in0];
        const sfpi::vFloat a_in2 = sfpi::dst_reg[off_in2];
        const sfpi::vFloat a_rem = a_in0 - a_in2 * a_in0;
        const sfpi::vFloat b_in0 = sfpi::dst_reg[off_in0 + 1];
        const sfpi::vFloat b_in2 = sfpi::dst_reg[off_in2 + 1];
        const sfpi::vFloat b_rem = b_in0 - b_in2 * b_in0;
        const sfpi::vFloat a_in1 = sfpi::dst_reg[off_in1];
        const sfpi::vFloat b_in1 = sfpi::dst_reg[off_in1 + 1];
        sfpi::vFloat a_res = a_in2 * a_in1 + a_rem;
        sfpi::vFloat b_res = b_in2 * b_in1 + b_rem;
        if constexpr (!is_fp32_dest_acc_en) {
            a_res = _lerp_bf16_rne_for_16b_store_(a_res, rne_bias, rne_one);
            b_res = _lerp_bf16_rne_for_16b_store_(b_res, rne_bias, rne_one);
        }
        sfpi::dst_reg[off_out] = a_res;
        sfpi::dst_reg[off_out + 1].mode(ADDR_MOD_6) = b_res;
    }
}

}  // namespace ckernel::sfpu
