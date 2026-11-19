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
// The lsb is a separate helper so the last row can issue the two integer adds apart.
sfpi_inline sfpi::vUInt _lerp_bf16_rne_lsb_(const sfpi::vFloat in) { return (sfpi::as<sfpi::vUInt>(in) << 15) >> 31; }
sfpi_inline sfpi::vFloat _lerp_bf16_rne_for_16b_store_(
    const sfpi::vFloat in, const sfpi::vUInt lsb, const sfpi::vUInt bias) {
    return sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(in) + bias + lsb);
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
    static_assert(ITERATIONS >= 2, "calculate_lerp() handles the first and the last row separately.");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    const auto off_in0 = dst_index_in0 * dst_tile_size_sfpi;
    const auto off_in1 = dst_index_in1 * dst_tile_size_sfpi;
    const auto off_in2 = dst_index_in2 * dst_tile_size_sfpi;
    const auto off_out = dst_index_out * dst_tile_size_sfpi;
    const sfpi::vFloat half = 0.5f;
    // RNE constant, loaded once per call. Unassigned unless rounding, which also gates its use.
    sfpi::vUInt rne_bias;
    if constexpr (!is_fp32_dest_acc_en) {
        rne_bias = 0x7fffU;
    }
    // lerp: out = input + weight * (end - input), evaluated at half scale as
    // 2 * (input / 2 + weight * (end / 2 - input / 2)): for weight in [0, 1] no intermediate can
    // overflow, and the power-of-two scalings are exact above the bottom binade, so the result is
    // the same as the direct form wherever that one is finite.
    //
    // One row per iteration, rotated: row d is formed while row d - 1 is scaled back and stored.
    // In the loop no instruction reads the result of the SFPMAD/SFPMULI right before it.
    sfpi::vFloat half_res;
    {
        const sfpi::vFloat in0 = sfpi::dst_reg[off_in0];
        const sfpi::vFloat in0_half = in0 * 0.5f;
        const sfpi::vFloat in1 = sfpi::dst_reg[off_in1];
        const sfpi::vFloat in2 = sfpi::dst_reg[off_in2];
        const sfpi::vFloat diff_half = in1 * half - in0_half;
        half_res = in2 * diff_half + in0_half;
    }
    // Unroll 16 records a 16-instruction replay body for fp32, the SFPU's whole share of the replay
    // buffer: check the disassembly when changing the loop.
#pragma GCC unroll 16
    for (int d = 1; d < ITERATIONS - 1; d++) {
        const sfpi::vFloat in0 = sfpi::dst_reg[off_in0 + 1];
        sfpi::vFloat prev = half_res * 2.0f;
        const sfpi::vFloat in0_half = in0 * 0.5f;
        if constexpr (!is_fp32_dest_acc_en) {
            prev = _lerp_bf16_rne_for_16b_store_(prev, _lerp_bf16_rne_lsb_(prev), rne_bias);
        }
        const sfpi::vFloat in1 = sfpi::dst_reg[off_in1 + 1];
        const sfpi::vFloat diff_half = in1 * half - in0_half;
        sfpi::dst_reg[off_out].mode(ADDR_MOD_5) = prev;
        // The store above advanced dest to row d.
        const sfpi::vFloat in2 = sfpi::dst_reg[off_in2];
        half_res = in2 * diff_half + in0_half;
    }
    {
        // Last row: the store of the row before goes between the final scale back and its store, and
        // for bf16 the last add of that row's rounding goes right after the last SFPMAD.
        const sfpi::vFloat in0 = sfpi::dst_reg[off_in0 + 1];
        sfpi::vFloat prev = half_res * 2.0f;
        const sfpi::vFloat in0_half = in0 * 0.5f;
        sfpi::vUInt prev_lsb;
        if constexpr (!is_fp32_dest_acc_en) {
            prev_lsb = _lerp_bf16_rne_lsb_(prev);
            prev = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(prev) + rne_bias);
        }
        const sfpi::vFloat in1 = sfpi::dst_reg[off_in1 + 1];
        const sfpi::vFloat diff_half = in1 * half - in0_half;
        const sfpi::vFloat in2 = sfpi::dst_reg[off_in2 + 1];
        const sfpi::vFloat last_half_res = in2 * diff_half + in0_half;
        if constexpr (!is_fp32_dest_acc_en) {
            prev = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(prev) + prev_lsb);
        }
        sfpi::vFloat last = last_half_res * 2.0f;
        sfpi::dst_reg[off_out].mode(ADDR_MOD_5) = prev;
        if constexpr (!is_fp32_dest_acc_en) {
            last = _lerp_bf16_rne_for_16b_store_(last, _lerp_bf16_rne_lsb_(last), rne_bias);
        }
        sfpi::dst_reg[off_out].mode(ADDR_MOD_5) = last;
    }
}

}  // namespace ckernel::sfpu
