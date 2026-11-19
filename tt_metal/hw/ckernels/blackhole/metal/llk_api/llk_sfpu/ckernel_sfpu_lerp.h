// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool FAST_INTERPOLATE = false>
inline void lerp_init() {
    // calculate_lerp's store advances dest by one sfpi row (two dest counter steps).
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_5);
    if constexpr (FAST_INTERPOLATE) {
        // The FAST_INTERPOLATE fp32 loop's second store advances dest by the two rows it wrote.
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    }
}

// fp32 -> bf16 round-to-nearest-even for a value stored straight to a 16-bit Dest: the low 16 bits
// are left as is, so the result must not be read back or kept in an fp32 Dest.
// The lsb is a separate helper so the last row can issue the two integer adds apart.
sfpi_inline sfpi::vUInt _lerp_bf16_rne_lsb_(const sfpi::vFloat in) { return (sfpi::as<sfpi::vUInt>(in) << 15) >> 31; }
sfpi_inline sfpi::vFloat _lerp_bf16_rne_for_16b_store_(
    const sfpi::vFloat in, const sfpi::vUInt lsb, const sfpi::vUInt bias) {
    return sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(in) + bias + lsb);
}

// fp32 -> bf16 round-to-nearest-even ("bits + 0x7fff + lsb") for the FAST_INTERPOLATE bf16 loop, same
// 16-bit store contract as above. bias (0x7fff) and one (1) are bound by the caller, which keeps them in
// LREGs across rows.
sfpi_inline sfpi::vFloat _lerp_fast_bf16_rne_for_16b_store_(
    const sfpi::vFloat in, const sfpi::vUInt bias, const sfpi::vUInt one) {
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);
    const sfpi::vUInt lsb = (bits >> 16) & one;
    bits = bits + bias + lsb;
    return sfpi::as<sfpi::vFloat>(bits);
}

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void _calculate_lerp_half_scale_(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_in2, const uint dst_index_out) {
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

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void _calculate_lerp_fast_interpolate_(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_in2, const uint dst_index_out) {
    static_assert(ITERATIONS % 2 == 0, "calculate_lerp<FAST_INTERPOLATE>() processes dest rows in pairs.");

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
    // lerp: out = (input - weight * input) + weight * end, for weight in [0, 1] only.
    // No (end - input) difference, so finite inputs cannot overflow for weight in [0, 1], and
    // weight == 0 / weight == 1 return input / end bit-exact. (1 - weight) * input is formed by
    // one SFPMAD as input - weight * input, so 1 - weight is never rounded on its own.
    //
    // Both loops are ordered so that no instruction reads the result of the SFPMAD right before it.
    if constexpr (is_fp32_dest_acc_en) {
        // Two rows (a, b) per iteration.
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
            const sfpi::vFloat a_res = a_in2 * a_in1 + a_rem;
            const sfpi::vFloat b_res = b_in2 * b_in1 + b_rem;
            sfpi::dst_reg[off_out] = a_res;
            sfpi::dst_reg[off_out + 1].mode(ADDR_MOD_6) = b_res;
        }
    } else {
        // One row per iteration, rotated: row d is formed while row d - 1 is rounded and stored,
        // which keeps the loop body within the 16 SFPU replay slots.
        const sfpi::vFloat f_in0 = sfpi::dst_reg[off_in0];
        const sfpi::vFloat f_in2 = sfpi::dst_reg[off_in2];
        const sfpi::vFloat f_rem = f_in0 - f_in2 * f_in0;
        const sfpi::vFloat f_in1 = sfpi::dst_reg[off_in1];
        sfpi::vFloat res = f_in2 * f_in1 + f_rem;
#pragma GCC unroll 8
        for (int d = 1; d < ITERATIONS; d++) {
            const sfpi::vFloat in0 = sfpi::dst_reg[off_in0 + 1];
            const sfpi::vFloat in2 = sfpi::dst_reg[off_in2 + 1];
            const sfpi::vFloat rem = in0 - in2 * in0;
            const sfpi::vFloat prev = _lerp_fast_bf16_rne_for_16b_store_(res, rne_bias, rne_one);
            const sfpi::vFloat in1 = sfpi::dst_reg[off_in1 + 1];
            sfpi::dst_reg[off_out].mode(ADDR_MOD_5) = prev;
            res = in2 * in1 + rem;
        }
        sfpi::dst_reg[off_out].mode(ADDR_MOD_5) = _lerp_fast_bf16_rne_for_16b_store_(res, rne_bias, rne_one);
    }
}

// Requires lerp_init<APPROXIMATION_MODE, FAST_INTERPOLATE>().
// FAST_INTERPOLATE: the caller guarantees weight in [0, 1]; the result is undefined outside it.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    DataFormat data_format,
    int ITERATIONS,
    bool FAST_INTERPOLATE = false>
inline void calculate_lerp(
    const uint dst_index_in0,  // input (start)
    const uint dst_index_in1,  // end
    const uint dst_index_in2,  // weight
    const uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");
    if constexpr (FAST_INTERPOLATE) {
        _calculate_lerp_fast_interpolate_<is_fp32_dest_acc_en, ITERATIONS>(
            dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out);
    } else {
        _calculate_lerp_half_scale_<is_fp32_dest_acc_en, ITERATIONS>(
            dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out);
    }
}

}  // namespace ckernel::sfpu
