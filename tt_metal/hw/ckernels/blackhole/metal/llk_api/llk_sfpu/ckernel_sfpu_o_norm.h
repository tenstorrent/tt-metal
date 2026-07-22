// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_sqrt.h"
#include "llk_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// ============================================================================
// Fused o_norm SFPU kernel (Kimi-K3 / KDA "o_norm" node), Blackhole.
//
//     o_norm = RMSNorm(o) * gamma2 * sigmoid(g_out)
//
// where, per head, RMSNorm reduces over the head-dim D:
//
//     rms(o)    = sqrt( (1/D) * sum_d o[d]^2 + eps )
//     o_norm[d] = ( o[d] / rms(o) ) * gamma2[d] * sigmoid(g_out[d])
//
// ---------------------------------------------------------------------------
// Tilization contract (IMPORTANT)
// ---------------------------------------------------------------------------
// Built from basic sfpi primitives (no SFPTRANSP shuffle). The reduction
// dimension (head-dim) runs along the Dest row/iteration axis and the heads
// occupy the SFPU lanes, so the per-head sum of squares accumulates per lane
// with no lane-to-lane communication. gamma2[d] is broadcast across lanes.
// D = NUM_REDUCE_TILES * 32 and the operands occupy NUM_REDUCE_TILES
// consecutive Dest tiles each.
//
// NOTE: the exact row/lane <-> logical-dimension mapping and the numeric
// tolerances must be validated on Blackhole hardware / simulator (mirrors the
// tt-llk reference kernel, ckernel_sfpu_o_norm.h).
// ============================================================================

// sigmoid(x) = 1 / (1 + exp(-x)) built from basic SFPU primitives. One Newton
// step is applied on top of the fast reciprocal approximation; it uses only a
// literal (2.0f), so it does not touch the programmable const register that the
// sqrt/rsqrt path programs.
sfpi_inline sfpi::vFloat _o_norm_sigmoid_(sfpi::vFloat x) {
    sfpi::vFloat denom = _sfpu_exp_(-x) + 1.0f;
    sfpi::vFloat y = sfpi::approx_recip(denom);
    // y <- y * (2 - denom * y), written to avoid a (float - vFloat) expression.
    sfpi::vFloat t = denom * y - 2.0f;
    y = y * (-t);
    return y;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int NUM_REDUCE_TILES>
inline void calculate_o_norm(
    const std::uint32_t dst_index_in0,  // o
    const std::uint32_t dst_index_in1,  // gamma2
    const std::uint32_t dst_index_in2,  // g_out
    const std::uint32_t dst_index_out,  // output
    const std::uint32_t eps_bits) {     // RMSNorm epsilon as an fp32 bit pattern
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Bfp8_b,
        "Unsupported data format for calculate_o_norm(). Only Float32, Float16_b and Bfp8_b are allowed.");

    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    constexpr int n_rows = NUM_REDUCE_TILES * static_cast<int>(dst_tile_size_sfpi);
    constexpr float inv_reduce_count = 1.0f / static_cast<float>(n_rows);
    const sfpi::vFloat eps = Converter::as_float(eps_bits);

    const std::uint32_t off_o = dst_index_in0 * dst_tile_size_sfpi;
    const std::uint32_t off_gamma = dst_index_in1 * dst_tile_size_sfpi;
    const std::uint32_t off_gout = dst_index_in2 * dst_tile_size_sfpi;
    const std::uint32_t off_out = dst_index_out * dst_tile_size_sfpi;

    // Pass 1: per-lane (per-head) sum of squares over the reduction rows.
    sfpi::vFloat acc = 0.0f;
    for (int r = 0; r < n_rows; r++) {
        sfpi::vFloat x = sfpi::dst_reg[off_o + r];
        acc = acc + x * x;
    }

    // rms^-1 = rsqrt( mean(o^2) + eps ), computed once per lane.
    sfpi::vFloat mean_sq = acc * inv_reduce_count + eps;
    sfpi::vFloat inv_rms = _calculate_sqrt_body_<APPROXIMATION_MODE, /*RECIPROCAL=*/true, /*FAST_APPROX=*/false>(mean_sq);

    // Pass 2: normalize and apply gamma2 * sigmoid(g_out).
    for (int r = 0; r < n_rows; r++) {
        sfpi::vFloat x = sfpi::dst_reg[off_o + r];
        sfpi::vFloat gamma = sfpi::dst_reg[off_gamma + r];
        sfpi::vFloat gout = sfpi::dst_reg[off_gout + r];

        sfpi::vFloat result = x * inv_rms * gamma * _o_norm_sigmoid_(gout);

        if constexpr (!is_fp32_dest_acc_en) {
            sfpi::dst_reg[off_out + r] = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        } else {
            sfpi::dst_reg[off_out + r] = result;
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void o_norm_init() {
    // Program the const registers used by the rsqrt approximation. The sigmoid
    // path (_sfpu_exp_ + approx_recip) needs no initialization.
    sqrt_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
