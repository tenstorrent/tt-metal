// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `binarypow-fresh` coverage row (metal
// ckernel_sfpu_binary_pow.h calculate_sfpu_binary_pow — the header's own
// kernel entry has zero standalone nodes; the tested SfpuElwpow dispatch goes
// through calculate_sfpu_binary instead).  Mathematical definition
// (torch.pow, elementwise vector exponent): y = a^b = 2^(b * log2(a)) on the
// suite's registered pow domain (base a in (1e-6, 3): the domain registry
// carves the non-positive-base hole out of operand A, so negative-base
// parity and 0^b never reach the golden).  log2 by the established rminimax
// cubic and 2^z by the exp_21f exponent/mantissa recombination — the
// calculate_unary_power_fresh_cpp machinery with the scalar exponent replaced
// by the second Dst tile; bf16 RNE store per the fresh float-body convention.
#include <cstdint>

namespace ckernel::sfpu
{

template <bool DST_ACCUM_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_binary_pow_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;
    // rminimax cubic over [1, 2) for ln(m) (production constants).
    constexpr float P3      = 0x2.44734p-4f;
    constexpr float P2      = -0xd.e712ap-4f;
    constexpr float P1      = 0x2.4f5388p+0f;
    constexpr float P0      = -0x1.952992p+0f;
    constexpr float ONE_LN2 = 1.4426950408889634f;
    // exp_21f fractional refinement (the calculate_exp_fresh_cpp constants).
    constexpr float C0 = 1.0017248f;
    constexpr float C1 = 7.839635491371155e-08f;
    constexpr float C2 = 4.791750143340323e-15f;

#pragma GCC unroll 0
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 0
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vFloat base = sfpi::dst_reg[0];
            const sfpi::vFloat pow  = sfpi::dst_reg[tile_rows];

            // Step 1: log2(base) = poly(mantissa)/ln2 + exponent (base > 0 by domain).
            const sfpi::vFloat m           = sfpi::setexp(base, 127);
            const sfpi::vFloat series      = m * (m * (m * P3 + P2) + P1) + P0;
            const sfpi::vFloat exp_f32     = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(sfpi::exexp(base)), sfpi::RoundMode::Nearest);
            const sfpi::vFloat log2_result = exp_f32 + series * ONE_LN2;

            // Step 2: 2^z by exponent/mantissa recombination (exp_21f).
            sfpi::vFloat zlog2   = pow * log2_result + 127.0f; // biased result exponent
            zlog2                = sfpi::min(zlog2, 255.0f);
            sfpi::vInt zi        = sfpi::shft(sfpi::exman(zlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(zlog2), sfpi::ShiftMode::Logical);
            const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(zi);

            sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
            frac              = (C2 * frac + C1) * frac + C0;

            sfpi::vFloat zc = z;
            v_if (zlog2 <= 0.0f)
            {
                zc = 0.0f;
            }
            v_endif;
            sfpi::vFloat y = sfpi::setexp(frac, sfpi::exexp(zc, sfpi::ExponentMode::Biased));

            if constexpr (!DST_ACCUM_MODE)
            {
                y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = y;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_binary_pow_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh binary pow expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh binary pow expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh binary pow expects full-tile vector mode");

    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_binary_pow_fresh_cpp<DST_ACCUM, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
