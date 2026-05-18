// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

sfpi_inline sfpi::vFloat _sfpu_exp2_fp32_accurate_(sfpi::vFloat val)
{
    sfpi::vFloat result = sfpi::vConst0;

    // Thresholds scaled for base-2
    constexpr float OVERFLOW_THRESHOLD  = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    sfpi::vInt exp_bits = sfpi::exexp(val);

    v_if (val >= OVERFLOW_THRESHOLD)
    {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif (val <= UNDERFLOW_THRESHOLD)
    {
        result = sfpi::vConst0;
    }
    v_elseif (exp_bits == 255)
    {
        result = val; // Propagate NaN / Inf
    }
    v_else
    {
        // k = round(val)
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_to_nearest_int32_(val, k_int);

        // r_frac = val - k
        sfpi::vFloat r_frac = k * -1.0f + val;

        // Multiply by ln(2) for polynomial evaluation (e^(r_frac * ln2))
        constexpr float LN2 = 0.6931471805599453f;
        sfpi::vFloat r = r_frac * LN2;

        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1,
            sfpi::vConst1,
            0.5f,
            1.0f / 6.0f,
            1.0f / 24.0f,
            1.0f / 120.0f,
            1.0f / 720.0f,
            1.0f / 5040.0f
        );

        sfpi::vInt p_exp   = sfpi::exexp(p, sfpi::ExponentMode::NoDebias);
        sfpi::vInt new_exp = p_exp + k_int;

        result = sfpi::setexp(p, new_exp);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _sfpu_exp2_21f_bf16_(sfpi::vFloat val)
{
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD  = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    sfpi::vInt exp_bits = sfpi::exexp(val);

    v_if (val >= OVERFLOW_THRESHOLD)
    {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif (val <= UNDERFLOW_THRESHOLD)
    {
        result = sfpi::vConst0;
    }
    v_elseif (exp_bits == 255)
    {
        result = val;
    }
    v_else
    {
        // xlog2 is directly val + 127.f (val is log2)
        sfpi::vFloat xlog2 = val + 127.f;

        sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);

        sfpi::vInt exponential_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z), sfpi::ExponentMode::NoDebias);
        sfpi::vInt fractional_part  = sfpi::exman(sfpi::reinterpret<sfpi::vFloat>(z));

        sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, sfpi::RoundMode::NearestEven);

        frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

        result = sfpi::setexp(frac, exponential_part);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en)
        {
            result = _sfpu_exp2_fp32_accurate_(v);
        }
        else
        {
            result = _sfpu_exp2_21f_bf16_<true>(v);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
    // No initialization needed for optimized direct evaluation
}

} // namespace ckernel::sfpu
