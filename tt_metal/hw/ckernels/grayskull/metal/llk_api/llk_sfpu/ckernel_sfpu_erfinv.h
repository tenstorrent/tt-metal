// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_erfinv_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::erfinv, APPROXIMATE>();
}

template <bool HAS_BASE_SCALING>
sfpi_inline void calculate_log_body_custom(const int log_base_scale_factor)
{
    vFloat in = dst_reg[0];
    vFloat x = setexp(in, 127);
    vFloat a = s2vFloat16a(0.1058F);
    vFloat series_result = x * (x * (x * a + s2vFloat16a(-0.7122f)) + s2vFloat16a(2.0869)) + s2vFloat16a(-1.4753f);
    vInt exp = 0;
    v_if (in != 0.0F) {
        exp = exexp(in);
        v_if (exp < 0) {
            exp = sfpi::abs(exp);
            in = setsgn(in, 1);
        }
        v_endif;
    }
    v_endif;
    vInt new_exp = 0;
    v_if (exp != 0) {
        new_exp = lz(exp);
        new_exp = ~new_exp;
        new_exp += 19;
        v_if (new_exp >= 0) {
            new_exp += 127;
        }
        v_endif;
    }
    v_endif;
    vFloat result = setexp(in, new_exp);
    vInt shift = lz(exp) + 1;
    result = setman(result, shft(reinterpret<vUInt>(exp), shift));
    result = result * vConst0p6929 + series_result;
    if constexpr (HAS_BASE_SCALING) {
        result *= s2vFloat16a(log_base_scale_factor);
    }
    v_if (dst_reg[0] == 0.0F) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;
    dst_reg[0] = result;
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_sqrt_custom(vFloat in)
{
    vFloat val = in;
    vFloat out;
    vUInt magic = reinterpret<vUInt>(vFloat(s2vFloat16b(0x5f37)));
    vFloat approx = reinterpret<vFloat>(magic - (reinterpret<vUInt>(val) >> 1));
    val = dst_reg[0];
    for (int r = 0; r < 2; r++)
    {
        approx = (approx * approx * val * vConstNeg0p5 + vConst1 + 0.5F) * approx;
    }
    out = approx * val;

    return out;
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_erfinv_body(vFloat in)
{
    vFloat log_value = in * in;
    log_value = 1 - log_value;
    dst_reg[0] = log_value;
    calculate_log_body_custom<false>(0);
    log_value = dst_reg[0];
    vFloat temp = dst_reg[0] * 0.5;
    temp = 4.5469 + temp;
    temp = -temp;
    vFloat calculated_value = (temp * temp) - (log_value * 7.1427);
    dst_reg[0] = calculated_value;
    vFloat intermediate_result = calculate_sqrt_custom<false>(dst_reg[0]);
    calculated_value = temp + intermediate_result;
    dst_reg[0] = calculated_value;
    log_value = calculate_sqrt_custom<false>(dst_reg[0]);
    dst_reg[0] = log_value;
    return log_value;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_erfinv()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        v_if (dst_reg[0] == 1.0f) {
            dst_reg[0] = std::numeric_limits<float>::infinity();
        }v_elseif (dst_reg[0] == -1.0f) {
            dst_reg[0] = -std::numeric_limits<float>::infinity();
        }v_elseif ((dst_reg[0] < -1.0f)||(dst_reg[0] > 1.0f)) {
            dst_reg[0] = std::numeric_limits<float>::quiet_NaN();
        }v_elseif (dst_reg[0] < 0.0f) {
            calculate_erfinv_body<true>(dst_reg[0]);
            dst_reg[0] = -dst_reg[0];
        }v_else {
            calculate_erfinv_body<true>(dst_reg[0]);
        }
        v_endif;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
