// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_log.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_sqrt_custom(vFloat in) {
    vFloat val = in;
    vFloat out;
    v_if(val != 0.0f) {
        vUInt magic = reinterpret<vUInt>(vFloat(s2vFloat16b(0x5f37)));
        vFloat approx = reinterpret<vFloat>(magic - (reinterpret<vUInt>(val) >> 1));
        for (int r = 0; r < 2; r++) {
            approx = ((approx * approx) * (val * -0.5f) + 1.5f) * approx;
        }
        out = approx * val;
    }
    v_else { out = val; }
    v_endif;
    return out;
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_erfinv_body(vFloat in) {
    vFloat log_value = in * in;
    log_value = 1 - log_value;
    dst_reg[0] = log_value;
    calculate_log_body<false>(0);
    log_value = dst_reg[0];
    vFloat temp = dst_reg[0] * 0.5;
    temp = 4.5469 + temp;
    temp = -temp;
    vFloat calculated_value = (temp * temp) - (log_value * 7.1427);
    vFloat intermediate_result = calculate_sqrt_custom<false>(calculated_value);
    calculated_value = temp + intermediate_result;
    log_value = calculate_sqrt_custom<false>(calculated_value);
    dst_reg[0] = log_value;
    return log_value;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    // SFPU microcode
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        v_if(v == 1.0f) { dst_reg[0] = std::numeric_limits<float>::infinity(); }
        v_elseif(v == -1.0f) { dst_reg[0] = -std::numeric_limits<float>::infinity(); }
        v_elseif((v < -1.0f) || (v > 1.0f)) {  // Nan not supported
            dst_reg[0] = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif(v < 0.0f) {
            calculate_erfinv_body<true>(v);
            dst_reg[0] = -dst_reg[0];
        }
        v_else { calculate_erfinv_body<true>(v); }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfinv_init() {
    vConstFloatPrgm0 = 0.692871f;  // ln2
    vConstFloatPrgm1 = 0.1058f;
    vConstFloatPrgm2 = -0.7166f;
}

}  // namespace sfpu
}  // namespace ckernel
