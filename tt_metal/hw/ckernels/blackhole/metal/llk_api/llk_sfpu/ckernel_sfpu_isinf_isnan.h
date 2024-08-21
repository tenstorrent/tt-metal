// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isfinite() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(
            v == std::numeric_limits<float>::infinity() || v == -std::numeric_limits<float>::infinity() ||
            v == std::numeric_limits<float>::quiet_NaN() || v == std::numeric_limits<float>::signaling_NaN()) {
            v = 0.0f;
        }
        v_else { v = 1.0f; }
        v_endif;

        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isinf() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v == std::numeric_limits<float>::infinity() || v == -std::numeric_limits<float>::infinity()) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isposinf() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v == std::numeric_limits<float>::infinity()) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isneginf() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v == -std::numeric_limits<float>::infinity()) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isnan() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v == std::numeric_limits<float>::quiet_NaN() || v == std::numeric_limits<float>::signaling_NaN()) {
            v = 1.0f;
        }
        v_else { v = 0.0f; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sfpu_isinf_isnan() {
    if constexpr (operation == SfpuType::isinf) {
        calculate_isinf<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::isposinf) {
        calculate_isposinf<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::isneginf) {
        calculate_isneginf<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::isnan) {
        calculate_isnan<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::isfinite) {
        calculate_isfinite<APPROXIMATION_MODE, ITERATIONS>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
