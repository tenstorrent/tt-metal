// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_sfpu_unary_max_min.h"
#include "cmath_common.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

enum { Max = true, Min = false };  // Clamp Mode

inline void clamp_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// out = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(std::uint32_t min_val, std::uint32_t max_val) {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_float(min_val);
        calculate_unary_max_min_float_body<Max>();
        load_value_param_float(max_val);
        calculate_unary_max_min_float_body<Min>();
        sfpi::dst_reg++;
    }
}

// Op class for clamp(x, min_val, max_val) on floats (fp32 bits).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Clamp : SfpuUnaryOp<Clamp<APPROXIMATION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t min_val, const std::uint32_t max_val) {
        calculate_clamp<APPROXIMATION_MODE, ITERATIONS>(min_val, max_val);
    }
    static inline __attribute__((always_inline)) void init_op() { clamp_init(); }
};

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp_int32(std::uint32_t min_val, std::uint32_t max_val) {
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_int(min_val);
        calculate_unary_max_min_int32_body<Max>(min_val);
        load_value_param_int(max_val);
        calculate_unary_max_min_int32_body<Min>(max_val);
        sfpi::dst_reg++;
    }
}

// Op class for clamp(x, min_val, max_val) on int32.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct ClampInt32 : SfpuUnaryOp<ClampInt32<APPROXIMATION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t min_val, const std::uint32_t max_val) {
        calculate_clamp_int32<APPROXIMATION_MODE, ITERATIONS>(min_val, max_val);
    }
    static inline __attribute__((always_inline)) void init_op() { clamp_init(); }
};

}  // namespace ckernel::sfpu
