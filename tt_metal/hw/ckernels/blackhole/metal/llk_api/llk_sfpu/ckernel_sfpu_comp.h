// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/ckernel_sfpu_is_fp16_zero.h"
#include "sfpu/ckernel_sfpu_load_config.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void greater_than_equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void greater_than_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void less_than_equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void less_than_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void not_equal_zero_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp() {
    // fp32 total-order comparison-to-zero. |v| is used to fold ±0 together and to
    // detect NaN: a NaN has |v| whose fp32 bit pattern is strictly greater than +inf
    // (0x7F800000), so `as<vInt>(|v|) > 0x7F800000` isolates it.
    constexpr int FP32_INF_BITS = 0x7F800000;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        vFloat abs_v = sfpi::abs(v);
        vInt abs_bits = as<vInt>(abs_v);
        vFloat result;

        // eqz: 1 where |v| == 0 (handles ±0; NaN has |v| != 0 → 0)
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            result = 0.0f;
            v_if(abs_v == 0.0f) { result = 1.0f; }
            v_endif;
        }

        // nez: 0 where |v| == 0 (handles ±0; NaN has |v| != 0 → 1)
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            result = 1.0f;
            v_if(abs_v == 0.0f) { result = 0.0f; }
            v_endif;
        }

        // ltz: (v < 0) AND (|v| != 0) → 1, then NaN → 0
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            result = 0.0f;
            v_if(v < 0.0f && abs_v != 0.0f) { result = 1.0f; }
            v_endif;
            v_if(abs_bits > FP32_INF_BITS) { result = 0.0f; }
            v_endif;
        }

        // gtz: (v >= 0) AND (|v| != 0) → 1, then NaN → 0
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            result = 0.0f;
            v_if(v >= 0.0f && abs_v != 0.0f) { result = 1.0f; }
            v_endif;
            v_if(abs_bits > FP32_INF_BITS) { result = 0.0f; }
            v_endif;
        }

        // gez: default 1; negatives (excl. -0) → 0; NaN → 0
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            result = 1.0f;
            v_if(v < 0.0f && abs_v != 0.0f) { result = 0.0f; }
            v_endif;
            v_if(abs_bits > FP32_INF_BITS) { result = 0.0f; }
            v_endif;
        }

        // lez: default 1; positives (excl. +0) → 0; NaN → 0
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            result = 1.0f;
            v_if(v >= 0.0f && abs_v != 0.0f) { result = 0.0f; }
            v_endif;
            v_if(abs_bits > FP32_INF_BITS) { result = 0.0f; }
            v_endif;
        }

        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_int() {
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt zero = 0;

        // a[i] == 0
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(v == zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] != 0
        if constexpr (COMP_MODE == SfpuType::not_equal_zero) {
            v_if(v == zero) { v = zero; }
            v_else { v = 1; }
            v_endif;
        }

        // a[i] < 0
        if constexpr (COMP_MODE == SfpuType::less_than_zero) {
            v_if(v < zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] > 0
        if constexpr (COMP_MODE == SfpuType::greater_than_zero) {
            v_if(v > zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] <= 0
        if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
            v_if(v <= zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        // a[i] >= 0
        if constexpr (COMP_MODE == SfpuType::greater_than_equal_zero) {
            v_if(v >= zero) { v = 1; }
            v_else { v = zero; }
            v_endif;
        }

        dst_reg[0] = v;
        dst_reg++;
    }
}

// NOTE: the uint16/uint32 comparison-to-zero paths below have no tt-llk python-test
// coverage (the comp-to-zero suite only exercises Float16_b/Float32). They mirror the
// original raw-TTI load/store modes (LO16 for uint16, INT32 for uint32) so behaviour is
// preserved; validate at the ttnn level before relying on them.
template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_uint16() {
    static_assert((COMP_MODE == SfpuType::equal_zero) or (COMP_MODE == SfpuType::not_equal_zero));
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0].mode<sfpi::DataLayout::U16>();
        vUInt result = 0;
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(v == 0) { result = 1; }
            v_endif;
        } else {
            v_if(v == 0) { result = 0; }
            v_else { result = 1; }
            v_endif;
        }
        dst_reg[0].mode<sfpi::DataLayout::U16>() = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_eqz_uint32() {
    // UInt32 values occupy the full dest word; DataLayout::U32 loads/stores them
    // directly (SFPLOAD/SFPSTORE mod = UINT32). eqz/nez are representation-agnostic
    // (only a compare against the all-zero word), so a plain unsigned compare works.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vUInt r = 0;
        v_if(v == 0) { r = 1; }
        v_endif;
        dst_reg[0].mode<sfpi::DataLayout::U32>() = r;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_nez_uint32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt v = dst_reg[0].mode<sfpi::DataLayout::U32>();
        vUInt r = 0;
        v_if(v != 0) { r = 1; }
        v_endif;
        dst_reg[0].mode<sfpi::DataLayout::U32>() = r;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp_unary_int(int scalar) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt v = dst_reg[0];
        vInt val = 0;

        // a[i] != scalar
        if constexpr (COMP_MODE == SfpuType::unary_ne) {
            v_if(v != scalar) { val = 1; }
            v_endif;
        }
        // a[i] == scalar
        else if constexpr (COMP_MODE == SfpuType::unary_eq) {
            v_if(v == scalar) { val = 1; }
            v_endif;
        }
        dst_reg[0] = val;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
