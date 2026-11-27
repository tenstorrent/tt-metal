// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

namespace sfpu {
void abs_init();
void bitwise_and_init();
void bitwise_or_init();
void bitwise_xor_init();
void hardmish_init();
void hardshrink_init();
void heaviside_init();
void i0_init();
void isinf_isnan_init();
void power_init();
void softcap_init();
void threshold_init();
void tiled_prod_init();
}  // namespace sfpu

// The bare init runs the common SFPU init, plus the op's own init for the ops whose Compute API
// calls the bare form (as on Blackhole).
template <SfpuType sfpu_op>
inline void llk_math_eltwise_unary_sfpu_init() {
    _llk_math_eltwise_sfpu_init_();
    if constexpr (sfpu_op == SfpuType::abs || sfpu_op == SfpuType::abs_int32) {
        // calculate_abs_int32 advances Dest through the ADDR_MOD_6 abs_init programs (ttnn reaches it
        // through abs_tile_init).
        sfpu::abs_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_and) {
        // calculate_sfpu_unary_bitwise advances Dest through the ADDR_MOD_6 these inits program.
        sfpu::bitwise_and_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_or) {
        sfpu::bitwise_or_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_xor) {
        sfpu::bitwise_xor_init();
    } else if constexpr (sfpu_op == SfpuType::hardmish) {
        // hardmish reads 0.5 from vConstFloatPrgm0 and advances Dest through ADDR_MOD_6, both set here.
        sfpu::hardmish_init();
    } else if constexpr (sfpu_op == SfpuType::hardshrink) {
        // calculate_hardshrink advances Dest through the ADDR_MOD_6 hardshrink_init programs.
        sfpu::hardshrink_init();
    } else if constexpr (sfpu_op == SfpuType::heaviside) {
        // calculate_heaviside advances Dest through the ADDR_MOD_6 heaviside_init programs.
        sfpu::heaviside_init();
    } else if constexpr (sfpu_op == SfpuType::i0) {
        sfpu::i0_init();
    } else if constexpr (
        sfpu_op == SfpuType::isinf || sfpu_op == SfpuType::isposinf || sfpu_op == SfpuType::isneginf ||
        sfpu_op == SfpuType::isnan || sfpu_op == SfpuType::isfinite) {
        // _calculate_sfpu_isinf_isnan_ advances Dest through the ADDR_MOD_6 isinf_isnan_init programs.
        sfpu::isinf_isnan_init();
    } else if constexpr (sfpu_op == SfpuType::power) {
        // calculate_unary_power_iterative advances Dest through the ADDR_MOD_6 power_init programs
        // (power_iterative_tile_init reaches it here; power_tile_init passes its own init).
        sfpu::power_init();
    } else if constexpr (sfpu_op == SfpuType::softcap) {
        sfpu::softcap_init();
    } else if constexpr (sfpu_op == SfpuType::threshold) {
        // _calculate_threshold_ advances Dest through the ADDR_MOD_6 threshold_init programs.
        sfpu::threshold_init();
    } else if constexpr (sfpu_op == SfpuType::tiled_prod) {
        // calculate_tiled_prod advances Dest through ADDR_MOD_6 and reads 1.0 from vConstFloatPrgm0,
        // both set by tiled_prod_init.
        sfpu::tiled_prod_init();
    }
}

// sfpu_op template parameter is unused, but kept for backwards compatibility
template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    _llk_math_eltwise_sfpu_init_();
    init_func(std::forward<ARGS>(args)...);
}

}  // namespace ckernel
