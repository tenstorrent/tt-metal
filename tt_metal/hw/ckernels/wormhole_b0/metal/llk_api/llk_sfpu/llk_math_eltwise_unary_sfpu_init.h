// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "llk_sfpu_types.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

// Kernel-invariant SFPU init (SFPU config register + invariant ADDR_MOD_7). Retained for the standalone tt-llk
// SFPU test harness, which bypasses the metal "full init" entry points and runs this itself. The metal compute
// path no longer hoists this: each per-op init below is self-contained (#50381), running the invariant per-op
// rather than once-per-kernel.
inline void llk_math_sfpu_init_once() { _llk_math_eltwise_unary_sfpu_init_once_(); }

namespace sfpu {
// Forward declarations of the co-located per-op inits (each defined next to its execute method in
// ckernel_sfpu_<op>.h). Declared -- not #included -- here so the dispatch below resolves these names
// under two-phase lookup WITHOUT pulling the op headers into every TRISC_MATH translation unit via
// compute_kernel_hw_startup. Each op's definition is in scope at its <op>_tile_init instantiation site.
void abs_init();
void acos_init();
void alt_complex_rotate90_init();
void asin_init();
void bitwise_and_init();
void bitwise_not_init();
void bitwise_or_init();
void bitwise_xor_init();
void celu_init();
void clamp_init();
void elu_init();
void equal_zero_init();
void greater_than_equal_zero_init();
void greater_than_zero_init();
void hardmish_init();
void hardshrink_init();
void hardtanh_init();
void heaviside_init();
void i0_init();
void left_shift_init();
void less_than_equal_zero_init();
void less_than_zero_init();
void logical_not_unary_init();
void mask_init();
void not_equal_zero_init();
void power_init();
void prelu_init();
void relu_max_init();
void reshuffle_rows_init();
void right_shift_init();
void selu_init();
void sign_init();
void softplus_init();
void softshrink_init();
void square_init();
void tiled_prod_init();
void unary_eq_init();
void unary_ge_init();
void unary_gt_init();
void unary_le_init();
void unary_lt_init();
void unary_ne_init();

// Residual per-op inits for ops used via bare SFPU_UNARY_INIT(OP) (no callback). config_reg + ADDR_MOD_7 are
// run per-op by the bare delegate below (_llk_math_eltwise_unary_sfpu_init_once_()), so these program only the
// op's residual state (op-specific ADDR_MOD_6 where needed + reset the RWC counters).
// Rounding-family ops (ceil/floor/trunc/frac/round): pure-arithmetic SFPI kernels with no LUT/ADDR_MOD_6
// state (production shares rounding_op_tile_init -> SFPU_UNARY_INIT(unused)); only reset the RWC counters.
inline void ceil_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void fill_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void floor_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void frac_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void round_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void trunc_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isfinite_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isinf_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isnan_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isneginf_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void isposinf_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void negative_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void silu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void threshold_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

inline void typecast_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void unused_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

}  // namespace sfpu

// Bare init entry point: delegates per SFPU op to its self-contained sfpu::<op>_init().
template <SfpuType sfpu_op>
inline void llk_math_eltwise_unary_sfpu_init() {
    // Per-op common SFPU init (config reg + invariant ADDR_MOD_7), formerly hoisted once-per-kernel via
    // llk_math_sfpu_init_once(). Consolidated back per-op (#50381) so each init is fully self-contained and
    // never depends on a separate once-init having run first. The co-located sfpu::<op>_init() below then
    // programs only the op's residual state (op-specific ADDR_MOD_6 where needed + counter reset).
    _llk_math_eltwise_unary_sfpu_init_once_();
    if constexpr (sfpu_op == SfpuType::abs) {
        sfpu::abs_init();
    } else if constexpr (sfpu_op == SfpuType::acos) {
        sfpu::acos_init();
    } else if constexpr (sfpu_op == SfpuType::alt_complex_rotate90) {
        sfpu::alt_complex_rotate90_init();
    } else if constexpr (sfpu_op == SfpuType::asin) {
        sfpu::asin_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_and) {
        sfpu::bitwise_and_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_not) {
        sfpu::bitwise_not_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_or) {
        sfpu::bitwise_or_init();
    } else if constexpr (sfpu_op == SfpuType::bitwise_xor) {
        sfpu::bitwise_xor_init();
    } else if constexpr (sfpu_op == SfpuType::ceil) {
        sfpu::ceil_init();
    } else if constexpr (sfpu_op == SfpuType::celu) {
        sfpu::celu_init();
    } else if constexpr (sfpu_op == SfpuType::clamp) {
        sfpu::clamp_init();
    } else if constexpr (sfpu_op == SfpuType::elu) {
        sfpu::elu_init();
    } else if constexpr (sfpu_op == SfpuType::equal_zero) {
        sfpu::equal_zero_init();
    } else if constexpr (sfpu_op == SfpuType::fill) {
        sfpu::fill_init();
    } else if constexpr (sfpu_op == SfpuType::floor) {
        sfpu::floor_init();
    } else if constexpr (sfpu_op == SfpuType::frac) {
        sfpu::frac_init();
    } else if constexpr (sfpu_op == SfpuType::round) {
        sfpu::round_init();
    } else if constexpr (sfpu_op == SfpuType::trunc) {
        sfpu::trunc_init();
    } else if constexpr (sfpu_op == SfpuType::greater_than_equal_zero) {
        sfpu::greater_than_equal_zero_init();
    } else if constexpr (sfpu_op == SfpuType::greater_than_zero) {
        sfpu::greater_than_zero_init();
    } else if constexpr (sfpu_op == SfpuType::hardmish) {
        sfpu::hardmish_init();
    } else if constexpr (sfpu_op == SfpuType::hardshrink) {
        sfpu::hardshrink_init();
    } else if constexpr (sfpu_op == SfpuType::hardtanh) {
        sfpu::hardtanh_init();
    } else if constexpr (sfpu_op == SfpuType::heaviside) {
        sfpu::heaviside_init();
    } else if constexpr (sfpu_op == SfpuType::i0) {
        sfpu::i0_init();
    } else if constexpr (sfpu_op == SfpuType::isfinite) {
        sfpu::isfinite_init();
    } else if constexpr (sfpu_op == SfpuType::isinf) {
        sfpu::isinf_init();
    } else if constexpr (sfpu_op == SfpuType::isnan) {
        sfpu::isnan_init();
    } else if constexpr (sfpu_op == SfpuType::isneginf) {
        sfpu::isneginf_init();
    } else if constexpr (sfpu_op == SfpuType::isposinf) {
        sfpu::isposinf_init();
    } else if constexpr (sfpu_op == SfpuType::left_shift) {
        sfpu::left_shift_init();
    } else if constexpr (sfpu_op == SfpuType::less_than_equal_zero) {
        sfpu::less_than_equal_zero_init();
    } else if constexpr (sfpu_op == SfpuType::less_than_zero) {
        sfpu::less_than_zero_init();
    } else if constexpr (sfpu_op == SfpuType::logical_not_unary) {
        sfpu::logical_not_unary_init();
    } else if constexpr (sfpu_op == SfpuType::mask) {
        sfpu::mask_init();
    } else if constexpr (sfpu_op == SfpuType::negative) {
        sfpu::negative_init();
    } else if constexpr (sfpu_op == SfpuType::not_equal_zero) {
        sfpu::not_equal_zero_init();
    } else if constexpr (sfpu_op == SfpuType::power) {
        sfpu::power_init();
    } else if constexpr (sfpu_op == SfpuType::prelu) {
        sfpu::prelu_init();
    } else if constexpr (sfpu_op == SfpuType::relu_max) {
        sfpu::relu_max_init();
    } else if constexpr (sfpu_op == SfpuType::reshuffle_rows) {
        sfpu::reshuffle_rows_init();
    } else if constexpr (sfpu_op == SfpuType::right_shift) {
        sfpu::right_shift_init();
    } else if constexpr (sfpu_op == SfpuType::selu) {
        sfpu::selu_init();
    } else if constexpr (sfpu_op == SfpuType::sign) {
        sfpu::sign_init();
    } else if constexpr (sfpu_op == SfpuType::silu) {
        sfpu::silu_init();
    } else if constexpr (sfpu_op == SfpuType::softplus) {
        sfpu::softplus_init();
    } else if constexpr (sfpu_op == SfpuType::softshrink) {
        sfpu::softshrink_init();
    } else if constexpr (sfpu_op == SfpuType::square) {
        sfpu::square_init();
    } else if constexpr (sfpu_op == SfpuType::threshold) {
        sfpu::threshold_init();
    } else if constexpr (sfpu_op == SfpuType::tiled_prod) {
        sfpu::tiled_prod_init();
    } else if constexpr (sfpu_op == SfpuType::unary_eq) {
        sfpu::unary_eq_init();
    } else if constexpr (sfpu_op == SfpuType::unary_ge) {
        sfpu::unary_ge_init();
    } else if constexpr (sfpu_op == SfpuType::unary_gt) {
        sfpu::unary_gt_init();
    } else if constexpr (sfpu_op == SfpuType::unary_le) {
        sfpu::unary_le_init();
    } else if constexpr (sfpu_op == SfpuType::unary_lt) {
        sfpu::unary_lt_init();
    } else if constexpr (sfpu_op == SfpuType::unary_ne) {
        sfpu::unary_ne_init();
    } else if constexpr (sfpu_op == SfpuType::unused) {
        sfpu::unused_init();
    } else if constexpr (sfpu_op == SfpuType::typecast) {
        sfpu::typecast_init();
    } else if constexpr (sfpu_op == SfpuType::exponential) {
        // SDPA and other direct LLK callers use this no-arg overload to get the generic unary SFPU
        // addrmod state (config reg + ADDR_MOD_7 + counter reset) without an op-specific init or a
        // preceding compute_kernel_hw_startup, so route it to the full generic init.
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::exponential>();
    } else {
        // Generic fallback (pre-restructuring behavior): ops without a self-contained sfpu::<op>_init()
        // get the full generic unary SFPU init (config reg + ADDR_MOD_7 + op-specific ADDR_MOD_6 via
        // eltwise_unary_sfpu_configure_addrmod<OP>, which has a default valid for any SfpuType + counter
        // reset). Keeps the bare no-arg delegate instantiable across the whole SfpuType set.
        _llk_math_eltwise_unary_sfpu_init_<sfpu_op>();
    }
}

// Callback init entry point (SFPU_UNARY_INIT_FN / _FN_ARGS / two-arg SFPU_UNARY_INIT). The per-op common init
// (config reg + ADDR_MOD_7 + counter reset) runs first, then the op-specific init_func. Consolidated per-op
// (#50381) rather than hoisted: re-asserting the shared SFPU config/addrmod state on every init -- on both the
// MATH and PACK threads -- keeps the exp(PACK)/fp32-reciprocal(MATH) shared macro/replay programming from
// interleaving destructively, which was the #50381 fp32 SDPA accuracy regression.
template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    _llk_math_eltwise_unary_sfpu_init_<sfpu_op>();
    init_func(std::forward<ARGS>(args)...);
}

}  // namespace ckernel
