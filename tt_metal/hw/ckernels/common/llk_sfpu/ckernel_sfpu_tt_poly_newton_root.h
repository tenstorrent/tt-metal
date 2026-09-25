// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_tt_poly_newton_root_core.h"
// Sibling stock wrappers may include another generated config while this
// shared runtime is being parsed. Expose its adapter interface first.
namespace ckernel::sfpu::ttpoly {
template <typename Config>
inline void init_newton_root();
template <typename Config, int Iterations = 8>
inline void calculate_newton_root();
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_NEWTON_ROOT_SELECTED_V1 1
#if defined(ARCH_WORMHOLE)
// rsqrt includes sqrt before declaring its callback; a composed sqrt config
// can therefore reach this runtime first. These declarations match the
// included stock definitions, without repeating their default arguments.
namespace ckernel::sfpu {
template <bool, int, bool, bool, bool>
inline void calculate_rsqrt();
template <bool, bool, int>
inline void calculate_cube_root();
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_rsqrt.h"
#include "ckernel_sfpu_cbrt.h"
#endif
namespace ckernel::sfpu::ttpoly {
template <typename Config>
inline void init_newton_root() {
    if constexpr (Config::kKind == 2) {
        sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kC0Bits);
    } else {
        sfpi::vConstIntPrgm0 = Config::kMagic;
    }
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kC1Bits);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kC2Bits);
#if defined(ARCH_BLACKHOLE)
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
#else
    if constexpr (Config::kKind == 0) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    }
#endif
}

template <typename Config, int Iterations>
inline void calculate_newton_root() {
    static_assert(Iterations == 8 && Config::kKind <= 2);
    static_assert(Config::kKind != 0 || Config::kNegativeExponentZeroOutput == 0x7f80u);
    static_assert(Config::kBodySlots == (Config::kKind == 0 ? 32u : Config::kKind == 1 ? 30u : 23u));
#if defined(ARCH_WORMHOLE)
    if constexpr (Config::kKind == 1) {
        ckernel::sfpu::calculate_rsqrt<false, Iterations, false, false, false>();
    } else if constexpr (Config::kKind == 2) {
        ckernel::sfpu::calculate_cube_root<false, false, Iterations>();
    } else {
        TTI_REPLAY(0, Config::kBodySlots, 1, 1);
        sfpi::newton_sqrt_body<false, false, true, ADDR_MOD_3, ADDR_MOD_2>();
#pragma GCC unroll 8
        for (int row = 1; row < Iterations; ++row) {
            TTI_REPLAY(0, Config::kBodySlots, 0, 0);
        }
    }
#elif defined(ARCH_BLACKHOLE)
    if constexpr (Config::kKind == 2) {
        constexpr float negative_inv_n_scaled = (-1.0f / 3.0f) / 256.0f;
        constexpr float magic = ((float)Config::kMagic) / 256.0f + 8388608.0f;
        constexpr uint32_t scale_bits = __builtin_bit_cast(uint32_t, negative_inv_n_scaled);
        constexpr uint32_t magic_bits = __builtin_bit_cast(uint32_t, magic);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, scale_bits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, scale_bits & 0xffffu);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, magic_bits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, magic_bits & 0xffffu);
    }
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    if constexpr (Config::kKind == 0) {
        static_assert(Config::kNegativeExponentZeroOutput == 0x7f80u);
        sfpi::newton_sqrt_body<true, true, false, ADDR_MOD_7, ADDR_MOD_6>();
    } else if constexpr (Config::kKind == 1) {
        sfpi::newton_rsqrt_body<ADDR_MOD_7, ADDR_MOD_6>();
    } else {
        sfpi::newton_cbrt_body<ADDR_MOD_7, ADDR_MOD_6>();
    }
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
#else
#error "selected Newton root requires BH or WH"
#endif
}
}  // namespace ckernel::sfpu::ttpoly
