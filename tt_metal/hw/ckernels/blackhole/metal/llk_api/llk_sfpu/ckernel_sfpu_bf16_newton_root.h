// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_newton_root_core.h"
// Sibling stock wrappers may include another generated config while this
// shared runtime is being parsed. Expose its adapter interface first.
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_newton_root();
template <typename Config, int Iterations = 8>
inline void calculate_newton_root();
}  // namespace ckernel::sfpu::bf16
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_newton_root() {
    if constexpr (Config::kKind == 2) {
        sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kC0Bits);
    } else {
        sfpi::vConstIntPrgm0 = Config::kMagic;
    }
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kC1Bits);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kC2Bits);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

// The selected rsqrt descriptor's one-step model (newton_root_iters = 1): the first
// correction y*t is already correctly rounded in BF16, so the rows end there. Negative
// inputs take NaN; then +-0 and +Inf take inf_bits - x_bits (+-Inf, 0), which also
// returns -0's -Inf where it reaches the SFPU signed. The zero test reads |x|, as
// SFPSETCC treats the raw -0 word as nonzero.
constexpr uint32_t kNewtonRsqrtCompactSlots = 22u;

inline void newton_rsqrt_compact_row() {
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);                                   // x
    TTI_SFPSHFT(0xFFF, p_sfpu::LREG1, p_sfpu::LREG0, 5);                            // bits(x) >> 1
    TTI_SFPIADD(0, p_sfpu::LREG12, p_sfpu::LREG0, 6);                               // y = MAGIC - i
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);   // xy
    TTI_SFPLOADI(p_sfpu::LREG3, 0, 0x7F80);                                         // +Inf
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 1);   // c = -(y*xy)
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG4, 2);                                 // x bits
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG14, p_sfpu::LREG2, p_sfpu::LREG5, 0);  // C2 + c
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG4, 6);                                // inf_bits - x_bits
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG13, p_sfpu::LREG2, 0);     // t = c*(C2+c) + C1
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG5, 1);                                 // |x|
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);   // y *= t
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);                // x < 0
    TTI_SFPLOADI(p_sfpu::LREG0, 0, 0x7FC0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);  // x != +Inf
    TTI_SFPSETCC(0, p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);  // x != +-0
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // +-Inf at +-0, 0 at +Inf
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);      // fp32 -> bf16 RNE
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_6, 0);  // dest += 2
}

template <typename Config, int Iterations>
inline void calculate_newton_root() {
    static_assert(Config::kKind <= 2);
    static_assert(Iterations == 8 || Iterations == 32);
    static_assert(Config::kKind != 0 || Config::kNegativeExponentZeroOutput == 0x7f80u);
    static_assert(
        Config::kBodySlots == (Config::kKind == 0   ? 32u
                               : Config::kKind == 1 ? kNewtonRsqrtCompactSlots
                                                    : 23u));
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
        newton_rsqrt_compact_row();
    } else {
        sfpi::newton_cbrt_body<ADDR_MOD_7, ADDR_MOD_6>();
    }
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16
