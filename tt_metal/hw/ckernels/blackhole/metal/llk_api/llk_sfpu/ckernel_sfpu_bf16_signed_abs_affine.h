// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Included after polynomial_horner.h and the target's existing LLK headers.
// Config carries the selected coefficient/terminal words. Callers own traversal.
namespace sfpi {
template <typename Config>
constexpr bool signed_abs_affine_replay_contract() {
    return Config::kDegree == 8u && Config::kBodySlots == 32u && Config::kCoefficientBits[0] == 0u &&
           Config::kCoefficientBits[1] == 0u && Config::kCoefficientBits[2] == 0u &&
           Config::kScaleBits == 0x3f800000u && Config::kBiasBits == 0xbf800000u &&
           Config::kLeftBiasBits == 0x3f800000u && Config::kRawEqualValue == 0x80ffu &&
           Config::kRepairedPatterns == 127u && Config::kMacroSequenceBits == 0x13850000u;
}

template <typename Config, uint32_t J>
constexpr uint32_t signed_abs_affine_addend_reg() {
    if constexpr (Config::kCoefficientBits[6u - J] == 0u) {
        return 9u;
    } else if constexpr (J == 0u) {
        return 12u;
    } else {
        return 8u - J;
    }
}

template <typename Config>
inline void signed_abs_affine_init() {
    static_assert(signed_abs_affine_replay_contract<Config>());
    vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kCoefficientBits[8]);
    vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kCoefficientBits[7]);
    vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kCoefficientBits[6]);
    ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ckernel::ADDR_MOD_6);
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, ckernel::p_sfpu::LREG0, 13, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, SFPLOADI_MOD0_UPPER, Config::kMacroSequenceBits >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, SFPLOADI_MOD0_LOWER, Config::kMacroSequenceBits & 0xffffu);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
    TTI_SFPNOP;
}

template <typename Config, uint32_t J>
inline void signed_abs_affine_pin() {
    if constexpr (signed_abs_affine_addend_reg<Config, J>() == 8u - J) {
        TTI_SFPLOADI(8u - J, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[6u - J] >> 16);
        TTI_SFPLOADI(8u - J, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[6u - J] & 0xffffu);
    }
}

template <typename Config>
inline void signed_abs_affine_pin_coefficients() {
    signed_abs_affine_pin<Config, 1>();
    signed_abs_affine_pin<Config, 2>();
    signed_abs_affine_pin<Config, 3>();
}

template <typename Config, uint32_t J>
inline void signed_abs_affine_rung() {
    if constexpr (J == 0u) {
        TTI_SFPXOR(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, 0);
    } else if constexpr (J == 1u) {
        TTI_SFPAND(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG3, 1);
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        (signed_abs_affine_addend_reg<Config, J>()),
        ckernel::p_sfpu::LREG2,
        0);
}

template <typename Config, uint32_t Hold, uint32_t Advance>
inline void signed_abs_affine_body() {
    static_assert(signed_abs_affine_replay_contract<Config>());
    TTI_SFPLOAD(ckernel::p_sfpu::LREG4, SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, SFPLOADI_MOD0_UPPER, Config::kBoundBits >> 16);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 1);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, SFPLOADI_MOD0_LOWER, Config::kBoundBits & 0xffffu);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 1);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, SFPLOADI_MOD0_USHORT, Config::kRawEqualValue);
    TTI_SFPMAD(ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG2, 0);
    signed_abs_affine_rung<Config, 0>();
    signed_abs_affine_rung<Config, 1>();
    signed_abs_affine_rung<Config, 2>();
    signed_abs_affine_rung<Config, 3>();
    signed_abs_affine_rung<Config, 4>();
    signed_abs_affine_rung<Config, 5>();
    signed_abs_affine_rung<Config, 6>();
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 1);
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, 9);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0, SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 1);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPLOADMACRO(2, 0, Advance, 0);
}

inline void signed_abs_affine_drain() {
    // Preserve the canonical five issued slots before counter/commit changes.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
}  // namespace sfpi
