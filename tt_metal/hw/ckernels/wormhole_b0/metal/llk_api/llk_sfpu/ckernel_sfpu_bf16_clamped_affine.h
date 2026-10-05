// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_relu.h"

namespace ckernel::sfpu::bf16 {
constexpr uint32_t kClampHold = ADDR_MOD_3;
constexpr uint32_t kClampAdvance = ADDR_MOD_2;

template <typename Config>
inline void init_clamped_affine() {
    if constexpr (Config::kRoute == 1u || Config::kRoute == 2u) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    }
    if constexpr (Config::kRoute == 1u) {
        // These ordinary registers must be pinned before recording, as in the
        // selected paired MAD body. They are not clamp operands (SWAP writes both).
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 0xffffu);
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, Config::kInterceptBits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, Config::kInterceptBits & 0xffffu);
    }
}

template <typename Config>
inline void clamped_affine_terminal_pair() {
    TTI_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_UINT16, kClampHold, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_UINT16, kClampHold, 2);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, Config::kTerminalEqual);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG5, 0);
    if constexpr (Config::kTerminalNonzero == 0u) {
        TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kTerminalOutput);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, Config::kTerminalOutput);
        TTI_SFPENCC(0, 0, 0, 0);
    } else {
        TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG1, 1);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kTerminalOutput);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG1, 1);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, Config::kTerminalOutput);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

template <typename Config, int Iterations>
inline void clamped_affine_pairs() {
    static_assert(Iterations % 2 == 0);
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    if constexpr (Config::kRoute == 1u) {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, kClampHold, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, 0, kClampHold, 2);
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG3, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 9);
        TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 9);
    } else {
        TTI_SFPLOAD(p_sfpu::LREG2, 0, kClampHold, 0);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, kClampHold, 2);
        TTI_SFPSWAP(0, p_sfpu::LCONST_neg1, p_sfpu::LREG2, 9);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, p_sfpu::LCONST_neg1, p_sfpu::LREG3, 9);
        TTI_SFPNOP;
    }
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG2, 1);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, 1);
    TTI_SFPNOP;
    if constexpr (Config::kHasTerminal) {
        clamped_affine_terminal_pair<Config>();
    }
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG2,
        p_sfpu::LREG2,
        p_sfpu::LREG2,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG3,
        p_sfpu::LREG3,
        p_sfpu::LREG3,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, kClampHold, 0);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, kClampAdvance, 2);
#pragma GCC unroll 4
    for (int pair = 1; pair < Iterations / 2; ++pair) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}

template <typename Config, int Iterations = 8>
inline void calculate_clamped_affine() {
    if constexpr (Config::kRoute == 0u) {
        // The selected lower-only identity shape owns the native callback.
        // Its surrounding upstream wrapper retains destination/face lifecycle.
        ckernel::sfpu::_relu_min_<sfpi::vFloat, false, Iterations, uint32_t>(0u);
    } else if constexpr (Config::kRoute == 1u || Config::kRoute == 2u) {
        clamped_affine_pairs<Config, Iterations>();
    } else {
#pragma GCC unroll 8
        for (int row = 0; row < Iterations; ++row) {
            // S60 retains only raw constants not already proved by this
            // target's decoded affine/clamp/round sequence.
            sfpi::vUInt raw;
            if constexpr (Config::kPosNanConstant || Config::kNegNanConstant) {
                raw = sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>();
            }
            sfpi::vFloat x = sfpi::dst_reg[0];
            constexpr float c0 = __builtin_bit_cast(float, Config::kInterceptBits);
            constexpr float c1 = __builtin_bit_cast(float, Config::kSlopeBits);
            sfpi::vFloat y = c1 * x + c0;
            if constexpr (Config::kHasLower) {
                sfpi::vFloat lower = __builtin_bit_cast(float, Config::kLowerBits);
                auto ordered = sfpi::min_max(lower, y);
                y = ordered.second;
            }
            if constexpr (Config::kHasUpper) {
                sfpi::vFloat upper = __builtin_bit_cast(float, Config::kUpperBits);
                auto ordered = sfpi::min_max(y, upper);
                y = ordered.first;
            }
            if constexpr (Config::kPosNanConstant || Config::kNegNanConstant) {
                v_if((raw & 0x7f00u) != 0u) {
                    if constexpr (Config::kPosNanConstant) {
                        v_if((raw & 0x80ffu) == 0x00ffu) { y = __builtin_bit_cast(float, Config::kPosNanBits); }
                        v_endif;
                    }
                    if constexpr (Config::kNegNanConstant) {
                        v_if((raw & 0x80ffu) == 0x80ffu) { y = __builtin_bit_cast(float, Config::kNegNanBits); }
                        v_endif;
                    }
                }
                v_endif;
            }
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = y;
            sfpi::dst_reg++;
        }
    }
}
}  // namespace ckernel::sfpu::bf16
