// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_softplus.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

constexpr auto sdpa_bits = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto sdpa_lo16 = [](float x) constexpr { return static_cast<std::uint16_t>(sdpa_bits(x) & 0xFFFFu); };
constexpr auto sdpa_hi16 = [](float x) constexpr { return static_cast<std::uint16_t>(sdpa_bits(x) >> 16); };

constexpr auto sdpa_addr_mod_x = ADDR_MOD_7;

ALWI void sdpa_insert_sfpnop() {}

template <bool USE_SFPARECIP_INSTR, int POLY_DEGREE>
constexpr bool sdpa_can_preload_ln2_constants() {
    return (USE_SFPARECIP_INSTR || POLY_DEGREE == 1 || POLY_DEGREE == 2);
}

// Compute the small online-softmax correction accurately with the full scale.
// The sum update subsequently forms correction - 1 for the L1 numerator update.
template <uint32_t scale_fp32>
inline void calculate_sdpa_exp_correction() {
    constexpr float scale = __builtin_bit_cast(float, scale_fp32);
    for (int d = 0; d < 4; ++d) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _sfpu_exp_fp32_accurate_(x * scale);
        sfpi::dst_reg += 2;
    }
}

// Investigation-only exact-exp and full-FP32 score-subtraction controls.
template <uint32_t scale_fp32, int iterations>
inline void calculate_sdpa_diag_exp() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float scale = __builtin_bit_cast(float, scale_fp32);
    for (int i = 0; i < iterations; ++i) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _sfpu_exp_fp32_accurate_(x * scale);
        sfpi::dst_reg++;
    }
}

template <uint32_t scale_fp32, bool apply_exp, int tiles = 1>
inline void calculate_sdpa_diag_sub() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float scale = __builtin_bit_cast(float, scale_fp32);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat maximum = sfpi::dst_reg[32 * tiles];
#pragma GCC unroll 2
        for (int tile = 0; tile < tiles; ++tile) {
            sfpi::vFloat score = sfpi::dst_reg[32 * tile];
            sfpi::vFloat x = score - maximum;
            if constexpr (apply_exp) {
                sfpi::dst_reg[32 * tile] = _sfpu_exp_fp32_accurate_(x * scale);
            } else {
                sfpi::dst_reg[32 * tile] = x;
            }
        }
        sfpi::dst_reg++;
    }
}

// Experimental negative-logit grid: trade exponent range for 10 fraction bits
// in the fast-exp macro's signed-magnitude INT16 encoding. Values below about
// -21.45 after scaling are zeroed by packer ReLU. This is not a general exp.
template <uint32_t scale_fp32>
inline void init_sdpa_exp_grid() {
    constexpr float a = 1024.0f * 1.4426950408889634f * __builtin_bit_cast(float, scale_fp32);
    constexpr float b = 31.0f * 1024.0f - 4.0f * (32512.0f - 32500.818359375f);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(a));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(a));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 0xA, 13);
    TTI_SFPLOADI(0, 0x8, 0);
    TTI_SFPCONFIG(0, 14, 0);
}

// Investigation-only fused two-score FP32 subtraction/grid/refinement. Keep
// the subtraction separate from the grid MAD, matching mode 4's FP32 rounding.
inline void calculate_sdpa_negate_max() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat negative = -sfpi::dst_reg[0];
        sfpi::dst_reg[0] = negative;
#ifdef SDPA_FP32_L1_REPEAT
        sfpi::dst_reg[32] = negative;
        sfpi::dst_reg[64] = negative;
        sfpi::dst_reg[96] = negative;
#endif
        sfpi::dst_reg++;
    }
}

template <uint32_t scale_fp32, bool init_only = false, bool reuse = false>
inline void calculate_sdpa_fused_sub_exp() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
#ifdef SDPA_FP32_EXTRA_CONST
#ifdef SDPA_FP32_L1_SUB
    constexpr uint32_t replay_length = 22;
#else
    constexpr uint32_t replay_length = 25;
#endif
    if constexpr (!init_only) {
        // L11 is programmable but SFPI reserves it for -1. This raw-instruction
        // body temporarily borrows it and restores -1 before returning.
        constexpr float grid_bias = 31.0f * 1024.0f - 4.0f * (32512.0f - 32500.818359375f);
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(grid_bias));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(grid_bias));
        TTI_SFPCONFIG(0, 11, 0);
    }
#else
    constexpr uint32_t replay_length = 27;
#endif
    if constexpr (!reuse) {
        constexpr float grid_a = 1024.0f * 1.4426950408889634f * __builtin_bit_cast(float, scale_fp32);
        constexpr float grid_b = 31.0f * 1024.0f - 4.0f * (32512.0f - 32500.818359375f);
        constexpr float a = -8.01081703839006e27f;
        constexpr float b = 5.39213985142374e28f;
        constexpr float c = -1.05653430577460e29f;
        constexpr float d = 1.38859037541378e29f;
        TTI_SFPLOADI(6, 0xA, sdpa_lo16(a));
        TTI_SFPLOADI(6, 0x8, sdpa_hi16(a));
        TTI_SFPLOADI(7, 0xA, sdpa_lo16(grid_a));
        TTI_SFPLOADI(7, 0x8, sdpa_hi16(grid_a));
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
        TTI_SFPCONFIG(0, 12, 0);
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(c));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(c));
        TTI_SFPCONFIG(0, 13, 0);
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(d));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(d));
        TTI_SFPCONFIG(0, 14, 0);
        // Each replay handles corresponding vectors from both score tiles and
        // their common maximum. The last store alone advances the DST address.
        TTI_REPLAY(0, replay_length, init_only ? 0 : 1, 1);
#ifndef SDPA_FP32_L1_SUB
        TTI_SFPLOAD(2, 0, ADDR_MOD_6, 128);
#endif
        TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
        TTI_SFPLOAD(1, 0, ADDR_MOD_6, 64);
#ifndef SDPA_FP32_L1_SUB
        TTI_SFPADD(10, 0, 2, 0, 2);
        TTI_SFPADD(10, 1, 2, 1, 2);
#endif
#ifdef SDPA_FP32_EXTRA_CONST
        TTI_SFPMAD(7, 0, 11, 0, 0);
        TTI_SFPMAD(7, 1, 11, 1, 0);
#else
        TTI_SFPLOADI(2, 0xA, sdpa_lo16(grid_b));
        TTI_SFPLOADI(2, 0x8, sdpa_hi16(grid_b));
        TTI_SFPMAD(7, 0, 2, 0, 0);
        TTI_SFPMAD(7, 1, 2, 1, 0);
#endif
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 7);
        TTI_SFP_STOCH_RND(0, 0, 1, 1, 1, 7);
        TTI_SFPSHFT(13, 0, 4, 5);
        TTI_SFPSHFT(13, 1, 5, 5);
        TTI_SFPSETSGN(0, 4, 0, 0);
        TTI_SFPSETSGN(1, 5, 1, 0);
        TTI_SFPSETEXP(127, 0, 2, 1);
        TTI_SFPSETEXP(127, 1, 3, 1);
        TTI_SFPMAD(2, 6, 12, 4, 0);
        TTI_SFPMAD(3, 6, 12, 5, 0);
        TTI_SFPMAD(2, 4, 13, 4, 0);
        TTI_SFPMAD(3, 5, 13, 5, 0);
        TTI_SFPMAD(2, 4, 14, 4, 0);
        TTI_SFPMAD(3, 5, 14, 5, 0);
        TTI_SFPMUL(0, 4, p_sfpu::LCONST_0, 0, 0);
        TTI_SFPMUL(1, 5, p_sfpu::LCONST_0, 1, 0);
        TTI_SFPSTORE(0, 0, ADDR_MOD_6, 0);
        TTI_SFPSTORE(1, 0, ADDR_MOD_7, 64);
    }
    if constexpr (!init_only) {
#pragma GCC unroll 8
        for (int i = reuse ? 0 : 1; i < 32; ++i) {
            lltt::replay(0, replay_length);
        }
#ifdef SDPA_FP32_EXTRA_CONST
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(-1.0f));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(-1.0f));
        TTI_SFPCONFIG(0, 11, 0);
#endif
    }
}

template <int pairs>
inline void calculate_sdpa_fp32_update() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat correction = sfpi::dst_reg[32 * pairs];
#pragma GCC unroll 2
        for (int p = 0; p < pairs; ++p) {
            sfpi::vFloat old = sfpi::dst_reg[32 * p];
            sfpi::dst_reg[32 * p] = old * correction;
        }
        sfpi::dst_reg++;
    }
}

inline void calculate_sdpa_fp32_update_first_col() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 4; ++i) {
        sfpi::vFloat old = sfpi::dst_reg[0];
        sfpi::vFloat correction = sfpi::dst_reg[32];
        sfpi::dst_reg[0] = old * correction;
        sfpi::dst_reg += 2;
    }
}

inline void calculate_sdpa_fp32_normalize() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat out = sfpi::dst_reg[0];
        sfpi::vFloat inv = sfpi::dst_reg[32];
        sfpi::dst_reg[0] = out * inv;
        sfpi::dst_reg++;
    }
}

inline void calculate_sdpa_fp32_recip() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    sfpi::vConstFloatPrgm0 = 2.0f;
    for (int d = 0; d < 4; ++d) {
        sfpi::dst_reg[0] = sfpu_reciprocal_iter<2>(sfpi::dst_reg[0]);
        sfpi::dst_reg += 2;
    }
}

template <int iterations>
inline void calculate_sdpa_exp_hifi2() {
    // Fast-exp leaves an auto-incrementing address modifier; SFPI advances explicitly.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    // Least-squares cubic for 2^(m-1), m in [1,2]. A common factor ensures p >= 1
    // on this discrete grid, as setexp replaces rather than adds the exponent.
    // This common factor cancels between softmax numerator and denominator.
    const sfpi::vFloat a = 0.07901999f * 1.0002f;
    // Programmable constants leave enough working registers for two independent
    // Horner chains. They overlap the fast macro's grid constants, which the
    // caller must restore before every subsequent exp tile.
    sfpi::vConstFloatPrgm0 = -0.01293332f * 1.0002f;
    sfpi::vConstFloatPrgm1 = 0.48564498f * 1.0002f;
    sfpi::vConstFloatPrgm2 = 0.44808037f * 1.0002f;
#pragma GCC unroll 4
    for (int i = 0; i < iterations; i += 2) {
        sfpi::vFloat linear = sfpi::dst_reg[0];
        sfpi::vFloat linear1 = sfpi::dst_reg[1];
        sfpi::vFloat m = sfpi::setexp(linear, 127);
        sfpi::vFloat m1 = sfpi::setexp(linear1, 127);
        sfpi::vFloat p = m * a + sfpi::vConstFloatPrgm0;
        sfpi::vFloat p1 = m1 * a + sfpi::vConstFloatPrgm0;
        p = m * p + sfpi::vConstFloatPrgm1;
        p1 = m1 * p1 + sfpi::vConstFloatPrgm1;
        p = m * p + sfpi::vConstFloatPrgm2;
        p1 = m1 * p1 + sfpi::vConstFloatPrgm2;
        sfpi::vInt exponent = sfpi::exexp(linear, sfpi::ExponentMode::Biased);
        // Restore the exponent bias removed to fit the wider grid in INT16.
        sfpi::vFloat y = sfpi::setexp(p, exponent + 96);
        // HiFi2 PV sees only six fraction bits of SrcB. Round to that grid before
        // both summing the weights and multiplying V, so the two paths agree.
        sfpi::vInt bits = sfpi::as<sfpi::vInt>(y) >> 1;
        y = sfpi::convert<sfpi::vFloat16b>(sfpi::as<sfpi::vFloat>(bits), sfpi::RoundMode::Nearest);
        y = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(y) << 1);
        sfpi::dst_reg[0] = sfpi::setsgn(y, linear);
        sfpi::vInt exponent1 = sfpi::exexp(linear1, sfpi::ExponentMode::Biased);
        sfpi::vFloat y1 = sfpi::setexp(p1, exponent1 + 96);
        sfpi::vInt bits1 = sfpi::as<sfpi::vInt>(y1) >> 1;
        y1 = sfpi::convert<sfpi::vFloat16b>(sfpi::as<sfpi::vFloat>(bits1), sfpi::RoundMode::Nearest);
        y1 = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(y1) << 1);
        sfpi::dst_reg[1] = sfpi::setsgn(y1, linear1);
        sfpi::dst_reg += 2;
    }
}

template <int iterations>
inline void calculate_sdpa_exp_grid_batch() {
    static_assert(iterations % 4 == 0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
#pragma GCC unroll 8
    for (int i = 0; i < iterations / 4; ++i) {
        lltt::replay(0, 8);
    }
    TTI_SFPNOP;
    TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
    TTI_SFPNOP;
    TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <int iterations>
inline void calculate_sdpa_exp_stream_effective() {
    static_assert(iterations % 2 == 0);
    // Replay slots 0..7 retain the fast-exp macro's repeating four-vector pattern.
    // Slots 8..21 hold the two-chain refiner. Only the second store advances DST.
    // Fit (0.995 * 2^(m-1) + 1/128) / m on [1,2], with 2^96 folded
    // into the coefficients. The linear grid carries the exponent. PV and the
    // LoFi denominator consume the same six-bit effective SrcB weights.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);
#if defined(SDPA_DIAG_EXP_MODE) && (SDPA_DIAG_EXP_MODE == 1 || SDPA_DIAG_EXP_MODE == 4)
    // Relative fit to exp2(m-1), without HiFi2's six-bit rounding bias.
    constexpr float a = -8.01081703839006e27f;
    constexpr float b = 5.39213985142374e28f;
    constexpr float c = -1.05653430577460e29f;
    constexpr float d = 1.38859037541378e29f;
#else
    constexpr float a = -8.106702647031e27f;
    constexpr float b = 5.446591617222e28f;
    constexpr float c = -1.069245718291e29f;
    constexpr float d = 1.399030410717e29f;
#endif
    TTI_SFPLOADI(6, 0xA, sdpa_lo16(a));
    TTI_SFPLOADI(6, 0x8, sdpa_hi16(a));
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(c));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(c));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(d));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(d));
    TTI_SFPCONFIG(0, 14, 0);
    TTI_REPLAY(8, 14, 1, 1);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(1, 0, ADDR_MOD_6, 2);
    TTI_SFPSETEXP(127, 0, 2, 1);
    TTI_SFPSETEXP(127, 1, 3, 1);
    TTI_SFPMAD(2, 6, 12, 4, 0);
    TTI_SFPMAD(3, 6, 12, 5, 0);
    TTI_SFPMAD(2, 4, 13, 4, 0);
    TTI_SFPMAD(3, 5, 13, 5, 0);
    TTI_SFPMAD(2, 4, 14, 4, 0);
    TTI_SFPMAD(3, 5, 14, 5, 0);
    TTI_SFPMUL(0, 4, p_sfpu::LCONST_0, 0, 0);
    TTI_SFPMUL(1, 5, p_sfpu::LCONST_0, 1, 0);
    TTI_SFPSTORE(0, 0, ADDR_MOD_6, 0);
    TTI_SFPSTORE(1, 0, ADDR_MOD_7, 2);
#pragma GCC unroll 8
    for (int i = 2; i < iterations; i += 2) {
        lltt::replay(8, 14);
    }
}

inline void init_sdpa_refine_loadmacros() {
    // Slot 0 is unused by the unclamped exp grid. SETEXP reads VC, so its
    // sequence must override VC (not VB) with the loaded register.
    TTI_SFPSETEXP(127, 0, 12, 1);
    TTI_SFPLOADI(0, 0xA, 0x0004);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 5, 0);
    // Final load/multiply/store: polynomial is in L4/L5. The macro uses the
    // loaded linear grid value as VB and captures its DST address for store.
    TTI_SFPLOADI(0, 0xA, 0x8500);
    TTI_SFPLOADI(0, 0x8, 0x1300);
    TTI_SFPCONFIG(0, 6, 0);
    TTI_SFPLOADI(0, 0xA, 0x8600);
    TTI_SFPLOADI(0, 0x8, 0x1300);
    TTI_SFPCONFIG(0, 7, 0);
}

inline void restore_sdpa_grid_macro_instructions() {
    TTI_SFPMAD(12, 0, 13, 13, 0);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 7);
}

template <int iterations>
inline void calculate_sdpa_exp_refine_loadmacro() {
    static_assert(iterations % 2 == 0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);
    constexpr float a = -8.01081703839006e27f;
    constexpr float b = 5.39213985142374e28f;
    constexpr float c = -1.05653430577460e29f;
    constexpr float d = 1.38859037541378e29f;
    TTI_SFPLOADI(6, 0xA, sdpa_lo16(a));
    TTI_SFPLOADI(6, 0x8, sdpa_hi16(a));
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(c));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(c));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(d));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(d));
    TTI_SFPCONFIG(0, 14, 0);
    TTI_SFPMUL(4, 0, p_sfpu::LCONST_0, 13, 0);
    TTI_SFPMUL(5, 0, p_sfpu::LCONST_0, 14, 0);
    TTI_REPLAY(8, 10, 1, 1);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(7, 0, ADDR_MOD_6, 2);
    TTI_SFPMAD(2, 6, 12, 4, 0);
    TTI_SFPMAD(3, 6, 12, 5, 0);
    TTI_SFPMAD(2, 4, 13, 4, 0);
    TTI_SFPMAD(3, 5, 13, 5, 0);
    TTI_SFPMAD(2, 4, 14, 4, 0);
    TTI_SFPMAD(3, 5, 14, 5, 0);
    TTI_SFPLOADMACRO(8, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(13, 0, ADDR_MOD_7, 2);
#pragma GCC unroll 8
    for (int i = 2; i < iterations; i += 2) {
        lltt::replay(8, 10);
    }
    // UnitDelayKind is elapsed SFPU instructions, as configured by exp init.
    // Retire the final two scheduled multiplies/stores before reconfiguration.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Approximate exp and add a half-ULP bias for HiFi2's six-bit SrcB grid.
// The caller MUST form the denominator from these same effective FPU weights.
template <int iterations>
inline void calculate_sdpa_exp_hifi2_effective() {
    // Fast-exp leaves an auto-incrementing address modifier; SFPI advances explicitly.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    // Relative least-squares fit of (0.995 * 2^(m-1) + 1/128) / m on [1,2].
    // Fold 2^96 into the coefficients and multiply by the fast macro's linear
    // result, preserving its exponent without EXEXP/IADD/SETEXP. The common
    // factor cancels in normalization. Negative underflow encodings remain
    // negative (the polynomial is positive on [-2,-1]) and packer ReLU zeros them.
    const sfpi::vFloat a = -8.106702647031e27f;
    // Programmable constants leave enough working registers for two independent
    // Horner chains. They overlap the fast macro's grid constants, which the
    // caller must restore before every subsequent exp tile.
    sfpi::vConstFloatPrgm0 = 5.446591617222e28f;
    sfpi::vConstFloatPrgm1 = -1.069245718291e29f;
    sfpi::vConstFloatPrgm2 = 1.399030410717e29f;
#pragma GCC unroll 4
    for (int i = 0; i < iterations; i += 2) {
        sfpi::vFloat linear = sfpi::dst_reg[0];
        sfpi::vFloat linear1 = sfpi::dst_reg[1];
        sfpi::vFloat m = sfpi::setexp(linear, 127);
        sfpi::vFloat m1 = sfpi::setexp(linear1, 127);
        sfpi::vFloat p = m * a + sfpi::vConstFloatPrgm0;
        sfpi::vFloat p1 = m1 * a + sfpi::vConstFloatPrgm0;
        p = m * p + sfpi::vConstFloatPrgm1;
        p1 = m1 * p1 + sfpi::vConstFloatPrgm1;
        p = m * p + sfpi::vConstFloatPrgm2;
        p1 = m1 * p1 + sfpi::vConstFloatPrgm2;
        sfpi::dst_reg[0] = linear * p;
        sfpi::dst_reg[1] = linear1 * p1;
        sfpi::dst_reg += 2;
    }
}

inline void calculate_sdpa_scale_sum_and_delta() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int d = 0; d < 4; ++d) {
        sfpi::vFloat sum = sfpi::dst_reg[0];
        sfpi::vFloat correction = sfpi::dst_reg[32];
        sfpi::dst_reg[0] = sum * correction;
        sfpi::dst_reg[32] = correction - 1.0f;
        sfpi::dst_reg += 2;
    }
}

// Each state is hi + lo, with both parts stored as BF16. Keep the online
// rescale/add in SFPU FP32 registers and re-split only at the store boundary.
// Inputs occupy triples (hi, lo, new_chunk), followed by one shared correction
// tile; up to two triples fit in the BF16 half-DST (seven of eight tiles).
inline void init_sdpa_compensated_state_macros() {
    // Round VC=L0/L3 into the load's register, then store at its captured
    // address. Override VB only: the rounding instruction reads fixed VC.
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 12, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(0, 0, 3, 3, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, 0xA, 0x0000);
    TTI_SFPLOADI(0, 0x8, 0x1384);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, 0x8, 0x1385);
    TTI_SFPCONFIG(0, 5, 0);
    TTI_SFPADD(10, 0, 1, 14, 0);
    TTI_SFPADD(10, 3, 2, 15, 0);
    TTI_SFPCONFIG(0x600, 6, 1);
    TTI_SFPCONFIG(0x700, 7, 1);
    TTI_SFPCONFIG(0xF00, 8, 1);
}

template <int pairs>
inline void calculate_sdpa_compensated_state_round_macro() {
    static_assert(pairs == 2);
    // This PACK-thread phase follows all logit exp tiles. The next K chunk
    // restores fast-exp's replay program before using it again.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
    constexpr uint32_t body_len = 15;
    // Record and execute the first vector; the final store advances DST.
    // Interleave the independent pairs to hide SFPMAD's dependent-use latency.
    // Preserve the previous SFPI arithmetic and nearest conversion instructions:
    // sum = (hi + lo) * correction + chunk; hi = bf16(sum); lo = bf16(sum - hi).
    // L0/L1/L2 and L3/L4/L5 hold the pairs; L6 holds their shared correction.
    // body_len must count every recorded instruction and fit the 32-entry buffer.
    TTI_REPLAY(0, body_len, 1, 1);
    TTI_SFPLOAD(6, 0, ADDR_MOD_6, 192 * pairs);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(3, 0, ADDR_MOD_6, 192);
    TTI_SFPLOADMACRO(9, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(14, 0, ADDR_MOD_6, 256);
    TTI_SFPLOAD(4, 0, ADDR_MOD_6, 128);
    TTI_SFPLOAD(5, 0, ADDR_MOD_6, 320);
    TTI_SFPMAD(1, 6, 4, 0, 0);
    if constexpr (pairs == 2) {
        TTI_SFPMAD(2, 6, 5, 3, 0);
    }
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 0);
    if constexpr (pairs == 2) {
        TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 192);
    }
    TTI_SFPADD(10, 0, 1, 0, 2);
    if constexpr (pairs == 2) {
        TTI_SFPADD(10, 3, 2, 3, 2);
    }
    TTI_SFPLOADMACRO(1, 0, pairs == 2 ? ADDR_MOD_6 : ADDR_MOD_7, 64);
    if constexpr (pairs == 2) {
        TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 256);
    }
#pragma GCC unroll 8
    for (int i = 1; i < 32; ++i) {
        TTI_REPLAY(0, body_len, 0, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <int pairs>
inline void calculate_sdpa_compensated_state() {
    static_assert(pairs == 1 || pairs == 2);
    if constexpr (pairs == 2) {
        calculate_sdpa_compensated_state_round_macro<pairs>();
        return;
    }
    // This PACK-thread phase follows all logit exp tiles. The next K chunk
    // restores fast-exp's replay program before using it again.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
    constexpr uint32_t body_len = pairs == 2 ? 21 : 11;
    // Record and execute the first vector; the final store advances DST.
    // Interleave the independent pairs to hide SFPMAD's dependent-use latency.
    // Preserve the previous SFPI arithmetic and nearest conversion instructions:
    // sum = (hi + lo) * correction + chunk; hi = bf16(sum); lo = bf16(sum - hi).
    // L0/L1/L2 and L3/L4/L5 hold the pairs; L6 holds their shared correction.
    // body_len must count every recorded instruction and fit the 32-entry buffer.
    TTI_REPLAY(0, body_len, 1, 1);
    TTI_SFPLOAD(6, 0, ADDR_MOD_6, 192 * pairs);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(1, 0, ADDR_MOD_6, 64);
    TTI_SFPLOAD(2, 0, ADDR_MOD_6, 128);
    if constexpr (pairs == 2) {
        TTI_SFPLOAD(3, 0, ADDR_MOD_6, 192);
        TTI_SFPLOAD(4, 0, ADDR_MOD_6, 256);
        TTI_SFPLOAD(5, 0, ADDR_MOD_6, 320);
    }
    TTI_SFPADD(10, 0, 1, 0, 0);
    if constexpr (pairs == 2) {
        TTI_SFPADD(10, 3, 4, 3, 0);
    }
    TTI_SFPMAD(0, 6, 2, 0, 0);
    if constexpr (pairs == 2) {
        TTI_SFPMAD(3, 6, 5, 3, 0);
    }
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    if constexpr (pairs == 2) {
        TTI_SFP_STOCH_RND(0, 0, 3, 3, 4, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    }
    TTI_SFPADD(10, 0, 1, 0, 2);
    if constexpr (pairs == 2) {
        TTI_SFPADD(10, 3, 4, 3, 2);
    }
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    if constexpr (pairs == 2) {
        TTI_SFP_STOCH_RND(0, 0, 3, 3, 3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    }
    TTI_SFPSTORE(1, 0, ADDR_MOD_6, 0);
    if constexpr (pairs == 2) {
        TTI_SFPSTORE(4, 0, ADDR_MOD_6, 192);
        TTI_SFPSTORE(0, 0, ADDR_MOD_6, 64);
        TTI_SFPSTORE(3, 0, ADDR_MOD_7, 256);
    } else {
        TTI_SFPSTORE(0, 0, ADDR_MOD_7, 64);
    }
#pragma GCC unroll 8
    for (int i = 1; i < 32; ++i) {
        TTI_REPLAY(0, body_len, 0, 0);
    }
}

inline void calculate_sdpa_zero_sum() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg++;
    }
}

template <bool legacy_compat, bool is_fp32_dest_acc_en>
inline void calculate_recip_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (legacy_compat) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat out = ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);
            if constexpr (!(is_fp32_dest_acc_en || APPROX)) {
                out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = out;
            sfpi::dst_reg += 2;
        }
    } else {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat out;

            if constexpr (is_fp32_dest_acc_en) {
                // The final normalization needs more than the hardware estimate,
                // even when approximate math is enabled for the matmuls.
                out = ckernel::sfpu::sfpu_reciprocal_iter<2>(in);
            } else if constexpr (APPROX) {
                out = ckernel::sfpu::sfpu_reciprocal_iter<0>(in);
            } else {
                out = ckernel::sfpu::sfpu_reciprocal_iter<1>(in);
                out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = out;
            sfpi::dst_reg += 2;
        }
    }
}

template <
    bool SCALE_EN,
    int ITERATIONS,
    bool USE_SFPARECIP_INSTR,
    int POLY_DEGREE,
    bool IS_FP32_DEST_ACC_EN,
    uint16_t SCALE_BF16>
inline void calculate_exponential_polynomial() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(sdpa_addr_mod_x);

    constexpr float LN2_RECIP = 1.44269504088896340736f;
    constexpr float M_LN2 = -0.69314718055994530942f;

    if constexpr (!USE_SFPARECIP_INSTR) {
        static_assert(POLY_DEGREE >= 1 && POLY_DEGREE <= 4);

        constexpr float c0 = (POLY_DEGREE == 1)   ? 1.03022936050163882354355235184958220293399209290987f
                             : (POLY_DEGREE == 2) ? 0.999848792924395313327307061545061386175496934006f
                             : (POLY_DEGREE == 3) ? 0.99992449655091231753798502608929170703152709521188f
                                                  : 1.0000001510806179002040134468008959160576106495165f;
        constexpr float c1 = (POLY_DEGREE == 1)   ? 1.0201394465967894800285756834161653337107187804001f
                             : (POLY_DEGREE == 2) ? 1.01508760098521056684783640695492761469306929535975f
                             : (POLY_DEGREE == 3) ? 0.99993960415029750534472970577402987498389428593233f
                                                  : 0.99996228117047652035114096488703457970402030983204f;
        constexpr float c2 = (POLY_DEGREE == 2)   ? 0.50628367056745568861842335616023694454759126020461f
                             : (POLY_DEGREE == 3) ? 0.50502329058055065591138054839814880512001604099324f
                                                  : 0.49998365704615426417337683145647067790385638465486f;
        constexpr float c3 = (POLY_DEGREE == 3) ? 0.16817330195731531429790827442800245470170482723302f
                                                : 0.16792157982882225102649214918047336097544632172075f;
        constexpr float c4 = 4.1959439860014343843000081999668024587178974865521e-2;

        if constexpr (POLY_DEGREE >= 4) {
            TTI_SFPLOADI(p_sfpu::LREG3, 0xA, sdpa_lo16(c4));
            TTI_SFPLOADI(p_sfpu::LREG3, 0x8, sdpa_hi16(c4));
        }
        if constexpr (POLY_DEGREE >= 3) {
            TTI_SFPLOADI(p_sfpu::LREG4, 0xA, sdpa_lo16(c3));
            TTI_SFPLOADI(p_sfpu::LREG4, 0x8, sdpa_hi16(c3));
        }
        if constexpr (POLY_DEGREE >= 2) {
            TTI_SFPLOADI(p_sfpu::LREG5, 0xA, sdpa_lo16(c2));
            TTI_SFPLOADI(p_sfpu::LREG5, 0x8, sdpa_hi16(c2));
        }
        if constexpr (POLY_DEGREE >= 1) {
            TTI_SFPLOADI(p_sfpu::LREG6, 0xA, sdpa_lo16(c1));
            TTI_SFPLOADI(p_sfpu::LREG6, 0x8, sdpa_hi16(c1));
            TTI_SFPLOADI(p_sfpu::LREG7, 0xA, sdpa_lo16(c0));
            TTI_SFPLOADI(p_sfpu::LREG7, 0x8, sdpa_hi16(c0));
        }
    }

    if constexpr (sdpa_can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
        TTI_SFPLOADI(p_sfpu::LREG3, 0xA, sdpa_lo16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG3, 0x8, sdpa_hi16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG4, 0xA, sdpa_lo16(M_LN2));
        TTI_SFPLOADI(p_sfpu::LREG4, 0x8, sdpa_hi16(M_LN2));
    }

    for (int d = 0; d < ITERATIONS; d++) {
        constexpr InstrModLoadStore input_type =
            IS_FP32_DEST_ACC_EN ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B;
        TTI_SFPLOAD(p_sfpu::LREG2, input_type, sdpa_addr_mod_x, 0);

        if constexpr (SCALE_EN) {
            TTI_SFPLOADI(p_sfpu::LREG0, 0, SCALE_BF16);
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
            sdpa_insert_sfpnop();
        }

        if constexpr (sdpa_can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        } else {
            TTI_SFPLOADI(p_sfpu::LREG1, 0xA, sdpa_lo16(LN2_RECIP));
            TTI_SFPLOADI(p_sfpu::LREG1, 0x8, sdpa_hi16(LN2_RECIP));
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        }
        sdpa_insert_sfpnop();
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

        if constexpr (USE_SFPARECIP_INSTR) {
            TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);
            TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 2);
            TTI_SFPENCC(0, 0, 0, 0);
            TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, 2);
        } else {
            if constexpr (sdpa_can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            } else {
                TTI_SFPLOADI(p_sfpu::LREG0, 0xA, sdpa_lo16(M_LN2));
                TTI_SFPLOADI(p_sfpu::LREG0, 0x8, sdpa_hi16(M_LN2));
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            }
            sdpa_insert_sfpnop();

            if constexpr (POLY_DEGREE == 1) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 2) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 3) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            }
            sdpa_insert_sfpnop();
        }

        TT_SFPADDI(0x42fe, p_sfpu::LREG1, 0);
        sdpa_insert_sfpnop();
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
        TTI_SFPSETEXP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        sdpa_insert_sfpnop();

        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 6);
        TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        if constexpr (!IS_FP32_DEST_ACC_EN) {
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }
        TTI_SFPSTORE(p_sfpu::LREG2, input_type, sdpa_addr_mod_x, 0);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16, bool is_fp32_dest_acc_en>
inline void calculate_exponential_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (SDPA_EXP_APPROX_MODE) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result =
                ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, is_fp32_dest_acc_en>(val, scale_bf16);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg += 2;
        }
    } else {
        constexpr int polynomial_degree = is_fp32_dest_acc_en ? 4 : 2;
        calculate_exponential_polynomial<
            true,
            ITERATIONS_HALF_FACE,
            false,
            polynomial_degree,
            is_fp32_dest_acc_en,
            scale_bf16>();
    }
}

template <bool is_fp32_dest_acc_en>
inline void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    constexpr uint32_t prev_max_base_idx = 0;
    constexpr uint32_t worker_max_base_idx = 32;
    constexpr uint32_t cur_max_base_idx = 64;
    constexpr uint32_t prev_sum_base_idx = 96;
    constexpr uint32_t worker_sum_base_idx = 128;

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat prev_max_vec = sfpi::dst_reg[prev_max_base_idx];
        sfpi::vFloat worker_max_vec = sfpi::dst_reg[worker_max_base_idx];
        sfpi::vFloat prev_sum_vec = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat worker_sum_vec = sfpi::dst_reg[worker_sum_base_idx];
        v_if(prev_max_vec < worker_max_vec) { sfpi::dst_reg[cur_max_base_idx] = worker_max_vec; }
        v_else { sfpi::dst_reg[cur_max_base_idx] = prev_max_vec; }
        v_endif;
        sfpi::vFloat cur_max = sfpi::dst_reg[cur_max_base_idx];

        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        sfpi::vFloat exp_prev =
            ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, is_fp32_dest_acc_en>(diff_prev, scale_bf16);
        sfpi::vFloat exp_worker =
            ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, is_fp32_dest_acc_en>(diff_worker, scale_bf16);

        sfpi::dst_reg[prev_max_base_idx] = exp_prev;
        sfpi::dst_reg[worker_max_base_idx] = exp_worker;

        sfpi::dst_reg[worker_sum_base_idx] = exp_worker * worker_sum_vec;
        sfpi::dst_reg[prev_sum_base_idx] = exp_prev * prev_sum_vec;
        sfpi::vFloat corr_worker_sum = sfpi::dst_reg[worker_sum_base_idx];
        sfpi::vFloat corr_prev_sum = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::dst_reg[prev_sum_base_idx] = corr_worker_sum + corr_prev_sum;
        sfpi::dst_reg += 2;
    }
}

template <bool is_fp32_dest_acc_en>
inline void calculate_softplus_first_column(uint param0, uint param1, uint param2) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    float beta = ckernel::sfpu::Converter::as_float(param0);
    float beta_reciprocal = ckernel::sfpu::Converter::as_float(param1);
    float threshold = ckernel::sfpu::Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        ckernel::sfpu::calculate_softplus_body<APPROX, is_fp32_dest_acc_en>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
