// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel::sfpu {

constexpr std::uint32_t sfpshft_mod1_arg_imm = 1;
constexpr std::uint32_t sfpshft_mod1_arg_imm_use_vc = sfpshft_mod1_arg_imm | 4;
constexpr std::uint32_t sfpsetsgn_mod1_arg_imm = 1;
constexpr std::uint32_t one_over_2_pow_31_bf16 = 0x3000;

template <std::uint32_t DEST>
inline void rand_prng() {
    constexpr std::uint32_t prng_source = 9;
    constexpr std::uint32_t sfpmov_mod1_from_special = 8;
    TTI_SFPMOV(0, prng_source, DEST, sfpmov_mod1_from_special);
}

template <bool APPROXIMATION_MODE>
inline void rand_init(std::uint32_t seed) {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    // The all-ones state is the lock-up state of the hardware XNOR LFSR.
    if (seed == 0xFFFFFFFF) {
        seed = 0xFFFFFFFE;
    }
    init_prng_seed(seed);
}

inline void make_lane_salt() {
    // rand_tile is a shared API and other SFPU operations may clobber every
    // mutable LREG between calls. Reconstruct a lane-specific salt from the
    // read-only lane ID at the start of each face instead of carrying it in an
    // LREG. Offset LTILEID first so lane zero also receives a nonzero salt.
    // The immediate offset also eliminates long-stream spatial ridges that
    // remained when the salt was derived directly from LTILEID.
    // All lane PRNGs receive the same seed, while nearby cores receive related
    // seeds. Salting before a bijective finalizer diffuses those relationships
    // through the 31 bits consumed by SFPCAST without biasing a lane's output.
    TTI_SFPIADD(407, p_sfpu::LTILEID, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT(14, p_sfpu::LREG4, p_sfpu::LREG5, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT(6, p_sfpu::LREG5, p_sfpu::LREG3, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG3, 0);
}

inline void begin_mix_uint32_mul24() {
    // Custom bijective 32-bit finalizer (not a named hash):
    //   x ^= x >> 8; x ^= x >> 16;
    //   x[22:0] *= 0x56594B (mod 2^23), preserving x[31:23];
    //   x ^= x << 8; x ^= x >> 14.
    // Blackhole's SFPMUL24 provides better diffusion than same-cost pure-ARX
    // candidates. LCONST_0_8373 has FP32 bits 0x3F56594B; its low 23 bits form
    // the odd multiplier above, so the modular multiply is bijective. The
    // search was constrained to bijective sequences fitting the instruction
    // and register budget. Candidates minimized avalanche RMS from 50% and
    // worst lane/offset Pearson correlation across seeds, while keeping
    // tile-mean dispersion near the IID expectation after FP32 conversion.
    // Finalists were also checked for long-stream spatial ridges.
    TTI_SFPSHFT((-8) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG5, sfpshft_mod1_arg_imm_use_vc);
}

inline void finish_mix_uint32_mul24() {
    TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG5, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG5, 0);

    // LOWER computes the low 23-bit product.
    TTI_SFPMUL24(p_sfpu::LREG5, p_sfpu::LCONST_0_8373, p_sfpu::LCONST_0, p_sfpu::LREG4, sfpi::SFPMUL24_MOD1_LOWER);
    // This independent PRNG read fills SFPMUL24's dependency slot.
    rand_prng<p_sfpu::LREG0>();
    // Restore the mixed input's upper nine bits in the low product.
    TTI_SFPSETMAN(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);

    TTI_SFPSHFT(8, p_sfpu::LREG4, p_sfpu::LREG5, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    TTI_SFPSHFT((-14) & 0xFFF, p_sfpu::LREG4, p_sfpu::LREG5, sfpshft_mod1_arg_imm_use_vc);
    TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
}

template <bool NORMALIZE_PER_ROW>
inline void rand_row() {
    finish_mix_uint32_mul24();
    // SFPCAST converts the low 31 bits as a sign-magnitude integer and rounds
    // directly to FP32. Clear its arbitrary sign; normalization by 2^-31 is
    // applied here or folded exactly into scale. This gives a correctly
    // rounded 31-bit uniform grid, including both mantissa parities and the
    // half-width upper-bound rounding basin.
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPSETSGN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpsetsgn_mod1_arg_imm);
    if constexpr (NORMALIZE_PER_ROW) {
        TTI_SFPMULI(one_over_2_pow_31_bf16, p_sfpu::LREG6, 0);
    }
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG6, 0);
    // Prime the following row's mixer in SFPMAD's dependency slot. This reads
    // LREG0 and writes LREG5, independently of SFPMAD's LREG6 result. The
    // speculative prime after the final row is harmless.
    begin_mix_uint32_mul24();
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
}

template <bool NORMALIZE_PER_ROW>
inline void rand_rows() {
    constexpr std::uint32_t row_instruction_count = NORMALIZE_PER_ROW ? 17 : 16;

    // One row fits in the 32-entry replay buffer. Record and execute it once,
    // then replay it for the remaining rows without scalar loop-control gaps.
    TTI_REPLAY(0, row_instruction_count, 1, 1);
    rand_row<NORMALIZE_PER_ROW>();
#pragma GCC unroll 7
    for (int d = 1; d < 8; d++) {
        TTI_REPLAY(0, row_instruction_count, 0, 0);
    }
}

template <bool APPROXIMATION_MODE>
inline void rand(std::uint32_t from, std::uint32_t scale) {
    constexpr std::uint32_t exponent_shift = 23;
    constexpr std::uint32_t exponent_mask = 0xFF;
    constexpr std::uint32_t normalization_exponent = 31;
    const std::uint32_t scale_exponent = (scale >> exponent_shift) & exponent_mask;
    const bool normalize_per_row = scale_exponent <= normalization_exponent || scale_exponent == exponent_mask;

    // For finite scales >= 2^-95, fold the exact power-of-two normalization
    // into scale once per face. Retain the per-row multiply when the adjusted
    // scale would be subnormal, or for non-finite inputs, preserving the
    // existing rand_tile behavior in both cases.
    if (!normalize_per_row) {
        scale -= normalization_exponent << exponent_shift;
    }

    // Load scale param to lreg1
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, from >> 16);

    make_lane_salt();
    rand_prng<p_sfpu::LREG0>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
    begin_mix_uint32_mul24();

    if (normalize_per_row) {
        rand_rows<true>();
    } else {
        rand_rows<false>();
    }
}
}  // namespace ckernel::sfpu
