// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

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
    // Reconstruct the salt for every face so rand_tile remains valid after
    // arbitrary SFPU operations have clobbered the mutable LREGs. LTILEID is
    // the ISA-defined LREG15, whose lane i contains 2*i. Offset LTILEID first
    // so lane zero also receives a nonzero salt. The immediate offset also
    // eliminates long-stream spatial ridges that remained when the salt was
    // derived directly from LTILEID. The immediate shift encodings select
    // LREG4 and LREG5 as their respective sources.
    // All lane PRNGs receive the same seed, while nearby cores receive related
    // seeds. Salting before a bijective finalizer diffuses those relationships
    // through the 31 bits consumed by SFPCAST without biasing a lane's output.
    TTI_SFPIADD(
        (-151) & 0xFFF, p_sfpu::LTILEID, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT2(20, 0, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPSHFT2(5, 0, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG3, 0);
}

inline void begin_mix_uint32_fast() {
    // Custom bijective 32-bit ARX finalizer (not a named hash):
    //   x ^= x >> 16; x += x << 10; x ^= x >> 6;
    //   x += x << 13; x ^= x >> 16.
    // Wormhole has no SFPMUL24, so this uses only shift, XOR, and modular add;
    // every stage is bijective. The search was constrained to bijective
    // sequences fitting the instruction and register budget. Candidates
    // minimized avalanche RMS from 50% and worst lane/offset Pearson
    // correlation across seeds, while keeping tile-mean dispersion near the
    // IID expectation after FP32 conversion. Finalists were also checked for
    // long-stream spatial ridges. Reusing shift 16 limits setup to four counts
    // and keeps the common replayed row at 16 instructions.
    //
    // SFPSHFT2's independent source and destination registers avoid staging
    // each input. LREG4, LREG7, LREG12, and LREG13 hold -16, 10, -6, and 13,
    // respectively. LREG1 holds x on entry and LREG0 holds the result on exit.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
}

inline void finish_mix_uint32_fast() {
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);

    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
}

template <bool NORMALIZE_PER_ROW>
inline void rand_row() {
    finish_mix_uint32_fast();
    // SFPCAST converts the low 31 bits as a sign-magnitude integer and rounds
    // directly to FP32. Clear its arbitrary sign; normalization by 2^-31 is
    // applied here or folded exactly into scale. This gives a correctly
    // rounded 31-bit uniform grid, including both mantissa parities and the
    // half-width upper-bound rounding basin.
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG6, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPSETSGN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpsetsgn_mod1_arg_imm);
    if constexpr (NORMALIZE_PER_ROW) {
        TTI_SFPMULI(one_over_2_pow_31_bf16, p_sfpu::LREG6, 0);
    }
    // Advance the PRNG while the preceding result becomes available.
    rand_prng<p_sfpu::LREG1>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG6, 0);
    // Prime the following row's mixer in SFPMAD's dependency slot. This reads
    // LREG1 and writes LREG0, independently of SFPMAD's LREG6 result. The
    // speculative prime after the final row is harmless.
    begin_mix_uint32_fast();
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP32, ADDR_MOD_2, 0);
}

template <bool NORMALIZE_PER_ROW>
inline void rand_rows() {
    constexpr std::uint32_t row_instruction_count = NORMALIZE_PER_ROW ? 17 : 16;

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

    make_lane_salt();
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_SHORT, 10);

    // Load scale param to lreg5
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);

    // Load from param to lreg2
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, from >> 16);

    // Keep the remaining shift counts outside the replayed row body.
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_SHORT, (-16) & 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_SHORT, (-6) & 0xFFFF);
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_SHORT, 13);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);

    rand_prng<p_sfpu::LREG1>();
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);
    begin_mix_uint32_fast();

    if (normalize_per_row) {
        rand_rows<true>();
    } else {
        rand_rows<false>();
    }
}
}  // namespace ckernel::sfpu
