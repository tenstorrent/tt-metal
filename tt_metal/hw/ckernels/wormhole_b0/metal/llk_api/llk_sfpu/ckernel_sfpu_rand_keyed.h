// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// Counter-based generator behind rand_keyed_tile and dropout_keyed_tile. Each element's 32-bit
// draw is a pure function of (seed, tile index, position in the tile), so it does not depend on
// which core computes the tile, on what ran on the core before, or on SFPU PRNG seeding.
//
// The draw is wang(wang(counter ^ key) + key), with wang Thomas Wang's 32-bit integer hash and key
// fmix32(seed). A single round leaves strong serial and neighbour-lane correlations between
// consecutive counters, and two rounds of the cheaper rand_tile finalizers still correlate draws
// 256 counters apart (one face). Blackhole computes the same function, so both architectures give
// identical bits, and the host can reproduce them exactly.
//
// The counter of an element is ((tile_index * 4 + face) * 8 + row) * 32 + lane, where row is the
// SFPU iteration within the face. It is unique for the first 2^22 tiles of a stream.
//
// Register use, per face: LREG0 holds x and LREG1 is scratch. LREG2 holds from (rand) or
// -probability (dropout), LREG3 the counter, LREG4 the key and LREG5 the scale. SFPSHFT2 takes its
// shift count from a register, so LREG6, LREG7, LREG12, LREG13 and LREG14 hold 15, -12, -4, 3 and
// -16; the shift by 2 copies and shifts in place instead. One Wang round is 16 instructions, recorded
// into replay slots 0-15 and replayed for the second round of every row.

constexpr std::uint32_t keyed_rand_elements_per_face = 256;
constexpr std::uint32_t keyed_rand_lanes = 32;
constexpr std::uint32_t keyed_rand_round_length = 16;

inline std::uint32_t keyed_rand_key(std::uint32_t seed) {
    seed ^= seed >> 16;
    seed *= 0x85EBCA6Bu;
    seed ^= seed >> 13;
    seed *= 0xC2B2AE35u;
    seed ^= seed >> 16;
    return seed;
}

inline void keyed_rand_wang_round() {
    constexpr std::uint32_t shft_lreg = sfpi::SFPSHFT2_MOD1_SHFT_LREG;
    constexpr std::uint32_t shft_imm = 1;
    constexpr std::uint32_t add = sfpi::SFPIADD_MOD1_CC_NONE;
    // x = ~x + (x << 15)
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LREG1, shft_lreg);
    TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, add);
    // x ^= x >> 12
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG1, shft_lreg);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    // x += x << 2
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPSHFT(2, 0, p_sfpu::LREG1, shft_imm);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, add);
    // x ^= x >> 4
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG1, shft_lreg);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    // x *= 2057, as x + (x << 3) + (x << 11)
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG1, shft_lreg);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, add);
    TTI_SFPSHFT(8, 0, p_sfpu::LREG1, shft_imm);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, add);
    // x ^= x >> 16
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG1, shft_lreg);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
}

// Loads the shift counts, key and counter for the face. Clobbers LREG0, LREG1 and LREG12-14.
inline void keyed_rand_face_setup(std::uint32_t key, std::uint32_t face_index) {
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_SHORT, 15);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_SHORT, (-12) & 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_SHORT, (-4) & 0xFFFF);
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_SHORT, 3);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_SHORT, (-16) & 0xFFFF);
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);

    TT_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, key & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, key >> 16);

    // LTILEID holds 2 * lane.
    const std::uint32_t base = face_index * keyed_rand_elements_per_face;
    TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG3, 0);
    TTI_SFPSHFT((-1) & 0xFFF, 0, p_sfpu::LREG3, 1);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, base & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, base >> 16);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_CC_NONE);
}

// Leaves the draw's top 31 bits in LREG0. The first row records the round it executes.
template <bool RECORD>
inline void keyed_rand_draw31() {
    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
    TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    if constexpr (RECORD) {
        TTI_REPLAY(0, keyed_rand_round_length, 1, 1);
        keyed_rand_wang_round();
    } else {
        TTI_REPLAY(0, keyed_rand_round_length, 0, 0);
    }
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_REPLAY(0, keyed_rand_round_length, 0, 0);
    TTI_SFPSHFT((-1) & 0xFFF, 0, p_sfpu::LREG0, 1);
}

template <bool RECORD, bool NORMALIZE_PER_ROW>
inline void rand_keyed_row() {
    constexpr std::uint32_t one_over_2_pow_31_bf16 = 0x3000;
    keyed_rand_draw31<RECORD>();
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    if constexpr (NORMALIZE_PER_ROW) {
        TTI_SFPMULI(one_over_2_pow_31_bf16, p_sfpu::LREG1, 0);
        // Fills the multiply's dependency slot.
        TTI_SFPIADD(
            keyed_rand_lanes, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPNOP;
    } else {
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // Fills SFPMAD's dependency slot.
        TTI_SFPIADD(
            keyed_rand_lanes, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    }
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
}

template <bool NORMALIZE_PER_ROW>
inline void rand_keyed_rows() {
    rand_keyed_row<true, NORMALIZE_PER_ROW>();
#pragma GCC unroll 7
    for (int d = 1; d < 8; d++) {
        rand_keyed_row<false, NORMALIZE_PER_ROW>();
    }
}

// face_index is the face's position in the stream (tile_index * 4 + face). The per-face loop
// passes it by reference, so each call advances it to the next face.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void rand_keyed(std::uint32_t from, std::uint32_t scale, std::uint32_t key, std::uint32_t& face_index) {
    static_assert(ITERATIONS == 8, "rand_keyed processes whole faces");
    constexpr std::uint32_t exponent_shift = 23;
    constexpr std::uint32_t exponent_mask = 0xFF;
    constexpr std::uint32_t normalization_exponent = 31;
    const std::uint32_t scale_exponent = (scale >> exponent_shift) & exponent_mask;
    // Fold the exact 2^-31 normalization into scale when the result stays a finite normal number.
    const bool normalize_per_row = scale_exponent <= normalization_exponent || scale_exponent == exponent_mask;
    if (!normalize_per_row) {
        scale -= normalization_exponent << exponent_shift;
    }

    // The store advances Dest through ADDR_MOD_2, which the SFPU addr-mod base maps to slot 6.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);

    keyed_rand_face_setup(key, face_index++);
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, from & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, from >> 16);

    if (normalize_per_row) {
        rand_keyed_rows<true>();
    } else {
        rand_keyed_rows<false>();
    }
}

template <bool RECORD>
inline void dropout_keyed_row() {
    keyed_rand_draw31<RECORD>();
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    // Fills SFPMUL's dependency slot.
    TTI_SFPIADD(
        keyed_rand_lanes, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    // draw31 - probability < 0 enables the lanes to drop.
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_LT0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, 0);
}

// Zeroes an element when its draw's top 31 bits are below probability (INT_MAX * p), and scales
// the rest by scale. With the same seed and tile index, an element is dropped exactly when
// rand_keyed with from = 0 and scale = 1 gives it a value below probability / 2^31.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void dropout_keyed(
    std::uint32_t probability, std::uint32_t scale, std::uint32_t key, std::uint32_t& face_index) {
    static_assert(ITERATIONS == 8, "dropout_keyed processes whole faces");
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);

    keyed_rand_face_setup(key, face_index++);
    const std::uint32_t negative_probability = 0u - probability;
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, negative_probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, negative_probability >> 16);

    dropout_keyed_row<true>();
#pragma GCC unroll 7
    for (int d = 1; d < 8; d++) {
        dropout_keyed_row<false>();
    }
}

}  // namespace ckernel::sfpu
