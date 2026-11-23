// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Counter-based generator behind rand_keyed_tile and dropout_keyed_tile. Each element's 32-bit
// draw is a pure function of (seed, tile index, position in the tile), so it does not depend on
// which core computes the tile, on what ran on the core before, or on SFPU PRNG seeding.
//
// The draw is wang(wang(counter ^ key) + key), with wang Thomas Wang's 32-bit integer hash and key
// fmix32(seed). A single round leaves strong serial and neighbour-lane correlations between
// consecutive counters, and two rounds of the cheaper rand_tile finalizers still correlate draws
// 256 counters apart (one face). Wormhole computes the same function with hand-scheduled
// instructions, so both architectures give identical bits, and the host can reproduce them exactly.
//
// The counter of an element is ((tile_index * 4 + face) * 8 + row) * 32 + lane, where row is the
// SFPU iteration within the face. It is unique for the first 2^22 tiles of a stream.

constexpr std::uint32_t keyed_rand_elements_per_face = 256;
constexpr std::uint32_t keyed_rand_lanes = 32;

inline std::uint32_t keyed_rand_key(std::uint32_t seed) {
    seed ^= seed >> 16;
    seed *= 0x85EBCA6Bu;
    seed ^= seed >> 13;
    seed *= 0xC2B2AE35u;
    seed ^= seed >> 16;
    return seed;
}

sfpi_inline sfpi::vUInt wang_hash32(sfpi::vUInt x) {
    x = ~x + (x << 15);
    x ^= x >> 12;
    x += x << 2;
    x ^= x >> 4;
    x += (x << 3) + (x << 11);
    x ^= x >> 16;
    return x;
}

sfpi_inline sfpi::vUInt keyed_rand_draw(sfpi::vUInt counter, sfpi::vUInt key) {
    return wang_hash32(wang_hash32(counter ^ key) + key);
}

// Counter of each lane's first element in the face. vConstTileId holds 2 * lane.
sfpi_inline sfpi::vUInt keyed_rand_face_counter(std::uint32_t face_index) {
    const sfpi::vUInt lane = sfpi::as<sfpi::vUInt>(sfpi::vConstTileId) >> 1;
    return lane + sfpi::vUInt(face_index * keyed_rand_elements_per_face);
}

// The top 31 bits of a draw, as a non-negative integer.
sfpi_inline sfpi::vInt keyed_rand_draw31(sfpi::vUInt draw) { return sfpi::as<sfpi::vInt>(draw >> 1); }

// face_index is the face's position in the stream (tile_index * 4 + face). The per-face loop
// passes it by reference, so each call advances it to the next face.
template <bool NORMALIZE_PER_ROW, int ITERATIONS>
inline void rand_keyed_rows(std::uint32_t from, std::uint32_t scale, std::uint32_t key, std::uint32_t face_index) {
    constexpr float two_pow_minus_31 = 4.656612873077393e-10f;
    const sfpi::vUInt vkey = key;
    const sfpi::vFloat vfrom = sfpi::as<sfpi::vFloat>(sfpi::vUInt(from));
    const sfpi::vFloat vscale = sfpi::as<sfpi::vFloat>(sfpi::vUInt(scale));
    sfpi::vUInt counter = keyed_rand_face_counter(face_index);
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vInt draw31 = keyed_rand_draw31(keyed_rand_draw(counter, vkey));
        // Correctly rounded to FP32, so u / 2^31 lies in [0, 1] (1 only by rounding).
        sfpi::vFloat u = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vMag>(draw31), sfpi::RoundMode::Nearest);
        if constexpr (NORMALIZE_PER_ROW) {
            u = u * two_pow_minus_31;
        }
        sfpi::dst_reg[0] = u * vscale + vfrom;
        counter += keyed_rand_lanes;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void rand_keyed(std::uint32_t from, std::uint32_t scale, std::uint32_t key, std::uint32_t& face_index) {
    constexpr std::uint32_t exponent_shift = 23;
    constexpr std::uint32_t exponent_mask = 0xFF;
    constexpr std::uint32_t normalization_exponent = 31;
    const std::uint32_t scale_exponent = (scale >> exponent_shift) & exponent_mask;
    // Fold the exact 2^-31 normalization into scale when the result stays a finite normal number.
    if (scale_exponent <= normalization_exponent || scale_exponent == exponent_mask) {
        rand_keyed_rows<true, ITERATIONS>(from, scale, key, face_index++);
    } else {
        rand_keyed_rows<false, ITERATIONS>(from, scale - (normalization_exponent << exponent_shift), key, face_index++);
    }
}

// Zeroes an element when its draw's top 31 bits are below probability (INT_MAX * p), and scales
// the rest by scale. With the same seed and tile index, an element is dropped exactly when
// rand_keyed with from = 0 and scale = 1 gives it a value below probability / 2^31.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void dropout_keyed(
    std::uint32_t probability, std::uint32_t scale, std::uint32_t key, std::uint32_t& face_index) {
    const sfpi::vUInt vkey = key;
    const sfpi::vInt vprobability = sfpi::vInt(probability);
    const sfpi::vFloat vscale = sfpi::as<sfpi::vFloat>(sfpi::vUInt(scale));
    sfpi::vUInt counter = keyed_rand_face_counter(face_index++);
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vInt draw31 = keyed_rand_draw31(keyed_rand_draw(counter, vkey));
        sfpi::vFloat value = sfpi::dst_reg[0] * vscale;
        v_if(draw31 < vprobability) { value = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = value;
        counter += keyed_rand_lanes;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
