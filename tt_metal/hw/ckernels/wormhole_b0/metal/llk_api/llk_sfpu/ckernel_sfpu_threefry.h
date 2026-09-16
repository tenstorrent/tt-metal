// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Threefry-2x32 counter-based generator. Each (counter, lane) pair maps to two words through a keyed
// bijection, so streams are disjoint by construction and no hardware PRNG state is involved.
constexpr std::uint32_t threefry_key_parity = 0x1BD11BDA;
constexpr int threefry_rotation[8] = {13, 15, 26, 6, 17, 29, 16, 24};
constexpr std::uint32_t threefry_words_per_face = 4;  // one 2x32 block per pair of dest rows

inline sfpi::vUInt threefry_rotl(sfpi::vUInt x, int r) { return (x << r) | (x >> (32 - r)); }

template <int ROUNDS>
inline void threefry2x32(sfpi::vUInt& x0, sfpi::vUInt& x1, const std::uint32_t (&ks)[3]) {
    x0 += ks[0];
    x1 += ks[1];
#pragma GCC unroll 32
    for (int r = 0; r < ROUNDS; ++r) {
        x0 += x1;
        x1 = threefry_rotl(x1, threefry_rotation[r & 7]);
        x1 ^= x0;
        if ((r & 3) == 3) {
            const int j = (r >> 2) + 1;
            x0 += ks[j % 3];
            x1 += ks[(j + 1) % 3] + static_cast<std::uint32_t>(j);
        }
    }
}

// `ctr` advances by threefry_words_per_face per call so RC mode's four face calls see distinct counters.
template <bool APPROXIMATION_MODE, int ROUNDS>
inline void threefry(
    std::uint32_t from, std::uint32_t scale, std::uint32_t key0, std::uint32_t key1, std::uint32_t& ctr) {
    constexpr std::uint32_t exponent_shift = 23;
    constexpr std::uint32_t exponent_mask = 0xFF;
    constexpr std::uint32_t normalization_exponent = 31;
    const std::uint32_t scale_exponent = (scale >> exponent_shift) & exponent_mask;
    const bool normalize_per_row = scale_exponent <= normalization_exponent || scale_exponent == exponent_mask;
    if (!normalize_per_row) {
        scale -= normalization_exponent << exponent_shift;
    }
    const std::uint32_t ks[3] = {key0, key1, key0 ^ key1 ^ threefry_key_parity};
    const sfpi::vFloat v_scale = __builtin_bit_cast(float, scale);
    const sfpi::vFloat v_from = __builtin_bit_cast(float, from);
    const sfpi::vFloat two_pow_m31 = __builtin_bit_cast(float, 0x30000000u);
    const sfpi::vUInt lane = sfpi::as<sfpi::vUInt>(sfpi::vConstTileId);

#pragma GCC unroll 4
    for (std::uint32_t pair = 0; pair < threefry_words_per_face; ++pair) {
        sfpi::vUInt x0 = ctr + pair;
        sfpi::vUInt x1 = lane;
        threefry2x32<ROUNDS>(x0, x1, ks);
        sfpi::vFloat f0 = sfpi::int32_to_float(sfpi::as<sfpi::vInt>(x0 >> 1), sfpi::RoundMode::Nearest);
        sfpi::vFloat f1 = sfpi::int32_to_float(sfpi::as<sfpi::vInt>(x1 >> 1), sfpi::RoundMode::Nearest);
        if (normalize_per_row) {
            f0 *= two_pow_m31;
            f1 *= two_pow_m31;
        }
        sfpi::dst_reg[2 * pair] = f0 * v_scale + v_from;
        sfpi::dst_reg[2 * pair + 1] = f1 * v_scale + v_from;
    }
    ctr += threefry_words_per_face;
}

}  // namespace ckernel::sfpu
