// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <tt_stl/assert.hpp>
#include "tt_metal/tools/profiler/tracy_debug_zones.hpp"

#include "impl/data_format/bfloat16_utils.hpp"

uint16_t fp32_to_bf16_bits_round_to_nearest_even(float val) {
    if (std::isnan(val)) {
        // NaN is represented when all exponent bits are 1 and mantissa is non-zero.
        // 0x7FC0  = 0 (sign) 11111111 (exponent) 1000000 (mantissa)
        return UINT16_C(0x7FC0);
    }
    uint32_t U32 = std::bit_cast<uint32_t>(val);
    // Rounding bias = 0111 1111 1111 1111 (0x7FFF) if last bit of mantissa ((U32 >> 16) & 1)
    // is 0, otherwise 1000 0000 0000 0000 (0x8000).
    // This ensures that we round to the nearest even number.
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
}

bfloat16 bfloat16::truncate(float float_num) {
    uint32_t U32 = std::bit_cast<uint32_t>(float_num);
    return std::bit_cast<bfloat16>(static_cast<uint16_t>(U32 >> 16));
}

// -- Arithmetic Operators ---
bfloat16& bfloat16::operator+=(bfloat16 rhs) noexcept {
    *this = *this + rhs;
    return *this;
}

bfloat16& bfloat16::operator-=(bfloat16 rhs) noexcept {
    *this = *this - rhs;
    return *this;
}

bfloat16& bfloat16::operator*=(bfloat16 rhs) noexcept {
    *this = *this * rhs;
    return *this;
}

bfloat16& bfloat16::operator/=(bfloat16 rhs) noexcept {
    *this = *this / rhs;
    return *this;
}

bfloat16 bfloat16::operator+(bfloat16 rhs) const {
    return bfloat16(static_cast<float>(*this) + static_cast<float>(rhs));
}

bfloat16 bfloat16::operator-(bfloat16 rhs) const {
    return bfloat16(static_cast<float>(*this) - static_cast<float>(rhs));
}

bfloat16 bfloat16::operator*(bfloat16 rhs) const {
    return bfloat16(static_cast<float>(*this) * static_cast<float>(rhs));
}

bfloat16 bfloat16::operator/(bfloat16 rhs) const {
    return bfloat16(static_cast<float>(*this) / static_cast<float>(rhs));
}

uint16_t bfloat16::from_float(float val) { return fp32_to_bf16_bits_round_to_nearest_even(val); }

std::ostream& operator<<(std::ostream& os, const bfloat16& bfp16) {
    os << std::bit_cast<uint16_t>(bfp16);
    return os;
}

bool operator==(const std::vector<bfloat16>& lhs, const std::vector<bfloat16>& rhs) {
    bool is_equal = lhs.size() == rhs.size();
    for (auto i = 0; i < lhs.size(); i++) {
        is_equal &= (std::bit_cast<uint16_t>(lhs[i]) == std::bit_cast<uint16_t>(rhs[i]));
    }
    return is_equal;
}

uint32_t pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16> two_bfloats) {
    // first -> lower 16
    // second -> upper 16
    return (uint32_t)std::bit_cast<uint16_t>(two_bfloats.first) |
           ((uint32_t)std::bit_cast<uint16_t>(two_bfloats.second) << 16);
}

std::vector<bfloat16> create_random_vector_of_bfloat16_native(
    size_t num_bytes, float rand_max_float, int seed, float offset) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    std::vector<bfloat16> vec(num_bytes / sizeof(bfloat16), 0);
    for (auto& elem : vec) {
        float num_1_float = rand_float() + offset;
        elem = bfloat16(num_1_float);
    }
    return vec;
}

std::vector<std::uint32_t> create_random_vector_of_bfloat16(
    size_t num_bytes, int rand_max_float, int seed, float offset) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    std::vector<std::uint32_t> vec(num_bytes / sizeof(std::uint32_t), 0);
    for (unsigned int& elem : vec) {
        float num_1_float = rand_float() + offset;
        float num_2_float = rand_float() + offset;

        bfloat16 num_1_bfloat16 = bfloat16(num_1_float);
        bfloat16 num_2_bfloat16 = bfloat16(num_2_float);

        // pack 2 uint16 into uint32
        elem = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }

    return vec;
}

/*
 * rk: Still won't handle the case where the number of elements is odd, except
 * if it's 1. Whatever, for now.
 */
std::vector<std::uint32_t> create_constant_vector_of_bfloat16(size_t num_bytes, float value) {
    const size_t num_elements_vec = std::max<size_t>(1ul, num_bytes / sizeof(std::uint32_t));  // always at least have 1
    std::vector<std::uint32_t> vec(num_elements_vec, 0);
    for (unsigned int& elem : vec) {
        bfloat16 num_1_bfloat16 = bfloat16(value);

        bfloat16 num_2_bfloat16 = num_elements_vec == 1 ? bfloat16(static_cast<float>(0.0)) : bfloat16(value);

        elem = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }

    return vec;
}

// creates a bfloat16 identity matrix with dims (rows x cols)
// each 2 cols will be packed as a single uint32_t
std::vector<bfloat16> create_identity_matrix(int rows, int cols, int num_ones) {
    std::vector<bfloat16> vec(rows * cols, (float)0);
    for (int i = 0; i < num_ones; i++) {
        vec.at((i * cols) + i) = bfloat16((float)1);
    }
    return vec;
}

std::vector<uint32_t> pack_bfloat16_vec_into_uint32_vec(const std::vector<bfloat16>& data) {
    TTZoneScopedD(DATA_FORMAT);
    TT_ASSERT(data.size() % 2 == 0);
    std::vector<uint32_t> result(data.size() / 2);
    std::memcpy(result.data(), data.data(), result.size() * sizeof(uint32_t));
    return result;
}

bfloat16 bfloat16_identity_transform(const bfloat16& input) { return input; }

std::vector<bfloat16> unpack_uint32_vec_into_bfloat16_vec(
    const std::vector<std::uint32_t>& data, const std::function<bfloat16(const bfloat16&)>& transform) {
    std::vector<bfloat16> result;
    result.reserve(data.size() * 2);
    for (unsigned int packed : data) {
        auto unpacked = unpack_two_bfloat16_from_uint32(packed);
        result.push_back(transform(unpacked.first));
        result.push_back(transform(unpacked.second));
    }
    return result;
}

// Equality functions
bool equal_within_n_sig_figs(float a, float b, int n) {
    TT_FATAL(
        n >= 1 && n <= std::numeric_limits<float>::max_digits10,
        "n must be between 1 and {}, got {}",
        std::numeric_limits<float>::max_digits10,
        n);

    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return a == b;
    }
    if (a == b) {
        return true;
    }
    if (std::signbit(a) != std::signbit(b) || a == 0.0f || b == 0.0f) {
        return false;
    }

    const auto leading_digits = [n](float value) {
        const double magnitude = std::fabs(static_cast<double>(value));
        const double decade = std::floor(std::log10(magnitude));
        const double normalized = magnitude / std::pow(10.0, decade);
        return static_cast<std::uint64_t>(normalized * std::pow(10.0, n - 1));
    };

    const double decade_a = std::floor(std::log10(std::fabs(static_cast<double>(a))));
    const double decade_b = std::floor(std::log10(std::fabs(static_cast<double>(b))));
    return decade_a == decade_b && leading_digits(a) == leading_digits(b);
}

bool is_within_n_sig_figs(float a, float b, int n) {
    TT_FATAL(
        n >= 1 && n <= std::numeric_limits<float>::max_digits10,
        "n must be between 1 and {}, got {}",
        std::numeric_limits<float>::max_digits10,
        n);

    // NaN has no significant figures, but two NaN results are the same outcome.
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return a == b;
    }

    const double abs_a = std::fabs(static_cast<double>(a));
    const double abs_b = std::fabs(static_cast<double>(b));
    const double scale = std::max(abs_a, abs_b);
    const double diff = std::fabs(static_cast<double>(a) - static_cast<double>(b));

    // log10(0) is undefined. Both zeros agree. For n >= 1 the decade tolerance
    // is smaller than any non-zero magnitude, so zero matches only zero.
    if (scale == 0.0) {
        return true;
    }

    // The float32 rounding of 1e-23 is 0x19416d9a. Its log10 is 7.82e-11 below
    // -23. A nudge of 1e-10 steps into the next decade and widens the tolerance
    // by 10x. 1e-12 sits above double log10 error (~1e-15) and below that gap.
    constexpr double kLog10TieEpsilon = 1e-12;
    const double decade = std::floor(std::log10(scale) + kLog10TieEpsilon);
    const double tolerance = 0.5 * std::pow(10.0, decade - static_cast<double>(n) + 1.0);
    return diff <= tolerance;
}

// this follows the implementation of numpy's is_close
bool is_close(float a, float b, float rtol, float atol) {
    if (a == b) {
        return true;
    }
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }

    // the idea is near zero we want absolute tolerance since relative doesn't make sense
    // (consider 1e-6f and 1.1e-6f)
    // elsewhere (not near zero) we want relative tolerance
    auto absdiff = fabsf(a - b);
    auto reldenom = fmaxf(fabsf(a), fabsf(b));
    auto result = (absdiff <= atol) || (absdiff <= rtol * reldenom);
    if (!result) {
        std::cout << "Discrepacy: A = " << a << " B = " << b << std::endl;
        std::cout << "   absdiff = " << absdiff << std::endl;
        std::cout << "   reldiff = " << absdiff / (reldenom + 1e-6f) << std::endl;
    }
    return result;
}
