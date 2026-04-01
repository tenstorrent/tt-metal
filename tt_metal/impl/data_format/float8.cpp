// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/float8.hpp>

#include <bit>
#include <cmath>
#include <limits>
#include <random>

namespace {

// Converts a float32 value to an FP8 E4M3 bit pattern.
// E4M3: bias=7, exponent range [1,14] normal, [0] subnormal, [15] NaN.
// Overflow (|val| > max representable ~240) clamps to max finite.
uint8_t fp32_to_fp8_e4m3_bits(float val) {
    constexpr int FP8_BIAS = 7;
    constexpr int FP8_EXP_MAX = 14;  // E=15 is reserved for NaN

    if (std::isnan(val)) {
        return 0x7F;  // canonical NaN: S=0, E=1111, M=111
    }

    uint32_t u = std::bit_cast<uint32_t>(val);
    uint8_t sign = static_cast<uint8_t>((u >> 31) & 1);

    if (std::isinf(val)) {
        // No infinity in FP8 E4M3 — clamp to max finite: E=1110, M=111
        return (sign << 7) | 0x77;
    }

    int32_t fp32_biased_exp = static_cast<int32_t>((u >> 23) & 0xFF);
    uint32_t fp32_mantissa = u & 0x7FFFFF;

    if (fp32_biased_exp == 0 && fp32_mantissa == 0) {
        return sign << 7;  // zero
    }

    if (fp32_biased_exp == 0) {
        // FP32 subnormal — too small for FP8 normal range, rounds to zero
        return sign << 7;
    }

    int32_t exp_unbiased = fp32_biased_exp - 127;
    int32_t fp8_biased_exp = exp_unbiased + FP8_BIAS;

    if (fp8_biased_exp > FP8_EXP_MAX) {
        return (sign << 7) | 0x77;  // overflow → max finite
    }

    uint8_t fp8_exp;
    uint8_t fp8_man;

    if (fp8_biased_exp <= 0) {
        // HW does not support FP8 subnormals (exp=0, mantissa!=0) and flushes them to zero.
        return sign << 7;
    } else {
        // Normal FP8: take the top 3 bits of the FP32 mantissa
        fp8_man = static_cast<uint8_t>((fp32_mantissa >> 20) & 0x7);
        // Round to nearest even
        uint32_t round_bit = (fp32_mantissa >> 19) & 1;
        uint32_t sticky = fp32_mantissa & 0x7FFFF;
        if (round_bit && (sticky || (fp8_man & 1))) {
            if (++fp8_man >= 8) {
                fp8_man = 0;
                if (++fp8_biased_exp > FP8_EXP_MAX) {
                    return (sign << 7) | 0x77;  // overflow → max finite
                }
            }
        }
        fp8_exp = static_cast<uint8_t>(fp8_biased_exp);
    }

    return (sign << 7) | (fp8_exp << 3) | fp8_man;
}

}  // namespace

// Widening conversion: FP8 E4M3 → float32
float8_e4m3::operator float() const {
    constexpr int FP8_BIAS = 7;

    uint8_t sign = (uint8_data >> 7) & 1;
    uint8_t exp = (uint8_data >> 3) & 0xF;
    uint8_t man = uint8_data & 0x7;

    if (exp == 15) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    if (exp == 0) {
        // HW flushes subnormals (exp=0, mantissa!=0) to zero.
        return sign ? -0.0f : 0.0f;
    }

    // Reconstruct float32 bit pattern: (-1)^sign * 2^(exp-bias) * (1 + man/8)
    uint32_t fp32_exp = static_cast<uint32_t>(exp - FP8_BIAS + 127);
    uint32_t fp32_bits = (static_cast<uint32_t>(sign) << 31) | (fp32_exp << 23) | (static_cast<uint32_t>(man) << 20);
    return std::bit_cast<float>(fp32_bits);
}

uint8_t float8_e4m3::from_float(float v) { return fp32_to_fp8_e4m3_bits(v); }

uint32_t pack_four_float8_e4m3_into_uint32(float8_e4m3 a, float8_e4m3 b, float8_e4m3 c, float8_e4m3 d) {
    return static_cast<uint32_t>(a.to_bits()) | (static_cast<uint32_t>(b.to_bits()) << 8) |
           (static_cast<uint32_t>(c.to_bits()) << 16) | (static_cast<uint32_t>(d.to_bits()) << 24);
}

std::vector<float8_e4m3> unpack_uint32_vec_into_float8_e4m3_vec(const std::vector<uint32_t>& data) {
    std::vector<float8_e4m3> result;
    result.reserve(data.size() * 4);
    for (uint32_t word : data) {
        result.push_back(float8_e4m3::from_bits(word & 0xFF));
        result.push_back(float8_e4m3::from_bits((word >> 8) & 0xFF));
        result.push_back(float8_e4m3::from_bits((word >> 16) & 0xFF));
        result.push_back(float8_e4m3::from_bits((word >> 24) & 0xFF));
    }
    return result;
}

std::vector<uint32_t> create_random_vector_of_float8_e4m3(
    size_t num_bytes, int rand_max_float, int seed, float offset) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0, rand_max_float);

    // num_bytes fp8 elements, packed 4 per uint32
    std::vector<uint32_t> result(num_bytes / sizeof(uint32_t), 0);
    for (uint32_t& word : result) {
        float8_e4m3 a(dist(rng) + offset);
        float8_e4m3 b(dist(rng) + offset);
        float8_e4m3 c(dist(rng) + offset);
        float8_e4m3 d(dist(rng) + offset);
        word = pack_four_float8_e4m3_into_uint32(a, b, c, d);
    }
    return result;
}
