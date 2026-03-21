// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

// FP8 E4M3: 1 sign bit (MSB), 4 exponent bits (bias=7), 3 mantissa bits.
class float8_e4m3 {
private:
    uint8_t uint8_data;

public:
    constexpr float8_e4m3() = default;

    float8_e4m3(float v) noexcept : uint8_data(from_float(v)) {}

    operator float() const;

    uint8_t to_bits() const { return uint8_data; }

    static float8_e4m3 from_bits(uint8_t bits) {
        float8_e4m3 f;
        f.uint8_data = bits;
        return f;
    }

private:
    static uint8_t from_float(float v);
};

uint32_t pack_four_float8_e4m3_into_uint32(float8_e4m3 a, float8_e4m3 b, float8_e4m3 c, float8_e4m3 d);

// Unpacks a packed uint32 vector (4 fp8 bytes per word) into float8_e4m3 values.
std::vector<float8_e4m3> unpack_uint32_vec_into_float8_e4m3_vec(const std::vector<uint32_t>& data);

// Generates num_bytes fp8_e4m3 values from a uniform distribution U(0, rand_max_float) + offset, packed 4 per uint32.
std::vector<uint32_t> create_random_vector_of_float8_e4m3(
    size_t num_bytes, int rand_max_float, int seed, float offset = 0.0f);
