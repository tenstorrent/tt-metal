// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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
