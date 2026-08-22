// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ostream>
#include <utility>
#include <vector>
#include <compare>
#include <bit>
#include <type_traits>

class bfloat16 {
private:
    uint16_t uint16_data;

public:
    // --- Constructors ---
    constexpr bfloat16() = default;

    // create from arithmetic type: tie-to-even rounding
    template <class T>
        requires std::is_arithmetic_v<T>
    constexpr bfloat16(T v) noexcept
        : uint16_data(from_float(static_cast<float>(v))) {}

    // create from float: truncate rounding
    static bfloat16 truncate(float float_num);

    // Widening conversion
    constexpr operator float() const {
        // move lower 16 to upper 16 (of 32) and convert to float
        uint32_t uint32_data = (uint32_t)uint16_data << 16;
        return std::bit_cast<float>(uint32_data);
    }

    // -- Comparison Operators ---
    constexpr bool operator==(bfloat16 rhs) const { return static_cast<float>(*this) == static_cast<float>(rhs); };
    constexpr std::partial_ordering operator<=>(bfloat16 rhs) const noexcept {
        return static_cast<float>(*this) <=> static_cast<float>(rhs);
    }

    // -- Arithmetic Operators ---
    bfloat16& operator+=(bfloat16 rhs) noexcept;
    bfloat16& operator-=(bfloat16 rhs) noexcept;
    bfloat16& operator*=(bfloat16 rhs) noexcept;
    bfloat16& operator/=(bfloat16 rhs) noexcept;

    bfloat16 operator+(bfloat16 rhs) const;
    bfloat16 operator-(bfloat16 rhs) const;
    bfloat16 operator*(bfloat16 rhs) const;
    bfloat16 operator/(bfloat16 rhs) const;

private:
    uint16_t from_float(float val);
};

std::ostream& operator<<(std::ostream& os, const bfloat16& bfp16);

bool operator==(const std::vector<bfloat16>& lhs, const std::vector<bfloat16>& rhs);

uint32_t pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16> two_bfloats);

std::vector<bfloat16> create_random_vector_of_bfloat16_native(
    size_t num_bytes, float rand_max_float, int seed, float offset = 0.0f);
