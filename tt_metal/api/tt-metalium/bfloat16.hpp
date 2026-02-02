// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include <compare>
#include <bit>

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

std::vector<std::uint32_t> create_random_vector_of_bfloat16(
    size_t num_bytes, int rand_max_float, int seed, float offset = 0.0f);

/*
 * rk: Still won't handle the case where the number of elements is odd, except
 * if it's 1. Whatever, for now.
 */
std::vector<std::uint32_t> create_constant_vector_of_bfloat16(size_t num_bytes, float value);

// creates a bfloat16 identity matrix with dims (rows x cols)
// each 2 cols will be packed as a single uint32_t
std::vector<bfloat16> create_identity_matrix(int rows, int cols, int num_ones);

std::vector<uint32_t> pack_bfloat16_vec_into_uint32_vec(const std::vector<bfloat16>& data);

bfloat16 bfloat16_identity_transform(const bfloat16& input);

std::vector<bfloat16> unpack_uint32_vec_into_bfloat16_vec(
    const std::vector<std::uint32_t>& data,
    const std::function<bfloat16(const bfloat16&)>& transform = bfloat16_identity_transform);

// Equality functions
bool equal_within_n_sig_figs(float a, float b, int n);

// this follows the implementation of numpy's is_close
bool is_close(float a, float b, float rtol = 0.01f, float atol = 0.001f);
