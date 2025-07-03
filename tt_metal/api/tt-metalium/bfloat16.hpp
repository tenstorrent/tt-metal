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

class bfloat16 {
private:
    uint16_t uint16_data;

public:
    static constexpr size_t SIZEOF = 2;
    bfloat16() = default;

    // create from float: no rounding, just truncate
    bfloat16(float float_num) {
        static_assert(sizeof float_num == 4, "float must have size 4");

        uint16_data = (*reinterpret_cast<uint32_t*>(&float_num)) >> 16;
    }

    // store lower 16 as 16-bit uint
    bfloat16(uint32_t uint32_data) { uint16_data = (uint16_t)uint32_data; }

    bfloat16(uint16_t uint16_data_) { uint16_data = uint16_data_; }

    // store lower 16 as 16-bit uint
    bfloat16(int int_data) { uint16_data = (uint16_t)int_data; }

    float to_float() const {
        // move lower 16 to upper 16 (of 32) and convert to float
        uint32_t uint32_data = (uint32_t)uint16_data << 16;
        float f;
        std::memcpy(&f, &uint32_data, sizeof(f));
        return f;
    }
    uint16_t to_packed() const { return uint16_data; }
    uint16_t to_uint16() const { return uint16_data; }
    bool operator==(const bfloat16 rhs) const { return uint16_data == rhs.uint16_data; }
    bool operator!=(const bfloat16 rhs) const { return not(*this == rhs); }
    bfloat16 operator*(const bfloat16 rhs) const { return bfloat16(this->to_float() * rhs.to_float()); }
};

std::ostream& operator<<(std::ostream& os, const bfloat16& bfp16);

bool operator==(const std::vector<bfloat16>& lhs, const std::vector<bfloat16>& rhs);

uint32_t pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16> two_bfloats);

std::pair<bfloat16, bfloat16> unpack_two_bfloat16_from_uint32(uint32_t uint32_data);

std::vector<std::uint32_t> create_arange_vector_of_bfloat16(uint32_t num_bytes, bool print = true);

std::vector<bfloat16> create_random_vector_of_bfloat16_native(
    uint32_t num_bytes, float rand_max_float, int seed, float offset = 0.0f);

void print_golden_metalium_vectors(std::vector<bfloat16>& golden_vec, std::vector<bfloat16>& result_vec);

std::vector<std::uint32_t> create_random_vector_of_bfloat16(
    uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f);

std::vector<std::uint32_t> create_random_vector_of_bfloat16_1_1(uint32_t num_bytes, int seed);

std::vector<std::uint32_t> create_random_vector_of_bfloat16_0_2(uint32_t num_bytes, int seed);

/*
 * rk: Still won't handle the case where the number of elements is odd, except
 * if it's 1. Whatever, for now.
 */
std::vector<std::uint32_t> create_constant_vector_of_bfloat16(uint32_t num_bytes, float value);

// creates a bfloat16 identity matrix with dims (rows x cols)
// each 2 cols will be packed as a single uint32_t
std::vector<bfloat16> create_identity_matrix(int rows, int cols, int num_ones);

// TODO(AP): duplication with above
std::vector<uint32_t> create_random_binary_vector_of_bfloat16(uint32_t num_bytes, int seed);

std::vector<uint16_t> u16_from_u32_vector(const std::vector<uint32_t>& in);

std::vector<uint32_t> u32_from_u16_vector(const std::vector<uint16_t>& in);

void print_vec_of_uint32_as_packed_bfloat16(
    const std::vector<std::uint32_t>& vec, int num_tiles, const std::string& name = "", int tile_print_offset = 0);

void print_vec_of_bfloat16(
    const std::vector<bfloat16>& vec, int num_tiles, const std::string& name = "", int tile_print_offset = 0);

void print_vec(
    const std::vector<uint32_t>& vec, int num_tiles, const std::string& name = "", int tile_print_offset = 0);

std::vector<uint32_t> pack_bfloat16_vec_into_uint32_vec(const std::vector<bfloat16>& data);

bfloat16 bfloat16_identity_transform(const bfloat16& input);

std::vector<bfloat16> unpack_uint32_vec_into_bfloat16_vec(
    const std::vector<std::uint32_t>& data,
    const std::function<bfloat16(const bfloat16&)>& transform = bfloat16_identity_transform);

// Equality functions
bool equal_within_n_sig_figs(float a, float b, int n);

bool equal_within_absolute_tolerance(float a, float b, float tol);

// this follows the implementation of numpy's is_close
bool is_close(float a, float b, float rtol = 0.01f, float atol = 0.001f);

bool packed_uint32_t_vector_comparison(
    const std::vector<uint32_t>& vec_a,
    const std::vector<uint32_t>& vec_b,
    const std::function<bool(float, float)>& comparison_function,
    int* argfail = nullptr);
