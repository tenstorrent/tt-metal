// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "tt_metal/common/assert.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

class bfloat16 {
 private:
    uint16_t uint16_data;

 public:
    static const size_t SIZEOF = 2;
    bfloat16() {
    }

    // create from float: no rounding, just truncate
    bfloat16(float float_num) {
        uint32_t uint32_data;
        TT_ASSERT (sizeof float_num == sizeof uint32_data);

        uint32_data = *reinterpret_cast<uint32_t*>(&float_num);
        // just move upper 16 to lower 16 (truncate)
        uint32_data = (uint32_data >> 16);

        // store lower 16 as 16-bit uint
        uint16_data = (uint16_t)uint32_data;
    }

    // store lower 16 as 16-bit uint
    bfloat16(uint32_t uint32_data) {
        uint16_data = (uint16_t)uint32_data;
    }

    bfloat16(uint16_t uint16_data_) {
        uint16_data = uint16_data_;
    }

    // store lower 16 as 16-bit uint
    bfloat16(int int_data) {
        uint16_data = (uint16_t)int_data;
    }

    float to_float() const {
        // move lower 16 to upper 16 (of 32) and convert to float
        uint32_t uint32_data = (uint32_t)uint16_data << 16;
        float f;
        std::memcpy(&f, &uint32_data, sizeof(f));
        return f;
    }
    uint16_t to_packed() const {
        return uint16_data;
    }
    uint16_t to_uint16() const {
        return uint16_data;
    }
    bool operator==(const bfloat16 rhs) const {
        return uint16_data == rhs.uint16_data;
    }
    bool operator!=(const bfloat16 rhs) const {
        return not (*this == rhs);
    }
    bfloat16 operator*(const bfloat16 rhs) const {
        return bfloat16(this->to_float() * rhs.to_float());
    }
};

inline std::ostream& operator<<(std::ostream& os, const bfloat16& bfp16)
{
    os << bfp16.to_uint16();
    return os;
}

inline bool operator==(const std::vector<bfloat16>& lhs, const std::vector<bfloat16>& rhs)
{
    bool is_equal = lhs.size() == rhs.size();
    for(auto i = 0 ; i < lhs.size(); i++) {
        is_equal &= (lhs[i].to_uint16() == rhs[i].to_uint16());
    }
    return is_equal;
}

inline uint32_t pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16> two_bfloats) {
    // first -> lower 16
    // second -> upper 16
    return (uint32_t)two_bfloats.first.to_uint16() | ((uint32_t)two_bfloats.second.to_uint16() << 16);
}

inline std::pair<bfloat16, bfloat16> unpack_two_bfloat16_from_uint32(uint32_t uint32_data) {
    std::pair<bfloat16, bfloat16> two_bfloats;

    two_bfloats.first = bfloat16(uint32_data & 0xffff); // lower 16
    two_bfloats.second = bfloat16(uint32_data >> 16); // upper 16

    return two_bfloats;
}

inline std::vector<std::uint32_t> create_arange_vector_of_bfloat16(uint32_t num_bytes, bool print = true) {
    std::vector<std::uint32_t> vec(num_bytes/sizeof(std::uint32_t), 0);
     for (int i = 0; i < vec.size(); i++) {
        float num_1_float = i * 2;
        float num_2_float = i * 2 + 1;

        bfloat16 num_1_bfloat16 = bfloat16(num_1_float);
        bfloat16 num_2_bfloat16 = bfloat16(num_2_float);

        if (print) {
            std::cout << "num_1_float: " << num_1_float << ", num_1_bfloat16 : " << num_1_bfloat16.to_float() << ", \t\t";
            std::cout << "num_2_float: " << num_2_float << ", num_2_bfloat16 : " << num_2_bfloat16.to_float() << std::endl;
        }

        // pack 2 uint16 into uint32
        vec.at(i) = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }

    return vec;
}

inline std::vector<bfloat16> create_random_vector_of_bfloat16_native(uint32_t num_bytes, float rand_max_float, int seed, float offset = 0.0f) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    std::vector<bfloat16> vec(num_bytes/sizeof(bfloat16), 0);
    for (int i = 0; i < vec.size(); i++) {
        float num_1_float = rand_float() + offset;
        vec[i] = bfloat16(num_1_float);
    }
    return vec;
}

inline void print_golden_metalium_vectors(std::vector<bfloat16>& golden_vec, std::vector<bfloat16>& result_vec) {
    std::cout << "-- index -- golden -- metalium --" << std::endl;
    for (int i = 0; i < result_vec.size(); i++) {
        float a1 = golden_vec[i].to_float();
        float a2 = result_vec[i].to_float();
        if (i % 128 == 0){
            std::cout << "-- " << i << " -- " << a1 << " <--> " << a2 << std::endl;
        }
    }
}


inline std::vector<std::uint32_t> create_random_vector_of_bfloat16(uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    std::vector<std::uint32_t> vec(num_bytes/sizeof(std::uint32_t), 0);
    for (int i = 0; i < vec.size(); i++) {
        float num_1_float = rand_float() + offset;
        float num_2_float = rand_float() + offset;

        bfloat16 num_1_bfloat16 = bfloat16(num_1_float);
        bfloat16 num_2_bfloat16 = bfloat16(num_2_float);

        //std::cout << "num_1_float: " << num_1_float << ", num_1_bfloat16 : " << num_1_bfloat16.to_float() << ", \t\t";
        //std::cout << "num_2_float: " << num_2_float << ", num_2_bfloat16 : " << num_2_bfloat16.to_float() << std::endl;

        // pack 2 uint16 into uint32
        vec.at(i) = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }

    return vec;
}

inline std::vector<std::uint32_t> create_random_vector_of_bfloat16_1_1(uint32_t num_bytes, int seed) {
    return create_random_vector_of_bfloat16(num_bytes, 2.0f, seed, -1.0f); // -1.0..1.0
}

inline std::vector<std::uint32_t> create_random_vector_of_bfloat16_0_2(uint32_t num_bytes, int seed) {
    return create_random_vector_of_bfloat16(num_bytes, 2.0f, seed); // 0.0f..2.0f
}


/*
 * rk: Still won't handle the case where the number of elements is odd, except
 * if it's 1. Whatever, for now.
 */
inline std::vector<std::uint32_t> create_constant_vector_of_bfloat16(uint32_t num_bytes, float value) {
    const uint32_t num_elements_vec = std::max(static_cast<uint32_t>(1), static_cast<uint32_t>(num_bytes/sizeof(std::uint32_t))); // always at least have 1
    std::vector<std::uint32_t> vec(num_elements_vec, 0);
    for (int i = 0; i < vec.size(); i++) {
        bfloat16 num_1_bfloat16 = bfloat16(value);

        bfloat16 num_2_bfloat16 = num_elements_vec == 1 ? bfloat16(static_cast<float>(0.0)) : bfloat16(value);

        vec.at(i) = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }

    return vec;
}


// creates a bfloat16 identity matrix with dims (rows x cols)
// each 2 cols will be packed as a single uint32_t
inline std::vector<bfloat16> create_identity_matrix(int rows, int cols, int num_ones) {
    std::vector<bfloat16> vec(rows * cols, (float)0);
    for(int i = 0; i < num_ones; i++) {
        vec.at(i * cols + i) = bfloat16((float)1);
    }
    return vec;
}


// TODO(AP): duplication with above
inline std::vector<uint32_t> create_random_binary_vector_of_bfloat16(uint32_t num_bytes, int seed) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, 1), std::mt19937(seed));

    std::vector<std::uint32_t> vec(num_bytes/sizeof(std::uint32_t), 0);
    for (int i = 0; i < vec.size(); i++) {
        float num_1_float = rand_float();
        float num_2_float = rand_float();

        num_1_float = (num_1_float > 0.5);
        num_2_float = (num_2_float > 0.5);

        bfloat16 num_1_bfloat16 = bfloat16(num_1_float);
        bfloat16 num_2_bfloat16 = bfloat16(num_2_float);

        vec.at(i) = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }
    return vec;
}

inline std::vector<uint16_t> u16_from_u32_vector(const std::vector<uint32_t>& in) {
    std::vector<uint16_t> result(in.size()*2);
    for (int i = 0; i < in.size(); i++) {
        uint32_t val = in.at(i);
        auto two_bfloats = unpack_two_bfloat16_from_uint32(val);
        result[i*2  ] = two_bfloats.first.to_uint16();
        result[i*2+1] = two_bfloats.second.to_uint16();
    }
    return result;
}

inline std::vector<uint32_t> u32_from_u16_vector(const std::vector<uint16_t>& in) {
    std::vector<uint32_t> result(in.size()/2);
    TT_ASSERT(in.size() % 2 == 0);
    for(auto i = 0; i < in.size(); i+=2) {
        auto val1 = bfloat16(in.at(i));
        auto val2 = bfloat16(in.at(i+1));
        auto packed = pack_two_bfloat16_into_uint32(std::make_pair(val1, val2));
        result[i/2] = packed;
    }
    return result;
}

inline void print_vec_of_uint32_as_packed_bfloat16(std::vector<std::uint32_t> vec, int num_tiles, std::string name = "", int tile_print_offset = 0) {
    int idx = 0;
    for (int i = 0; i < num_tiles; i++) {
        std::cout << name << " tile " << i + tile_print_offset << std::endl;
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k+=2) {
                uint32_t uint32_data = vec.at(idx);
                std::pair<bfloat16, bfloat16> two_bfloats = unpack_two_bfloat16_from_uint32(uint32_data);
                std::cout << two_bfloats.first.to_float() << ", " << two_bfloats.second.to_float() << ", ";
                idx++;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

inline void print_vec_of_bfloat16(std::vector<bfloat16> vec, int num_tiles, std::string name = "", int tile_print_offset = 0) {
    int idx = 0;
    for (int i = 0; i < num_tiles; i++) {
        std::cout << name << " tile " << i + tile_print_offset << std::endl;
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k++) {
                std::cout << vec.at(idx).to_float() << ", " ;
                idx++;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

inline void print_vec(std::vector<uint32_t> vec, int num_tiles, std::string name = "", int tile_print_offset = 0) {
    int idx = 0;
    for (int i = 0; i < num_tiles; i++) {
        std::cout << name << " tile " << i + tile_print_offset << std::endl;
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k++) {
                std::cout << vec.at(idx) << ", " ;
                idx++;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

inline std::vector<uint32_t> pack_bfloat16_vec_into_uint32_vec(const std::vector<bfloat16>& data) {
    ZoneScoped;
    TT_ASSERT(data.size() % 2 == 0);
    std::vector<uint32_t> result(data.size()/2);
    std::memcpy (result.data(), data.data(), result.size()*sizeof(uint32_t));
    return result;
}

inline bfloat16 bfloat16_identity_transform(const bfloat16 &input) {
    return input;
}

inline std::vector<bfloat16> unpack_uint32_vec_into_bfloat16_vec(
    const std::vector<std::uint32_t>& data,
    std::function<bfloat16(const bfloat16 &)> transform = bfloat16_identity_transform
) {
    std::vector<bfloat16> result;
    for(auto i = 0 ; i < data.size(); i++) {
        auto unpacked = unpack_two_bfloat16_from_uint32(data[i]);
        result.push_back(transform(unpacked.first));
        result.push_back(transform(unpacked.second));
    }
    return result;
}

// Equality functions
inline bool equal_within_n_sig_figs(float a, float b, int n) {
    std::string str_a = std::to_string(a);
    std::string str_b = std::to_string(b);

    // Iterate until no more zeroes
    int i = 0;
    while (i < std::min(str_a.size(), str_b.size()) and (str_a.at(i) == '0' or str_a.at(i) == '.')) {
        i++;
    }

    // Compare sig figs
    int num_correct_sig_figs = 0;
    for (; i < std::min(str_a.size(), str_b.size()); i++) {
        char cur_char = str_a.at(i);

        if (cur_char == str_b.at(i)) {
            if (cur_char != '.') { // Ignore decimal point
                num_correct_sig_figs++;
            }
        } else {
            std::cout << "Floats being compared: A: " << a << ", B: " << b << std::endl;
            return false;
        }

        if (num_correct_sig_figs == n) {
            break;
        }
    }

    return true;
};

inline bool equal_within_absolute_tolerance(float a, float b, float tol) {
    return std::abs(a - b) < tol;
}

// this follows the implementation of numpy::is_close
inline bool is_close(float a, float b, float rtol = 0.01f, float atol = 0.001f) {
    // the idea is near zero we want absolute tolerance since relative doesn't make sense
    // (consider 1e-6f and 1.1e-6f)
    // elsewhere (not near zero) we want relative tolerance
    auto absdiff = fabsf(a - b);
    auto reldenom = fmaxf(fabsf(a), fabsf(b));
    auto result = (absdiff <= atol) || (absdiff <= rtol*reldenom);
    if (result != true) {
        std::cout << "Discrepacy: A = " << a << " B = " << b << std::endl;
        std::cout << "   absdiff = " << absdiff << std::endl;
        std::cout << "   reldiff = " << absdiff/(reldenom+1e-6f) << std::endl;
    }
    return result;
}

inline bool packed_uint32_t_vector_comparison(
    const std::vector<uint32_t> &vec_a, const std::vector<uint32_t> &vec_b,
    std::function<bool(float, float)> comparison_function,
    int* argfail = nullptr
) {
    if (vec_a.size() != vec_b.size()) {
        std::cout << "Sizes don't match, returning false" << std::endl;
        return false;
    }

    for (int i = 0; i < vec_a.size(); i++) {
        std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(vec_a.at(i));
        std::pair<bfloat16, bfloat16> bs = unpack_two_bfloat16_from_uint32(vec_b.at(i));

        float a1 = as.first.to_float();
        float b1 = bs.first.to_float();

        float a2 = as.second.to_float();
        float b2 = bs.second.to_float();

        if (not (comparison_function(a1, b1) and comparison_function(a2, b2)))  {
            if (argfail) {
                *argfail = i;
                std::cout << "a1 = " << std::hex << a1 << std::endl;
                std::cout << "b1 = " << std::hex << b1 << std::endl;
                std::cout << "a2 = " << std::hex << a2 << std::endl;
                std::cout << "b2 = " << std::hex << b2 << std::endl;
            }
            return false;
        }
    }

    return true;
}
