// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <bit>

#include "bfloat16_utils.hpp"

#include <tt_stl/assert.hpp>

std::pair<bfloat16, bfloat16> unpack_two_bfloat16_from_uint32(uint32_t uint32_data) {
    std::pair<bfloat16, bfloat16> two_bfloats;

    two_bfloats.first = std::bit_cast<bfloat16>(static_cast<uint16_t>(uint32_data & 0xffff));  // lower 16
    two_bfloats.second = std::bit_cast<bfloat16>(static_cast<uint16_t>(uint32_data >> 16));    // upper 16

    return two_bfloats;
}

std::vector<std::uint32_t> create_arange_vector_of_bfloat16(size_t num_bytes, bool print) {
    std::vector<std::uint32_t> vec(num_bytes / sizeof(std::uint32_t), 0);
    for (size_t i = 0; i < vec.size(); i++) {
        float num_1_float = i * 2;
        float num_2_float = (i * 2) + 1;

        bfloat16 num_1_bfloat16 = bfloat16(num_1_float);
        bfloat16 num_2_bfloat16 = bfloat16(num_2_float);

        if (print) {
            std::cout << "num_1_float: " << num_1_float << ", num_1_bfloat16 : " << static_cast<float>(num_1_bfloat16)
                      << ", \t\t";
            std::cout << "num_2_float: " << num_2_float << ", num_2_bfloat16 : " << static_cast<float>(num_2_bfloat16)
                      << std::endl;
        }

        // pack 2 uint16 into uint32
        vec.at(i) = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }

    return vec;
}

std::vector<uint16_t> u16_from_u32_vector(const std::vector<uint32_t>& in) {
    std::vector<uint16_t> result(in.size() * 2);
    for (size_t i = 0; i < in.size(); i++) {
        uint32_t val = in.at(i);
        auto two_bfloats = unpack_two_bfloat16_from_uint32(val);
        result[i * 2] = std::bit_cast<uint16_t>(two_bfloats.first);
        result[(i * 2) + 1] = std::bit_cast<uint16_t>(two_bfloats.second);
    }
    return result;
}

std::vector<uint32_t> u32_from_u16_vector(const std::vector<uint16_t>& in) {
    std::vector<uint32_t> result(in.size() / 2);
    TT_ASSERT(in.size() % 2 == 0);
    for (size_t i = 0; i < in.size(); i += 2) {
        auto val1 = std::bit_cast<bfloat16>(in[i]);
        auto val2 = std::bit_cast<bfloat16>(in.at(i + 1));
        auto packed = pack_two_bfloat16_into_uint32(std::make_pair(val1, val2));
        result[i / 2] = packed;
    }
    return result;
}

void print_vec_of_uint32_as_packed_bfloat16(
    const std::vector<std::uint32_t>& vec, int num_tiles, const std::string& name, int tile_print_offset) {
    int idx = 0;
    for (int i = 0; i < num_tiles; i++) {
        std::cout << name << " tile " << i + tile_print_offset << std::endl;
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k += 2) {
                uint32_t uint32_data = vec.at(idx);
                std::pair<bfloat16, bfloat16> two_bfloats = unpack_two_bfloat16_from_uint32(uint32_data);
                std::cout << static_cast<float>(two_bfloats.first) << ", " << static_cast<float>(two_bfloats.second)
                          << ", ";
                idx++;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void print_vec_of_bfloat16(
    const std::vector<bfloat16>& vec, int num_tiles, const std::string& name, int tile_print_offset) {
    int idx = 0;
    for (int i = 0; i < num_tiles; i++) {
        std::cout << name << " tile " << i + tile_print_offset << std::endl;
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k++) {
                std::cout << static_cast<float>(vec.at(idx)) << ", ";
                idx++;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

bool packed_uint32_t_vector_comparison(
    const std::vector<uint32_t>& vec_a,
    const std::vector<uint32_t>& vec_b,
    const std::function<bool(float, float)>& comparison_function,
    int* argfail) {
    if (vec_a.size() != vec_b.size()) {
        std::cout << "Sizes don't match, returning false" << std::endl;
        return false;
    }

    for (size_t i = 0; i < vec_a.size(); i++) {
        std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(vec_a.at(i));
        std::pair<bfloat16, bfloat16> bs = unpack_two_bfloat16_from_uint32(vec_b.at(i));

        float a1 = static_cast<float>(as.first);
        float b1 = static_cast<float>(bs.first);

        float a2 = static_cast<float>(as.second);
        float b2 = static_cast<float>(bs.second);

        if (not(comparison_function(a1, b1) and comparison_function(a2, b2))) {
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
