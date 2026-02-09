// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <utility>
#include <functional>
#include <tt-metalium/bfloat16.hpp>

std::pair<bfloat16, bfloat16> unpack_two_bfloat16_from_uint32(uint32_t uint32_data);

std::vector<std::uint32_t> create_arange_vector_of_bfloat16(size_t num_bytes, bool print = true);

std::vector<uint16_t> u16_from_u32_vector(const std::vector<uint32_t>& in);

std::vector<uint32_t> u32_from_u16_vector(const std::vector<uint16_t>& in);

void print_vec_of_uint32_as_packed_bfloat16(
    const std::vector<std::uint32_t>& vec, int num_tiles, const std::string& name = "", int tile_print_offset = 0);

void print_vec_of_bfloat16(
    const std::vector<bfloat16>& vec, int num_tiles, const std::string& name = "", int tile_print_offset = 0);

bool packed_uint32_t_vector_comparison(
    const std::vector<uint32_t>& vec_a,
    const std::vector<uint32_t>& vec_b,
    const std::function<bool(float, float)>& comparison_function,
    int* argfail = nullptr);
