// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tt-metalium/float8.hpp>

// Unpacks a packed uint32 vector (4 fp8 bytes per word) into float8_e4m3 values.
std::vector<float8_e4m3> unpack_uint32_vec_into_float8_e4m3_vec(const std::vector<uint32_t>& data);

// Generates num_bytes fp8_e4m3 values from a uniform distribution U(0, rand_max_float) + offset, packed 4 per uint32.
std::vector<uint32_t> create_random_vector_of_float8_e4m3(
    size_t num_bytes, int rand_max_float, int seed, float offset = 0.0f);
