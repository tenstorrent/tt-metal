// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

uint32_t pack_four_int8_into_uint32(int8_t a, int8_t b, int8_t c, int8_t d);

// Generates num_bytes int8 values from a uniform distribution, packed 4 per uint32.
// Note: num_bytes must be divisible by 4.
std::vector<uint32_t> create_random_vector_of_int8(size_t num_bytes, int seed);
