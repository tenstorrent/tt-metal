// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

uint32_t pack_four_uint8_into_uint32(uint8_t a, uint8_t b, uint8_t c, uint8_t d);

// Generates num_bytes uint8 values from a uniform distribution, packed 4 per uint32.
std::vector<uint32_t> create_random_vector_of_uint8(size_t num_bytes, int min_val, int max_val, int seed);
