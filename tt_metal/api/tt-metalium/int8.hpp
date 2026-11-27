// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

[[deprecated("This API is deprecated without replacement, it will be removed after 2026-11-12.")]]
uint32_t pack_four_int8_into_uint32(int8_t a, int8_t b, int8_t c, int8_t d);

[[deprecated("This API is deprecated without replacement, it will be removed after 2026-11-12.")]]
std::vector<uint32_t> create_random_vector_of_int8(size_t num_bytes, int seed);
