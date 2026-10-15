// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief Pack four signed bytes into a word, with a in the least significant byte.
 * @deprecated No replacement; remove calls to this API. This API will be removed after 2026-11-12.
 */
[[deprecated("No replacement; remove calls to this API. This API will be removed after 2026-11-12.")]]
uint32_t pack_four_int8_into_uint32(int8_t a, int8_t b, int8_t c, int8_t d);

/**
 * @brief Generate num_bytes random int8 values, packed four per uint32.
 * @pre num_bytes must be divisible by four.
 * @deprecated No replacement; remove calls to this API. This API will be removed after 2026-11-12.
 */
[[deprecated("No replacement; remove calls to this API. This API will be removed after 2026-11-12.")]]
std::vector<uint32_t> create_random_vector_of_int8(size_t num_bytes, int seed);
