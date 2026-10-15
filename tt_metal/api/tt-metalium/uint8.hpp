// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief Pack four unsigned bytes into a word, with a in the least significant byte.
 * @deprecated No replacement; remove calls to this API. This API will be removed after 2026-11-12.
 */
[[deprecated("No replacement; remove calls to this API. This API will be removed after 2026-11-12.")]]
uint32_t pack_four_uint8_into_uint32(uint8_t a, uint8_t b, uint8_t c, uint8_t d);

/**
 * @brief Generate num_bytes random uint8 values, packed four per uint32.
 * @pre num_bytes must be divisible by four.
 * @deprecated No replacement; remove calls to this API. This API will be removed after 2026-11-12.
 */
[[deprecated("No replacement; remove calls to this API. This API will be removed after 2026-11-12.")]]
std::vector<uint32_t> create_random_vector_of_uint8(size_t num_bytes, int seed);
