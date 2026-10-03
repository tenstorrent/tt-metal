// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

uint32_t pack_four_uint8_into_uint32(uint8_t a, uint8_t b, uint8_t c, uint8_t d);

// Compatibility shim; use tests/tt_metal/test_utils/uint8_utils.hpp instead.
[[deprecated(
    "Use tt::test_utils::create_random_packed_uint8 from tests/tt_metal/test_utils/uint8_utils.hpp instead. "
    "This API will be removed after 2026-11-01.")]]
std::vector<uint32_t> create_random_vector_of_uint8(size_t num_bytes, int seed);
