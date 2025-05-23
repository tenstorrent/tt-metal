// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

// Number of elements needs to be divisible by 4 for the time being
std::vector<std::uint32_t> create_constant_vector_of_uint8(uint32_t num_bytes, uint8_t value);
