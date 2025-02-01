// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Si KB Prefix
constexpr auto operator""_KB(const unsigned long long v) -> uint32_t { return 1024 * v; }

// Returns the size rounded up to the given alignment
inline uint32_t round_size(uint32_t sz, uint32_t alignment) {
    return ((sz + alignment - 1) / alignment * alignment);
}

}  // namespace tt::tt_metal
