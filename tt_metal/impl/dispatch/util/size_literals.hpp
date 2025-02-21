// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// KiB Prefix literal
constexpr auto operator""_KB(const unsigned long long v) -> uint32_t { return 1024 * v; }

// MiB prefix literal
constexpr auto operator""_MB(const unsigned long long v) -> uint32_t { return 1024 * 1024 * v; }

// GiB prefix literal
constexpr auto operator""_GB(const unsigned long long v) -> uint32_t { return 1024 * 1024 * 1024 * v; }

// Returns the size rounded up to the given alignment
inline uint32_t round_size(uint32_t sz, uint32_t alignment) { return ((sz + alignment - 1) / alignment * alignment); }

}  // namespace tt::tt_metal
