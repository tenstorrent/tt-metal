// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt {

inline constexpr uint32_t align(uint32_t addr, uint32_t alignment) { return ((addr - 1) | (alignment - 1)) + 1; }

}  // namespace tt
