// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt_stl/aligned_allocator.hpp>

namespace tt::tt_metal {

static constexpr uint32_t MEMCPY_ALIGNMENT = 16;  // sizeof(__m128i);

template <typename T>
using vector_aligned = std::vector<T, tt::stl::aligned_allocator<T, MEMCPY_ALIGNMENT>>;

}  // namespace tt::tt_metal
