// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>

#include "allocator.hpp"

namespace tt {

namespace tt_metal {

struct BasicAllocator : Allocator {
    BasicAllocator(const AllocatorConfig& alloc_config);
};

}  // namespace tt_metal

}  // namespace tt
