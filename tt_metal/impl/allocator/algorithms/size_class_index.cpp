// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/allocator/algorithms/size_class_index.hpp"

#include <algorithm>
#include <bit>

namespace tt::tt_metal::allocator {

namespace {

size_t num_size_classes(DeviceAddr max_size_bytes) {
    // One class per bit of max_size_bytes / base_size, at least 2. A last class starting at 128 MiB
    // should be enough, and more classes cost time to iterate.
    const auto count = static_cast<size_t>(std::bit_width(max_size_bytes / SizeClassIndex::base_size));
    const auto max_count = static_cast<size_t>(std::bit_width(128 * 1024 * 1024 / SizeClassIndex::base_size));
    return std::clamp(count, size_t{2}, max_count);
}

}  // namespace

SizeClassIndex::SizeClassIndex(DeviceAddr max_size_bytes) :
    num_classes_{num_size_classes(max_size_bytes)}, classes_{num_classes_} {}

void SizeClassIndex::reserve(size_t blocks_per_class) {
    for (auto& free_blocks : classes_) {
        free_blocks.reserve(blocks_per_class);
    }
}

void SizeClassIndex::clear() {
    for (auto& free_blocks : classes_) {
        free_blocks.clear();
    }
}

}  // namespace tt::tt_metal::allocator
