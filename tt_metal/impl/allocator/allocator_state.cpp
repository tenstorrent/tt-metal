// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "allocator_state.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>
#include <enchantum/enchantum.hpp>
#include "tt_metal/impl/allocator/algorithms/address_ranges.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace tt::tt_metal {

void AllocatorState::BufferTypeState::merge(const BufferTypeState& other) {
    // Validate compatibility before merging
    TT_FATAL(
        is_compatible_with(other),
        "Cannot merge incompatible allocator states: buffer_type {}, other.buffer_type {}",
        enchantum::to_string(buffer_type),
        enchantum::to_string(other.buffer_type));

    // Merge allocated regions
    allocated_regions.reserve(allocated_regions.size() + other.allocated_regions.size());
    allocated_regions.insert(allocated_regions.end(), other.allocated_regions.begin(), other.allocated_regions.end());

    // Normalize to sort and coalesce regions
    normalize();
}

void AllocatorState::BufferTypeState::normalize() {
    if (allocated_regions.empty()) {
        return;
    }

    allocated_regions = allocator::coalesce(std::move(allocated_regions));
}

bool AllocatorState::BufferTypeState::is_compatible_with(const BufferTypeState& other) const {
    // Check buffer type
    if (buffer_type != other.buffer_type) {
        return false;
    }

    // Check bank configuration
    if (num_banks != other.num_banks) {
        return false;
    }

    if (bank_size != other.bank_size) {
        return false;
    }

    if (alignment_bytes != other.alignment_bytes) {
        return false;
    }

    if (interleaved_address_limit != other.interleaved_address_limit) {
        return false;
    }

    // Check bank offsets match
    if (bank_id_to_bank_offset.size() != other.bank_id_to_bank_offset.size()) {
        return false;
    }

    for (const auto& [bank_id, offset] : bank_id_to_bank_offset) {
        auto it = other.bank_id_to_bank_offset.find(bank_id);
        if (it == other.bank_id_to_bank_offset.end() || it->second != offset) {
            return false;
        }
    }

    return true;
}

DeviceAddr AllocatorState::BufferTypeState::total_allocated_size() const {
    DeviceAddr total = 0;
    for (const auto& [start, end] : allocated_regions) {
        total += (end - start);
    }
    return total;
}

void AllocatorState::merge(const AllocatorState& other) {
    // Merge each buffer type
    for (const auto& [buffer_type, other_state] : other.states_per_buffer_type_) {
        auto it = states_per_buffer_type_.find(buffer_type);
        if (it != states_per_buffer_type_.end()) {
            // Buffer type exists in both, merge states
            it->second.merge(other_state);
        } else {
            // Buffer type only in other, copy it
            states_per_buffer_type_[buffer_type] = other_state;
        }
    }
}

DeviceAddr AllocatorState::total_allocated_size() const {
    DeviceAddr total = 0;
    for (const auto& [buffer_type, state] : states_per_buffer_type_) {
        total += state.total_allocated_size();
    }
    return total;
}

const std::vector<std::pair<DeviceAddr, DeviceAddr>>& AllocatorState::get_allocated_regions(
    BufferType buffer_type) const {
    static const std::vector<std::pair<DeviceAddr, DeviceAddr>> empty_vector;
    auto it = states_per_buffer_type_.find(buffer_type);
    if (it == states_per_buffer_type_.end()) {
        return empty_vector;
    }
    return it->second.allocated_regions;
}

}  // namespace tt::tt_metal
