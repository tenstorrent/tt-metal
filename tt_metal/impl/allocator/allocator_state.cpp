// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/allocator_state.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>
#include <enchantum/enchantum.hpp>

#include <algorithm>
#include <vector>

namespace tt {
namespace tt_metal {

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

    // Merge source tracking if both states track sources
    if (!region_source_allocator_ids.empty() && !other.region_source_allocator_ids.empty()) {
        region_source_allocator_ids.reserve(
            region_source_allocator_ids.size() + other.region_source_allocator_ids.size());
        region_source_allocator_ids.insert(
            region_source_allocator_ids.end(),
            other.region_source_allocator_ids.begin(),
            other.region_source_allocator_ids.end());
    } else {
        // If either state doesn't track sources, disable tracking
        region_source_allocator_ids.clear();
    }

    // Normalize to sort and coalesce regions
    normalize();
}

void AllocatorState::BufferTypeState::normalize() {
    if (allocated_regions.empty()) {
        return;
    }

    // Sort by start address
    // If tracking sources, we need to sort both vectors in parallel
    if (!region_source_allocator_ids.empty() && region_source_allocator_ids.size() == allocated_regions.size()) {
        // Create index vector for indirect sorting
        std::vector<size_t> indices(allocated_regions.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        // Sort indices by allocated_regions start addresses
        std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
            return allocated_regions[a].first < allocated_regions[b].first;
        });

        // Apply sorted order to both vectors
        std::vector<std::pair<DeviceAddr, DeviceAddr>> sorted_regions;
        std::vector<uint32_t> sorted_sources;
        sorted_regions.reserve(allocated_regions.size());
        sorted_sources.reserve(region_source_allocator_ids.size());

        for (size_t idx : indices) {
            sorted_regions.push_back(allocated_regions[idx]);
            sorted_sources.push_back(region_source_allocator_ids[idx]);
        }

        allocated_regions = std::move(sorted_regions);
        region_source_allocator_ids = std::move(sorted_sources);
    } else {
        // No source tracking or size mismatch, simple sort
        region_source_allocator_ids.clear();
        std::sort(allocated_regions.begin(), allocated_regions.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
    }

    // Coalesce overlapping and adjacent regions
    std::vector<std::pair<DeviceAddr, DeviceAddr>> coalesced;
    coalesced.reserve(allocated_regions.size());
    coalesced.push_back(allocated_regions[0]);

    for (size_t i = 1; i < allocated_regions.size(); ++i) {
        const auto& curr = allocated_regions[i];
        auto& last = coalesced.back();

        // If current region overlaps or is adjacent to last, merge them
        // Overlapping means curr.first < last.second, adjacent means curr.first == last.second (no gap between regions)
        // The condition below handles both cases.
        if (curr.first <= last.second) {
            last.second = std::max(last.second, curr.second);
        } else {
            coalesced.push_back(curr);
        }
    }

    allocated_regions = std::move(coalesced);

    // After coalescing, source tracking is no longer accurate, so clear it
    region_source_allocator_ids.clear();
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

    // Merge buffer pointers
    all_allocated_buffers_.insert(
        all_allocated_buffers_.end(), other.all_allocated_buffers_.begin(), other.all_allocated_buffers_.end());
}

DeviceAddr AllocatorState::total_allocated_size() const {
    DeviceAddr total = 0;
    for (const auto& [buffer_type, state] : states_per_buffer_type_) {
        total += state.total_allocated_size();
    }
    return total;
}

bool AllocatorState::has_buffer_type(BufferType buffer_type) const {
    return states_per_buffer_type_.find(buffer_type) != states_per_buffer_type_.end();
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

AllocatorState::BufferTypeState& AllocatorState::get_or_create_buffer_type_state(BufferType buffer_type) {
    auto it = states_per_buffer_type_.find(buffer_type);
    if (it == states_per_buffer_type_.end()) {
        it = states_per_buffer_type_.emplace(buffer_type, BufferTypeState{}).first;
        it->second.buffer_type = buffer_type;
    }
    return it->second;
}

const AllocatorState::BufferTypeState* AllocatorState::get_buffer_type_state(BufferType buffer_type) const {
    auto it = states_per_buffer_type_.find(buffer_type);
    if (it == states_per_buffer_type_.end()) {
        return nullptr;
    }
    return &it->second;
}

}  // namespace tt_metal
}  // namespace tt
