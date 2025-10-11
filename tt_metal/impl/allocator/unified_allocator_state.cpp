// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/unified_allocator_state.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/assert.hpp>
#include <enchantum/enchantum.hpp>

#include <algorithm>
#include <vector>

namespace tt {
namespace tt_metal {

// ============================================================================
// UnifiedAllocatorState Implementation
// ============================================================================

void UnifiedAllocatorState::merge(const UnifiedAllocatorState& other) {
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

void UnifiedAllocatorState::normalize() {
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
        // Adjacent means curr.first == last.second (no gap between regions)
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

bool UnifiedAllocatorState::is_compatible_with(const UnifiedAllocatorState& other) const {
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

DeviceAddr UnifiedAllocatorState::total_allocated_size() const {
    DeviceAddr total = 0;
    for (const auto& [start, end] : allocated_regions) {
        total += (end - start);
    }
    return total;
}

bool UnifiedAllocatorState::has_conflict(DeviceAddr start_addr, DeviceAddr end_addr) const {
    // Check if [start_addr, end_addr) overlaps with any allocated region
    for (const auto& [region_start, region_end] : allocated_regions) {
        // Check for overlap: ranges overlap if start1 < end2 && start2 < end1
        if (start_addr < region_end && region_start < end_addr) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// CompleteUnifiedAllocatorState Implementation
// ============================================================================

void CompleteUnifiedAllocatorState::merge(const CompleteUnifiedAllocatorState& other) {
    // Merge each buffer type
    for (const auto& [buffer_type, other_state] : other.states_per_buffer_type) {
        auto it = states_per_buffer_type.find(buffer_type);
        if (it != states_per_buffer_type.end()) {
            // Buffer type exists in both, merge states
            it->second.merge(other_state);
        } else {
            // Buffer type only in other, copy it
            states_per_buffer_type[buffer_type] = other_state;
        }
    }

    // Merge buffer pointers
    all_allocated_buffers.insert(
        all_allocated_buffers.end(), other.all_allocated_buffers.begin(), other.all_allocated_buffers.end());
}

DeviceAddr CompleteUnifiedAllocatorState::total_allocated_size() const {
    DeviceAddr total = 0;
    for (const auto& [buffer_type, state] : states_per_buffer_type) {
        total += state.total_allocated_size();
    }
    return total;
}

bool CompleteUnifiedAllocatorState::has_buffer_type(BufferType buffer_type) const {
    return states_per_buffer_type.find(buffer_type) != states_per_buffer_type.end();
}

// ============================================================================
// Global Helper Functions
// ============================================================================

CompleteUnifiedAllocatorState compute_unified_state(const std::vector<Allocator*>& allocators) {
    CompleteUnifiedAllocatorState unified_state;

    if (allocators.empty()) {
        return unified_state;
    }

    // Merge states for each buffer type
    std::array<BufferType, 4> buffer_types = {
        BufferType::DRAM, BufferType::L1, BufferType::L1_SMALL, BufferType::TRACE};

    for (const auto& buffer_type : buffer_types) {
        UnifiedAllocatorState type_state;
        bool first = true;

        for (Allocator* allocator : allocators) {
            if (!allocator) {
                continue;
            }

            try {
                auto state = allocator->extract_state(buffer_type);
                if (first) {
                    type_state = std::move(state);
                    first = false;
                } else {
                    type_state.merge(state);
                }
            } catch (...) {
                // Skip allocators that don't have this buffer type or encounter errors
                continue;
            }
        }

        if (!first) {  // At least one allocator had this buffer type
            unified_state.states_per_buffer_type[buffer_type] = std::move(type_state);
        }
    }

    // Collect all buffer pointers
    for (Allocator* allocator : allocators) {
        if (!allocator) {
            continue;
        }

        auto buffers = allocator->get_allocated_buffers();
        unified_state.all_allocated_buffers.reserve(unified_state.all_allocated_buffers.size() + buffers.size());
        unified_state.all_allocated_buffers.insert(
            unified_state.all_allocated_buffers.end(), buffers.begin(), buffers.end());
    }

    return unified_state;
}

UnifiedAllocatorState compute_unified_state(const std::vector<Allocator*>& allocators, const BufferType& buffer_type) {
    UnifiedAllocatorState unified_state;
    bool first = true;

    for (Allocator* allocator : allocators) {
        if (!allocator) {
            continue;
        }

        try {
            auto state = allocator->extract_state(buffer_type);
            if (first) {
                unified_state = std::move(state);
                first = false;
            } else {
                unified_state.merge(state);
            }
        } catch (...) {
            // Skip allocators that don't have this buffer type or encounter errors
            continue;
        }
    }

    return unified_state;
}

UnifiedAllocatorState compute_unified_state(
    const std::vector<const Allocator*>& allocators, const BufferType& buffer_type) {
    // Convert const vector to non-const for reuse
    // (extract_state is const, so this is safe)
    std::vector<Allocator*> non_const_allocators;
    non_const_allocators.reserve(allocators.size());
    for (const Allocator* alloc : allocators) {
        non_const_allocators.push_back(const_cast<Allocator*>(alloc));
    }

    return compute_unified_state(non_const_allocators, buffer_type);
}

}  // namespace tt_metal
}  // namespace tt
