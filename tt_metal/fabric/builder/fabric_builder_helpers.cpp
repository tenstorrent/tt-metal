// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_builder_helpers.hpp"
#include <algorithm>

namespace tt::tt_fabric {

/**
 * Implementation of memory region subtraction for defragmentation.
 */
std::vector<MemoryRegion> subtract_memory_regions(
    const std::vector<MemoryRegion>& available_regions, const std::vector<MemoryRegion>& consumed_regions) {
    // Create a list of all consumed address ranges
    std::vector<std::pair<size_t, size_t>> consumed_ranges;
    for (const auto& consumed : consumed_regions) {
        consumed_ranges.emplace_back(consumed.get_start_address(), consumed.get_end_address());
    }

    // Sort consumed ranges by start address
    std::sort(consumed_ranges.begin(), consumed_ranges.end());

    std::vector<MemoryRegion> remaining_regions;
    // Pre-allocate a reasonable capacity to avoid reallocations
    remaining_regions.reserve(available_regions.size() * 2);

    // For each available region, subtract consumed regions
    for (const auto& available : available_regions) {
        size_t current_start = available.get_start_address();
        size_t available_end = available.get_end_address();

        // Find all consumed regions that overlap with this available region
        for (const auto& consumed : consumed_ranges) {
            size_t consumed_start = consumed.first;
            size_t consumed_end = consumed.second;

            // If consumed region ends before current start, skip it
            if (consumed_end <= current_start) {
                continue;
            }

            // If consumed region starts after available end, we're done with this available region
            if (consumed_start >= available_end) {
                break;
            }

            // If there's a gap before the consumed region, add it as remaining
            if (current_start < consumed_start) {
                remaining_regions.emplace_back(current_start, consumed_start - current_start);
            }

            // Move current start past the consumed region
            current_start = std::max(current_start, consumed_end);
        }

        // Add any remaining part after the last consumed region
        if (current_start < available_end) {
            remaining_regions.emplace_back(current_start, available_end - current_start);
        }
    }

    // Merge contiguous regions
    if (!remaining_regions.empty()) {
        std::sort(remaining_regions.begin(), remaining_regions.end(), [](const MemoryRegion& a, const MemoryRegion& b) {
            return a.get_start_address() < b.get_start_address();
        });

        std::vector<MemoryRegion> merged_regions;
        MemoryRegion current = remaining_regions[0];

        for (size_t i = 1; i < remaining_regions.size(); ++i) {
            if (current.get_end_address() == remaining_regions[i].get_start_address()) {
                // Merge contiguous regions
                current =
                    MemoryRegion(current.get_start_address(), current.get_size() + remaining_regions[i].get_size());
            } else {
                merged_regions.push_back(current);
                current = remaining_regions[i];
            }
        }
        merged_regions.push_back(current);
        remaining_regions = std::move(merged_regions);
    }

    return remaining_regions;
}

}  // namespace tt::tt_fabric
