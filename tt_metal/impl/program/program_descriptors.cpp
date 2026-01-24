// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tile.hpp>

#include "impl/buffers/semaphore.hpp"
#include "tt_stl/overloaded.hpp"

#include <set>

namespace tt::tt_metal {

TileDescriptor::TileDescriptor(const Tile& tile) :
    height(tile.get_height()), width(tile.get_width()), transpose(tile.get_transpose_of_faces()) {}

std::optional<uint32_t> ProgramDescriptor::find_available_semaphore_id(
    const CoreCoord& core, CoreType core_type) const {
    std::bitset<NUM_SEMAPHORES> used_semaphores;

    // check existing semaphores
    for (const auto& sem_desc : semaphores) {
        if (sem_desc.core_type == core_type && sem_desc.core_ranges.contains(core)) {
            used_semaphores.set(sem_desc.id);
        }
    }

    // find first available semaphore ID
    for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
        if (!used_semaphores.test(i)) {
            return i;
        }
    }
    return std::nullopt;
}

namespace {

// Helper function to check if two CoreRangeSets overlap
bool core_ranges_overlap(const CoreRangeSet& a, const CoreRangeSet& b) {
    for (const auto& range_a : a.ranges()) {
        for (const auto& range_b : b.ranges()) {
            if (range_a.intersects(range_b)) {
                return true;
            }
        }
    }
    return false;
}

// Helper function to collect all core ranges from kernels in a descriptor
CoreRangeSet collect_all_kernel_core_ranges(const ProgramDescriptor& desc) {
    std::set<CoreRange> all_ranges;
    for (const auto& kernel : desc.kernels) {
        for (const auto& range : kernel.core_ranges.ranges()) {
            all_ranges.insert(range);
        }
    }
    return CoreRangeSet(all_ranges);
}

}  // namespace

void ProgramDescriptor::merge(const ProgramDescriptor& other) {
    // Check for overlapping core ranges in kernels
    CoreRangeSet this_ranges = collect_all_kernel_core_ranges(*this);
    CoreRangeSet other_ranges = collect_all_kernel_core_ranges(other);

    TT_FATAL(
        !core_ranges_overlap(this_ranges, other_ranges),
        "Cannot merge ProgramDescriptors with overlapping kernel core ranges. "
        "Ensure that each descriptor operates on a distinct set of cores.");

    // Merge kernels
    for (const auto& kernel : other.kernels) {
        kernels.push_back(kernel);
    }

    // Merge semaphores
    for (const auto& sem : other.semaphores) {
        semaphores.push_back(sem);
    }

    // Merge circular buffers
    for (const auto& cb : other.cbs) {
        cbs.push_back(cb);
    }

    // Custom program hash is invalidated after merge since it's no longer a single operation
    custom_program_hash = std::nullopt;
}

ProgramDescriptor ProgramDescriptor::merge_descriptors(const std::vector<ProgramDescriptor>& descriptors) {
    if (descriptors.empty()) {
        return ProgramDescriptor{};
    }

    if (descriptors.size() == 1) {
        return descriptors[0];
    }

    // Check all pairs of descriptors for overlapping core ranges
    for (size_t i = 0; i < descriptors.size(); ++i) {
        CoreRangeSet ranges_i = collect_all_kernel_core_ranges(descriptors[i]);
        for (size_t j = i + 1; j < descriptors.size(); ++j) {
            CoreRangeSet ranges_j = collect_all_kernel_core_ranges(descriptors[j]);
            TT_FATAL(
                !core_ranges_overlap(ranges_i, ranges_j),
                "Cannot merge ProgramDescriptors with overlapping kernel core ranges between descriptor {} and {}. "
                "Ensure that each descriptor operates on a distinct set of cores.",
                i,
                j);
        }
    }

    // Create the merged descriptor starting from the first one
    ProgramDescriptor result = descriptors[0];

    // Merge all subsequent descriptors
    for (size_t i = 1; i < descriptors.size(); ++i) {
        const auto& other = descriptors[i];

        // Merge kernels
        for (const auto& kernel : other.kernels) {
            result.kernels.push_back(kernel);
        }

        // Merge semaphores
        for (const auto& sem : other.semaphores) {
            result.semaphores.push_back(sem);
        }

        // Merge circular buffers
        for (const auto& cb : other.cbs) {
            result.cbs.push_back(cb);
        }
    }

    // Custom program hash is invalidated after merge
    result.custom_program_hash = std::nullopt;

    return result;
}

}  // namespace tt::tt_metal
