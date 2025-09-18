// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <allocator_types.hpp>
#include <buffer_types.hpp>
#include <stdint.h>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <functional>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/strong_type.hpp>

#include "algorithms/allocator_algorithm.hpp"
#include "core_coord.hpp"
#include "hal_types.hpp"

namespace tt {

namespace tt_metal {
enum class BufferType;
namespace allocator {
class Algorithm;
}  // namespace allocator

class BankManager {
public:
    BankManager() = default;

    /**
     * @brief Describes dependencies between multiple allocators which share the same memory space.
     *
     * Allocator dependencies are bidirectional and are stored as undirected adjacency lists.
     * Eg. If allocator A depends on allocator B, then A and B cannot allocate in regions occupied by the other.
     * It is used by BankManager to differentiate between different allocators and query dependencies between them.

     * AllocatorDependencies is created from an unordered map of allocator IDs with their dependencies.
     * Some nuances:
        - Default value (AllocatorDependencies() or AllocatorDependencies{}) represents a single independent allocator.
        - The presence of alloctor IDs in keys or values implies the existence of previous allocator IDs.
          Eg. 3: {0, 1} (read as: 3 depends on 0 and 1) implies that 0: {3}, 1: {3}, 2: {}, 3: {0, 1}.
        - Undirected adjacency lists means that 0: 1 implies 1: 0 (as seen in the example above).
          Eg. 0: {1}, 1: {2}, 2: {3} implies 0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}.
     */
    struct AllocatorDependencies {
        using AllocatorID = ttsl::StrongType<uint32_t, struct AllocatorIDTag>;
        using AdjacencyList = ttsl::SmallVector<ttsl::SmallVector<AllocatorID>>;

        // Position of each state in the adjacency list corresponds to the allocator ID of that state
        // Dependencies per state are sorted in order of allocator IDs
        AdjacencyList dependencies{{}};  // Default: single allocator (0) with no dependencies

        AllocatorDependencies();
        explicit AllocatorDependencies(
            const std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>>& dependencies_map);

        uint32_t num_allocators() const;
        ttsl::SmallVector<AllocatorID> allocator_ids() const;

        bool operator==(const AllocatorDependencies& other) const noexcept;
        bool operator!=(const AllocatorDependencies& other) const noexcept { return !(*this == other); }
    };

    BankManager(
        const BufferType& buffer_type,
        const std::vector<int64_t>& bank_descriptors,
        DeviceAddr size_bytes,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false,
        const AllocatorDependencies& dependencies = AllocatorDependencies());
    BankManager(
        const BufferType& buffer_type,
        const std::unordered_map<uint32_t, int64_t>& bank_id_to_descriptor,
        DeviceAddr size_bytes,
        DeviceAddr interleaved_address_limit,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false,
        const AllocatorDependencies& dependencies = AllocatorDependencies());
    BankManager& operator=(BankManager&& that) noexcept;
    uint32_t num_banks() const;

    DeviceAddr bank_size() const;

    int64_t bank_offset(uint32_t bank_id) const;

    DeviceAddr allocate_buffer(
        DeviceAddr size,
        DeviceAddr page_size,
        bool bottom_up,
        const CoreRangeSet& compute_grid,
        std::optional<uint32_t> num_shards,
        AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0});

    void deallocate_buffer(
        DeviceAddr address, AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0});
    void deallocate_all();

    void clear();

    std::optional<DeviceAddr> lowest_occupied_address(
        uint32_t bank_id,
        AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0}) const;

    Statistics get_statistics(
        AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0}) const;

    void dump_blocks(
        std::ofstream& out,
        AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0}) const;

    MemoryBlockTable get_memory_block_table(
        AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0}) const;

    void shrink_size(
        DeviceAddr shrink_size,
        bool bottom_up = true,
        AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0});
    void reset_size(AllocatorDependencies::AllocatorID allocator_id = AllocatorDependencies::AllocatorID{0});

private:
    /*********************************
     * Allocator-independent members *
     *********************************/
    // Type of buffers allocated in the banks (same across allocators)
    BufferType buffer_type_{0};
    // This is to store offsets for any banks that share a core or node (dram in wh/storage core), so we can view all
    // banks using only bank_id. Set to 0 for cores/nodes with only 1 bank.
    std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset_;

    DeviceAddr interleaved_address_limit_{};
    uint32_t alignment_bytes_{};

    /*******************************
     * Allocator-dependent members *
     *******************************/
    // Dependencies between allocators (also encodes number of allocators)
    AllocatorDependencies allocator_dependencies_{};

    // Track allocations per allocator
    ttsl::SmallVector<std::unordered_set<DeviceAddr>> allocated_buffers_{};
    ttsl::SmallVector<std::unique_ptr<allocator::Algorithm>> allocators_{};

    // Per-allocator cache of: merged allocated ranges of all other dependent allocators
    ttsl::SmallVector<std::optional<std::vector<std::pair<DeviceAddr, DeviceAddr>>>> allocated_ranges_cache_{};

    /*********************************
     * Allocator-independent methods *
     *********************************/
    void validate_bank_id(uint32_t bank_id) const;
    void init_allocators(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset);

    /*******************************
     * Allocator-dependent methods *
     *******************************/
    // Returns allocator for the given allocator ID; returns nullptr if allocator ID is invalid
    allocator::Algorithm* get_allocator_from_id(AllocatorDependencies::AllocatorID allocator_id);
    const allocator::Algorithm* get_allocator_from_id(AllocatorDependencies::AllocatorID allocator_id) const;

    // Invalidate caches stored on allocators that depend on the given allocator
    void invalidate_allocated_ranges_cache_for_dependent_allocators(AllocatorDependencies::AllocatorID allocator_id);

    // Compute and cache the merged allocated ranges of all dependent allocators for the given allocator
    const std::vector<std::pair<DeviceAddr, DeviceAddr>>& compute_merged_allocated_ranges(
        AllocatorDependencies::AllocatorID allocator_id);

    // Compute available address ranges for the given allocator and request, after subtracting merged neighbor
    // allocations
    std::vector<std::pair<DeviceAddr, DeviceAddr>> compute_available_addresses(
        AllocatorDependencies::AllocatorID allocator_id, DeviceAddr size_per_bank, DeviceAddr address_limit);
};

}  // namespace tt_metal

}  // namespace tt
