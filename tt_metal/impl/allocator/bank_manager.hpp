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
#include <map>
#include <tt_stl/small_vector.hpp>

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

    struct StateDependencies {
        struct StateId {
            uint32_t value{0};
            bool operator==(const StateId& other) const noexcept { return value == other.value; }
            bool operator!=(const StateId& other) const noexcept { return value != other.value; }
        };

        struct Hasher {
            size_t operator()(const StateId& s) const noexcept { return std::hash<uint32_t>{}(s.value); }
        };

        std::unordered_map<StateId, tt::stl::SmallVector<StateId>, Hasher> adjacency{};

        StateDependencies();
        explicit StateDependencies(
            const std::unordered_map<StateId, tt::stl::SmallVector<StateId>, Hasher>& dependencies_map);

        uint32_t num_states() const;
    };

    BankManager(
        const BufferType& buffer_type,
        const std::vector<int64_t>& bank_descriptors,
        DeviceAddr size_bytes,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false,
        const StateDependencies& dependencies = StateDependencies());
    BankManager(
        const BufferType& buffer_type,
        const std::unordered_map<uint32_t, int64_t>& bank_id_to_descriptor,
        DeviceAddr size_bytes,
        DeviceAddr interleaved_address_limit,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false,
        const StateDependencies& dependencies = StateDependencies());
    BankManager&& operator=(BankManager&& that) noexcept;
    ~BankManager();
    uint32_t num_banks() const;

    DeviceAddr bank_size() const;

    int64_t bank_offset(uint32_t bank_id) const;

    DeviceAddr allocate_buffer(
        DeviceAddr size,
        DeviceAddr page_size,
        bool bottom_up,
        const CoreRangeSet& compute_grid,
        std::optional<uint32_t> num_shards,
        StateDependencies::StateId state = StateDependencies::StateId{0});

    void deallocate_buffer(DeviceAddr address, StateDependencies::StateId state = StateDependencies::StateId{0});
    void deallocate_all(StateDependencies::StateId state = StateDependencies::StateId{0});

    void clear(StateDependencies::StateId state = StateDependencies::StateId{0});

    std::optional<DeviceAddr> lowest_occupied_address(
        uint32_t bank_id, StateDependencies::StateId state = StateDependencies::StateId{0}) const;

    Statistics get_statistics(StateDependencies::StateId state = StateDependencies::StateId{0}) const;

    void dump_blocks(std::ofstream& out, StateDependencies::StateId state = StateDependencies::StateId{0}) const;

    MemoryBlockTable get_memory_block_table(StateDependencies::StateId state = StateDependencies::StateId{0}) const;

    void shrink_size(
        DeviceAddr shrink_size,
        bool bottom_up = true,
        StateDependencies::StateId state = StateDependencies::StateId{0});
    void reset_size(StateDependencies::StateId state = StateDependencies::StateId{0});

private:
    // State-independent members
    // Type of buffers allocated in the banks (same across states)
    BufferType buffer_type_{0};
    // This is to store offsets for any banks that share a core or node (dram in wh/storage core), so we can view all
    // banks using only bank_id. Set to 0 for cores/nodes with only 1 bank.
    std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset_;

    DeviceAddr interleaved_address_limit_{};
    uint32_t alignment_bytes_{};

    // State-dependent members
    // Dependencies between states (also encodes number of states)
    StateDependencies dependencies_{};
    // Reverse edges: for each state, which states depend on it
    std::unordered_map<uint32_t, tt::stl::SmallVector<uint32_t>> dependents_{};

    // Track allocations per state: base address -> size_per_bank
    std::vector<std::unordered_map<DeviceAddr, DeviceAddr>> allocated_buffers_{};
    std::vector<std::unique_ptr<allocator::Algorithm>> allocators_{};
    // Remember allocator offsets per state
    std::vector<DeviceAddr> allocator_offsets_{};
    // Foreign occupancy overlay per state, represented as difference map
    struct Overlay {
        std::map<DeviceAddr, int64_t> delta;
        void add(DeviceAddr start, DeviceAddr end);
        void remove(DeviceAddr start, DeviceAddr end);
        std::vector<std::pair<DeviceAddr, DeviceAddr>> occupied() const;
        std::vector<std::pair<DeviceAddr, DeviceAddr>> subtract(
            const std::vector<std::pair<DeviceAddr, DeviceAddr>>& free_ranges) const;
    };
    std::vector<Overlay> overlays_{};

    // State-independent methods
    void validate_bank_id(uint32_t bank_id) const;
    void init_allocators_across_states(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset);

    // State-dependent methods
    void deallocate_buffer_(DeviceAddr address, StateDependencies::StateId state);
    void assert_valid_state(StateDependencies::StateId state) const;

    static DeviceAddr align_up(DeviceAddr addr, DeviceAddr alignment) {
        if (alignment == 0) {
            return addr;
        }
        DeviceAddr factor = (addr + alignment - 1) / alignment;
        return factor * alignment;
    }
};

}  // namespace tt_metal

}  // namespace tt
