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

    BankManager(
        const BufferType& buffer_type,
        const std::vector<int64_t>& bank_descriptors,
        DeviceAddr size_bytes,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false,
        uint32_t num_states = 1);
    BankManager(
        const BufferType& buffer_type,
        const std::unordered_map<uint32_t, int64_t>& bank_id_to_descriptor,
        DeviceAddr size_bytes,
        DeviceAddr interleaved_address_limit,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false,
        uint32_t num_states = 1);
    BankManager&& operator=(BankManager&& that) noexcept;
    ~BankManager();
    uint32_t num_banks(uint32_t state = 0) const;

    DeviceAddr bank_size(uint32_t state = 0) const;

    int64_t bank_offset(uint32_t bank_id, uint32_t state = 0) const;

    DeviceAddr allocate_buffer(
        DeviceAddr size,
        DeviceAddr page_size,
        bool bottom_up,
        const CoreRangeSet& compute_grid,
        std::optional<uint32_t> num_shards,
        uint32_t state = 0);

    void deallocate_buffer(DeviceAddr address, uint32_t state = 0);
    void deallocate_all(uint32_t state = 0);

    void clear(uint32_t state = 0);

    std::optional<DeviceAddr> lowest_occupied_address(uint32_t bank_id, uint32_t state = 0) const;

    Statistics get_statistics(uint32_t state = 0) const;

    void dump_blocks(std::ofstream& out, uint32_t state = 0) const;

    MemoryBlockTable get_memory_block_table(uint32_t state = 0) const;

    void shrink_size(DeviceAddr shrink_size, bool bottom_up = true, uint32_t state = 0);
    void reset_size(uint32_t state = 0);

private:
    void deallocate_buffer_(DeviceAddr address);

    // Number of independent states tracked by this manager
    uint32_t num_states_{};

    // Type of buffers allocated in the banks (same across states)
    BufferType buffer_type_{};
    std::unordered_set<DeviceAddr> allocated_buffers_{};
    // This is to store offsets for any banks that share a core or node (dram in wh/storage core), so we can view all
    // banks using only bank_id. Set to 0 for cores/nodes with only 1 bank.
    std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset_{};
    std::vector<std::unique_ptr<allocator::Algorithm>> allocators_{};
    DeviceAddr interleaved_address_limit_{};
    uint32_t alignment_bytes_{};

    void validate_bank_id(uint32_t bank_id, uint32_t state) const;
    void assert_single_state(uint32_t state) const;

    void init_allocator(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset, uint32_t state);
};

}  // namespace tt_metal

}  // namespace tt
