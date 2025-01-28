// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include <unordered_set>

#include "allocator_types.hpp"
#include "assert.hpp"
#include "core_coord.hpp"
#include "allocator_algorithm.hpp"
#include "hal.hpp"

namespace tt {

namespace tt_metal {

inline namespace v0 {

class Buffer;

}  // namespace v0

// Fwd declares
enum class BufferType;

namespace allocator {

class BankManager {
public:
    BankManager() {}

    BankManager(
        const BufferType& buffer_type,
        const std::vector<int64_t>& bank_descriptors,
        DeviceAddr size_bytes,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false);
    BankManager(
        const BufferType& buffer_type,
        const std::unordered_map<uint32_t, int64_t>& bank_id_to_descriptor,
        DeviceAddr size_bytes,
        DeviceAddr interleaved_address_limit,
        uint32_t alignment_bytes,
        DeviceAddr alloc_offset = 0,
        bool disable_interleaved = false);
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
        std::optional<uint32_t> num_shards);

    void deallocate_buffer(DeviceAddr address);
    void deallocate_all();

    void clear();

    std::optional<DeviceAddr> lowest_occupied_address(uint32_t bank_id) const;

    Statistics get_statistics() const;

    void dump_blocks(std::ofstream& out) const;

    MemoryBlockTable get_memory_block_table() const;

    void shrink_size(DeviceAddr shrink_size, bool bottom_up = true);
    void reset_size();

private:
    void deallocate_buffer_(DeviceAddr address);

    // Types of buffers allocated in the banks
    BufferType buffer_type_;
    std::unordered_set<DeviceAddr> allocated_buffers_;
    // This is to store offsets for any banks that share a core or node (dram in wh/storage core), so we can view all
    // banks using only bank_id Set to 0 for cores/nodes with only 1 bank
    std::unordered_map<uint32_t, int64_t> bank_id_to_bank_offset_;
    std::unique_ptr<Algorithm> allocator_;
    DeviceAddr interleaved_address_limit_;
    uint32_t alignment_bytes_;
    void validate_bank_id(uint32_t bank_id) const;

    void init_allocator(DeviceAddr size_bytes, uint32_t alignment_bytes, DeviceAddr offset);
};

}  // namespace allocator

class Allocator {
public:
    Allocator(const AllocatorConfig& alloc_config);

    ~Allocator();

    DeviceAddr allocate_buffer(Buffer* buffer);

    void deallocate_buffer(Buffer* buffer);
    void deallocate_buffers();

    const std::unordered_set<Buffer*>& get_allocated_buffers() const;

    uint32_t get_num_banks(const BufferType& buffer_type) const;
    DeviceAddr get_bank_size(const BufferType& buffer_type) const;

    uint32_t get_dram_channel_from_bank_id(uint32_t bank_id) const;
    CoreCoord get_logical_core_from_bank_id(uint32_t bank_id) const;

    int32_t get_bank_offset(BufferType buffer_type, uint32_t bank_id) const;

    const std::vector<uint32_t>& get_bank_ids_from_dram_channel(uint32_t dram_channel) const;
    const std::vector<uint32_t>& get_bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord& logical_core) const;

    DeviceAddr get_unreserved_base_address(const HalMemType& mem_type) const;

    const AllocatorConfig& get_config() const;

    allocator::Statistics get_statistics(const BufferType& buffer_type) const;
    MemoryBlockTable get_memory_block_table(const BufferType& buffer_type) const;
    void dump_memory_blocks(const BufferType& buffer_type, std::ofstream& out) const;

    std::optional<DeviceAddr> get_lowest_occupied_l1_address(uint32_t bank_id) const;

    void shrink_allocator_size(const BufferType& buffer_type, DeviceAddr shrink_size, bool bottom_up = true);
    void reset_allocator_size(const BufferType& buffer_type);

    void mark_allocations_unsafe();
    void mark_allocations_safe();

    void clear();

protected:
    // Initializers for mapping banks to DRAM channels / L1 banks
    void init_one_bank_per_channel();
    void init_one_bank_per_l1();
    void init_compute_and_storage_l1_bank_manager();

private:
    void verify_safe_allocation() const;

    // Set to true if allocating a buffer is unsafe. This happens when a live trace on device can corrupt
    // memory allocated by the user (memory used by trace is not tracked in the allocator once the trace is captured).
    bool allocations_unsafe_ = false;
    allocator::BankManager dram_manager_;
    allocator::BankManager l1_manager_;
    allocator::BankManager l1_small_manager_;
    allocator::BankManager trace_buffer_manager_;

    std::unordered_map<uint32_t, uint32_t> bank_id_to_dram_channel_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> dram_channel_to_bank_ids_;
    std::unordered_map<uint32_t, CoreCoord> bank_id_to_logical_core_;
    std::unordered_map<BufferType, std::unordered_map<CoreCoord, std::vector<uint32_t>>> logical_core_to_bank_ids_;
    std::unordered_set<Buffer*> allocated_buffers_;

    AllocatorConfig config_;
};

}  // namespace tt_metal

}  // namespace tt
