// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt-metalium/allocator_types.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt {

namespace tt_metal {

class BankManager;
class Buffer;
// Fwd declares
enum class BufferType;

// THREAD SAFETY: Allocator is thread safe.
class Allocator {
public:
    Allocator(const AllocatorConfig& alloc_config);

    ~Allocator();

    DeviceAddr allocate_buffer(Buffer* buffer);

    void deallocate_buffer(Buffer* buffer);
    void deallocate_buffers();

    std::unordered_set<Buffer*> get_allocated_buffers() const;
    size_t get_num_allocated_buffers() const;

    uint32_t get_num_banks(const BufferType& buffer_type) const;
    DeviceAddr get_bank_size(const BufferType& buffer_type) const;

    uint32_t get_dram_channel_from_bank_id(uint32_t bank_id) const;
    CoreCoord get_logical_core_from_bank_id(uint32_t bank_id) const;

    int32_t get_bank_offset(BufferType buffer_type, uint32_t bank_id) const;

    const std::vector<uint32_t>& get_bank_ids_from_dram_channel(uint32_t dram_channel) const;
    const std::vector<uint32_t>& get_bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord& logical_core) const;

    DeviceAddr get_base_allocator_addr(const HalMemType& mem_type) const;

    const AllocatorConfig& get_config() const;
    // Alignment can be pulled out of the AllocatorConfig but this getter is a helper
    // so client code does not need to condition based on BufferType
    uint32_t get_alignment(BufferType buffer_type) const;

    Statistics get_statistics(const BufferType& buffer_type) const;
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

    void validate_bank_assignments() const;

private:
    void verify_safe_allocation() const;

    mutable std::mutex mutex_;

    // Set to true if allocating a buffer is unsafe. This happens when a live trace on device can corrupt
    // memory allocated by the user (memory used by trace is not tracked in the allocator once the trace is captured).
    bool allocations_unsafe_ = false;
    std::unique_ptr<BankManager> dram_manager_;
    std::unique_ptr<BankManager> l1_manager_;
    std::unique_ptr<BankManager> l1_small_manager_;
    std::unique_ptr<BankManager> trace_buffer_manager_;

    std::unordered_map<uint32_t, uint32_t> bank_id_to_dram_channel_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> dram_channel_to_bank_ids_;
    std::unordered_map<uint32_t, CoreCoord> bank_id_to_logical_core_;
    std::unordered_map<BufferType, std::unordered_map<CoreCoord, std::vector<uint32_t>>> logical_core_to_bank_ids_;
    std::unordered_set<Buffer*> allocated_buffers_;

    const AllocatorConfig config_;
};

}  // namespace tt_metal

}  // namespace tt
