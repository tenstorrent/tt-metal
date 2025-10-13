// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/math.hpp>

namespace tt {

namespace tt_metal {

/*
MemoryBlockTable is a list of memory blocks in the following format:
[{"blockID": "0", "address": "0", "size": "0", "prevID": "0", "nextID": "0", "allocated": true}]
address: bytes
size: bytes
*/
using MemoryBlockTable = std::vector<std::unordered_map<std::string, std::string>>;

struct Statistics {
    size_t total_allocatable_size_bytes = 0;
    size_t total_allocated_bytes = 0;
    size_t total_free_bytes = 0;
    size_t largest_free_block_bytes = 0;
    // addresses (relative to bank) that can hold the largest_free_block_bytes
    std::vector<uint32_t> largest_free_block_addrs;
};

// Fwd declares
class BankManager;
class Buffer;
// These are supplied from impl
enum class BufferType;
struct AllocatorConfig;
class AllocatorState;

// THREAD SAFETY: Allocator is thread safe.
class Allocator {
public:
    // AllocatorConfig is not in the API directory, thus Allocator currently cannot be constructed publicly,
    // this is because we are in the middle of moving Allocator into implementation details.
    // This initiative is established from our analysis that Allocator is only used to query memory profiles
    // (e.g. how much memory is left in L1?)
    // but not for allocator-specific operations (managing allocations).
    //
    // While in the middle of this refactor,
    // runtime (river) splits moving AllocatorConfig out of public API,
    // coming up with a memory profile access interface to replace current Allocator API into two (or more) PRs.
    //
    // See: #29569
    explicit Allocator(const AllocatorConfig& alloc_config);

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

    // This a proxy of get_config().worker_l1_size,
    // this helper function is made for reports.cpp in TTNN and act as a transient member function
    // before we figure out a good memory profile accessor.
    size_t get_worker_l1_size() const;

    Statistics get_statistics(const BufferType& buffer_type) const;
    MemoryBlockTable get_memory_block_table(const BufferType& buffer_type) const;
    void dump_memory_blocks(const BufferType& buffer_type, std::ostream& out) const;

    std::optional<DeviceAddr> get_lowest_occupied_l1_address(uint32_t bank_id) const;

    void shrink_allocator_size(const BufferType& buffer_type, DeviceAddr shrink_size, bool bottom_up = true);
    void reset_allocator_size(const BufferType& buffer_type);

    void mark_allocations_unsafe();
    void mark_allocations_safe();

    void clear();

    // AllocatorState Methods
    AllocatorState extract_state() const;
    void override_state(const AllocatorState& state);

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

    // config_ is stored in a unique_ptr because AllocatorConfig is current an incomplete type in API directory.
    //
    // TODO(river): revert this to inplace storage if we can shove Allocator into impl.
    std::unique_ptr<AllocatorConfig> config_;
};

namespace detail {

// This is only used by the move operation in ttnn and is not intended for public use
// (it's in the detail namespace)
constexpr DeviceAddr calculate_bank_size_spread(
    DeviceAddr size_bytes, DeviceAddr page_size_bytes, uint32_t num_banks, uint32_t alignment_bytes) {
    TT_ASSERT(
        page_size_bytes == 0 ? size_bytes == 0 : size_bytes % page_size_bytes == 0,
        "Page size {} should be divisible by buffer size {}",
        page_size_bytes,
        size_bytes);
    DeviceAddr num_pages = page_size_bytes == 0 ? 0 : size_bytes / page_size_bytes;
    DeviceAddr num_equally_distributed_pages = num_pages == 0 ? 0 : 1 + ((num_pages - 1) / num_banks);
    return num_equally_distributed_pages * round_up(page_size_bytes, static_cast<DeviceAddr>(alignment_bytes));
}

}  // namespace detail

}  // namespace tt_metal

}  // namespace tt
