// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>

#include "impl/allocator/allocator_types.hpp"
#include "impl/allocator/bank_manager.hpp"

namespace tt::tt_metal {

// THREAD SAFETY: Allocator is thread safe.
class AllocatorImpl {
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
    explicit AllocatorImpl(const AllocatorConfig& alloc_config);

    ~AllocatorImpl();

    DeviceAddr allocate_buffer(Buffer* buffer);

    void deallocate_buffer(Buffer* buffer);
    void deallocate_buffers();

    std::unordered_set<Buffer*> get_allocated_buffers() const;
    size_t get_num_allocated_buffers() const;

    std::uint32_t get_num_banks(const BufferType& buffer_type) const;
    DeviceAddr get_bank_size(const BufferType& buffer_type) const;

    std::uint32_t get_dram_channel_from_bank_id(std::uint32_t bank_id) const;
    CoreCoord get_logical_core_from_bank_id(std::uint32_t bank_id) const;

    int32_t get_bank_offset(BufferType buffer_type, std::uint32_t bank_id) const;

    const std::vector<std::uint32_t>& get_bank_ids_from_dram_channel(std::uint32_t dram_channel) const;
    const std::vector<std::uint32_t>& get_bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord& logical_core) const;

    DeviceAddr get_base_allocator_addr(const HalMemType& mem_type) const;

    const AllocatorConfig& get_config() const;
    // Alignment can be pulled out of the AllocatorConfig but this getter is a helper
    // so client code does not need to condition based on BufferType
    std::uint32_t get_alignment(BufferType buffer_type) const;

    // This a proxy of get_config().worker_l1_size,
    // this helper function is made for reports.cpp in TTNN and act as a transient member function
    // before we figure out a good memory profile accessor.
    size_t get_worker_l1_size() const;

    Statistics get_statistics(const BufferType& buffer_type) const;
    MemoryBlockTable get_memory_block_table(const BufferType& buffer_type) const;
    void dump_memory_blocks(const BufferType& buffer_type, std::ostream& out) const;

    std::optional<DeviceAddr> get_lowest_occupied_l1_address(std::uint32_t bank_id) const;

    void shrink_allocator_size(const BufferType& buffer_type, DeviceAddr shrink_size, bool bottom_up = true);
    void reset_allocator_size(const BufferType& buffer_type);

    void mark_allocations_unsafe();
    void mark_allocations_safe();

    // what does clear even mean on an allocator???
    void clear();

    // AllocatorState Methods
    // Extracts the current state of the allocator.
    AllocatorState extract_state() const;

    // Overrides the current state with the given state, deallocating all of existing buffers.
    void override_state(const AllocatorState& state);

    // We likely won't need to perform heap allocation just to expose the user side of Allocator,
    // this is to ease transition so we keep the pointer-to-allocator semantics.
    const std::unique_ptr<Allocator>& view() const;

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

    std::unordered_map<std::uint32_t, std::uint32_t> bank_id_to_dram_channel_;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> dram_channel_to_bank_ids_;
    std::unordered_map<std::uint32_t, CoreCoord> bank_id_to_logical_core_;
    std::unordered_map<BufferType, std::unordered_map<CoreCoord, std::vector<std::uint32_t>>> logical_core_to_bank_ids_;
    std::unordered_set<Buffer*> allocated_buffers_;

    // config_ is stored in a unique_ptr because AllocatorConfig is currently an incomplete type in API directory.
    //
    // TODO(river): revert this to inplace storage if we can shove Allocator into impl.
    std::unique_ptr<AllocatorConfig> config_;

    // External view of the allocator, this shouldn't need to be a unique_ptr, but currently kept as so to preserve API
    // stability
    // TODO(river): Revisit during API refactor.
    std::unique_ptr<Allocator> view_;
};

}  // namespace tt::tt_metal
