// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <unordered_set>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

struct Statistics {
    size_t total_allocatable_size_bytes = 0;
    size_t total_allocated_bytes = 0;
    size_t total_free_bytes = 0;
    size_t largest_free_block_bytes = 0;
    // addresses (relative to bank) that can hold the largest_free_block_bytes
    std::vector<uint32_t> largest_free_block_addrs;
};

class Buffer;
enum class BufferType : std::uint8_t;
class AllocatorState;

// This is the internal representation for Allocator, not exposed for general usage.
class AllocatorImpl;

// Looking at the member functions, there is a way better name we can give to this struct.
// Currently kept as "Allocator" to avoid breaking anyone.
class Allocator {
public:
    // AllocatorImpl is internal to Runtime, this effectively means there's no way
    // to construct Allocator publicly.
    explicit Allocator(AllocatorImpl* _impl);

    void deallocate_buffers();
    std::unordered_set<Buffer*> get_allocated_buffers() const;
    uint32_t get_num_banks(const BufferType& buffer_type) const;
    DeviceAddr get_bank_size(const BufferType& buffer_type) const;
    CoreCoord get_logical_core_from_bank_id(uint32_t bank_id) const;
    int32_t get_bank_offset(BufferType buffer_type, uint32_t bank_id) const;
    const std::vector<uint32_t>& get_bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord& logical_core) const;
    DeviceAddr get_base_allocator_addr(const HalMemType& mem_type) const;
    uint32_t get_alignment(BufferType buffer_type) const;
    // This a proxy of get_config().worker_l1_size,
    // this helper function is made for reports.cpp in TTNN and act as a transient member function.
    size_t get_worker_l1_size() const;
    Statistics get_statistics(const BufferType& buffer_type) const;
    // AllocatorState Methods
    // Extracts the current state of the allocator.
    AllocatorState extract_state() const;
    // Overrides the current state with the given state, deallocating all of existing buffers.
    void override_state(const AllocatorState& state);

private:
    AllocatorImpl* impl;
};

namespace detail {

// This is only used by the move operation in ttnn and is not intended for public use
// (it's in the detail namespace)
DeviceAddr calculate_bank_size_spread(
    DeviceAddr size_bytes, DeviceAddr page_size_bytes, uint32_t num_banks, uint32_t alignment_bytes);

}  // namespace detail

}  // namespace tt::tt_metal
