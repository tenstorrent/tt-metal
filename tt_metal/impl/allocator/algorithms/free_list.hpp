/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/allocator/allocator_types.hpp"
#include "tt_metal/common/concurrency_interface.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

using block_offset_ptr_t =  boost::interprocess::offset_ptr<concurrent::block_t>;
class FreeList {
   public:
    enum class SearchPolicy {
        BEST = 0,
        FIRST = 1
    };

    FreeList(std::string name, uint64_t max_size_bytes, uint64_t offset_bytes, uint64_t min_allocation_size, uint64_t alignment, SearchPolicy search_policy);

    uint64_t max_size_bytes() const;

    std::optional<uint64_t> lowest_occupied_address() const;

    // bottom_up=true indicates that allocation grows from address 0
    std::optional<uint64_t> allocate(uint64_t size_bytes, bool bottom_up=true);

    std::optional<uint64_t> allocate_at_address(uint64_t absolute_start_address, uint64_t size_bytes);

    void deallocate(uint64_t absolute_address);

    void clear();

    Statistics get_statistics() const;

    void dump_blocks(std::ofstream &out) const;

   private:

    void dump_block(const block_offset_ptr_t block, std::ofstream &out) const;

    bool is_allocated(block_offset_ptr_t block) const;

    block_offset_ptr_t search_best(uint64_t size_bytes, bool bottom_up);

    block_offset_ptr_t search_first(uint64_t size_bytes, bool bottom_up);

    block_offset_ptr_t search(uint64_t size_bytes, bool bottom_up);

    void allocate_entire_free_block(block_offset_ptr_t free_block_to_allocate);

    void update_left_aligned_allocated_block_connections(block_offset_ptr_t free_block, block_offset_ptr_t allocated_block);

    void update_right_aligned_allocated_block_connections(block_offset_ptr_t free_block, block_offset_ptr_t allocated_block);

    block_offset_ptr_t allocate_slice_of_free_block(block_offset_ptr_t free_block, uint64_t offset, uint64_t size_bytes);

    block_offset_ptr_t find_block(uint64_t address);

    void reset();

    void update_lowest_occupied_address();

    void update_lowest_occupied_address(uint64_t address);

    void debug_dump_blocks() const;

    std::string name_;  // corresponds to name of allocated/free block tracker in shared memory
    uint64_t max_size_bytes_;
    uint64_t offset_bytes_;
    uint64_t min_allocation_size_;
    uint64_t alignment_;
    std::optional<uint64_t> lowest_occupied_address_;

    SearchPolicy search_policy_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
