// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>

#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"
#include <boost/smart_ptr/local_shared_ptr.hpp>

namespace tt {
namespace tt_metal {
namespace allocator {
class FreeList : public Algorithm {
public:
    enum class SearchPolicy { BEST = 0, FIRST = 1 };

    FreeList(
        DeviceAddr max_size_bytes,
        DeviceAddr offset_bytes,
        DeviceAddr min_allocation_size,
        DeviceAddr alignment,
        SearchPolicy search_policy);
    ~FreeList();
    void init();

    std::vector<std::pair<DeviceAddr, DeviceAddr>> available_addresses(DeviceAddr size_bytes) const;

    std::optional<DeviceAddr> allocate(DeviceAddr size_bytes, bool bottom_up = true, DeviceAddr address_limit = 0);

    std::optional<DeviceAddr> allocate_at_address(DeviceAddr absolute_start_address, DeviceAddr size_bytes);

    void deallocate(DeviceAddr absolute_address);

    void clear();

    Statistics get_statistics() const;

    void dump_blocks(std::ostream& out) const;

    std::vector<std::unordered_map<std::string, std::string>> get_block_table() const;

    void shrink_size(DeviceAddr shrink_size, bool bottom_up = true);

    void reset_size();

private:
    struct Block {
        Block(DeviceAddr address, DeviceAddr size) : address(address), size(size) {}
        Block(
            DeviceAddr address,
            DeviceAddr size,
            boost::local_shared_ptr<Block> prev_block,
            boost::local_shared_ptr<Block> next_block,
            boost::local_shared_ptr<Block> prev_free,
            boost::local_shared_ptr<Block> next_free) :
            address(address),
            size(size),
            prev_block(prev_block),
            next_block(next_block),
            prev_free(prev_free),
            next_free(next_free) {}
        DeviceAddr address;
        DeviceAddr size;
        boost::local_shared_ptr<Block> prev_block = nullptr;
        boost::local_shared_ptr<Block> next_block = nullptr;
        boost::local_shared_ptr<Block> prev_free = nullptr;
        boost::local_shared_ptr<Block> next_free = nullptr;
    };

    void dump_block(const boost::local_shared_ptr<Block>& block, std::ostream& out) const;

    bool is_allocated(const boost::local_shared_ptr<Block>& block) const;

    boost::local_shared_ptr<Block> search_best(DeviceAddr size_bytes, bool bottom_up);

    boost::local_shared_ptr<Block> search_first(DeviceAddr size_bytes, bool bottom_up);

    boost::local_shared_ptr<Block> search(DeviceAddr size_bytes, bool bottom_up);

    void allocate_entire_free_block(const boost::local_shared_ptr<Block>& free_block_to_allocate);

    void update_left_aligned_allocated_block_connections(
        const boost::local_shared_ptr<Block>& free_block, const boost::local_shared_ptr<Block>& allocated_block);

    void update_right_aligned_allocated_block_connections(
        const boost::local_shared_ptr<Block>& free_block, const boost::local_shared_ptr<Block>& allocated_block);

    boost::local_shared_ptr<Block> allocate_slice_of_free_block(
        boost::local_shared_ptr<Block> free_block, DeviceAddr offset, DeviceAddr size_bytes);

    boost::local_shared_ptr<Block> find_block(DeviceAddr address);

    void update_lowest_occupied_address();

    void update_lowest_occupied_address(DeviceAddr address);

    SearchPolicy search_policy_;
    boost::local_shared_ptr<Block> block_head_;
    boost::local_shared_ptr<Block> block_tail_;
    boost::local_shared_ptr<Block> free_block_head_;
    boost::local_shared_ptr<Block> free_block_tail_;
};

}  // namespace allocator
}  // namespace tt_metal
}  // namespace tt
