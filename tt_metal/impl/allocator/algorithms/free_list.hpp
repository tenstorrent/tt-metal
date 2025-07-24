// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <boost/smart_ptr/local_shared_ptr.hpp>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "allocator_algorithm.hpp"
#include "allocator_types.hpp"
#include "hal_types.hpp"
#include "hostdevcommon/common_values.hpp"

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
    ~FreeList() override;
    void init() override;

    std::vector<std::pair<DeviceAddr, DeviceAddr>> available_addresses(DeviceAddr size_bytes) const override;

    std::optional<DeviceAddr> allocate(
        DeviceAddr size_bytes, bool bottom_up = true, DeviceAddr address_limit = 0) override;

    std::optional<DeviceAddr> allocate_at_address(DeviceAddr absolute_start_address, DeviceAddr size_bytes) override;

    void deallocate(DeviceAddr absolute_address) override;

    void clear() override;

    Statistics get_statistics() const override;

    void dump_blocks(std::ostream& out) const override;

    MemoryBlockTable get_memory_block_table() const override;

    void shrink_size(DeviceAddr shrink_size, bool bottom_up = true) override;

    void reset_size() override;

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
