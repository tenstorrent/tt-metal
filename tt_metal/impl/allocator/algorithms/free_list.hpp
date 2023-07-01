#pragma once

#include <string>

#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/allocator/algorithms/allocator_algorithm.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

class FreeList : public Algorithm {
   public:
    enum class SearchPolicy {
        BEST = 0,
        FIRST = 1
    };

    FreeList(u64 max_size_bytes, u64 offset_bytes, u64 min_allocation_size, u64 alignment, SearchPolicy search_policy);

    ~FreeList();

    void init();

    std::vector<std::pair<u64, u64>> available_addresses(u64 size_bytes) const;

    std::optional<u64> allocate(u64 size_bytes, bool bottom_up=true);

    std::optional<u64> allocate_at_address(u64 absolute_start_address, u64 size_bytes);

    void deallocate(u64 absolute_address);

    void clear();

    Statistics get_statistics() const;

    void dump_blocks(std::ofstream &out) const;

   private:
    struct Block {
        u64 address;
        u64 size;
        Block *prev_block = nullptr;
        Block *next_block = nullptr;
        Block *prev_free = nullptr;
        Block *next_free = nullptr;
    };

    void dump_block(const Block *block, std::ofstream &out) const;

    bool is_allocated(const Block *block) const;

    Block *search_best(u64 size_bytes, bool bottom_up);

    Block *search_first(u64 size_bytes, bool bottom_up);

    Block *search(u64 size_bytes, bool bottom_up);

    void allocate_entire_free_block(Block *free_block_to_allocate);

    void update_left_aligned_allocated_block_connections(Block *free_block, Block *allocated_block);

    void update_right_aligned_allocated_block_connections(Block *free_block, Block *allocated_block);

    Block *allocate_slice_of_free_block(Block *free_block, u64 offset, u64 size_bytes);

    Block *find_block(u64 address);

    void reset();

    void update_lowest_occupied_address();

    void update_lowest_occupied_address(u64 address);

    SearchPolicy search_policy_;
    Block *block_head_;
    Block *block_tail_;
    Block *free_block_head_;
    Block *free_block_tail_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
