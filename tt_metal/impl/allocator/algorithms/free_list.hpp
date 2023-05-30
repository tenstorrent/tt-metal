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

    FreeList(u32 max_size_bytes, u32 min_allocation_size, u32 alignment, SearchPolicy search_policy);

    ~FreeList();

    void init();

    std::vector<std::pair<u32, u32>> available_addresses(u32 size_bytes) const;

    std::optional<u32> allocate(u32 size_bytes, bool bottom_up=true);

    std::optional<u32> allocate_at_address(u32 start_address, u32 size_bytes);

    void deallocate(u32 address);

    void clear();

   private:
    struct Block {
        u32 address;
        u32 size;
        Block *prev_block = nullptr;
        Block *next_block = nullptr;
        Block *prev_free = nullptr;
        Block *next_free = nullptr;
    };

    void dump_block(const Block *block, const std::string &preamble) const;

    void dump_blocks() const;

    bool is_allocated(const Block *block) const;

    Block *search_best(u32 size_bytes, bool bottom_up);

    Block *search_first(u32 size_bytes, bool bottom_up);

    Block *search(u32 size_bytes, bool bottom_up);

    void allocate_entire_free_block(Block *free_block_to_allocate);

    void update_left_aligned_allocated_block_connections(Block *free_block, Block *allocated_block);

    void update_right_aligned_allocated_block_connections(Block *free_block, Block *allocated_block);

    Block *allocate_slice_of_free_block(Block *free_block, u32 offset, u32 size_bytes);

    Block *find_block(u32 address);

    void reset();

    SearchPolicy search_policy_;
    Block *block_head_;
    Block *block_tail_;
    Block *free_block_head_;
    Block *free_block_tail_;
};

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
