#pragma once

#include <string>

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

    FreeList(uint32_t max_size_bytes, uint32_t min_allocation_size, uint32_t alignment, SearchPolicy search_policy);

    ~FreeList();

    void init();

    std::vector<std::pair<uint32_t, uint32_t>> available_addresses(uint32_t size_bytes) const;

    std::optional<uint32_t> allocate(uint32_t size_bytes, bool bottom_up=true);

    std::optional<uint32_t> allocate_at_address(uint32_t start_address, uint32_t size_bytes, bool bottom_up=true);

    void deallocate(uint32_t address);

    void clear();

   private:
    struct Block {
        uint32_t address;
        uint32_t size;
        bool grows_up = true;    // if true range of block is [address, address + size) otherwise range is [address, address - size)
        Block *prev_block = nullptr;
        Block *next_block = nullptr;
        Block *prev_free = nullptr;
        Block *next_free = nullptr;
    };

    void dump_block(const Block *block, const std::string &preamble) const;

    void dump_blocks() const;

    bool is_allocated(const Block *block) const;

    bool allocated_neighbour_grows_in_opposite_direction(const Block *allocation_candidate, bool bottom_up) const;

    Block *search_best(uint32_t size_bytes, bool bottom_up, bool offset_added=false);

    Block *search_first(uint32_t size_bytes, bool bottom_up, bool offset_added=false);

    Block *search(uint32_t size_bytes, bool bottom_up);

    void allocate_entire_free_block(Block *free_block_to_allocate, bool grows_up);

    void update_left_aligned_allocated_block_connections(Block *free_block, Block *allocated_block, bool bottom_up);

    void update_right_aligned_allocated_block_connections(Block *free_block, Block *allocated_block);

    Block *allocate_slice_of_free_block(Block *free_block, uint32_t offset, uint32_t size_bytes, bool allocated_block_grows_up);

    Block *find_block(uint32_t address);

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
