// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/assert.hpp"
#include "common/math.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

#include "tt_metal/impl/allocator/algorithms/free_list.hpp"

namespace tt {

namespace tt_metal {

namespace allocator {

FreeList::FreeList(std::string name, uint64_t max_size_bytes, uint64_t offset_bytes, uint64_t min_allocation_size, uint64_t alignment, FreeList::SearchPolicy search_policy)
    : name_(name), max_size_bytes_(max_size_bytes), offset_bytes_(offset_bytes), min_allocation_size_(min_allocation_size), alignment_(alignment), lowest_occupied_address_(std::nullopt), search_policy_(search_policy) {
    if (offset_bytes % this->alignment_ != 0) {
        log_fatal("Error in initializing allocator, offset {} should be {} B aligned", offset_bytes, this->alignment_);
    }
    if (this->min_allocation_size_ % this->alignment_ != 0) {
        log_fatal("Error in initializing allocator, minimum allocation size {} should be {} B aligned", this->min_allocation_size_, this->alignment_);
    }
    concurrent::initialize_allocator(name, max_size_bytes);
}

uint64_t FreeList::max_size_bytes() const { return max_size_bytes_; }

std::optional<uint64_t> FreeList::lowest_occupied_address() const {
    if (not this->lowest_occupied_address_.has_value()) {
        return this->lowest_occupied_address_;
    }
    return this->lowest_occupied_address_.value() + this->offset_bytes_;
}

bool FreeList::is_allocated(block_offset_ptr_t block) const {
    concurrent::allocator_t *memory_manager = concurrent::get_allocator(this->name_.c_str()).first;

    int32_t block_idx = (block->address / this->min_allocation_size_);
    return block->prev_free == 0 and block->next_free == 0 and block != memory_manager->free_block_head and block != memory_manager->free_block_tail;
}

block_offset_ptr_t FreeList::search_best(uint64_t size_bytes, bool bottom_up) {
    concurrent::allocator_t *memory_manager = concurrent::get_allocator(this->name_.c_str()).first;

    block_offset_ptr_t best_block = 0;
    block_offset_ptr_t curr_block = bottom_up ? memory_manager->free_block_head : memory_manager->free_block_tail;
    while (curr_block != 0) {
        if (curr_block->size == size_bytes) {
            best_block = curr_block;
            break;
        } else if (curr_block->size >= size_bytes) {
            if (best_block == 0 or curr_block->size < best_block->size) {
                best_block = curr_block;
            }
        }
        curr_block = bottom_up ? curr_block->next_free : curr_block->prev_free;
    }

    return best_block;
}

block_offset_ptr_t FreeList::search_first(uint64_t size_bytes, bool bottom_up) {
    concurrent::allocator_t *memory_manager = concurrent::get_allocator(this->name_.c_str()).first;

    block_offset_ptr_t first_fit = 0;
    block_offset_ptr_t curr_block = bottom_up ? memory_manager->free_block_head : memory_manager->free_block_tail;
    while (curr_block != 0) {
        if (curr_block->size >= size_bytes) {
            first_fit = curr_block;
            break;
        }
        curr_block = bottom_up ? curr_block->next_free : curr_block->prev_free;
    }

    return first_fit;
}

block_offset_ptr_t FreeList::search(uint64_t size_bytes, bool bottom_up) {
    switch (this->search_policy_) {
        case FreeList::SearchPolicy::BEST:
            return search_best(size_bytes, bottom_up);
        break;
        case FreeList::SearchPolicy::FIRST:
            return search_first(size_bytes, bottom_up);
        break;
        default:
            TT_ASSERT(false && "Unsupported search policy");
    }
    return 0;
}

void FreeList::allocate_entire_free_block(block_offset_ptr_t free_block_to_allocate) {
    concurrent::allocator_t *memory_manager = concurrent::get_allocator(this->name_.c_str()).first;

    TT_ASSERT(not is_allocated(free_block_to_allocate));
    if (free_block_to_allocate->prev_free != 0) {
        free_block_to_allocate->prev_free->next_free = free_block_to_allocate->next_free;
    }
    if (free_block_to_allocate->next_free != 0) {
        free_block_to_allocate->next_free->prev_free = free_block_to_allocate->prev_free;
    }
    if (free_block_to_allocate == memory_manager->free_block_head) {
        if (free_block_to_allocate->next_free == 0) {
            memory_manager->free_block_head = 0;
        } else {
            memory_manager->free_block_head = free_block_to_allocate->next_free;
        }
    }
    if (free_block_to_allocate == memory_manager->free_block_tail) {
        if (free_block_to_allocate->prev_free == 0) {
            memory_manager->free_block_tail = 0;
        } else {
            memory_manager->free_block_tail = free_block_to_allocate->prev_free;
        }
    }
    free_block_to_allocate->prev_free = 0;
    free_block_to_allocate->next_free = 0;
}

// free_block range: [a, b)
// allocated_block range: [a, c), where c < b
void FreeList::update_left_aligned_allocated_block_connections(block_offset_ptr_t free_block, block_offset_ptr_t allocated_block) {
    concurrent::allocator_t *memory_manager = concurrent::get_allocator(this->name_.c_str()).first;

    allocated_block->prev = free_block->prev;
    allocated_block->next = free_block;
    if (free_block->prev != 0) {
        free_block->prev->next = allocated_block;
    }
    if (free_block == memory_manager->block_head) {
        memory_manager->block_head = allocated_block;
    }
    // next_free and prev_free connections of free_block are still valid
    free_block->prev = allocated_block;
    free_block->address = allocated_block->address + allocated_block->size;
    free_block->size -= allocated_block->size;
}

// free_block range: [a, b)
// allocated_block range: [c, b), where c > a
void FreeList::update_right_aligned_allocated_block_connections(block_offset_ptr_t free_block, block_offset_ptr_t allocated_block) {
    allocated_block->prev = free_block;
    allocated_block->next = free_block->next;
    if (free_block->next != 0) {
        free_block->next->prev = allocated_block;
    }
    // next_free and prev_free connections of free_block are still valid
    free_block->next = allocated_block;
    free_block->size -= allocated_block->size;
}

// Offset marks the start of the allocated block
block_offset_ptr_t FreeList::allocate_slice_of_free_block(block_offset_ptr_t free_block, uint64_t offset, uint64_t size_bytes) {
    concurrent::allocator_t *memory_manager = concurrent::get_allocator(this->name_.c_str()).first;

    TT_ASSERT(free_block->address + offset + size_bytes <= free_block->address + free_block->size);

    // Allocated slice spans the entire space of free_block
    if (offset == 0 and size_bytes == free_block->size) {
        this->allocate_entire_free_block(free_block);
        return free_block;
    }

    block_offset_ptr_t allocated_block = static_cast<concurrent::block_t *>(concurrent::get_shared_mem_segment().allocate(sizeof(concurrent::block_t)));
    allocated_block->address = free_block->address + offset;
    allocated_block->size = size_bytes;
    allocated_block->prev = 0; allocated_block->next = 0; allocated_block->prev_free = 0; allocated_block->next_free = 0;

    // Allocated slice takes up a portion of free_block, three cases to consider:
    // 1. allocated_block is left aligned with free_block with free space remaining on the right
    // 2. allocated_block is right aligned with free_block with free space remaining on the left
    // 3. allocated_block is in the middle of free_block with free space on left and right sides
    bool case_one = offset == 0 and size_bytes < free_block->size;
    bool case_two = offset > 0 and ((free_block->address + offset + size_bytes) == (free_block->address + free_block->size));
    bool case_three = offset > 0 and ((free_block->address + offset + size_bytes) < (free_block->address + free_block->size));
    TT_ASSERT((int)(case_one + case_two + case_three) == 1);

    if (case_one) {
        this->update_left_aligned_allocated_block_connections(free_block, allocated_block);
    } else if (case_two) {
        this->update_right_aligned_allocated_block_connections(free_block, allocated_block);
    } else {
        TT_ASSERT(case_three);
        // Original: | .................... free_block ....................|
        // Result:   | free_block_mod | allocated_block | next_free_block  |
        uint64_t next_free_block_addr = free_block->address + offset + size_bytes;
        uint64_t next_free_block_size = (free_block->address + free_block->size) - next_free_block_addr;

        block_offset_ptr_t next_free_block = static_cast<concurrent::block_t *>(concurrent::get_shared_mem_segment().allocate(sizeof(concurrent::block_t)));
        next_free_block->address = next_free_block_addr;
        next_free_block->size = next_free_block_size;
        next_free_block->prev = allocated_block;
        next_free_block->next = free_block->next;
        next_free_block->prev_free = free_block;
        next_free_block->next_free = free_block->next_free;

        if (free_block->next != 0) {
            free_block->next->prev = next_free_block;
        }
        if (free_block->next_free != 0) {
            free_block->next_free->prev_free = next_free_block;
        }
        if (memory_manager->free_block_tail == free_block) {
            memory_manager->free_block_tail = next_free_block;
        }
        free_block->next_free = next_free_block;
        free_block->next = allocated_block;

        allocated_block->prev = free_block;
        allocated_block->next = next_free_block;

        free_block->size -= (allocated_block->size + next_free_block->size);
    }

    return allocated_block;
}

void FreeList::update_lowest_occupied_address(uint64_t address) {
    if (not this->lowest_occupied_address_.has_value()) {
        this->lowest_occupied_address_ = address;
    } else {
        this->lowest_occupied_address_ = std::min(this->lowest_occupied_address_.value(), address);
    }
}

std::optional<uint64_t> FreeList::allocate(uint64_t size_bytes, bool bottom_up) {
    concurrent::allocator_and_lock_pair_t memory_manager_and_lock = concurrent::get_allocator(this->name_.c_str());
    concurrent::allocator_t *memory_manager = memory_manager_and_lock.first;
    boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(memory_manager_and_lock.second);

    uint64_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : size_bytes;
    alloc_size = tt::round_up(alloc_size, this->alignment_);
    block_offset_ptr_t free_block = search(alloc_size, bottom_up);

    if (free_block == 0) {
        return std::nullopt;
    }

    // offset denotes where allocation starts relative to free_block start
    uint64_t offset = bottom_up ? 0 : (((free_block->address + free_block->size) - alloc_size) - free_block->address);
    block_offset_ptr_t allocated_block = allocate_slice_of_free_block(free_block, offset, alloc_size);

    this->update_lowest_occupied_address(allocated_block->address);

    return allocated_block->address + this->offset_bytes_;
}

std::optional<uint64_t> FreeList::allocate_at_address(uint64_t absolute_start_address, uint64_t size_bytes) {
    TT_ASSERT(absolute_start_address % this->alignment_ == 0, "Requested address " + std::to_string(absolute_start_address) + " should be " + std::to_string(this->alignment_) + "B aligned");

    concurrent::allocator_and_lock_pair_t memory_manager_and_lock = concurrent::get_allocator(this->name_.c_str());
    concurrent::allocator_t *memory_manager = memory_manager_and_lock.first;
    boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(memory_manager_and_lock.second);

    auto start_address = absolute_start_address - this->offset_bytes_;
    block_offset_ptr_t curr_block = memory_manager->free_block_head;
    uint64_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : size_bytes;
    alloc_size = tt::round_up(alloc_size, this->alignment_);
    // Look for a free block of size at least size_bytes that encompasses start_address
    while (curr_block != 0) {
        if (curr_block->size >= alloc_size) {
            if (curr_block->address == start_address) {
                allocate_slice_of_free_block(curr_block, /*offset=*/0, alloc_size);
                break;
            } else if ((start_address > curr_block->address) and ((start_address + alloc_size) <= (curr_block->address + curr_block->size))) {
                uint64_t start_offset = start_address - curr_block->address;
                allocate_slice_of_free_block(curr_block, start_offset, alloc_size);
                break;
            }
        }
        curr_block = curr_block->next_free;
    }

    if (curr_block == 0) {
        return std::nullopt;
    }
    this->update_lowest_occupied_address(start_address);
    return absolute_start_address;
}

block_offset_ptr_t FreeList::find_block(uint64_t address) {
    concurrent::allocator_t *memory_manager = concurrent::get_allocator(this->name_.c_str()).first;

    block_offset_ptr_t block = 0;
    block_offset_ptr_t curr_block = memory_manager->block_head;
    while (curr_block != 0) {
        if (curr_block->address == address) {
            return curr_block;
        }
        curr_block = curr_block->next;
    }
    return block;
}

void FreeList::update_lowest_occupied_address() {
    concurrent::allocator_t *memory_manager = concurrent::get_allocator(this->name_.c_str()).first;

    block_offset_ptr_t block = memory_manager->block_head;
    while (block != 0) {
        if (this->is_allocated(block)) {
            break;
        }
        block = block->next;
    }
    if (block == 0) {
        this->lowest_occupied_address_ = std::nullopt;
    } else {
        this->lowest_occupied_address_ = block->address;
    }
}

void FreeList::deallocate(uint64_t absolute_address) {
    concurrent::allocator_and_lock_pair_t memory_manager_and_lock = concurrent::get_allocator(this->name_.c_str());
    concurrent::allocator_t *memory_manager = memory_manager_and_lock.first;
    boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(memory_manager_and_lock.second);

    uint64_t address = absolute_address - this->offset_bytes_;
    block_offset_ptr_t block_to_free = find_block(address);
    if (block_to_free == 0 or not this->is_allocated(block_to_free)) {
        return;
    }

    block_offset_ptr_t prev_block = block_to_free->prev;
    block_offset_ptr_t next_block = block_to_free->next;

    bool merged_prev = false;
    bool merged_next = false;
    if (prev_block != nullptr and not is_allocated(prev_block)) {
        prev_block->next = block_to_free->next;
        if (block_to_free->next != nullptr) {
            block_to_free->next->prev = prev_block;
        }
        prev_block->size += block_to_free->size;
        concurrent::get_shared_mem_segment().deallocate(block_to_free.get());
        block_to_free = prev_block;
        merged_prev = true;
    }

    if (next_block != nullptr and not is_allocated(next_block)) {
        block_to_free->next = next_block->next;
        if (next_block->next != 0) {
            next_block->next->prev = block_to_free;
        }
        if (next_block == memory_manager->free_block_head) {
            memory_manager->free_block_head = block_to_free;
        }
        if (next_block == memory_manager->free_block_tail) {
            memory_manager->free_block_tail = block_to_free;
        }
        block_to_free->next_free = next_block->next_free;
        if (next_block->next_free != 0) {
            next_block->next_free->prev_free = block_to_free;
        }
        if (not merged_prev) {
            block_to_free->prev_free = next_block->prev_free;
            if (next_block->prev_free != 0) {
                next_block->prev_free->next_free = block_to_free;
            }
        }
        block_to_free->size += next_block->size;
        concurrent::get_shared_mem_segment().deallocate(next_block.get());
        merged_next = true;
    }

    if (not merged_prev and not merged_next) {
        // Find where to include deallocated block in free list
        block_offset_ptr_t prev_free_block = block_to_free->prev;
        while (prev_free_block != 0 and is_allocated(prev_free_block)) {
            prev_free_block = prev_free_block->prev;
        }
        block_offset_ptr_t next_free_block = block_to_free->next;
        while (next_free_block != nullptr and is_allocated(next_free_block)) {
            next_free_block = next_free_block->next;
        }
        block_to_free->prev_free = prev_free_block;
        if (prev_free_block != 0) {
            prev_free_block->next_free = block_to_free;
        } else {
            memory_manager->free_block_head = block_to_free;
        }

        block_to_free->next_free = next_free_block;
        if (next_free_block != 0) {
            next_free_block->prev_free = block_to_free;
        } else {
            memory_manager->free_block_tail = block_to_free;
        }
    }

    if (address == this->lowest_occupied_address_) {
        this->update_lowest_occupied_address();
    }
}

void FreeList::clear() {
    concurrent::allocator_and_lock_pair_t memory_manager_and_lock = concurrent::get_allocator(this->name_.c_str());
    concurrent::allocator_t *memory_manager = memory_manager_and_lock.first;
    boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(memory_manager_and_lock.second);

    block_offset_ptr_t prev_block = 0, curr_block;
    curr_block = memory_manager->block_head;
    while (curr_block != 0) {
        prev_block = curr_block;
        curr_block = curr_block->next;
        concurrent::get_shared_mem_segment().deallocate(prev_block.get());
    }

    block_offset_ptr_t block = static_cast<concurrent::block_t *>(concurrent::get_shared_mem_segment().allocate(sizeof(concurrent::block_t)));
    block->address = 0;
    block->size = this->max_size_bytes_;
    block->prev = 0; block->next = 0; block->prev_free = 0; block->next_free = 0;
    memory_manager->block_head = block;
    memory_manager->free_block_head = block;
    memory_manager->free_block_tail = block;
}

// Feeds memory reports
Statistics FreeList::get_statistics() const {
    concurrent::allocator_and_lock_pair_t memory_manager_and_lock = concurrent::get_allocator(this->name_.c_str());
    concurrent::allocator_t *memory_manager = memory_manager_and_lock.first;
    boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(memory_manager_and_lock.second);

    Statistics stats{
        .total_allocatable_size_bytes = this->max_size_bytes_,
        .total_allocated_bytes = 0,
        .total_free_bytes = 0,
        .largest_free_block_bytes = 0
    };

    block_offset_ptr_t curr_block = memory_manager->block_head;
    while (curr_block != 0) {
        if (this->is_allocated(curr_block)) {
            stats.total_allocated_bytes += curr_block->size;
        } else {
            stats.total_free_bytes += curr_block->size;
            if (curr_block->size >= stats.largest_free_block_bytes) {
                stats.largest_free_block_bytes = curr_block->size;
                stats.largest_free_block_addrs.push_back(curr_block->address + this->offset_bytes_);
            }
        }
        curr_block = curr_block->next;
    }
    if (stats.total_allocated_bytes == 0) {
        stats.total_free_bytes = this->max_size_bytes_;
        stats.largest_free_block_bytes = this->max_size_bytes_;
    }
    return stats;
}

void FreeList::dump_block(block_offset_ptr_t block, std::ofstream &out) const {
    auto alloc_status = this->is_allocated(block) ? "Y" : "N";
    out << ",,,Address (B):," << (block->address + this->offset_bytes_) << "\n"
        << ",,,Size (B):," << (block->size) << "\n"
        << ",,,Allocated (Y/N):," << alloc_status << "\n";
}

void FreeList::dump_blocks(std::ofstream &out) const {
    concurrent::allocator_and_lock_pair_t memory_manager_and_lock = concurrent::get_allocator(this->name_.c_str());
    concurrent::allocator_t *memory_manager = memory_manager_and_lock.first;
    // Lock while dumping current state so its correct
    boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(memory_manager_and_lock.second);

    out << ",,Memory Blocks:\n";
    block_offset_ptr_t curr_block = memory_manager->block_head;
    while (curr_block != 0) {
        this->dump_block(curr_block, out);
        curr_block = curr_block->next;
    }
    out << "\n";
}

void FreeList::debug_dump_blocks() const {
#ifdef DEBUG
    concurrent::allocator_and_lock_pair_t memory_manager_and_lock = concurrent::get_allocator(this->name_.c_str());
    concurrent::allocator_t *memory_manager = memory_manager_and_lock.first;

    std::cout << "------------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "Memory Blocks:\n";
    block_offset_ptr_t curr_block = memory_manager->block_head;
    while (curr_block != 0) {
        auto alloc_status = this->is_allocated(curr_block) ? "Y" : "N";
        std::cout << "Address (B): " << (curr_block->address)
                  << "\tSize (B): " << (curr_block->size)
                  << "\tAllocated (Y/N): " << alloc_status
                  << "\tCurr Offset: " << curr_block.get()
                  << "\tPrev Offset: " << curr_block->prev.get()
                  << "\tNext Offset: " << curr_block->next.get()
                  << "\tPrev Free Offset: " << curr_block->prev_free.get()
                  << "\tNext Free Offset: " << curr_block->next_free.get()
                  << "\n";
        curr_block = curr_block->next;
    }
#endif
}

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
