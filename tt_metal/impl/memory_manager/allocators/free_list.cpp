#include "tt_metal/impl/memory_manager/allocators/free_list.hpp"
#include "common/assert.hpp"

#include <cmath>

namespace tt {

namespace tt_metal {

namespace allocator {

FreeList::FreeList(uint32_t max_size_bytes, uint32_t min_allocation_size, FreeList::SearchPolicy search_policy)
    : search_policy_(search_policy), Allocator(max_size_bytes, min_allocation_size) {
    this->init();
}

void FreeList::init() {
    auto block = new FreeList::Block{.address = 0, .size = this->max_size_bytes_};
    this->block_head_ = block;
    this->free_block_head_ = block;
}

bool FreeList::is_allocated(Block *block) const {
    return block->prev_free == nullptr and block->next_free == nullptr and block != this->free_block_head_;
}

std::vector<std::pair<uint32_t, uint32_t>> FreeList::available_addresses(uint32_t size_bytes) const {
    std::vector<std::pair<uint32_t, uint32_t>> addresses;
    FreeList::Block *curr_block = this->free_block_head_;
    while (curr_block != nullptr) {
        if (curr_block->size >= size_bytes) {
            uint32_t end_range = (curr_block->address + curr_block->size) - size_bytes;
            addresses.push_back({curr_block->address, end_range});
        }
        curr_block = curr_block->next_free;
    }
    return addresses;
}

FreeList::Block *FreeList::search_best(uint32_t size_bytes) {
    FreeList::Block *best_block = nullptr;
    FreeList::Block *curr_block = this->free_block_head_;
    while (curr_block != nullptr) {
        if (curr_block->size == size_bytes) {
            best_block = curr_block;
            break;
        } else if (curr_block->size >= size_bytes) {
            if (best_block == nullptr or curr_block->size < best_block->size) {
                best_block = curr_block;
            }
        }
        curr_block = curr_block->next_free;
    }
    return best_block;
}

FreeList::Block *FreeList::search_first(uint32_t size_bytes) {
    FreeList::Block *curr_block = this->free_block_head_;
    while (curr_block != nullptr) {
        if (curr_block->size >= size_bytes) {
            return curr_block;
        }
        curr_block = curr_block->next_free;
    }
    return nullptr;
}

FreeList::Block *FreeList::search(uint32_t size_bytes) {
    switch (this->search_policy_) {
        case FreeList::SearchPolicy::BEST:
            return search_best(size_bytes);
        break;
        case FreeList::SearchPolicy::FIRST:
            return search_first(size_bytes);
        break;
        default:
            TT_ASSERT(false && "Unsupported search policy");
    }
    return nullptr;
}

void FreeList::split_free_block(Block *to_be_allocated, uint32_t size_bytes) {
    uint32_t next_address = to_be_allocated->address + size_bytes;
    uint32_t split_block_size = to_be_allocated->size - size_bytes;
    auto split_block = new FreeList::Block{
        .address = next_address,
        .size = split_block_size,
        .prev_block = to_be_allocated,
        .next_block = to_be_allocated->next_block
    };
    split_block->prev_free = to_be_allocated->prev_free;
    split_block->next_free = to_be_allocated->next_free;
    if (to_be_allocated->prev_free != nullptr) {
        to_be_allocated->prev_free->next_free = split_block;
    }
    if (to_be_allocated->next_free != nullptr) {
        to_be_allocated->next_free->prev_free = split_block;
    }
    if (to_be_allocated == this->free_block_head_) {
        this->free_block_head_ = split_block;
    }
    if (to_be_allocated->next_block != nullptr) {
        to_be_allocated->next_block->prev_block = split_block;
    }
    to_be_allocated->next_block = split_block;
}

void FreeList::allocate_free_block(Block *to_be_allocated, uint32_t size_bytes) {
    if (to_be_allocated->size > size_bytes) {
        split_free_block(to_be_allocated, size_bytes);
    } else {
        if (to_be_allocated->prev_free != nullptr) {
            to_be_allocated->prev_free->next_free = to_be_allocated->next_free;
        }
        if (to_be_allocated->next_free != nullptr) {
            to_be_allocated->next_free->prev_free = to_be_allocated->prev_free;
        }
        if (to_be_allocated == this->free_block_head_) {
            this->free_block_head_ = to_be_allocated->next_free;
            this->free_block_head_->prev_free = nullptr;
        }
    }
    to_be_allocated->size = size_bytes;
    to_be_allocated->prev_free = nullptr;
    to_be_allocated->next_free = nullptr;
}

uint32_t FreeList::allocate(uint32_t size_bytes) {
    uint32_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : size_bytes;
    auto first_fit_block = search_first(alloc_size);

    if (first_fit_block == nullptr) {
        TT_THROW("Not enough memory to allocate " + std::to_string(size_bytes) + " bytes");
    }

    allocate_free_block(first_fit_block, alloc_size);

    return first_fit_block->address;
}

void FreeList::segment_free_block(Block *to_be_split, uint32_t start_address, uint32_t size_bytes) {
    TT_ASSERT(
        start_address >= to_be_split->address and start_address + size_bytes <= to_be_split->address + to_be_split->size);
    auto allocated_block = new FreeList::Block{
        .address = start_address,
        .size = size_bytes,
        .prev_block = to_be_split,
    };
    uint32_t next_address = start_address + size_bytes;
    FreeList::Block *next;
    if (next_address == to_be_split->address + to_be_split->size) {
        next = to_be_split->next_block;
        if (next != nullptr) {
            next->prev_block = allocated_block;
        }
    } else {
        uint32_t next_free_size = (to_be_split->size - next_address) + to_be_split->address;
        next = new FreeList::Block{
            .address = next_address,
            .size = next_free_size,
            .prev_block = allocated_block,
            .next_block = to_be_split->next_block,
            .prev_free = to_be_split,
            .next_free = to_be_split->next_free
        };
        if (to_be_split->next_block != nullptr) {
            to_be_split->next_block->prev_block = next;
        }
        if (to_be_split->next_free != nullptr) {
            to_be_split->next_free->prev_free = next;
        }
        to_be_split->next_free = next;
    }
    allocated_block->next_block = next;
    to_be_split->next_block = allocated_block;
    to_be_split->size -= ((to_be_split->size - start_address) + to_be_split->address);
}

uint32_t FreeList::reserve(uint32_t start_address, uint32_t size_bytes) {
    FreeList::Block *curr_block = this->free_block_head_;
    uint32_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : size_bytes;
    while (curr_block != nullptr) {
        if (curr_block->size >= alloc_size) {
            if (curr_block->address == start_address) {
                allocate_free_block(curr_block, alloc_size);
                break;
            } else if ((start_address > curr_block->address) and ((start_address + alloc_size) <= (curr_block->address + curr_block->size))) {
                segment_free_block(curr_block, start_address, alloc_size);
                break;
            }
        }
        curr_block = curr_block->next_free;
    }

    if (curr_block == nullptr) {
        TT_THROW("Cannot reserve " + std::to_string(size_bytes) + " at " + std::to_string(start_address) + ". It is already reserved!");
    }
    return start_address;
}

FreeList::Block *FreeList::find_block(uint32_t address) {
    FreeList::Block *block = nullptr;
    FreeList::Block *curr_block = this->block_head_;
    while (curr_block != nullptr) {
        if (curr_block->address == address) {
            return curr_block;
        }
        curr_block = curr_block->next_block;
    }
    return block;
}

void FreeList::deallocate(uint32_t address) {
    FreeList::Block *block_to_free = find_block(address);
    if (block_to_free == nullptr or not this->is_allocated(block_to_free)) {
        return;
    }

    auto prev = block_to_free->prev_block;
    auto next = block_to_free->next_block;

    bool merged_prev = false;
    bool merged_next = false;
    if (prev != nullptr and not is_allocated(prev)) {
        prev->next_block = block_to_free->next_block;
        if (block_to_free->next_block != nullptr) {
            block_to_free->next_block->prev_block = prev;
        }
        prev->size += block_to_free->size;
        block_to_free = prev;
        merged_prev = true;
    }

    if (next != nullptr and not is_allocated(next)) {
        block_to_free->next_block = next->next_block;
        if (next->next_block != nullptr) {
            next->next_block->prev_block = block_to_free;
        }
        if (next == this->free_block_head_) {
            this->free_block_head_ = block_to_free;
        }
        block_to_free->next_free = next->next_free;
        if (next->next_free != nullptr) {
            next->next_free->prev_free = block_to_free;
        }
        if (not merged_prev) {
            block_to_free->prev_free = next->prev_free;
            if (next->prev_free != nullptr) {
                next->prev_free->next_free = block_to_free;
            }
        }
        block_to_free->size += next->size;
        merged_next = true;
    }

    if (not merged_prev and not merged_next) {
        // Find where to include deallocated block in free list
        auto prev_free_block = block_to_free->prev_block;
        while (prev_free_block != nullptr and is_allocated(prev_free_block)) {
            prev_free_block = prev_free_block->prev_block;
        }
        auto next_free_block = block_to_free->next_block;
        while (next_free_block != nullptr and is_allocated(next_free_block)) {
            next_free_block = next_free_block->next_block;
        }
        block_to_free->prev_free = prev_free_block;
        if (prev_free_block != nullptr) {
            prev_free_block->next_free = block_to_free;
        } else {
            this->free_block_head_ = block_to_free;
        }

        block_to_free->next_free = next_free_block;
        if (next_free_block != nullptr) {
            next_free_block->prev_free = block_to_free;
        }
    }
}

void FreeList::reset() {
    Block *curr_block = this->block_head_;
    Block *next;
    while (curr_block != nullptr) {
        next = curr_block->next_block;
        delete curr_block;
        curr_block = next;
    }
    this->block_head_ = nullptr;
    this->free_block_head_ = nullptr;
}

void FreeList::clear() {
    this->reset();
    this->init();
}

FreeList::~FreeList() {
    this->reset();
}

void FreeList::dump_blocks() const {
    std::cout << "DUMPING MEMORY BLOCKS:" << std::endl;
    Block *curr_block = this->block_head_;
    if (this->block_head_ != nullptr) {
        auto status = this->is_allocated(this->block_head_) ? " allocated " : " free ";
        std::cout << "Block head address: " << this->block_head_->address
                  << " size: " << this->block_head_->size
                  << status << std::endl;
    }
    if (this->free_block_head_ != nullptr) {
        auto status = this->is_allocated(this->free_block_head_) ? " allocated " : " free ";
        std::cout << "Free block head address: " << this->free_block_head_->address
                  << " size: " << this->free_block_head_->size
                  << status << std::endl;
    }
    while (curr_block != nullptr) {
        auto status = this->is_allocated(curr_block) ? " allocated " : " free ";
        std::cout << "\tBlock address: " << curr_block->address
                  << " size: " << curr_block->size << " bytes"
                  << status << std::endl;
        curr_block = curr_block->next_block;
    }
    std::cout << "\n";
}

}  // namespace allocator

}  // namespace tt_metal

}  // namespace tt
