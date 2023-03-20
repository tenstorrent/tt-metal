#include "tt_metal/impl/memory_manager/memory_manager.hpp"
#include "common/assert.hpp"

#include <cmath>

namespace tt {

namespace tt_metal {

MemoryManager::MemoryManager(uint32_t max_size_bytes) : max_size_bytes_(max_size_bytes) {
    constexpr static uint32_t min_allocation_size = sizeof(uint32_t);
    this->min_allocation_size_ = min_allocation_size;
    this->free_tree_ = red_black_tree(max_size_bytes);
    this->head_ = nullptr;
    this->tail_ = nullptr;
}

uint32_t MemoryManager::get_address(uint32_t size_bytes) {
    if (this->head_ == nullptr) {
        this->head_ = new Block{.address = 0, .size = size_bytes};
        this->tail_ = this->head_;
        return this->head_->address;
    }

    // Blocks are sorted based on address
    if (this->head_->address != 0 and (this->head_->address >= size_bytes)) {
        Block *new_block = new Block{.address = (this->head_->address - size_bytes), .size = size_bytes};
        new_block->next = this->head_;
        this->head_->prev = new_block;
        this->head_ = new_block;
        return this->head_->address;
    }

    Block *prev_block = nullptr;
    Block *curr_block = this->head_;
    while (curr_block != nullptr) {
        if (curr_block->size != 0) {
            prev_block = curr_block;
            curr_block = curr_block->next;
        } else if (curr_block->next->address - curr_block->address >= size_bytes) {
            curr_block->size = size_bytes;
            return curr_block->address;
        }
    }
    uint32_t address = prev_block->address + prev_block->size;
    Block *new_block = new Block{.address = address, .size = size_bytes};
    prev_block->next = new_block;
    new_block->prev = prev_block;
    this->tail_ = new_block;
    return address;
}

uint32_t MemoryManager::reserve_free_space(uint32_t size_bytes) {
    if (size_bytes > this->max_size_bytes_) {
        TT_THROW(size_bytes + " bytes is larger than maximum available size of " + std::to_string(this->max_size_bytes_) + " bytes");
    }

    uint32_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : size_bytes;

    red_black_tree::node_t *free_node = this->free_tree_.search_best(alloc_size);
    if (free_node == nullptr) {
        TT_THROW("Not enough memory to allocate " + std::to_string(size_bytes) + " bytes!");
    }

    this->free_tree_.remove(free_node);

    if (free_node->key > alloc_size) {
        uint32_t available_chunk_size = free_node->key - alloc_size;
        if (available_chunk_size >= this->min_allocation_size_) {
            this->free_tree_.insert(available_chunk_size);
        }
    }
    return alloc_size;
}

uint32_t MemoryManager::malloc(uint32_t size_bytes) {
    uint32_t alloc_size = reserve_free_space(size_bytes);
    uint32_t address = get_address(alloc_size);
    return address;
}

void MemoryManager::insert_block(Block *block) {
    if (this->head_ == nullptr) {
        this->head_ = block;
        this->tail_ = block;
        return;
    }

    if (this->head_->address > block->address) {
        block->next = this->head_;
        block->next->prev = block;
        this->head_ = block;
        return;
    }

    Block *curr_block = this->head_;
    while (curr_block->next != nullptr and curr_block->next->address < block->address) {
        curr_block = curr_block->next;
    }
    block->next = curr_block->next;
    if (curr_block->next != nullptr) {
        block->next->prev = block;
    } else {
        this->tail_ = block;
    }

    curr_block->next = block;
    block->prev = curr_block;
}

uint32_t MemoryManager::reserve(uint32_t start_address, uint32_t size_bytes) {
    uint32_t alloc_size = reserve_free_space(size_bytes);

    Block *curr_block = this->head_;
    while (curr_block != nullptr) {
        if (curr_block->address == start_address) {
            if (curr_block->size != 0) {
                TT_THROW("Cannot reserve " + std::to_string(size_bytes) + " bytes at address " + std::to_string(start_address) + "! It is already reserved.");
            } else if (curr_block->next != nullptr and (curr_block->next->address - curr_block->address < alloc_size)) {
                TT_THROW("Cannot fit " + std::to_string(size_bytes) + "  bytes at address " + std::to_string(start_address));
            } else {
                curr_block->size = alloc_size;
                return start_address;
            }
        }
        curr_block = curr_block->next;
    }

    // Didn't find the address in the list so we need to insert it
    Block *new_block = new Block{.address = start_address, .size = alloc_size};
    insert_block(new_block);
    return start_address;
}

uint32_t MemoryManager::peak() const {
    if (this->head_ == nullptr) {
        TT_ASSERT(this->tail_ == nullptr);
        return 0;
    }
    Block *curr_block = this->tail_;
    while (curr_block != nullptr) {
        if (curr_block->size != 0) {
            return curr_block->address + curr_block->size;
        }
        curr_block = curr_block->prev;
    }
    return 0;
}

uint32_t MemoryManager::coalesce(Block *block_to_free) {
    uint32_t freed_space = block_to_free->size;
    block_to_free->size = 0;

    Block *prev_block = block_to_free->prev;
    Block *next_block = block_to_free->next;

    if (block_to_free == this->head_ and block_to_free == this->tail_) {
        return this->max_size_bytes_;
    }

    Block *block = block_to_free;
    bool collapsed_block_to_free = false;
    if (prev_block != nullptr and prev_block->size == 0) {
        freed_space += (block->address - prev_block->address);
        prev_block->next = block->next;
        if (block->next != nullptr) {
            block->next->prev = prev_block;
        } else {
            this->tail_ = prev_block;
        }
        collapsed_block_to_free = true;
        block = prev_block;
    }

    bool collapsed_next_block = false;
    if (next_block != nullptr and next_block->size == 0) {
        if (next_block->next == nullptr) {
            freed_space += (this->max_size_bytes_ - next_block->address);
        } else {
            freed_space += (next_block->next->address - next_block->address);
        }
        block->next = next_block->next;
        if (next_block->next != nullptr) {
            next_block->next->prev = block;
        } else {
            this->tail_ = block;
        }
        collapsed_next_block = true;
    }

    if (collapsed_block_to_free) {
        block_to_free->prev = nullptr;
        block_to_free->next = nullptr;
        delete block_to_free;
    }

    if (collapsed_next_block) {
        next_block->prev = nullptr;
        next_block->next = nullptr;
        delete next_block;
    }

    return freed_space;
}

void MemoryManager::free(uint32_t address) {
    Block *block_to_free = this->head_;
    while (block_to_free != nullptr) {
        if (block_to_free->address == address) {
            break;
        }
        block_to_free = block_to_free->next;
    }
    if (block_to_free == nullptr) {
        return;
    }
    uint32_t freed_space = coalesce(block_to_free);
    if (freed_space >= this->min_allocation_size_) {
        this->free_tree_.insert(freed_space);
    }
}

void MemoryManager::clear() {
    this->free_tree_ = red_black_tree(this->max_size_bytes_);
    this->head_ = nullptr;
    this->tail_ = nullptr;
}

}  // namespace tt_metal

}  // namespace tt
