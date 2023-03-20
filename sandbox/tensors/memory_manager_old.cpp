#include "tt_metal/impl/device/memory_manager.hpp"
#include "common/assert.hpp"

#include <cmath>

namespace tt {

namespace tt_metal {

constexpr uint32_t get_level(uint32_t max_size, uint32_t min_size) {
    return std::log2(max_size) - std::log2(min_size);
}

constexpr uint32_t level_offset(uint32_t level) {
    return (1 << level) - 1;
}

constexpr uint32_t nearest_power_of_two(uint32_t val) {
    uint32_t log2_val = std::log2(val);
    if (std::pow(2, log2_val) == val) {
        return val;
    }
    return std::pow(2, log2_val + 1);
}

constexpr bool is_power_of_two(uint32_t val) {
    if (val <= 0) {
        return false;
    }
    return ((val & (val - 1)) == 0);
}

constexpr uint32_t level_of_index(uint32_t block_index) {
    return std::floor(std::log2(block_index + 1));
}

constexpr uint32_t address_of_block_index(uint32_t block_index, uint32_t level, uint32_t memory_size) {
    uint32_t level_start = level_offset(level);
    uint32_t block_offset = block_index - level_start;
    return block_offset << ((uint32_t)std::log2(memory_size) - level);
}

constexpr uint32_t block_index_of_address(uint32_t address, uint32_t size_bytes, uint32_t memory_size) {
    uint32_t level = get_level(memory_size, size_bytes);
    return (1 << level) + (address >> ((uint32_t)std::log2(memory_size) - level)) - 1;
}

constexpr uint32_t parent_block_index(uint32_t child_index) {
    return (child_index - 1) / 2;
}

MemoryManager::MemoryManager(uint32_t max_size_bytes) : max_size_bytes_(max_size_bytes) {
    constexpr static uint32_t min_allocation_size = sizeof(uint32_t);

    this->num_levels_ = get_level(max_size_bytes, min_allocation_size) + 1;
    this->num_blocks_ = level_offset(this->num_levels_);
    this->min_allocation_size_ = min_allocation_size;

    this->in_use_.resize(this->num_blocks_, 0);
    this->is_split_.resize(this->num_blocks_, 0);

    // std::cout << "Initializing memory manager with " << max_size_bytes
    //           << " min allocation size " << this->min_allocation_size_
    //           << " num levels: " << this->num_levels_
    //           << " num blocks: " << this->num_blocks_ << std::endl;
}

bool MemoryManager::ancestor_in_use(uint32_t child_level, uint32_t child_index) {
    if (child_level == 0) {
        return false;
    }

    uint32_t block_index = child_index;
    bool in_use = false;
    for (int parent_level = child_level - 1; parent_level >= 0; parent_level--) {
        block_index = parent_block_index(block_index);
        if (this->in_use_[block_index]) {
            in_use = true;
        }
    }

    return in_use;
}

uint32_t MemoryManager::level_of_used_ancestor(uint32_t child_level, uint32_t child_index) {
    if (child_level == 0) {
        return 0;
    }

    uint32_t block_index = child_index;
    uint32_t level = child_level;
    for (int parent_level = child_level - 1; parent_level >=0; parent_level--) {
        block_index = parent_block_index(block_index);
        if (this->in_use_[block_index]) {
            level = parent_level;
        }
    }
    return level;
}

void MemoryManager::mark_ancestors_as_split(uint32_t child_level, uint32_t child_index) {
    if (child_level == 0) {
        return;
    }

    uint32_t block_index = child_index;
    for (int parent_level = child_level - 1; parent_level >=0; parent_level--) {
        block_index = parent_block_index(block_index);
        if (this->is_split_[block_index]) {
            break;
        }
        this->is_split_[block_index] = 1;
    }
}

uint32_t MemoryManager::malloc(uint32_t size_bytes) {
    if (size_bytes > this->max_size_bytes_) {
        TT_THROW(size_bytes + " bytes is larger than maximum available size of " + std::to_string(this->max_size_bytes_) + " bytes");
    }

    uint32_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : nearest_power_of_two(size_bytes);
    uint32_t level = get_level(this->max_size_bytes_, alloc_size);
    uint32_t block_index = level_offset(level);
    uint32_t level_end = level_offset(level + 1);

    // std::cout << "Requesting " << size_bytes << " bytes "
    //         << "Will allocate " << alloc_size << " bytes "
    //         << "Target level " << level << " "
    //         << "Level starts at " << block_index << " "
    //         << "Level ends at " << level_end << std::endl;

    while (block_index < level_end) {
        if (this->ancestor_in_use(level, block_index)) {
            // std::cout << "At level " << level << " original block_index " << block_index << " ancestor is in use";
            block_index += (1 << (level - this->level_of_used_ancestor(level, block_index)));
            // std::cout << " block index jumping to: " << block_index << std::endl;
        } else if (not this->in_use_[block_index]) {
            this->in_use_[block_index] = 1;
            mark_ancestors_as_split(level, block_index);
            // std::cout << "Block index " << block_index << " is not in use, allocate it and mark parent ancestors as being split" << std::endl;
            break;
        } else {
            block_index++;
        }
    }

    if (block_index >= level_end) {
        TT_THROW("Not enough memory to allocate " + std::to_string(size_bytes) + " bytes!");
    }

    uint32_t address = address_of_block_index(block_index, level, this->max_size_bytes_);
    return address;
}

uint32_t MemoryManager::reserve(uint32_t start_address, uint32_t size_bytes) {
    TT_ASSERT(is_power_of_two(size_bytes) && "Can only reserve power of two sizes");

    uint32_t block_index = block_index_of_address(start_address, size_bytes, this->max_size_bytes_);
    if (this->in_use_[block_index]) {
        TT_THROW("Cannot allocate " + std::to_string(size_bytes) + " bytes starting at " + std::to_string(start_address) + ". It's already in use!");
    }

    this->in_use_[block_index] = 1;
    uint32_t parent_index = parent_block_index(block_index);
    this->is_split_[parent_index] = 1;
    // std::cout << "Block index " << block_index << " is being marked as in use" << std::endl;
    return start_address;
}

uint32_t MemoryManager::peak() const {
    auto found = std::find(this->in_use_.rbegin(), this->in_use_.rend(), 1);
    if (found == this->in_use_.rend()) {
        return 0;
    }

    uint32_t max_used_index = (this->in_use_.rend() - found) - 1;
    uint32_t level = level_of_index(max_used_index);
    uint32_t address = address_of_block_index(max_used_index, level, this->max_size_bytes_);
    // std::cout << "Max used index: " << max_used_index << " level " << level << " address " << address << std::endl;
    return address;
}

void MemoryManager::free(uint32_t address, uint32_t size_bytes) {
    uint32_t alloc_size = size_bytes < this->min_allocation_size_ ? this->min_allocation_size_ : nearest_power_of_two(size_bytes);
    uint32_t block_index = block_index_of_address(address, alloc_size, this->max_size_bytes_);
    // std::cout << "Requesting to free " << address << " of size " << size_bytes << " which is at block index " << block_index << std::endl;
    if (not this->in_use_[block_index]) {
        TT_THROW("Attempting to double free memory at address " + std::to_string(address) + "that is already free!");
    }

    this->in_use_[block_index] = 0;
    int level = level_of_index(block_index);
    for (int curr_level = level; curr_level >= 0; curr_level--) {
        // std::cout << "block index " << block_index << " level " << curr_level << std::endl;
        if (this->in_use_[block_index] or this->is_split_[block_index]) {
            // std::cout << " in use " << std::to_string(this->in_use_[block_index])
            //           << " is split " << std::to_string(this->is_split_[block_index]) << " cannot free any further!" << std::endl;
            break;
        }
        uint32_t block_index_sibling = block_index + (block_index % 2 == 0 ? -1 : 1);
        if (this->in_use_[block_index_sibling] or this->is_split_[block_index_sibling]) {
            // std::cout << "Block index sibling " << block_index_sibling
            //           << " in use? " << std::to_string(this->in_use_[block_index_sibling])
            //           << " is split? " << std::to_string(this->is_split_[block_index_sibling])
            //           << " cannot merge!" << std::endl;
            break;
        }
        block_index = parent_block_index(block_index);
        this->is_split_[block_index] = 0;
        if (block_index == 0) {
            break;
        }
    }
}

void MemoryManager::clear() {
    std::fill(this->in_use_.begin(), this->in_use_.end(), 0);
    std::fill(this->is_split_.begin(), this->is_split_.end(), 0);
}

}  // namespace tt_metal

}  // namespace tt
