#include "tt_metal/impl/memory_manager/memory_manager.hpp"
#include "common/assert.hpp"

#include <cmath>

namespace tt {

namespace tt_metal {

MemoryManager::MemoryManager(uint32_t max_size_bytes) {
    constexpr static uint32_t min_allocation_size_bytes = 16;
    this->allocator_ = new allocator::FreeList(
        max_size_bytes,
        min_allocation_size_bytes,
        allocator::FreeList::SearchPolicy::FIRST);
}

uint32_t MemoryManager::allocate(uint32_t size_bytes) {
    return this->allocator_->allocate(size_bytes);
}

uint32_t MemoryManager::reserve(uint32_t start_address, uint32_t size_bytes) {
    return this->allocator_->reserve(start_address, size_bytes);
}

void MemoryManager::deallocate(uint32_t address) {
    this->allocator_->deallocate(address);
}

std::vector<std::pair<uint32_t, uint32_t>> MemoryManager::available_addresses(uint32_t size_bytes) const {
    return this->allocator_->available_addresses(size_bytes);
}

void MemoryManager::clear() {
    this->allocator_->clear();
}

MemoryManager::~MemoryManager() {
    delete this->allocator_;
    this->allocator_ = nullptr;
}

}  // namespace tt_metal

}  // namespace tt
