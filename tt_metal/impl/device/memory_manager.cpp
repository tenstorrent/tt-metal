#include "tt_metal/impl/device/memory_manager.hpp"

#include "common/assert.hpp"

namespace tt {

namespace tt_metal {

uint32_t MemoryManager::malloc(uint32_t size_bytes) {
    uint32_t curr_address = (uint32_t)start_ptr_ + offset_;
    if (curr_address + size_bytes > max_size_bytes_) {
        TT_THROW("Cannot allocate buffer of size " + std::to_string(size_bytes) + " bytes in DRAM");
    }
    auto next_address = offset_;
    offset_ += size_bytes;
    return next_address;
}

void MemoryManager::free(uint32_t address) {
    TT_ASSERT(false, "Unimplemented");
}

// Jump
uint32_t MemoryManager::reserve(uint32_t start_address, uint32_t size_bytes) {
    TT_ASSERT(start_address >= offset_ && "Requested chunk conflicts with allocated memory, cannot reserve!");
    offset_ = (start_address + size_bytes);
    return start_address;
}

void MemoryManager::clear() {
    offset_ = 0;
}

}  // namespace tt_metal

}  // namespace tt
