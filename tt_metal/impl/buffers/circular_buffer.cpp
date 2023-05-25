#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

CircularBuffer::CircularBuffer(
    Device *device,
    const CoreRangeSet &core_range_set,
    uint32_t buffer_index,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format) :
    device_(device), core_range_set_(core_range_set), buffer_index_(buffer_index), num_tiles_(num_tiles), size_(size_in_bytes), address_(0), data_format_(data_format), allocated_on_device_(true) {
    uint32_t address = std::numeric_limits<uint32_t>::max();
    for (auto core_range : this->core_range_set_.ranges()) {
        uint32_t l1_address = allocator::get_address_for_circular_buffers_across_core_range(*device->allocator_, core_range, size_in_bytes);
        if (address == std::numeric_limits<uint32_t>::max()) {
            address = l1_address;
        } else if (l1_address != address) {
            TT_THROW("Cannot allocate CBs in core range " + core_range.str() + " at desired address  " + std::to_string(address));
        }
    }
    this->address_ = address;
    this->reserve();
}

CircularBuffer::CircularBuffer(
    Device *device,
    const CoreRangeSet &core_range_set,
    uint32_t buffer_index,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t address,
    DataFormat data_format) :
    device_(device), core_range_set_(core_range_set), buffer_index_(buffer_index), num_tiles_(num_tiles), size_(size_in_bytes), address_(address), data_format_(data_format), allocated_on_device_(false) {
    // TODO (abhullar): Enable invoking allocator when we have a spec for overlapping circular buffers in L1
    TT_ASSERT(address_ >= UNRESERVED_BASE, "First " + std::to_string(UNRESERVED_BASE) + " bytes in L1 are reserved");
    // This assertion is only added for circular buffers because DRAM buffers and Interleaved DRAM buffers invoke mem manager
    // to reserve specific addresses which checks for aligned addresses.
    TT_ASSERT(address % 32 == 0, "Requested address " + std::to_string(address) + " should be 32B aligned");
}

CircularBuffer::CircularBuffer(CircularBuffer &&other)
    : device_(other.device_),
      core_range_set_(other.core_range_set_),
      buffer_index_(other.buffer_index_),
      num_tiles_(other.num_tiles_),
      size_(other.size_),
      address_(other.address_),
      data_format_(other.data_format_),
      allocated_on_device_(other.allocated_on_device_) {
    other.device_ = nullptr;
    other.allocated_on_device_ = false;
}

CircularBuffer &CircularBuffer::operator=(CircularBuffer &&other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->core_range_set_ = other.core_range_set_;
        this->buffer_index_ = other.buffer_index_;
        this->num_tiles_ = other.num_tiles_;
        this->size_ = other.size_;
        this->address_ = other.address_;
        this->data_format_ = other.data_format_;
        this->allocated_on_device_ = other.allocated_on_device_;
        other.device_ = nullptr;
        other.allocated_on_device_ = false;
    }
    return *this;
}

bool CircularBuffer::is_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_range_set_.core_coord_in_core_ranges(logical_core);
}

void CircularBuffer::reserve() {
    for (auto core_range : this->core_range_set_.ranges()) {
        auto start = core_range.start;
        auto end = core_range.end;
        for (auto x = start.x; x <= end.x; x++) {
            for (auto y = start.y; y <= end.y; y++) {
                CoreCoord logical_core({.x=x, .y=y});
                uint32_t address = allocator::allocate_circular_buffer(*this->device_->allocator_, logical_core, this->address_, this->size_);
                TT_ASSERT(address == this->address_);
            }
        }
    }
}

void CircularBuffer::deallocate() {
    if (this->allocated_on_device_) {
        if (this->device_ == nullptr or this->device_->closed_) {
            return;
        }
        for (auto core_range : this->core_range_set_.ranges()) {
            auto start = core_range.start;
            auto end = core_range.end;
            for (auto x = start.x; x <= end.x; x++) {
                for (auto y = start.y; y <= end.y; y++) {
                    CoreCoord logical_core({.x=x, .y=y});
                    auto bank_ids = this->device_->bank_ids_from_logical_core(logical_core);
                    TT_ASSERT(bank_ids.size() == 1);
                    allocator::deallocate_buffer(*this->device_->allocator_, bank_ids.at(0), this->address_, BufferType::L1);
                }
            }
        }
        this->allocated_on_device_ = false;
    }
}

CircularBuffer::~CircularBuffer() {
    this->deallocate();
}

}  // namespace tt_metal

}  // namespace tt
