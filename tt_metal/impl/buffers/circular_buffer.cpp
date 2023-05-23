#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

CircularBuffer::CircularBuffer(
    Device *device,
    const CoreCoord &logical_core,
    uint32_t buffer_index,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format) :
    device_(device), logical_core_(logical_core), buffer_index_(buffer_index), num_tiles_(num_tiles), size_(size_in_bytes), address_(0), data_format_(data_format), allocated_on_device_(true) {
    this->address_ = allocator::allocate_circular_buffer(*device->allocator_, logical_core, size_in_bytes);
}

CircularBuffer::CircularBuffer(
    Device *device,
    const CoreCoord &logical_core,
    uint32_t buffer_index,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t address,
    DataFormat data_format) :
    device_(device), logical_core_(logical_core), buffer_index_(buffer_index), num_tiles_(num_tiles), size_(size_in_bytes), address_(address), data_format_(data_format), allocated_on_device_(false) {
    // TODO (abhullar): Enable invoking allocator when we have a spec for overlapping circular buffers in L1
    TT_ASSERT(address_ >= UNRESERVED_BASE, "First " + std::to_string(UNRESERVED_BASE) + " bytes in L1 are reserved");
    // This assertion is only added for circular buffers because DRAM buffers and Interleaved DRAM buffers invoke mem manager
    // to reserve specific addresses which checks for aligned addresses.
    TT_ASSERT(address % 32 == 0, "Requested address " + std::to_string(address) + " should be 32B aligned");
}

CircularBuffer::CircularBuffer(CircularBuffer &&other)
    : device_(other.device_),
      logical_core_(other.logical_core_),
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
        this->logical_core_ = other.logical_core_;
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

CoreCoord CircularBuffer::noc_coordinates() const {
    return this->device_->worker_core_from_logical_core(this->logical_core_);
}

void CircularBuffer::reserve() {
    auto address = allocator::allocate_circular_buffer(*this->device_->allocator_, this->logical_core_, this->size_, this->address_);
    TT_ASSERT(address == this->address_);
}

void CircularBuffer::deallocate() {
    if (this->allocated_on_device_) {
        if (this->device_ == nullptr or this->device_->closed_) {
            return;
        }
        auto bank_ids = this->device_->bank_ids_from_logical_core(this->logical_core_);
        TT_ASSERT(bank_ids.size() == 1);
        allocator::deallocate_buffer(*this->device_->allocator_, bank_ids.at(0), this->address_, BufferType::L1);
        this->allocated_on_device_ = false;
    }
}

CircularBuffer::~CircularBuffer() {
    this->deallocate();
}

}  // namespace tt_metal

}  // namespace tt
