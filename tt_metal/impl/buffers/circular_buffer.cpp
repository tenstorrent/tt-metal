#include "tt_metal/impl/buffers/circular_buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

CircularBuffer::CircularBuffer(
    Device *device,
    const tt_xy_pair &logical_core,
    uint32_t buffer_index,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format) :
    logical_core_(logical_core), buffer_index_(buffer_index), num_tiles_(num_tiles), data_format_(data_format), Buffer(device, size_in_bytes, -1, true) {
    this->address_ = device->allocate_circular_buffer(logical_core, size_in_bytes);
}

CircularBuffer::CircularBuffer(
    Device *device,
    const tt_xy_pair &logical_core,
    uint32_t buffer_index,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t address,
    DataFormat data_format) :
    logical_core_(logical_core), buffer_index_(buffer_index), num_tiles_(num_tiles), data_format_(data_format), Buffer(device, size_in_bytes, address, false) {
    // TODO (abhullar): Enable invoking allocator when we have a spec for overlapping circular buffers in L1
    TT_ASSERT(address_ >= UNRESERVED_BASE, "First " + std::to_string(UNRESERVED_BASE) + " bytes in L1 are reserved");
    // This assertion is only added for circular buffers because DRAM buffers and Interleaved DRAM buffers invoke mem manager
    // to reserve specific addresses which checks for aligned addresses.
    TT_ASSERT(address % 32 == 0, "Requested address " + std::to_string(address) + " should be 32B aligned");
}

Buffer *CircularBuffer::clone() {
    return new CircularBuffer(
        this->device_, this->logical_core_, this->buffer_index_, this->num_tiles_, this->size_in_bytes_, this->data_format_);
}

tt_xy_pair CircularBuffer::noc_coordinates() const {
    return this->device_->worker_core_from_logical_core(this->logical_core_);
}

void CircularBuffer::reserve() {
    auto address = this->device_->allocate_circular_buffer(this->logical_core_, this->size_in_bytes_, this->address_);
    TT_ASSERT(address == this->address_);
}

void CircularBuffer::free() {
    if (this->allocated_on_device_) {
        this->device_->free_l1_buffer(this->logical_core_, this->address_);
        this->allocated_on_device_ = false;
    }
}

CircularBuffer::~CircularBuffer() {
    this->free();
}

}  // namespace tt_metal

}  // namespace tt
