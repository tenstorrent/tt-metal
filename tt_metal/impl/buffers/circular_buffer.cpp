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
    buffer_index_(buffer_index), num_tiles_(num_tiles), data_format_(data_format), L1Buffer(device, logical_core, size_in_bytes) {
}

CircularBuffer::CircularBuffer(
    Device *device,
    const tt_xy_pair &logical_core,
    uint32_t buffer_index,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t address,
    DataFormat data_format) :
    buffer_index_(buffer_index), num_tiles_(num_tiles), data_format_(data_format), L1Buffer(device, logical_core, size_in_bytes, address) {
}

Buffer *CircularBuffer::clone() {
    return new CircularBuffer(
        this->device_, this->logical_core_, this->buffer_index_, this->num_tiles_, this->size_in_bytes_, this->data_format_);
}

CircularBuffer::~CircularBuffer() {
    this->free();
}

}  // namespace tt_metal

}  // namespace tt
