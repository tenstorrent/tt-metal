#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

void validate_buffer_indices(const std::set<u32> &buffer_indices) {
    log_assert(buffer_indices.size() <= NUM_CIRCULAR_BUFFERS, "Number of circular buffers requested ({}) exceeds max number of circular buffers allowed on a core ({})", buffer_indices.size(), NUM_CIRCULAR_BUFFERS);
    // check that they're between 0 to NUM_CIRCULAR_BUFFERS - 1
    for (u32 buffer_index : buffer_indices) {
        log_assert(buffer_index < NUM_CIRCULAR_BUFFERS, "Buffer index can only be up to {}", NUM_CIRCULAR_BUFFERS - 1);
    }
}

CircularBuffer::CircularBuffer(
    Device *device,
    const CoreRangeSet &core_range_set,
    const std::set<u32> &buffer_indices,
    u32 num_tiles,
    u32 size_in_bytes,
    DataFormat data_format) :
    device_(device), core_range_set_(core_range_set), buffer_indices_(buffer_indices), num_tiles_(num_tiles), size_(size_in_bytes), address_(std::numeric_limits<uint32_t>::max()), data_format_(data_format), allocated_on_device_(false) {
    validate_buffer_indices(buffer_indices);
    this->reserve(device);
}

CircularBuffer::CircularBuffer(
    Device *device,
    const CoreRangeSet &core_range_set,
    const std::set<u32> &buffer_indices,
    u32 num_tiles,
    u32 size_in_bytes,
    u32 address,
    DataFormat data_format) :
    device_(device), core_range_set_(core_range_set), buffer_indices_(buffer_indices), num_tiles_(num_tiles), size_(size_in_bytes), address_(address), data_format_(data_format), allocated_on_device_(false) {
    validate_buffer_indices(buffer_indices);
    this->reserve(device);
}

bool CircularBuffer::is_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_range_set_.core_coord_in_core_ranges(logical_core);
}


void CircularBuffer::reserve(Device* device) {
    TT_ASSERT(not this->allocated_on_device_, "Cannot re-allocate the buffer");
    this->device_ = device;
    // First find space on each core the CB is on because each core's memory can differ. Address has to be the same on each core.
    // Don't need to do this for manually specified addresses
    if (this->address_ == std::numeric_limits<uint32_t>::max()) {
        uint32_t address = std::numeric_limits<uint32_t>::max();
        for (auto core_range : this->core_range_set_.ranges()) {
            uint32_t l1_address = allocator::get_address_for_circular_buffers_across_core_range(*this->device_->allocator_, core_range, this->size_);
            if (address == std::numeric_limits<uint32_t>::max()) {
                address = l1_address;
            } else if (l1_address != address) {
                TT_THROW("Cannot allocate CBs in core range " + core_range.str() + " at desired address  " + std::to_string(address));
            }
        }
        this->address_ = address;
    }

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
    this->allocated_on_device_ = true;
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
