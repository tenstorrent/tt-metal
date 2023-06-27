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
    device_(device), core_range_set_(core_range_set), buffer_indices_(buffer_indices), num_tiles_(num_tiles), size_(size_in_bytes), data_format_(data_format), state_(State::UNALLOCATED) {
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
    device_(device), core_range_set_(core_range_set), buffer_indices_(buffer_indices), num_tiles_(num_tiles), size_(size_in_bytes), address_(address), data_format_(data_format), state_(State::ADDRESS_SPECIFIED) {
    validate_buffer_indices(buffer_indices);
    this->reserve(device);
}

bool CircularBuffer::is_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_range_set_.core_coord_in_core_ranges(logical_core);
}


void CircularBuffer::reserve(Device* device) {
    log_assert(this->state_ != State::ALLOCATED, "Cannot re-allocate the buffer");
    this->device_ = device;
    // First find space on each core the CB is on because each core's memory can differ. Address has to be the same on each core.
    // Don't need to do this for manually specified addresses
    if (this->state_ != State::ADDRESS_SPECIFIED) {
        uint32_t address = 0;
        bool addr_set = false;
        for (auto core_range : this->core_range_set_.ranges()) {
            uint32_t l1_address = allocator::get_address_for_circular_buffers_across_core_range(*this->device_->allocator_, core_range, this->size_);
            if (not addr_set) {
                address = l1_address;
                addr_set = true;
            } else if (l1_address != address) {
                TT_THROW("Cannot allocate CBs in core range " + core_range.str() + " at desired address  " + std::to_string(address));
            }
        }
        this->address_ = address;
        this->state_ = State::ADDRESS_SPECIFIED;
    }

    for (auto core_range : this->core_range_set_.ranges()) {
        auto start = core_range.start;
        auto end = core_range.end;
        for (auto x = start.x; x <= end.x; x++) {
            for (auto y = start.y; y <= end.y; y++) {
                CoreCoord logical_core({.x=x, .y=y});
                uint32_t address = allocator::allocate_circular_buffer(*this->device_->allocator_, logical_core, this->address_, this->size_);
                log_assert(address == this->address_, "Allocator failed: allocated address {} is not the same as specified address {}", address, this->address_);
            }
        }
    }
    this->state_ = State::ALLOCATED;
}

void CircularBuffer::deallocate() {
    if (this->state_ == State::ALLOCATED) {
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
        this->state_ = State::UNALLOCATED;
    }
}

CircularBuffer::~CircularBuffer() {
    this->deallocate();
}

}  // namespace tt_metal

}  // namespace tt
