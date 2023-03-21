#include "tt_metal/impl/buffers/interleaved_buffer.hpp"

namespace tt {

namespace tt_metal {

InterleavedDramBuffer::InterleavedDramBuffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry)
    : Buffer(device, num_bank_units * num_entries_per_bank_unit * num_bytes_per_entry, 0, true) {
    this->set_address();

    int num_equally_distributed_units = num_bank_units / this->device_->num_dram_banks();
    int remaining_units_after_equally_distributing = num_bank_units % this->device_->num_dram_banks();

    uint32_t total_allocated = 0;
    for (int dram_bank = 0; dram_bank < this->device_->num_dram_banks(); dram_bank++) {
        int num_units_in_bank = num_equally_distributed_units;
        if (remaining_units_after_equally_distributing > 0) {
            num_units_in_bank += 1;
            remaining_units_after_equally_distributing -= 1;
        }
        uint32_t buffer_size = num_units_in_bank * (num_entries_per_bank_unit * num_bytes_per_entry);
        auto dram_buffer = new DramBuffer(device, dram_bank, buffer_size, this->address_);
        this->bank_to_buffer_.insert({dram_bank, dram_buffer});
        total_allocated += buffer_size;
        if (total_allocated == this->size_in_bytes_) {
            break;
        }
    }
}

InterleavedDramBuffer::InterleavedDramBuffer(Device *device, uint32_t total_size_bytes, const std::unordered_map<int, DramBuffer *> &bank_to_buffer)
    : Buffer(device, total_size_bytes, 0, true) {
    this->set_address();

    for (auto &[dram_bank, dram_buffer] : bank_to_buffer) {
        auto new_dram_buffer = new DramBuffer(device, dram_buffer->dram_channel(), dram_buffer->size(), this->address_);
        this->bank_to_buffer_.insert({dram_bank, new_dram_buffer});
    }
}

void InterleavedDramBuffer::set_address() {
    this->address_ = 0;
    for (int dram_bank = 0; dram_bank < this->device_->num_dram_banks(); dram_bank++) {
        this->address_ = std::max(this->address_, this->device_->banked_dram_manager_.at(dram_bank)->peak());
    }
}

Buffer *InterleavedDramBuffer::clone() {
    return new InterleavedDramBuffer(this->device_, this->size(), this->bank_to_buffer_);
}

uint32_t InterleavedDramBuffer::buffer_size(int dram_channel) const {
    if (this->bank_to_buffer_.find(dram_channel) == this->bank_to_buffer_.end()) {
        TT_THROW("No buffer has been allocated at DRAM channel " + std::to_string(dram_channel));
    }
    return this->bank_to_buffer_.at(dram_channel)->size();
}

tt_xy_pair InterleavedDramBuffer::noc_coordinates() const {
    return this->noc_coordinates(0);
}

tt_xy_pair InterleavedDramBuffer::noc_coordinates(int dram_channel) const {
    if (this->bank_to_buffer_.find(dram_channel) == this->bank_to_buffer_.end()) {
        TT_THROW("No buffer has been allocated at DRAM channel " + std::to_string(dram_channel));
    }
    return this->bank_to_buffer_.at(dram_channel)->noc_coordinates();
}

std::vector<tt_xy_pair> InterleavedDramBuffer::interleaved_noc_coordinates() const {
    std::vector<tt_xy_pair> dram_noc_coordinates;
    for (int dram_bank = 0; dram_bank < this->device_->num_dram_banks(); dram_bank++) {
        if (this->bank_to_buffer_.find(dram_bank) != this->bank_to_buffer_.end()) {
            dram_noc_coordinates.push_back(this->bank_to_buffer_.at(dram_bank)->noc_coordinates());
        }
    }
    TT_ASSERT(not dram_noc_coordinates.empty());
    return dram_noc_coordinates;
}

void InterleavedDramBuffer::free() {
    for (auto &[dram_bank, dram_buffer] : bank_to_buffer_) {
        dram_buffer->free();
    }
    this->allocated_on_device_ = false;
    bank_to_buffer_.clear();
}

InterleavedDramBuffer::~InterleavedDramBuffer() {
    if (this->allocated_on_device_) {
        this->free();
    }
}

}  // namespace tt_metal

}  // namespace tt
