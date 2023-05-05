#include "tt_metal/impl/buffers/interleaved_l1_buffer.hpp"

#include <algorithm>

namespace tt {

namespace tt_metal {

InterleavedL1Buffer::InterleavedL1Buffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry)
    : num_bank_units_(num_bank_units),
      num_entries_per_bank_unit_(num_entries_per_bank_unit),
      num_bytes_per_entry_(num_bytes_per_entry),
      Buffer(device, num_bank_units * num_entries_per_bank_unit * num_bytes_per_entry, 0, true) {
    this->l1_bank_to_relative_address_ = this->device_->allocate_interleaved_l1_buffer(num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
    this->address_ = this->l1_bank_to_relative_address_.at(0).second;
}

Buffer *InterleavedL1Buffer::clone() {
    return new InterleavedL1Buffer(this->device_, this->num_bank_units_, this->num_entries_per_bank_unit_, this->num_bytes_per_entry_);
}

tt_xy_pair InterleavedL1Buffer::logical_core() const {
    if (this->l1_bank_to_relative_address_.empty()) {
        TT_THROW("Interleaved L1 buffer has not been allocated!");
    }
    L1Bank l1_bank_holding_first_bank_unit = this->l1_bank_to_relative_address_.at(0).first;
    return  l1_bank_holding_first_bank_unit.logical_core;
}

tt_xy_pair InterleavedL1Buffer::noc_coordinates() const {
    if (this->l1_bank_to_relative_address_.empty()) {
        TT_THROW("Interleaved L1 buffer has not been allocated!");
    }
    L1Bank l1_bank_holding_first_bank_unit = this->l1_bank_to_relative_address_.at(0).first;
    return this->device_->worker_core_from_logical_core(l1_bank_holding_first_bank_unit.logical_core);
}

L1Bank InterleavedL1Buffer::bank(int bank_index) const {
    if (bank_index >= this->num_banks()) {
        TT_THROW("Bank index " + std::to_string(bank_index) + " exceeds number of banks!");
    }
    return this->l1_bank_to_relative_address_.at(bank_index).first;
}

uint32_t InterleavedL1Buffer::address_of_bank_unit(int bank_index, int bank_unit_index) const {
    if (bank_index >= this->num_banks()) {
        TT_THROW("Bank index " + std::to_string(bank_index) + " exceeds number of banks!");
    }
    auto l1_bank = this->bank(bank_index);
    auto relative_address = this->l1_bank_to_relative_address_.at(bank_index).second;
    int units_read_in_bank = (int)bank_unit_index / this->num_banks();
    auto absolute_address = l1_bank.offset_bytes + relative_address;
    uint32_t offset = (this->bank_unit_size() * units_read_in_bank);
    absolute_address += offset;
    return absolute_address;
}

void InterleavedL1Buffer::free() {
    for (auto &l1_bank_addr_pair : this->l1_bank_to_relative_address_) {
        auto l1_bank = l1_bank_addr_pair.first;
        auto relative_address = l1_bank_addr_pair.second;
        this->device_->free_l1_buffer(l1_bank.logical_core, relative_address + l1_bank.offset_bytes);
    }
    this->allocated_on_device_ = false;
}

InterleavedL1Buffer::~InterleavedL1Buffer() {
    if (this->allocated_on_device_) {
        this->free();
    }
}

}  // namespace tt_metal

}  // namespace tt
