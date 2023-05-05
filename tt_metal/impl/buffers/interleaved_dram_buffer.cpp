#include "tt_metal/impl/buffers/interleaved_dram_buffer.hpp"

#include <algorithm>
#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

InterleavedDramBuffer::InterleavedDramBuffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry)
    : num_bank_units_(num_bank_units),
      num_entries_per_bank_unit_(num_entries_per_bank_unit),
      num_bytes_per_entry_(num_bytes_per_entry),
      Buffer(device, num_bank_units * num_entries_per_bank_unit * num_bytes_per_entry, 0, true) {
    this->dram_bank_to_relative_address_ = this->device_->allocate_interleaved_dram_buffer(num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
    this->address_ = this->dram_bank_to_relative_address_.at(0).second;
}

Buffer *InterleavedDramBuffer::clone() {
    return new InterleavedDramBuffer(this->device_, this->num_bank_units_, this->num_entries_per_bank_unit_, this->num_bytes_per_entry_);
}

int InterleavedDramBuffer::dram_channel() const {
    if (this->dram_bank_to_relative_address_.empty()) {
        TT_THROW("Interleaved DRAM buffer has not been allocated!");
    }
    DramBank dram_bank_holding_first_bank_unit = this->dram_bank_to_relative_address_.at(0).first;
    return  dram_bank_holding_first_bank_unit.channel;
}

tt_xy_pair InterleavedDramBuffer::noc_coordinates() const {
    if (this->dram_bank_to_relative_address_.empty()) {
        TT_THROW("Interleaved DRAM buffer has not been allocated!");
    }
    DramBank dram_bank_holding_first_bank_unit = this->dram_bank_to_relative_address_.at(0).first;
    return llrt::get_core_for_dram_channel(this->device_->cluster(), dram_bank_holding_first_bank_unit.channel, this->device_->pcie_slot());
}

DramBank InterleavedDramBuffer::bank(int bank_index) const {
    if (bank_index >= this->num_banks()) {
        TT_THROW("Bank index " + std::to_string(bank_index) + " exceeds number of banks!");
    }
    return this->dram_bank_to_relative_address_.at(bank_index).first;
}

uint32_t InterleavedDramBuffer::address_of_bank_unit(int bank_index, int bank_unit_index) const {
    if (bank_index >= this->num_banks()) {
        TT_THROW("Bank index " + std::to_string(bank_index) + " exceeds number of banks!");
    }
    auto dram_bank = this->bank(bank_index);
    auto relative_address = this->dram_bank_to_relative_address_.at(bank_index).second;
    int units_read_in_bank = (int)bank_unit_index / this->num_banks();
    auto absolute_address = dram_bank.offset_bytes + relative_address;
    uint32_t offset = (this->bank_unit_size() * units_read_in_bank);
    return absolute_address + offset;
}

void InterleavedDramBuffer::free() {
    for (auto &dram_bank_addr_pair : this->dram_bank_to_relative_address_) {
        auto dram_bank = dram_bank_addr_pair.first;
        auto relative_address = dram_bank_addr_pair.second;
        this->device_->free_dram_buffer(dram_bank.channel, relative_address + dram_bank.offset_bytes);
    }
    this->allocated_on_device_ = false;
}

InterleavedDramBuffer::~InterleavedDramBuffer() {
    if (this->allocated_on_device_) {
        this->free();
    }
}

}  // namespace tt_metal

}  // namespace tt
