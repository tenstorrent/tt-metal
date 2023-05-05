#pragma once

#include <map>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

class InterleavedL1Buffer : public Buffer {
   public:
    InterleavedL1Buffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry);

    ~InterleavedL1Buffer();

    Buffer *clone();

    int bank_unit_size() const { return this->num_entries_per_bank_unit_ * this->num_bytes_per_entry_; }

    int num_entries_per_bank_unit() const { return this->num_entries_per_bank_unit_; }

    // logical core coordinates of the first bank holding the data
    tt_xy_pair logical_core() const;

    // NoC coordinates of first bank holding the data
    tt_xy_pair noc_coordinates() const;

    int num_banks() const { return this->l1_bank_to_relative_address_.size(); }

    L1Bank bank(int bank_index) const;

    uint32_t address_of_bank_unit(int bank_index, int bank_unit_index) const;

   private:
    void free();
    friend void DeallocateBuffer(Buffer *buffer);

    int num_bank_units_;
    int num_entries_per_bank_unit_;
    int num_bytes_per_entry_;
    std::vector<L1BankAddrPair> l1_bank_to_relative_address_;
};

}  // namespace tt_metal

}  // namespace tt
