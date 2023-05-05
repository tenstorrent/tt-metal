#pragma once

#include <map>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

class InterleavedDramBuffer : public Buffer {
   public:
    InterleavedDramBuffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry);

    ~InterleavedDramBuffer();

    Buffer *clone();

    int bank_unit_size() const { return this->num_entries_per_bank_unit_ * this->num_bytes_per_entry_; }

    int num_entries_per_bank_unit() const { return this->num_entries_per_bank_unit_; }

    // DRAM channel of first bank holding the data
    int dram_channel() const;

    // NoC coordinates of first bank holding the data
    tt_xy_pair noc_coordinates() const;

    int num_banks() const { return this->dram_bank_to_relative_address_.size(); }

    DramBank bank(int bank_index) const;

    uint32_t address_of_bank_unit(int bank_index, int bank_unit_index) const;

   private:
    void free();
    friend void DeallocateBuffer(Buffer *buffer);

    int num_bank_units_;
    int num_entries_per_bank_unit_;
    int num_bytes_per_entry_;
    std::vector<DramBankAddrPair> dram_bank_to_relative_address_;
};

}  // namespace tt_metal

}  // namespace tt
