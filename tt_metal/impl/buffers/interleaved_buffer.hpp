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

    // Size in bytes of buffer in given DRAM channel
    uint32_t buffer_size(int dram_channel) const;

    // NoC coordinates of DRAM channel 0
    tt_xy_pair noc_coordinates() const;

    // NoC coordinates of given DRAM channel
    tt_xy_pair noc_coordinates(int dram_channel) const;

    std::vector<tt_xy_pair> interleaved_noc_coordinates() const;

   private:
    InterleavedDramBuffer(
        Device *device,
        int num_bank_units,
        int num_entries_per_bank_unit,
        int num_bytes_per_entry,
        const std::map<int, DramBuffer *> &bank_to_buffer
    );

    void free();
    friend void DeallocateBuffer(Buffer *buffer);

    int num_bank_units_;
    int num_entries_per_bank_unit_;
    int num_bytes_per_entry_;
    std::map<int, DramBuffer *> bank_to_buffer_;
};

}  // namespace tt_metal

}  // namespace tt
