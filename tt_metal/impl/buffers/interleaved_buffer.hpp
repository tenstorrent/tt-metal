#pragma once

#include <unordered_map>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

class InterleavedDramBuffer : public Buffer {
   public:
    InterleavedDramBuffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry);

    // Size in bytes of buffer in given DRAM channel
    uint32_t buffer_size(int dram_channel) const;

    // NoC coordinates of DRAM channel 0
    tt_xy_pair noc_coordinates() const;

    // NoC coordinates of given DRAM channel
    tt_xy_pair noc_coordinates(int dram_channel) const;

    std::vector<tt_xy_pair> interleaved_noc_coordinates() const;

   private:
    std::unordered_map<int, DramBuffer *> bank_to_buffer_;
};

}  // namespace tt_metal

}  // namespace tt
