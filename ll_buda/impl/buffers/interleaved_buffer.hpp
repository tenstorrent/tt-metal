#pragma once

#include <unordered_map>

#include "ll_buda/impl/buffers/buffer.hpp"
#include "ll_buda/impl/device/device.hpp"

namespace tt {

namespace ll_buda {

class InterleavedDramBuffer {
   public:
    InterleavedDramBuffer(Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry);

    // Size in bytes of all buffers across all DRAM channels
    uint32_t size() const { return total_size_bytes_; }

    // Size in bytes of buffer in given DRAM channel
    uint32_t buffer_size(int dram_channel) const;

    // Buffers across all DRAM channels share the same address
    uint32_t address() const { return address_; }

    // NoC coordinates of given DRAM channel
    tt_xy_pair noc_coordinates(int dram_channel) const;

    std::vector<tt_xy_pair> noc_coordinates() const;

   private:
    Device *device_;
    uint32_t total_size_bytes_;     // Total size in bytes
    uint32_t address_;              // Address
    std::unordered_map<int, DramBuffer *> bank_to_buffer_;
};

}  // namespace ll_buda

}  // namespace tt
