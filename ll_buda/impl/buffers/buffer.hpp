#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/tt_xy_pair.h"
#include "common/assert.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "ll_buda/impl/device/device.hpp"

namespace tt {

namespace ll_buda {

class Buffer {
   public:
    Buffer(Device *device, uint32_t size_in_bytes, uint32_t address) : device_(device), size_in_bytes_(size_in_bytes), address_(address) {}

    Device *device() const { return device_; }

    // Returns size of buffer in bytes.
    uint32_t size() const { return size_in_bytes_; }

    uint32_t address() const { return address_; }

    virtual tt_xy_pair noc_coordinates() const = 0;

   protected:
    Device *device_;
    uint32_t size_in_bytes_;    // Size in bytes
    uint32_t address_;          // L1 address
};

class DramBuffer : public Buffer {
   public:
    DramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes, uint32_t address) : dram_channel_(dram_channel), Buffer(device, size_in_bytes, address) {}

    int dram_channel() const { return dram_channel_; }

    tt_xy_pair noc_coordinates() const;

   private:
    int dram_channel_;          // DRAM channel ID
};

class L1Buffer : public Buffer {
   public:
    L1Buffer(Device *device, const tt_xy_pair &logical_core, uint32_t size_in_bytes, uint32_t address) : logical_core_(logical_core), Buffer(device, size_in_bytes, address) {
        TT_ASSERT(address_ >= UNRESERVED_BASE, "First " + std::to_string(UNRESERVED_BASE) + " bytes in L1 are reserved");
    }

    tt_xy_pair logical_core() const { return logical_core_; }

    tt_xy_pair noc_coordinates() const;

   private:
    tt_xy_pair logical_core_;          // Logical core
};

}  // namespace ll_buda

}  // namespace tt
