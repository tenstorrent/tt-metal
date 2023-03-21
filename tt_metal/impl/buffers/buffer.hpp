#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/tt_xy_pair.h"
#include "common/assert.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

class Buffer {
   public:
    Buffer(Device *device, uint32_t size_in_bytes, uint32_t address, bool allocated_on_device)
        : device_(device), size_in_bytes_(size_in_bytes), address_(address), allocated_on_device_(allocated_on_device) {}

    virtual ~Buffer() {}

    virtual Buffer *clone() = 0;

    Device *device() const { return device_; }

    // Returns size of buffer in bytes.
    uint32_t size() const { return size_in_bytes_; }

    uint32_t address() const { return address_; }

    virtual tt_xy_pair noc_coordinates() const = 0;

   protected:
    virtual void free() = 0;
    friend void FreeBuffer(Buffer *buffer);

    Device *device_;
    uint32_t size_in_bytes_;    // Size in bytes
    uint32_t address_;          // L1 address
    bool allocated_on_device_;  // Indicates if buffer space has been explicitly allocated
};

class DramBuffer : public Buffer {
   public:
    DramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes);

    DramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes, uint32_t address);

    ~DramBuffer() {}

    Buffer *clone();

    int dram_channel() const { return dram_channel_; }

    tt_xy_pair noc_coordinates() const;

   private:
    void free();
    friend void FreeBuffer(Buffer *buffer);

    friend class InterleavedDramBuffer;

    int dram_channel_;          // DRAM channel ID
};

class L1Buffer : public Buffer {
   public:
    L1Buffer(Device *device, const tt_xy_pair &logical_core, uint32_t size_in_bytes, uint32_t address) : logical_core_(logical_core), Buffer(device, size_in_bytes, address, false) {
        TT_ASSERT(address_ >= UNRESERVED_BASE, "First " + std::to_string(UNRESERVED_BASE) + " bytes in L1 are reserved");
    }

    ~L1Buffer() {}

    Buffer *clone();

    tt_xy_pair logical_core() const { return logical_core_; }

    tt_xy_pair noc_coordinates() const;

   private:
    void free() { TT_ASSERT(false && "Freeing L1 is unimplemented"); }
    friend void FreeBuffer(Buffer *buffer);

    tt_xy_pair logical_core_;          // Logical core
};

}  // namespace tt_metal

}  // namespace tt
