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

    // TODO: Buffer clone should do memcopy
    virtual Buffer *clone() = 0;

    Device *device() const { return device_; }

    // Returns size of buffer in bytes.
    uint32_t size() const { return size_in_bytes_; }

    uint32_t address() const { return address_; }

    virtual tt_xy_pair noc_coordinates() const = 0;

   protected:
    virtual void free() = 0;
    friend void DeallocateBuffer(Buffer *buffer);

    Device *device_;
    uint32_t size_in_bytes_;    // Size in bytes
    uint32_t address_;          // L1 address
    bool allocated_on_device_;  // Indicates if buffer space has been explicitly allocated
};

class DramBuffer : public Buffer {
   public:
    DramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes);

    DramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes, uint32_t address);

    ~DramBuffer();

    Buffer *clone();

    int dram_channel() const { return dram_channel_; }

    tt_xy_pair noc_coordinates() const;

   private:
    void free();
    friend void DeallocateBuffer(Buffer *buffer);

    friend class InterleavedDramBuffer;

    int dram_channel_;          // DRAM channel ID
};

// Fwd declares
class Program;
class CircularBuffer;

class L1Buffer : public Buffer {
   public:
    L1Buffer(Device *device, const tt_xy_pair &logical_core, uint32_t size_in_bytes);

    L1Buffer(Device *device, const tt_xy_pair &logical_core, uint32_t size_in_bytes, uint32_t address);

    ~L1Buffer();

    Buffer *clone();

    tt_xy_pair logical_core() const { return logical_core_; }

    tt_xy_pair noc_coordinates() const;

   protected:
    void reserve();
    friend std::vector<L1Buffer *> CreateL1Buffers(Program *program, Device *device, const CoreRange &core_range, uint32_t size_in_bytes);
    friend std::vector<CircularBuffer *> CreateCircularBuffers(
        Program *program,
        Device *device,
        uint32_t buffer_index,
        const CoreRange &core_range,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format
    );

    void free();
    friend void DeallocateBuffer(Buffer *buffer);

    tt_xy_pair logical_core_;          // Logical core
};

}  // namespace tt_metal

}  // namespace tt
