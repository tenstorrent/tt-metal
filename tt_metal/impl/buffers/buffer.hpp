#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "common/assert.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY
};

class Buffer {
   public:
    Buffer() : device_(nullptr) {}

    Buffer(Device *device, u64 size, u64 address, u64 page_size, const BufferType buffer_type);

    Buffer(Device *device, u64 size, u64 page_size, const BufferType buffer_type);

    Buffer(const Buffer &other);
    Buffer& operator=(const Buffer &other);

    Buffer(Buffer &&other);
    Buffer& operator=(Buffer &&other);

    ~Buffer();

    Device *device() const { return device_; }

    u32 size() const { return static_cast<u32>(size_); }

    // Returns address of buffer in the first bank
    u32 address() const { return static_cast<u32>(address_); }

    u64 page_size() const { return page_size_; }

    BufferType buffer_type() const { return buffer_type_; }

    u32 dram_channel_from_bank_id(u32 bank_id) const;

    CoreCoord logical_core_from_bank_id(u32 bank_id) const;

    CoreCoord noc_coordinates(u32 bank_id) const;

    // returns NoC coordinates of first bank buffer is in
    CoreCoord noc_coordinates() const;

    u64 page_address(u32 bank_id, u32 page_index) const;

   private:
    void allocate();

    void deallocate();
    friend void DeallocateBuffer(Buffer &buffer);

    Device *device_;
    u64 size_;                 // Size in bytes
    u64 address_;              // Address of buffer
    u64 page_size_;            // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type_;
};

}  // namespace tt_metal

}  // namespace tt
