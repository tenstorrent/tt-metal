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

    Buffer(Device *device, uint32_t size, uint32_t address, uint32_t starting_bank_id, uint32_t page_size, const BufferType buffer_type);

    Buffer(Device *device, uint32_t size, uint32_t starting_bank_id, uint32_t page_size, const BufferType buffer_type);

    Buffer(const Buffer &other);
    Buffer& operator=(const Buffer &other);

    Buffer(Buffer &&other);
    Buffer& operator=(Buffer &&other);

    ~Buffer();

    Device *device() const { return device_; }

    uint32_t size() const { return size_; }

    // Returns address of buffer in the first bank
    uint32_t address() const { return address_; }

    uint32_t starting_bank_id() const { return starting_bank_id_; }

    uint32_t page_size() const { return page_size_; }

    BufferType buffer_type() const { return buffer_type_; }

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    CoreCoord noc_coordinates(uint32_t bank_id) const;

    // returns NoC coordinates of first bank buffer is in
    CoreCoord noc_coordinates() const;

    uint32_t page_address(uint32_t bank_id, uint32_t page_index) const;

   private:
    void allocate();

    void deallocate();
    friend void DeallocateBuffer(Buffer &buffer);

    Device *device_;
    uint32_t size_;                 // Size in bytes
    uint32_t address_;              // Address of buffer in starting_bank
    uint32_t starting_bank_id_;
    uint32_t page_size_;            // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type_;
    BankIdToRelativeAddress bank_id_to_relative_address_;
};

}  // namespace tt_metal

}  // namespace tt
