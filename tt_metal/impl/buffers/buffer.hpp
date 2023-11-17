// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "hostdevcommon/common_values.hpp"

namespace tt {

namespace tt_metal {

class Device;

enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY
};

class Buffer {
   public:
    Buffer() : device_(nullptr) {}

    Buffer(Device *device, uint64_t size, uint64_t page_size, const BufferType buffer_type);

    Buffer(const Buffer &other);
    Buffer& operator=(const Buffer &other);

    Buffer(Buffer &&other);
    Buffer& operator=(Buffer &&other);

    ~Buffer();

    Device *device() const { return device_; }

    uint32_t size() const { return static_cast<uint32_t>(size_); }

    // Returns address of buffer in the first bank
    uint32_t address() const { return static_cast<uint32_t>(address_); }

    uint32_t page_size() const { return page_size_; }

    uint32_t num_pages() const { return this->size() / this->page_size(); }

    BufferType buffer_type() const { return buffer_type_; }

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    CoreCoord noc_coordinates(uint32_t bank_id) const;

    // returns NoC coordinates of first bank buffer is in
    CoreCoord noc_coordinates() const;

    uint64_t page_address(uint32_t bank_id, uint32_t page_index) const;

   private:
    void allocate();

    void deallocate();
    friend void DeallocateBuffer(Buffer &buffer);

    Device *device_;
    uint64_t size_;                 // Size in bytes
    uint64_t address_;              // Address of buffer
    uint64_t page_size_;            // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type_;
};

}  // namespace tt_metal

}  // namespace tt
