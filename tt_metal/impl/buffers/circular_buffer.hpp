#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/tt_xy_pair.h"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

class CircularBuffer {
   public:
    CircularBuffer(
        Device *device,
        const tt_xy_pair &logical_core,
        uint32_t buffer_index,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t address,
        DataFormat data_format) :
        logical_core_(logical_core), buffer_index_(buffer_index), num_tiles_(num_tiles), size_in_bytes_(size_in_bytes), address_(address), data_format_(data_format) {}

    tt_xy_pair logical_core() const { return logical_core_; }

    uint32_t buffer_index() const { return buffer_index_; }

    uint32_t num_tiles() const { return num_tiles_; }

    // Returns size of buffer in bytes.
    uint32_t size() const { return size_in_bytes_; }

    uint32_t address() const { return address_; }

    DataFormat data_format() const { return data_format_; }

    tt_xy_pair noc_coordinates() const {
        return device_->worker_core_from_logical_core(logical_core_);
    }

   private:
    Device *device_;
    tt_xy_pair logical_core_;             // Logical Tensix core
    uint32_t buffer_index_;               // A buffer ID unique within a Tensix core (0 to 32)
    uint32_t num_tiles_;                  // Size in tiles
    uint32_t size_in_bytes_;              // Size in bytes
    uint32_t address_;                    // L1 address
    DataFormat data_format_;              // e.g. fp16, bfp8
};

}  // namespace tt_metal

}  // namespace tt
