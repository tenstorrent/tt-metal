#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/tt_xy_pair.h"

namespace tt {

namespace ll_buda {

class CircularBuffer {
   public:
    CircularBuffer(
        const tt_xy_pair &core,
        uint32_t buffer_index,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t address,
        DataFormat data_format) :
        core_(core), buffer_index_(buffer_index), num_tiles_(num_tiles), size_in_bytes_(size_in_bytes), address_(address), data_format_(data_format) {}

    tt_xy_pair core() const { return core_; }

    uint32_t buffer_index() const { return buffer_index_; }

    uint32_t num_tiles() const { return num_tiles_; }

    // Returns size of buffer in bytes.
    uint32_t size() const { return size_in_bytes_; }

    uint32_t address() const { return address_; }

    DataFormat data_format() const { return data_format_; }

   private:
    tt_xy_pair core_;                     // Logical Tensix core
    uint32_t buffer_index_;               // A buffer ID unique within a Tensix core (0 to 32)
    uint32_t num_tiles_;                  // Size in tiles
    uint32_t size_in_bytes_;              // Size in bytes
    uint32_t address_;                    // L1 address
    DataFormat data_format_;              // e.g. fp16, bfp8
};

}  // namespace ll_buda

}  // namespace tt
