#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/tt_xy_pair.h"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

class CircularBuffer : public Buffer {
   public:
    CircularBuffer(
        Device *device,
        const tt_xy_pair &logical_core,
        uint32_t buffer_index,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format);

    CircularBuffer(
        Device *device,
        const tt_xy_pair &logical_core,
        uint32_t buffer_index,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t address,
        DataFormat data_format);

    ~CircularBuffer();

    Buffer *clone();

    tt_xy_pair logical_core() const { return logical_core_; }

    uint32_t buffer_index() const { return buffer_index_; }

    uint32_t num_tiles() const { return num_tiles_; }

    DataFormat data_format() const { return data_format_; }

    tt_xy_pair noc_coordinates() const;

   private:
    void reserve();
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

    tt_xy_pair logical_core_;             // Logical core
    uint32_t buffer_index_;               // A buffer ID unique within a Tensix core (0 to 32)
    uint32_t num_tiles_;                  // Size in tiles
    DataFormat data_format_;              // e.g. fp16, bfp8
};




}  // namespace tt_metal

}  // namespace tt
