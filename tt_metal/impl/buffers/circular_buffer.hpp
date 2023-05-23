#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

class CircularBuffer {
   public:
    CircularBuffer(
        Device *device,
        const CoreCoord &logical_core,
        uint32_t buffer_index,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format);

    CircularBuffer(
        Device *device,
        const CoreCoord &logical_core,
        uint32_t buffer_index,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t address,
        DataFormat data_format);

    // TODO (abhullar): Copy ctor semantics for CB should allocate same size in new space. Uplift when redesigning CB and CB config
    CircularBuffer(const CircularBuffer &other) = delete;
    CircularBuffer& operator=(const CircularBuffer &other) = delete;

    CircularBuffer(CircularBuffer &&other);
    CircularBuffer& operator=(CircularBuffer &&other);

    ~CircularBuffer();

    CoreCoord logical_core() const { return logical_core_; }

    uint32_t buffer_index() const { return buffer_index_; }

    uint32_t num_tiles() const { return num_tiles_; }

    uint32_t size() const { return size_; }

    uint32_t address() const { return address_; }

    DataFormat data_format() const { return data_format_; }

    CoreCoord noc_coordinates() const;

   private:
    void reserve();
    friend std::vector<CircularBuffer *> CreateCircularBuffers(
        Program &program,
        Device *device,
        uint32_t buffer_index,
        const CoreRange &core_range,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format
    );

    void deallocate();
    friend void DeallocateBuffer(Buffer &buffer);

    Device *device_;
    CoreCoord logical_core_;             // Logical core
    uint32_t buffer_index_;               // A buffer ID unique within a Tensix core (0 to 32)
    uint32_t num_tiles_;                  // Size in tiles
    uint32_t size_;
    uint32_t address_;
    DataFormat data_format_;              // e.g. fp16, bfp8
    // TODO: Remove this when CBs can have multiple buffer indices
    bool allocated_on_device_;
};

}  // namespace tt_metal

}  // namespace tt
