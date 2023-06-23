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
        const CoreRangeSet &core_range_set,
        const std::set<u32> &buffer_indices,
        u32 num_tiles,
        u32 size_in_bytes,
        DataFormat data_format);

    CircularBuffer(
        Device *device,
        const CoreRangeSet &core_range_set,
        const std::set<u32> &buffer_indices,
        u32 num_tiles,
        u32 size_in_bytes,
        u32 address,
        DataFormat data_format);

    // TODO (abhullar): Copy ctor semantics for CB should allocate same size in new space. Uplift when redesigning CB and CB config
    CircularBuffer(const CircularBuffer &other) = delete;
    CircularBuffer& operator=(const CircularBuffer &other) = delete;

    CircularBuffer(CircularBuffer &&other) = default;
    CircularBuffer& operator=(CircularBuffer &&other) = default;

    ~CircularBuffer();

    CoreRangeSet core_range_set() const { return core_range_set_; }

    std::set<u32> buffer_indices() const { return buffer_indices_; }

    u32 num_tiles() const { return num_tiles_; }

    u32 size() const { return size_; }

    u32 address() const { return address_; }

    DataFormat data_format() const { return data_format_; }

    bool is_on_logical_core(const CoreCoord &logical_core) const;

    void reserve(Device*);
    void deallocate();

    bool is_allocated() {
        return this->allocated_on_device_;
    }

   private:

    friend void DeallocateBuffer(Buffer &buffer);

    Device *device_;
    CoreRangeSet core_range_set_;
    const std::set<u32> buffer_indices_;        // Buffer IDs unique within a Tensix core (0 to 32)
    u32 num_tiles_;                  // Size in tiles
    u32 size_;
    u32 address_;
    DataFormat data_format_;              // e.g. fp16, bfp8
    bool allocated_on_device_;
};

}  // namespace tt_metal

}  // namespace tt
