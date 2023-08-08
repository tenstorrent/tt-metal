#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "tt_metal/impl/device/device.hpp"

namespace tt::tt_metal {

class LocalBuffer {
   public:
    LocalBuffer() : core_range_set_({}), size_(0), address_(0) {}

    LocalBuffer(const CoreRangeSet &core_range_set, u64 size_in_bytes, u64 address);

    LocalBuffer(const LocalBuffer &other) = default;
    LocalBuffer& operator=(const LocalBuffer &other) = default;

    LocalBuffer(LocalBuffer &&other) = default;
    LocalBuffer& operator=(LocalBuffer &&other) = default;

    const CoreRangeSet &core_range_set() const { return core_range_set_; }

    // Returns size in bytes
    u32 size() const { return size_; }

    u32 address() const { return address_; }

   private:
    CoreRangeSet core_range_set_;
    u64 size_;
    u64 address_;
};

}  // namespace tt::tt_metal
