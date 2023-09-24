/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"

namespace tt {

namespace tt_metal {

class CircularBuffer {
   public:
    CircularBuffer() : id_(reinterpret_cast<uintptr_t>(this)), core_range_set_({}), num_tiles_(0), size_(0), address_(0), data_format_(DataFormat::Invalid) {}

    CircularBuffer(
        const CoreRangeSet &core_range_set,
        const std::set<u32> &buffer_indices,
        u32 num_tiles,
        u32 size_in_bytes,
        u32 address,
        DataFormat data_format);

    const CircularBufferID id() const { return id_; }

    CoreRangeSet core_range_set() const { return core_range_set_; }

    const std::set<u32> &buffer_indices() const { return buffer_indices_; }

    u32 num_tiles() const { return num_tiles_; }

    u32 size() const { return size_; }

    u32 address() const { return address_; }

    DataFormat data_format() const { return data_format_; }

    bool is_on_logical_corerange( const CoreRange & logical_cr ) const;

    bool is_on_logical_core(const CoreCoord &logical_core) const;

   private:
    const uintptr_t id_;
    CoreRangeSet core_range_set_;
    std::set<u32> buffer_indices_;        // Buffer IDs unique within a Tensix core (0 to 32)
    u32 num_tiles_;                             // Size in tiles
    u32 size_;
    u32 address_;
    DataFormat data_format_;                    // e.g. fp16, bfp8
};

}  // namespace tt_metal

}  // namespace tt
