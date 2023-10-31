/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "hostdevcommon/common_values.hpp"
#include <variant>
#include <optional>
#include "tt_metal/common/constants.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt {

namespace tt_metal {

class Device;

enum class BufferStorage {
    DRAM,
    L1,
    SYSTEM_MEMORY
};

enum class TensorMemoryLayout {
    INTERLEAVED,
    SINGLE_BANK,
    HEIGHT_SHARDED,
    WIDTH_SHARDED,
    BLOCK_SHARDED,
};

enum class ShardOrientation {
    ROW_MAJOR,
    COL_MAJOR,
};


struct ShardSpec {
    CoreRangeSet shard_grid;
    std::array<uint32_t, 2> shard_shape;
    ShardOrientation shard_orientation;
    bool halo = false;
    std::optional<uint32_t> element_size;
    std::optional< std::array<u32, 2 > > page_strides;
    const uint32_t num_cores() const { return this->shard_grid.num_cores(); }
    const uint32_t numel() const { return this->shard_shape[0] * this->shard_shape[1]; }
    tt::stl::reflection::Attributes attributes() const;
};

bool is_sharded(const TensorMemoryLayout & layout);
class Buffer {
   public:
    Buffer() : device_(nullptr) {}

    Buffer(Device *device, u64 size, u64 page_size, const BufferStorage buffer_storage,
        const TensorMemoryLayout buffer_layout=TensorMemoryLayout::INTERLEAVED,
        std::optional<ShardSpec> shard_parameter = std::nullopt
        );

    Buffer(const Buffer &other);
    Buffer& operator=(const Buffer &other);

    Buffer(Buffer &&other);
    Buffer& operator=(Buffer &&other);

    ~Buffer();

    Device *device() const { return device_; }

    u32 size() const { return static_cast<u32>(size_); }

    // Returns address of buffer in the first bank
    u32 address() const {
        TT_ASSERT(!is_sharded(buffer_layout_) , "Sharded buffer requires corecoord");
        return static_cast<u32>(address_);
    }

    u32 address(CoreCoord core) const {
        TT_ASSERT(is_sharded(buffer_layout_) , "Buffer is not sharded and does not require corecoord, use buffer.address() instead");
        return static_cast<u32>(address_);
    }

    u64 page_size() const { return page_size_; }

    BufferStorage buffer_storage() const { return buffer_storage_; }

    TensorMemoryLayout buffer_layout() const { return buffer_layout_; }

    u32 dram_channel_from_bank_id(u32 bank_id) const;

    CoreCoord logical_core_from_bank_id(u32 bank_id) const;

    CoreCoord noc_coordinates(u32 bank_id) const;

    // returns NoC coordinates of first bank buffer is in
    CoreCoord noc_coordinates() const;

    u64 page_address(u32 bank_id, u32 page_index) const;

    CoreRangeSet shard_grid() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return shard_parameters_.value().shard_grid;
    }

    ShardOrientation shard_orientation() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return shard_parameters_.value().shard_orientation;
    }

    u32 element_size_multiplier() const{
        return sizeof(uint32_t) / shard_parameters_.value().element_size.value();
    }

    u32 shard_size() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        auto num_elements_total = shard_parameters_.value().shard_shape[0] * shard_parameters_.value().shard_shape[1];
        auto size_in_pages = num_elements_total/constants::TILE_HW * element_size_multiplier();
        return size_in_pages;
    }

    std::array<u32,2> shard_shape() const {
        TT_ASSERT(is_sharded(this->buffer_layout_) , "Buffer not sharded");
        return {shard_parameters_.value().shard_shape[0]/constants::TILE_HEIGHT * element_size_multiplier() , shard_parameters_.value().shard_shape[1]/constants::TILE_WIDTH * element_size_multiplier()};
    }



   private:
    void allocate();

    void deallocate();
    friend void DeallocateBuffer(Buffer &buffer);

    Device *device_;
    u64 size_;                 // Size in bytes
    u64 address_;              // Address of buffer
    u64 page_size_;            // Size of unit being interleaved.
                               // For non-interleaved buffers: size == page_size
                               // For sharded, refers to sharded page size
                               // shard size refers
    BufferStorage buffer_storage_;
    TensorMemoryLayout buffer_layout_;
    std::optional <ShardSpec> shard_parameters_;
};

}  // namespace tt_metal

}  // namespace tt
