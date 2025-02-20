// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <nlohmann/json.hpp>

#include "bfloat16.hpp"
#include "core_coord.hpp"
#include "buffer_constants.hpp"
#include "hal.hpp"

namespace tt::tt_metal {
inline namespace v0 {

class IDevice;

}  // namespace v0

class Allocator;

bool is_sharded(const TensorMemoryLayout& layout);

struct BufferPageMapping {
    std::vector<CoreCoord> all_cores_;
    std::vector<uint32_t> core_bank_indices_;
    std::vector<std::vector<uint32_t>> core_host_page_indices_;
    std::vector<uint32_t> dev_page_to_core_mapping_;

    // some dev pages don't have mapping to host (in case of padding)
    std::vector<std::optional<uint32_t>> dev_page_to_host_page_mapping_;
    std::vector<uint32_t> host_page_to_dev_page_mapping_;
    std::unordered_map<CoreCoord, uint32_t> core_to_core_id_;
    std::vector<uint32_t> host_page_to_local_shard_page_mapping_;
    std::vector<std::array<uint32_t, 2>> core_shard_shape_;
};

struct ShardSpec {
    /* The individual cores the shard grid is mapped to */
    CoreRangeSet grid;

    /* Canonical tensor shape where the depth dimensions ([:-2] are folded along y) */
    std::array<uint32_t, 2> shape;

    /* The sequence order of the grid cores that the shards are layed out onto. */
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;

    // In ShardMode::PHYSICAL, physical_shard_shape will always be std::nullopt
    ShardMode mode = ShardMode::PHYSICAL;
    std::optional<std::array<uint32_t, 2>> physical_shard_shape = std::nullopt;

    ShardSpec(
        const CoreRangeSet& core_sets_,
        const std::array<uint32_t, 2>& shard_shape_,
        const ShardOrientation& shard_orientation_ = ShardOrientation::ROW_MAJOR,
        const ShardMode& shard_mode_ = ShardMode::PHYSICAL) :
        grid(core_sets_),
        shape(shard_shape_),
        orientation(shard_orientation_),
        mode(shard_mode_),
        physical_shard_shape(std::nullopt) {}

    ShardSpec(
        const CoreRangeSet& core_sets_,
        const std::array<uint32_t, 2>& shard_shape_,
        const std::array<uint32_t, 2>& physical_shard_shape_,
        const ShardOrientation& shard_orientation_ = ShardOrientation::ROW_MAJOR) :
        grid(core_sets_),
        shape(shard_shape_),
        orientation(shard_orientation_),
        mode(ShardMode::LOGICAL),
        physical_shard_shape(physical_shard_shape_) {
        TT_FATAL(
            physical_shard_shape_[0] >= shard_shape_[0] and physical_shard_shape_[1] >= shard_shape_[1],
            "Physical shard shape ({}, {}) must be greater or equal to logical shard shape ({}, {})!",
            physical_shard_shape_[0],
            physical_shard_shape_[1],
            shard_shape_[0],
            shard_shape_[1]);
    }

    const uint32_t num_cores() const { return this->grid.num_cores(); }
    const uint32_t numel() const { return this->shape[0] * this->shape[1]; }

    bool operator==(const ShardSpec& other) const;
    bool operator!=(const ShardSpec& other) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("grid", "shape", "orientation", "mode", "physical_shard_shape");
    constexpr auto attribute_values() const {
        return std::forward_as_tuple(
            this->grid, this->shape, this->orientation, this->mode, this->physical_shard_shape);
    }
};

std::ostream& operator<<(std::ostream& os, const ShardSpec& spec);

struct ShardSpecBuffer {
    ShardSpec tensor_shard_spec;
    std::array<uint32_t, 2> page_shape;
    std::array<uint32_t, 2> tensor2d_shape_in_pages;
    ShardSpecBuffer(
        const CoreRangeSet& core_sets_,
        const std::array<uint32_t, 2>& shard_shape_,
        const ShardOrientation& shard_orientation_,
        const std::array<uint32_t, 2>& page_shape,
        const std::array<uint32_t, 2>& tensor2d_shape_in_pages) :
        tensor_shard_spec(core_sets_, shard_shape_, shard_orientation_) {
        this->page_shape = page_shape;
        this->tensor2d_shape_in_pages = tensor2d_shape_in_pages;
    }
    ShardSpecBuffer(
        const ShardSpec& shard_spec,
        const std::array<uint32_t, 2>& page_shape,
        const std::array<uint32_t, 2>& tensor2d_shape_in_pages) :
        tensor_shard_spec(shard_spec) {
        this->page_shape = page_shape;
        this->tensor2d_shape_in_pages = tensor2d_shape_in_pages;
    }
    CoreRangeSet grid() const { return tensor_shard_spec.grid; }
    std::array<uint32_t, 2> shape() const { return tensor_shard_spec.shape; }
    ShardOrientation orientation() const { return tensor_shard_spec.orientation; }
    void set_shard_spec(const ShardSpec& shard_spec) { tensor_shard_spec = shard_spec; };

    /* Shape in pages of the full shard */
    std::array<uint32_t, 2> shape_in_pages() const;
    DeviceAddr num_pages() const;
};

inline namespace v0 {

struct BufferConfig {
    IDevice* device;
    DeviceAddr size;       // Size in bytes
    DeviceAddr page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED;
};

typedef BufferConfig InterleavedBufferConfig;

// copied from above instead of using inheritance such that we can use
// designator constructor
struct ShardedBufferConfig {
    IDevice* device;
    DeviceAddr size;       // Size in bytes
    DeviceAddr page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type = BufferType::L1;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardSpecBuffer shard_parameters;
};

struct BufferRegion {
    DeviceAddr offset = 0;
    DeviceAddr size = 0;

    BufferRegion() = delete;
    BufferRegion(DeviceAddr offset, DeviceAddr size) : offset(offset), size(size) {}
};

using HostDataType = std::variant<
    const std::shared_ptr<std::vector<uint8_t>>,
    const std::shared_ptr<std::vector<uint16_t>>,
    const std::shared_ptr<std::vector<int32_t>>,
    const std::shared_ptr<std::vector<uint32_t>>,
    const std::shared_ptr<std::vector<float>>,
    const std::shared_ptr<std::vector<bfloat16>>,
    const void*>;

}  // namespace v0

}  // namespace tt::tt_metal

namespace tt::stl::json {
template <>
struct from_json_t<tt_metal::ShardSpec> {
    tt_metal::ShardSpec operator()(const nlohmann::json& json_object) const;
};
}  // namespace tt::stl::json
