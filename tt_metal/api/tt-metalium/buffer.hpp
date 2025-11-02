// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json_fwd.hpp>
#include <tt_stl/concepts.hpp>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/xy_pair.hpp>

namespace tt {
namespace stl {
namespace json {
template <typename T>
struct from_json_t;
}  // namespace json
}  // namespace stl
}  // namespace tt

namespace tt::tt_metal {

class Allocator;
class IDevice;

struct ShardSpec {
    /* The individual cores the shard grid is mapped to */
    CoreRangeSet grid;

    /* Canonical tensor shape where the depth dimensions ([:-2] are folded along y) */
    std::array<uint32_t, 2> shape;

    /* The sequence order of the grid cores that the shards are layed out onto. */
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;

    ShardSpec(
        const CoreRangeSet& core_sets_,
        const std::array<uint32_t, 2>& shard_shape_,
        const ShardOrientation& shard_orientation_ = ShardOrientation::ROW_MAJOR) :
        grid(core_sets_), shape(shard_shape_), orientation(shard_orientation_) {}

    uint32_t num_cores() const { return this->grid.num_cores(); }
    uint32_t numel() const { return this->shape[0] * this->shape[1]; }

    bool operator==(const ShardSpec& other) const;
    bool operator!=(const ShardSpec& other) const;

    static constexpr auto attribute_names = std::forward_as_tuple("grid", "shape", "orientation");
    constexpr auto attribute_values() const {
        return std::forward_as_tuple(this->grid, this->shape, this->orientation);
    }
};

std::ostream& operator<<(std::ostream& os, const ShardSpec& spec);

struct ShardSpecBuffer {
    ShardSpec tensor_shard_spec;
    std::array<uint32_t, 2> page_shape{};
    std::array<uint32_t, 2> tensor2d_shape_in_pages{};
    ShardSpecBuffer(
        const CoreRangeSet& core_sets_,
        const std::array<uint32_t, 2>& shard_shape_,
        const ShardOrientation& shard_orientation_,
        const std::array<uint32_t, 2>& page_shape,
        const std::array<uint32_t, 2>& tensor2d_shape_in_pages) :
        tensor_shard_spec(core_sets_, shard_shape_, shard_orientation_),
        page_shape(page_shape),
        tensor2d_shape_in_pages(tensor2d_shape_in_pages) {}
    ShardSpecBuffer(
        const ShardSpec& shard_spec,
        const std::array<uint32_t, 2>& page_shape,
        const std::array<uint32_t, 2>& tensor2d_shape_in_pages) :
        tensor_shard_spec(shard_spec), page_shape(page_shape), tensor2d_shape_in_pages(tensor2d_shape_in_pages) {}
    CoreRangeSet grid() const { return tensor_shard_spec.grid; }
    std::array<uint32_t, 2> shape() const { return tensor_shard_spec.shape; }
    ShardOrientation orientation() const { return tensor_shard_spec.orientation; }
    void set_shard_spec(const ShardSpec& shard_spec) { tensor_shard_spec = shard_spec; };

    /* Shape in pages of the full shard */
    std::array<uint32_t, 2> shape_in_pages() const;
    DeviceAddr num_pages() const;
};

struct InterleavedBufferConfig {
    IDevice* device;
    DeviceAddr size;       // Size in bytes
    DeviceAddr page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type;
};

// copied from above instead of using inheritance such that we can use
// designator constructor
struct ShardedBufferConfig {
    IDevice* device{};
    DeviceAddr size{};       // Size in bytes
    DeviceAddr page_size{};  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type = BufferType::L1;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardSpecBuffer shard_parameters;
};

class BufferShardingArgs {
public:
    BufferShardingArgs() = default;
    BufferShardingArgs(std::nullopt_t) {}

    BufferShardingArgs(BufferDistributionSpec buffer_distribution_spec) :
        buffer_distribution_spec_(std::move(buffer_distribution_spec)),
        buffer_layout_(TensorMemoryLayout::BLOCK_SHARDED) {}
    BufferShardingArgs(std::optional<BufferDistributionSpec> buffer_distribution_spec) :
        buffer_distribution_spec_(std::move(buffer_distribution_spec)),
        buffer_layout_(
            buffer_distribution_spec_.has_value() ? TensorMemoryLayout::BLOCK_SHARDED
                                                  : TensorMemoryLayout::INTERLEAVED) {}

    BufferShardingArgs(ShardSpecBuffer shard_spec, TensorMemoryLayout buffer_layout) :
        shard_spec_(std::move(shard_spec)), buffer_layout_(buffer_layout) {}
    BufferShardingArgs(std::optional<ShardSpecBuffer> shard_spec, TensorMemoryLayout buffer_layout) :
        shard_spec_(std::move(shard_spec)), buffer_layout_(buffer_layout) {}

    BufferShardingArgs(
        std::optional<BufferDistributionSpec> buffer_distribution_spec,
        std::optional<ShardSpecBuffer> shard_spec,
        TensorMemoryLayout buffer_layout) :
        buffer_distribution_spec_(std::move(buffer_distribution_spec)),
        shard_spec_(std::move(shard_spec)),
        buffer_layout_(buffer_layout) {}

    const std::optional<BufferDistributionSpec>& buffer_distribution_spec() const { return buffer_distribution_spec_; }

    const std::optional<ShardSpecBuffer>& shard_spec() const { return shard_spec_; }

    TensorMemoryLayout buffer_layout() const { return buffer_layout_; }

private:
    std::optional<BufferDistributionSpec> buffer_distribution_spec_;
    std::optional<ShardSpecBuffer> shard_spec_;
    TensorMemoryLayout buffer_layout_ = TensorMemoryLayout::INTERLEAVED;
};

bool is_sharded(const TensorMemoryLayout& layout);

struct BufferRegion {
    DeviceAddr offset = 0;
    DeviceAddr size = 0;

    BufferRegion() = delete;
    BufferRegion(DeviceAddr offset, DeviceAddr size) : offset(offset), size(size) {}
};

class BufferImpl;
class Buffer final : public std::enable_shared_from_this<Buffer> {
public:
    static std::shared_ptr<Buffer> create(
        IDevice* device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        const BufferShardingArgs& sharding_args = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);
    static std::shared_ptr<Buffer> create(
        IDevice* device,
        DeviceAddr address,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        const BufferShardingArgs& sharding_args = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);

    Buffer(const Buffer& other) = delete;
    Buffer& operator=(const Buffer& other) = delete;
    Buffer(Buffer&& other) = delete;
    Buffer& operator=(Buffer&& other) = delete;
    ~Buffer();

    IDevice* device() const;
    // Single usage by reports.cpp
    Allocator* allocator() const;
    DeviceAddr size() const;

    // Returns address of buffer in the first bank
    uint32_t address() const;

    DeviceAddr page_size() const;
    // Single Usage from view op
    void set_page_size(DeviceAddr page_size);

    uint32_t num_pages() const;

    BufferType buffer_type() const;
    CoreType core_type() const;

    bool is_l1() const;
    bool is_dram() const;

    TensorMemoryLayout buffer_layout() const;

    // Single Usage from reports.cpp
    DeviceAddr page_address(DeviceAddr bank_id, DeviceAddr page_index) const;

    uint32_t alignment() const;
    DeviceAddr aligned_page_size() const;
    DeviceAddr aligned_size_per_bank() const;

    // SHARDED API STARTS HERE
    const std::optional<BufferDistributionSpec>& buffer_distribution_spec() const;
    ShardSpecBuffer shard_spec() const;
    // Single Usage from view op
    void set_shard_spec(const ShardSpecBuffer& shard_spec);
    std::optional<uint32_t> num_cores() const;
    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping();

    // Single usage from graph_processor
    size_t unique_id() const;

    BufferImpl* impl();
    const BufferImpl* impl() const;

    Buffer(
        IDevice* device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        const BufferShardingArgs& sharding_args,
        std::optional<bool> bottom_up,
        std::optional<SubDeviceId> sub_device_id,
        bool owns_data);

private:
    std::unique_ptr<BufferImpl> pimpl_;
};

}  // namespace tt::tt_metal

namespace ttsl::json {
template <>
struct from_json_t<tt::tt_metal::ShardSpec> {
    tt::tt_metal::ShardSpec operator()(const nlohmann::json& json_object) const;
};
}  // namespace ttsl::json
