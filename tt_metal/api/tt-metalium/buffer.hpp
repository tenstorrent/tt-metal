// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json_fwd.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <tuple>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
// UMD: re-exports CoreType (used in Buffer::core_type return type).
#include <umd/device/types/core_coordinates.hpp>

namespace ttsl::json {
template <typename T>
struct from_json_t;
}  // namespace ttsl::json

namespace tt::tt_metal {

class Allocator;
class BufferImpl;
class IDevice;

// Forward declarations for friended free functions in the experimental namespace.
// These are used to access experimental config params, which are not part of the official public API.
class Buffer;
class BufferShardingArgs;
namespace experimental::per_core_allocation {
BufferShardingArgs& set_per_core_allocation(BufferShardingArgs& args, bool enable);
bool is_per_core_allocation(const BufferShardingArgs& args);
}  // namespace experimental::per_core_allocation

struct ShardSpec {
    /* The individual cores the shard grid is mapped to */
    CoreRangeSet grid;

    /* Canonical tensor shape where the depth dimensions ([:-2] are folded along y) */
    std::array<uint32_t, 2> shape;

    /* The sequence order of the grid cores that the shards are laid out onto. */
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

    /* Shape in pages of the full shard */
    std::array<uint32_t, 2> shape_in_pages() const;
    DeviceAddr num_pages() const;
};

struct BufferConfig {
    IDevice* device;
    DeviceAddr size;       // Size in bytes
    DeviceAddr page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type;
};

using InterleavedBufferConfig = BufferConfig;

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

    // per_core_allocation is experimental functionality
    // access is through experimental::per_core_allocation free functions
    friend BufferShardingArgs& experimental::per_core_allocation::set_per_core_allocation(BufferShardingArgs&, bool);
    friend bool experimental::per_core_allocation::is_per_core_allocation(const BufferShardingArgs&);

private:
    std::optional<BufferDistributionSpec> buffer_distribution_spec_;
    std::optional<ShardSpecBuffer> shard_spec_;
    TensorMemoryLayout buffer_layout_ = TensorMemoryLayout::INTERLEAVED;
    // per_core_allocation is experimental functionality
    // access is through experimental::per_core_allocation free functions
    bool per_core_allocation_ = false;
};

bool is_sharded(const TensorMemoryLayout& layout);

struct BufferRegion {
    DeviceAddr offset = 0;
    DeviceAddr size = 0;

    BufferRegion() = delete;
    BufferRegion(DeviceAddr offset, DeviceAddr size) : offset(offset), size(size) {}
};

class Buffer final : public std::enable_shared_from_this<Buffer> {
public:
    explicit Buffer(BufferImpl impl);

    Buffer(const Buffer& other) = delete;
    Buffer& operator=(const Buffer& other) = delete;
    Buffer(Buffer&& other) = delete;
    Buffer& operator=(Buffer&& other) = delete;
    ~Buffer();

    IDevice* device() const;
    Allocator* allocator() const;
    DeviceAddr size() const;

    // Returns address of buffer in the first bank
    uint32_t address() const;

    DeviceAddr page_size() const;

    uint32_t num_pages() const;
    uint32_t num_dev_pages() const;

    BufferType buffer_type() const;
    CoreType core_type() const;

    bool is_l1() const;
    bool is_dram() const;

    TensorMemoryLayout buffer_layout() const;

    DeviceAddr page_address(DeviceAddr bank_id, DeviceAddr page_index) const;

    uint32_t alignment() const;
    DeviceAddr aligned_page_size() const;
    DeviceAddr aligned_size() const;
    DeviceAddr aligned_size_per_bank() const;

    const std::optional<BufferDistributionSpec>& buffer_distribution_spec() const;
    bool has_shard_spec() const;
    ShardSpecBuffer shard_spec() const;
    std::optional<uint32_t> num_cores() const;
    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping();

    size_t unique_id() const;

    BufferImpl& impl();
    const BufferImpl& impl() const;

private:
    DeviceAddr translate_page_address(DeviceAddr offset, uint32_t bank_id) const;

    std::unique_ptr<BufferImpl> impl_;
};

}  // namespace tt::tt_metal

namespace ttsl::json {
template <>
struct from_json_t<tt::tt_metal::ShardSpec> {
    tt::tt_metal::ShardSpec operator()(const nlohmann::json& json_object) const;
};
}  // namespace ttsl::json
