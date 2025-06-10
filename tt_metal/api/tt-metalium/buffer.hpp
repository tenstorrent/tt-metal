// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_soc_descriptor.h>
#include <umd/device/types/xy_pair.h>

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

    uint32_t num_cores() const { return this->grid.num_cores(); }
    uint32_t numel() const { return this->shape[0] * this->shape[1]; }

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

struct BufferConfig {
    IDevice* device;
    DeviceAddr size;       // Size in bytes
    DeviceAddr page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED;
};

using InterleavedBufferConfig = BufferConfig;

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

bool is_sharded(const TensorMemoryLayout& layout);

struct BufferRegion {
    DeviceAddr offset = 0;
    DeviceAddr size = 0;

    BufferRegion() = delete;
    BufferRegion(DeviceAddr offset, DeviceAddr size) : offset(offset), size(size) {}
};

class Buffer final {
    // Used in public Buffer constructors so they are only callable within Buffer
    // Buffer constructors are public so we can call std::make_shared on Buffer
    struct Private {
        explicit Private() = default;
    };

public:
    // Buffer::create APIs provide single entry point for creating buffers with ShardSpec or BufferDistributionSpec
    // - Validation is done in Buffer constructor with validate_buffer_parameters
    // - Only one of ShardSpec or BufferDistributionSpec can be set
    // TODO: buffer_layout should not be needed with BufferDistributionSpec since layout is implicit in the spec
    // - ie. It's possible to fully unify interleaved and sharding
    // - For now, must pass TensorMemoryLayout::BLOCK_SHARDED (seems like the most sensible option)
    // TODO: Unify Buffer parameters with MeshBuffer (ie. use DeviceLocalBufferConfig as well)
    // - BufferDistributionSpec is added to the end with default std::nullopt value to avoid breaking existing usages
    // - Eventually, do we even need to expose single-device Buffer::create to users?
    static std::shared_ptr<Buffer> create(
        IDevice* device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
        const std::optional<std::variant<ShardSpecBuffer, BufferDistributionSpec>>& shard_parameter = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);
    static std::shared_ptr<Buffer> create(
        IDevice* device,
        DeviceAddr address,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
        const std::optional<std::variant<ShardSpecBuffer, BufferDistributionSpec>>& shard_parameter = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);

    Buffer(const Buffer& other) = delete;
    Buffer& operator=(const Buffer& other) = delete;
    Buffer(Buffer&& other) = delete;
    Buffer& operator=(Buffer&& other) = delete;
    ~Buffer();

    IDevice* device() const { return device_; }
    Allocator* allocator() const { return allocator_; }
    DeviceAddr size() const { return size_; }
    bool is_allocated() const;

    // Returns address of buffer in the first bank
    uint32_t address() const;

    DeviceAddr page_size() const;
    void set_page_size(DeviceAddr page_size);

    uint32_t num_pages() const;
    uint32_t num_dev_pages() const;

    BufferType buffer_type() const { return buffer_type_; }
    HalMemType memory_type() const;
    CoreType core_type() const;

    bool is_l1() const;
    bool is_dram() const;
    bool is_trace() const;

    bool is_valid_region(const BufferRegion& region) const;
    bool is_valid_partial_region(const BufferRegion& region) const;

    TensorMemoryLayout buffer_layout() const { return buffer_layout_; }

    bool bottom_up() const { return bottom_up_; }

    DeviceAddr page_address(uint32_t bank_id, uint32_t page_index) const;

    DeviceAddr bank_local_page_address(uint32_t bank_id, uint32_t page_index) const;
    uint32_t alignment() const;
    DeviceAddr aligned_page_size() const;
    DeviceAddr aligned_size() const;
    DeviceAddr aligned_size_per_bank() const;

    // SHARDED API STARTS HERE
    // If buffer contains BufferDistributionSpec, it is considered ND sharded
    bool is_nd_sharded() const;
    const std::optional<BufferDistributionSpec>& buffer_distribution_spec() const;

    // TODO: WILL SEPARATE INTO SHARDED BUFFER CLASS

    DeviceAddr sharded_page_address(uint32_t bank_id, uint32_t page_index) const;

    ShardSpecBuffer shard_spec() const;
    void set_shard_spec(const ShardSpecBuffer& shard_spec);

    // TODO: Consolidate with interleaved and delete this (maybe get from BufferDistributionSpec)
    std::optional<uint32_t> num_cores() const;

    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping();

    std::optional<SubDeviceId> sub_device_id() const { return sub_device_id_; }

    size_t unique_id() const { return unique_id_; }

    // Mark the buffer as deallocated, without releasing underlying device memory
    void mark_as_deallocated();

    Buffer(
        IDevice* device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout,
        const std::optional<std::variant<ShardSpecBuffer, BufferDistributionSpec>>& shard_parameter,
        std::optional<bool> bottom_up,
        std::optional<SubDeviceId> sub_device_id,
        bool owns_data,
        Private);

private:
    enum class AllocationStatus : uint8_t {
        ALLOCATION_REQUESTED,
        ALLOCATED,
        DEALLOCATED,
    };

    void allocate_impl();

    // Deallocate is allowed to be called multiple times on the same buffer
    void deallocate();
    void deallocate_impl();
    friend void DeallocateBuffer(Buffer& buffer);

    DeviceAddr translate_page_address(uint64_t offset, uint32_t bank_id) const;

    IDevice* const device_;
    const DeviceAddr size_;  // Size in bytes
    const BufferType buffer_type_;
    const TensorMemoryLayout buffer_layout_;
    const bool bottom_up_;
    const std::optional<SubDeviceId> sub_device_id_;
    const bool owns_data_;

    std::optional<SubDeviceManagerId> sub_device_manager_id_;
    Allocator* allocator_;

    AllocationStatus allocation_status_ = AllocationStatus::ALLOCATION_REQUESTED;
    bool hooked_allocation_ = false;
    DeviceAddr address_ = 0;

    // These members must be only accessed on the device worker thread
    DeviceAddr page_size_;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    std::optional<ShardSpecBuffer> shard_parameters_;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping_;

    std::optional<BufferDistributionSpec> buffer_distribution_spec_;

    size_t unique_id_ = 0;
    static std::atomic<size_t> next_unique_id;
};

BufferPageMapping generate_buffer_page_mapping(const Buffer& buffer);

using HostDataType = std::variant<
    const std::shared_ptr<std::vector<uint8_t>>,
    const std::shared_ptr<std::vector<uint16_t>>,
    const std::shared_ptr<std::vector<int32_t>>,
    const std::shared_ptr<std::vector<uint32_t>>,
    const std::shared_ptr<std::vector<float>>,
    const std::shared_ptr<std::vector<bfloat16>>,
    const void*>;

}  // namespace tt::tt_metal

namespace tt::stl::json {
template <>
struct from_json_t<tt_metal::ShardSpec> {
    tt_metal::ShardSpec operator()(const nlohmann::json& json_object) const;
};
}  // namespace tt::stl::json
