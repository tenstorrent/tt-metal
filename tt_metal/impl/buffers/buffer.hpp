// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/core_coord.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/impl/sub_device/sub_device_types.hpp"
#include "umd/device/tt_soc_descriptor.h"
#include "umd/device/types/xy_pair.h"
#include "tt_metal/tt_stl/concepts.hpp"
#include "tt_metal/common/assert.hpp"
#include <nlohmann/json.hpp>

#include "llrt/hal.hpp"

namespace tt::tt_metal {
inline namespace v0 {

class Device;

}  // namespace v0

class Allocator;

struct ShardSpec {
    /* The individual cores the shard grid is mapped to */
    CoreRangeSet grid;

    /* Canonical tensor shape where the depth dimensions ([:-2] are folded along y) */
    std::array<uint32_t, 2> shape;

    /* The sequence order of the grid cores that the shards are layed out onto. */
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
    bool halo = false;

    // In ShardMode::PHYSICAL, physical_shard_shape will always be std::nullopt
    ShardMode mode = ShardMode::PHYSICAL;
    std::optional<std::array<uint32_t, 2>> physical_shard_shape = std::nullopt;

    ShardSpec(
        const CoreRangeSet &core_sets_,
        const std::array<uint32_t, 2> &shard_shape_,
        const ShardOrientation &shard_orientation_ = ShardOrientation::ROW_MAJOR,
        const bool &halo_ = false,
        const ShardMode &shard_mode_ = ShardMode::PHYSICAL) :
        grid(core_sets_), shape(shard_shape_), orientation(shard_orientation_), halo(halo_), mode(shard_mode_), physical_shard_shape(std::nullopt) {
    }

    ShardSpec(
        const CoreRangeSet &core_sets_,
        const std::array<uint32_t, 2> &shard_shape_,
        const std::array<uint32_t, 2> &physical_shard_shape_,
        const ShardOrientation &shard_orientation_ = ShardOrientation::ROW_MAJOR,
        const bool &halo_ = false) :
        grid(core_sets_), shape(shard_shape_), orientation(shard_orientation_), halo(halo_), mode(ShardMode::LOGICAL), physical_shard_shape(physical_shard_shape_) {
        TT_FATAL(physical_shard_shape_[0] >= shard_shape_[0] and physical_shard_shape_[1] >= shard_shape_[1], "Physical shard shape ({}, {}) must be greater or equal to logical shard shape ({}, {})!", physical_shard_shape_[0], physical_shard_shape_[1], shard_shape_[0], shard_shape_[1]);
    }

    const uint32_t num_cores() const { return this->grid.num_cores(); }
    const uint32_t numel() const { return this->shape[0] * this->shape[1]; }

    bool operator==(const ShardSpec& other) const;
    bool operator!=(const ShardSpec& other) const;

    static constexpr auto attribute_names = std::forward_as_tuple("grid", "shape", "orientation", "halo", "mode", "physical_shard_shape");
    constexpr auto attribute_values() const {
        return std::forward_as_tuple(this->grid, this->shape, this->orientation, this->halo, this->mode, this->physical_shard_shape);
    }
};

std::ostream& operator<<(std::ostream& os, const ShardSpec& spec);

struct ShardSpecBuffer {
    ShardSpec tensor_shard_spec;
    std::array<uint32_t, 2> page_shape;
    std::array<uint32_t, 2> tensor2d_shape;
    ShardSpecBuffer(
        const CoreRangeSet &core_sets_,
        const std::array<uint32_t, 2> &shard_shape_,
        const ShardOrientation &shard_orientation_,
        const bool &halo_,
        const std::array<uint32_t, 2> &page_shape,
        const std::array<uint32_t, 2> &tensor2d_shape) :
        tensor_shard_spec(core_sets_, shard_shape_, shard_orientation_, halo_) {
        this->page_shape = page_shape;
        this->tensor2d_shape = tensor2d_shape;
    }
    ShardSpecBuffer(
        const ShardSpec &shard_spec,
        const std::array<uint32_t, 2> &page_shape,
        const std::array<uint32_t, 2> &tensor2d_shape) :
        tensor_shard_spec(shard_spec) {
        this->page_shape = page_shape;
        this->tensor2d_shape = tensor2d_shape;
    }
    CoreRangeSet grid() const { return tensor_shard_spec.grid; }
    std::array<uint32_t, 2> shape() const { return tensor_shard_spec.shape; }
    ShardOrientation orientation() const { return tensor_shard_spec.orientation; }
    bool halo() const { return tensor_shard_spec.halo; }
    void set_shard_spec(const ShardSpec& shard_spec) { tensor_shard_spec = shard_spec; };

    /* Shape in pages of the full tensor, not per core */
    std::array<uint32_t, 2> shape_in_pages() const;
    DeviceAddr size() const;
};

inline namespace v0 {

struct BufferConfig {
    Device *device;
    DeviceAddr size;       // Size in bytes
    DeviceAddr page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED;
};

typedef BufferConfig InterleavedBufferConfig;

// copied from above instead of using inheritance such that we can use
// designator constructor
struct ShardedBufferConfig {
    Device *device;
    DeviceAddr size;       // Size in bytes
    DeviceAddr page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type = BufferType::L1;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    ShardSpecBuffer shard_parameters;
};

}  // namespace v0

bool is_sharded(const TensorMemoryLayout &layout);

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

inline namespace v0 {

struct BufferRegion {
    DeviceAddr offset;
    DeviceAddr size;

    BufferRegion() = delete;
    BufferRegion(DeviceAddr offset, DeviceAddr size) : offset(offset), size(size) {}
};

class Buffer final {
    struct Private { explicit Private() = default; };

   public:
    static std::shared_ptr<Buffer> create(
        Device *device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
        const std::optional<ShardSpecBuffer>& shard_parameter = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);
    static std::shared_ptr<Buffer> create(
        Device *device,
        DeviceAddr address,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
        const std::optional<ShardSpecBuffer>& shard_parameter = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);

    Buffer(const Buffer &other) = delete;
    Buffer &operator=(const Buffer &other) = delete;
    Buffer(Buffer &&other) = delete;
    Buffer &operator=(Buffer &&other) = delete;

    Device *device() const { return device_; }
    Allocator *allocator() const { return allocator_; }
    DeviceAddr size() const { return size_; }
    bool is_allocated() const;

    // Returns address of buffer in the first bank
    uint32_t address() const;

    DeviceAddr page_size() const;
    void set_page_size(DeviceAddr page_size);

    uint32_t num_pages() const;
    uint32_t num_dev_pages() const;

    BufferType buffer_type() const { return buffer_type_; }
    CoreType core_type() const;

    bool is_l1() const;
    bool is_dram() const;
    bool is_trace() const;

    bool is_valid_region(const BufferRegion& region) const;
    bool is_partial_region(const BufferRegion& region) const;

    TensorMemoryLayout buffer_layout() const { return buffer_layout_; }

    bool bottom_up() const { return bottom_up_; }

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    DeviceAddr page_address(uint32_t bank_id, uint32_t page_index) const;

    DeviceAddr bank_local_page_address(uint32_t bank_id, uint32_t page_index) const;
    uint32_t alignment() const;
    DeviceAddr aligned_page_size() const;
    DeviceAddr aligned_size() const;
    DeviceAddr aligned_size_per_bank() const;

    // SHARDED API STARTS HERE
    // TODO: WILL SEPARATE INTO SHARDED BUFFER CLASS

    DeviceAddr sharded_page_address(uint32_t bank_id, uint32_t page_index) const;

    ShardSpecBuffer shard_spec() const;
    void set_shard_spec(const ShardSpecBuffer& shard_spec);

    std::optional<uint32_t> num_cores() const;

    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping();

    std::optional<SubDeviceId> sub_device_id() const { return sub_device_id_; }
    std::optional<SubDeviceManagerId> sub_device_manager_id() const { return sub_device_manager_id_; }

    size_t unique_id() const { return unique_id_; }

    Buffer(
        Device *device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout,
        const std::optional<ShardSpecBuffer>& shard_parameter,
        std::optional<bool> bottom_up,
        std::optional<SubDeviceId> sub_device_id,
        bool owns_data,
        Private);

   private:
    enum class AllocationStatus : uint8_t {
        ALLOCATION_REQUESTED,
        ALLOCATION_FAILED,
        ALLOCATED,
        DEALLOCATED,
    };

    // Deallocate is allowed to be called multiple times on the same buffer
    void deallocate();
    static void deleter(Buffer* buffer);
    void deallocate_impl();
    friend void DeallocateBuffer(Buffer &buffer);

    DeviceAddr translate_page_address(uint64_t offset, uint32_t bank_id) const;

    Device * const device_;
    const DeviceAddr size_; // Size in bytes
    const BufferType buffer_type_;
    const TensorMemoryLayout buffer_layout_;
    const bool bottom_up_;
    const std::optional<SubDeviceId> sub_device_id_;
    const bool owns_data_;

    std::optional<SubDeviceManagerId> sub_device_manager_id_;
    Allocator * allocator_;

    std::atomic<AllocationStatus> allocation_status_ = AllocationStatus::ALLOCATION_REQUESTED;
    DeviceAddr address_ = 0;
    mutable std::mutex allocation_mutex_;
    mutable std::condition_variable allocation_cv_;
    // Used exclusively for is_allocated() method
    std::atomic<bool> deallocation_requested_ = false;

    // These members must be only accessed on the device worker thread
    DeviceAddr page_size_; // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    std::optional<ShardSpecBuffer> shard_parameters_;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping_;

    std::weak_ptr<Buffer> weak_self;
    size_t unique_id_ = 0;
    static std::atomic<size_t> next_unique_id;
};

}  // namespace v0

BufferPageMapping generate_buffer_page_mapping(const Buffer &buffer);

inline namespace v0 {

using HostDataType = std::variant<
    const std::shared_ptr<std::vector<uint8_t>>,
    const std::shared_ptr<std::vector<uint16_t>>,
    const std::shared_ptr<std::vector<int32_t>>,
    const std::shared_ptr<std::vector<uint32_t>>,
    const std::shared_ptr<std::vector<float>>,
    const std::shared_ptr<std::vector<bfloat16>>,
    const void *>;

}  // namespace v0
}  // namespace tt::tt_metal

namespace tt::stl::json {
template <>
struct from_json_t<tt_metal::ShardSpec> {
    tt_metal::ShardSpec operator()(const nlohmann::json &json_object) const;
};
}  // namespace tt::stl::json
