// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <mutex>
#include <optional>
#include <condition_variable>

#include "common/bfloat16.hpp"
#include "common/core_coord.hpp"
#include "common/tt_backend_api_types.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/impl/allocator/allocator_types.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h" // For CoreType
#include "tt_metal/tt_stl/concepts.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal {
inline namespace v0 {

class Device;

}  // namespace v0

struct ShardSpec {
    /* The individual cores the shard grid is mapped to */
    CoreRangeSet grid;

    /* Canonical tensor shape where the depth dimensions ([:-2] are folded along y) */
    std::array<uint32_t, 2> shape;

    /* The sequence order of the grid cores that the shards are layed out onto. */
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
    bool halo = false;

    ShardSpec(
        const CoreRangeSet &core_sets_,
        const std::array<uint32_t, 2> &shard_shape_,
        const ShardOrientation &shard_orientation_ = ShardOrientation::ROW_MAJOR,
        const bool &halo_ = false) :
        grid(core_sets_), shape(shard_shape_), orientation(shard_orientation_), halo(halo_) {
    }

    const uint32_t num_cores() const { return this->grid.num_cores(); }
    const uint32_t numel() const { return this->shape[0] * this->shape[1]; }

    bool operator==(const ShardSpec& other) const;
    bool operator!=(const ShardSpec& other) const;

    static constexpr auto attribute_names = std::forward_as_tuple("grid", "shape", "orientation", "halo");
    constexpr auto attribute_values() const {
        return std::forward_as_tuple(this->grid, this->shape, this->orientation, this->halo);
    }
};

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
    bool allocate = true;
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
    bool allocate = true;
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

struct AllocBufferMetadata;
void EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md);

inline namespace v0 {

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
        bool allocate = true);

    Buffer(const Buffer &other) = delete;
    Buffer &operator=(const Buffer &other) = delete;
    Buffer(Buffer &&other) = delete;
    Buffer &operator=(Buffer &&other) = delete;

    Device *device() const { return device_; }
    DeviceAddr size() const { return size_; }

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

    TensorMemoryLayout buffer_layout() const { return buffer_layout_; }

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    CoreCoord noc_coordinates(uint32_t bank_id) const;

    // returns NoC coordinates of first bank buffer is in
    CoreCoord noc_coordinates() const;

    DeviceAddr page_address(uint32_t bank_id, uint32_t page_index) const;

    uint32_t alignment() const;
    DeviceAddr aligned_page_size() const;
    DeviceAddr aligned_size() const;

    // SHARDED API STARTS HERE
    // TODO: WILL SEPARATE INTO SHARDED BUFFER CLASS

    DeviceAddr sharded_page_address(uint32_t bank_id, uint32_t page_index) const;

    ShardSpecBuffer shard_spec() const;
    void set_shard_spec(const ShardSpecBuffer& shard_spec);

    uint32_t num_cores() const;

    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping();
    BufferPageMapping generate_buffer_page_mapping() const;

    // Private
    Buffer(
        Device *device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout,
        const std::optional<ShardSpecBuffer>& shard_parameter,
        std::optional<bool> bottom_up,
        Private);

   private:
    void allocate();
    void deallocate();
    static void deallocateAndDelete(Buffer* buffer);
    void set_address(uint64_t addr);

    friend void DeallocateBuffer(Buffer &buffer);
    friend void tt_metal::EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md);

    ShardSpecBuffer shard_spec_locked() const;
    DeviceAddr page_size_locked() const;
    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping_locked();
    DeviceAddr aligned_page_size_locked() const;
    DeviceAddr aligned_size_locked() const;
    DeviceAddr sharded_page_address_locked(uint32_t bank_id, uint32_t page_index) const;
    DeviceAddr page_address_locked(uint32_t bank_id, uint32_t page_index) const;
    uint32_t num_pages_locked() const;
    uint32_t num_dev_pages_locked() const;
    uint32_t num_cores_locked() const;
    BufferPageMapping generate_buffer_page_mapping_locked() const;

    DeviceAddr translate_page_address(uint64_t offset, uint32_t bank_id) const;

    Device * const device_;
    const DeviceAddr size_; // Size in bytes
    DeviceAddr address_ = 0;    // Address of buffer
    const BufferType buffer_type_;
    const TensorMemoryLayout buffer_layout_;
    const std::optional<bool> bottom_up_;

    bool is_allocated_ = false;

    mutable std::mutex config_mutex_;
    DeviceAddr page_size_;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    std::optional<ShardSpecBuffer> shard_parameters_;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping_;

    std::weak_ptr<Buffer> weak_self;
};

}  // namespace v0

BufferPageMapping generate_buffer_page_mapping(const Buffer &buffer);

namespace detail {
using Deviceid = uint32_t;

class buffer_map_t {
   public:
    void insert(std::tuple<Deviceid, DeviceAddr> buf_attr, Buffer *buffer) {
        std::scoped_lock<std::mutex> lock(this->map_mutex);
        this->map.insert({buf_attr, buffer});
    }

    void erase(std::tuple<Deviceid, DeviceAddr> buf_attr) {
        std::scoped_lock<std::mutex> lock(this->map_mutex);
        this->map.erase(buf_attr);
    }

    std::map<std::tuple<Deviceid, DeviceAddr>, Buffer *> value() {
        std::scoped_lock<std::mutex> lock(this->map_mutex);
        return this->map;
    }

    ~buffer_map_t() { TT_ASSERT(this->map.empty(), "Not all buffers deallocated by runtime!"); }

   private:
    std::mutex map_mutex;
    std::map<std::tuple<Deviceid, DeviceAddr>, Buffer *> map = {};
};

extern buffer_map_t BUFFER_MAP;
}  // namespace detail

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
