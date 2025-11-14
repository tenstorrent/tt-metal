// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <variant>
#include <cstdint>
#include <vector>
#include <optional>

#include <tt-metalium/device.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/bfloat16.hpp>

namespace tt::tt_metal {
using HostDataType = std::variant<
    const std::shared_ptr<std::vector<uint8_t>>,
    const std::shared_ptr<std::vector<uint16_t>>,
    const std::shared_ptr<std::vector<int32_t>>,
    const std::shared_ptr<std::vector<uint32_t>>,
    const std::shared_ptr<std::vector<float>>,
    const std::shared_ptr<std::vector<bfloat16>>,
    const void*>;

class BufferImpl {
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

    // Creates a view of the region of the buffer.
    // The view is a new buffer (unless the region is the entire buffer) that shares the same underlying device memory.
    // The view keeps the underlying buffer alive as long as the view is alive.
    std::shared_ptr<Buffer> view(const BufferRegion& region);

    BufferImpl(const Buffer& other) = delete;
    BufferImpl& operator=(const BufferImpl& other) = delete;
    BufferImpl(BufferImpl&& other) = delete;
    BufferImpl& operator=(BufferImpl&& other) = delete;
    ~BufferImpl();

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

    DeviceAddr page_address(DeviceAddr bank_id, DeviceAddr page_index) const;

    uint32_t alignment() const;
    DeviceAddr aligned_page_size() const;
    DeviceAddr aligned_size() const;
    DeviceAddr aligned_size_per_bank() const;

    // SHARDED API STARTS HERE
    const std::optional<BufferDistributionSpec>& buffer_distribution_spec() const;
    bool has_shard_spec() const { return shard_spec_.has_value(); }
    ShardSpecBuffer shard_spec() const;
    void set_shard_spec(const ShardSpecBuffer& shard_spec);
    std::optional<uint32_t> num_cores() const;
    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping();

    // Returns the buffer that owns the underlying device memory.
    // Typically returns itself unless the buffer was created with a view method.
    std::shared_ptr<Buffer> root_buffer();
    BufferRegion root_buffer_region() const { return BufferRegion(root_buffer_offset_, size_); }

    std::optional<SubDeviceId> sub_device_id() const { return sub_device_id_; }

    size_t unique_id() const { return unique_id_; }

    // Mark the buffer as deallocated, without releasing underlying device memory
    void mark_as_deallocated();

    BufferImpl(
        IDevice* device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        const BufferShardingArgs& sharding_args,
        std::optional<bool> bottom_up,
        std::optional<SubDeviceId> sub_device_id,
        bool owns_data,
        Buffer* pimpl_owner_);

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

    DeviceAddr translate_page_address(DeviceAddr offset, uint32_t bank_id) const;

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
    std::optional<ShardSpecBuffer> shard_spec_;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping_;

    std::optional<BufferDistributionSpec> buffer_distribution_spec_;

    // The root buffer is the buffer that owns the underlying device memory.
    // The root buffer is populated only when the buffer was created with a view method.
    std::shared_ptr<Buffer> root_buffer_;
    // Offset of the current view buffer in the root buffer
    DeviceAddr root_buffer_offset_ = 0;

    size_t unique_id_ = 0;
    static std::atomic<size_t> next_unique_id;

    Buffer* const pimpl_owner;
};
}
