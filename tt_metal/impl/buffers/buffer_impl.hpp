// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>

#include <atomic>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>

namespace tt::tt_metal {

class AllocatorImpl;
class IDevice;

class BufferImpl {
public:
    enum class AllocationStatus : uint8_t {
        ALLOCATION_REQUESTED,
        ALLOCATED,
        DEALLOCATED,
    };

    BufferImpl(
        IDevice* device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        const BufferShardingArgs& sharding_args,
        std::optional<bool> bottom_up,
        std::optional<SubDeviceId> sub_device_id,
        bool owns_data);

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

    std::shared_ptr<Buffer> view(Buffer& self, const BufferRegion& region);

    bool is_allocated() const { return allocation_status_ == AllocationStatus::ALLOCATED && allocation_live(); }
    // False once the allocator is destroyed or has released the allocation this buffer holds
    // (AllocatorImpl::lifetime()).
    bool allocation_live() const { return allocation_live(*allocator_lifetime_, allocation_generation_); }
    static bool allocation_live(const std::atomic<std::uint64_t>& lifetime, std::optional<std::uint64_t> generation) {
        const std::uint64_t current = lifetime.load(std::memory_order_relaxed);
        return current != kAllocatorDestroyed && (!generation || *generation == current);
    }
    // The value of AllocatorImpl::lifetime() once its allocator is destroyed; no generation reaches it.
    static constexpr std::uint64_t kAllocatorDestroyed = std::numeric_limits<std::uint64_t>::max();
    HalMemType memory_type() const;
    bool is_valid_region(const BufferRegion& region) const;
    bool is_valid_partial_region(const BufferRegion& region) const;
    bool bottom_up() const { return bottom_up_; }

    std::shared_ptr<Buffer> root_buffer(Buffer& self);
    BufferRegion root_buffer_region() const { return BufferRegion(root_buffer_offset_, size_); }
    std::optional<SubDeviceId> sub_device_id() const { return sub_device_id_; }
    void mark_as_deallocated() { allocation_status_ = AllocationStatus::DEALLOCATED; }

    void allocate_impl(Buffer& self);
    void deallocate(Buffer& self);
    void deallocate_impl(Buffer& self);
    DeviceAddr translate_page_address(const Buffer& self, DeviceAddr offset, uint32_t bank_id) const;

    void set_per_core_addresses(std::unordered_map<CoreCoord, DeviceAddr> addrs) {
        per_core_addresses_ = std::move(addrs);
    }

    IDevice* const device_;
    const DeviceAddr size_;
    const BufferType buffer_type_;
    const TensorMemoryLayout buffer_layout_;
    const bool bottom_up_;
    const std::optional<SubDeviceId> sub_device_id_;
    const bool owns_data_;

    std::optional<SubDeviceManagerId> sub_device_manager_id_;
    AllocatorImpl* allocator_ = nullptr;
    // See AllocatorImpl::lifetime(). Only allocate_buffer() sets the generation; buffers without an allocation have
    // none.
    std::shared_ptr<const std::atomic<std::uint64_t>> allocator_lifetime_;
    std::optional<std::uint64_t> allocation_generation_;
    // device_->id(), for cleanup that runs when device_ may be gone.
    int device_id_ = 0;

    AllocationStatus allocation_status_ = AllocationStatus::ALLOCATION_REQUESTED;
    bool hooked_allocation_ = false;
    DeviceAddr address_ = 0;

    DeviceAddr page_size_;
    std::optional<ShardSpecBuffer> shard_spec_;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping_;

    std::optional<BufferDistributionSpec> buffer_distribution_spec_;

    bool per_core_allocation_ = false;
    std::unordered_map<CoreCoord, DeviceAddr> per_core_addresses_;

    // Lockstep only across the cores this buffer occupies, rather than every core on the device.
    bool range_lockstep_allocation_ = false;

    std::shared_ptr<Buffer> root_buffer_;
    DeviceAddr root_buffer_offset_ = 0;

    size_t unique_id_ = 0;
    static std::atomic<size_t> next_unique_id;
};

}  // namespace tt::tt_metal
