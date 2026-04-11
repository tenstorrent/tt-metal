// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Internal header — not part of the public API. Do not include from public headers.

#include <tt-metalium/buffer.hpp>
#include <atomic>
#include <memory>
#include <optional>
#include <unordered_map>

namespace tt::tt_metal {

class AllocatorImpl;

// Internal implementation struct for Buffer (pimpl idiom).
// All fields are public — access control is enforced by the Buffer class.
struct BufferImpl {
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

    IDevice* const device_;
    const DeviceAddr size_;
    const BufferType buffer_type_;
    const TensorMemoryLayout buffer_layout_;
    const bool bottom_up_;
    const std::optional<SubDeviceId> sub_device_id_;
    const bool owns_data_;

    std::optional<SubDeviceManagerId> sub_device_manager_id_;
    AllocatorImpl* allocator_ = nullptr;

    AllocationStatus allocation_status_ = AllocationStatus::ALLOCATION_REQUESTED;
    bool hooked_allocation_ = false;
    DeviceAddr address_ = 0;

    // These members must only be accessed on the device worker thread
    DeviceAddr page_size_;
    std::optional<ShardSpecBuffer> shard_spec_;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping_;

    std::optional<BufferDistributionSpec> buffer_distribution_spec_;

    // Per-core allocation state (experimental)
    bool per_core_allocation_ = false;
    std::unordered_map<CoreCoord, DeviceAddr> per_core_addresses_;

    // Root buffer for views
    std::shared_ptr<Buffer> root_buffer_;
    DeviceAddr root_buffer_offset_ = 0;

    size_t unique_id_ = 0;
    static std::atomic<size_t> next_unique_id;
};

}  // namespace tt::tt_metal
