// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <memory>
#include <optional>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {
class HostTensor;
class MemoryConfig;
class MeshTensor;
namespace experimental {
class PinnedMemory;
class ShardDataTransferHelper;
}  // namespace experimental
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {

class MeshDevice;
class MeshEvent;
class MeshWorkload;
class ShardDataTransfer;

// THREAD SAFETY: All methods are thread safe.
class MeshCommandQueue {
    // Main interface to dispatch data and workloads to a MeshDevice
protected:
    MeshDevice* mesh_device_ = nullptr;
    uint32_t id_ = 0;

    MeshCommandQueue(MeshDevice* mesh_device, uint32_t id) : mesh_device_(mesh_device), id_(id) {}

public:
    MeshCommandQueue(const MeshCommandQueue& other) = delete;
    MeshCommandQueue& operator=(const MeshCommandQueue& other) = delete;

    virtual ~MeshCommandQueue() = default;

    MeshDevice* device() const { return mesh_device_; }
    uint32_t id() const { return id_; }
    virtual std::optional<MeshTraceId> trace_id() const = 0;
    virtual void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) = 0;

    // If PinnedMemory is set on a ShardDataTransfer, the contents of the memory must not be modified until the
    // enqueue_write has completed on the device. This may be checked by any of
    // * calling lock() on the PinnedMemory
    // * setting the blocking parameter to true
    // * calling finish() on the MeshCommandQueue
    // * calling enqueue_record_event_to_host() and then waiting for the event to complete on the host.
    virtual void enqueue_write_shards(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
        bool blocking) = 0;

    // MeshTensor Read/Write APIs
    tt::tt_metal::HostTensor enqueue_read_tensor(const tt::tt_metal::MeshTensor& device_tensor, bool blocking = true);
    void enqueue_read_tensor(
        const tt::tt_metal::MeshTensor& device_tensor, tt::tt_metal::HostTensor& host_tensor, bool blocking = true);
    tt::tt_metal::MeshTensor enqueue_write_tensor(const tt::tt_metal::HostTensor& host_tensor);
    tt::tt_metal::MeshTensor enqueue_write_tensor(
        const tt::tt_metal::HostTensor& host_tensor, const tt::tt_metal::MemoryConfig& memory_config);
    void enqueue_write_tensor(const tt::tt_metal::HostTensor& host_tensor, tt::tt_metal::MeshTensor& device_tensor);

    virtual MeshEvent enqueue_record_event(
        ttsl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual MeshEvent enqueue_record_event_to_host(
        ttsl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual void enqueue_wait_for_event(const MeshEvent& sync_event) = 0;
    virtual void finish(ttsl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
};

// Specifies host data to be written to or read from a MeshBuffer shard.
class ShardDataTransfer {
private:
    MeshCoordinate shard_coord_;
    void* host_data_ = nullptr;
    std::optional<BufferRegion> region_;
    std::shared_ptr<experimental::PinnedMemory> pinned_memory_ = nullptr;
    friend class experimental::ShardDataTransferHelper;

public:
    explicit ShardDataTransfer(const MeshCoordinate& shard_coord) : shard_coord_(shard_coord) {}

    MeshCoordinate shard_coord() const { return shard_coord_; }
    void* host_data() const { return host_data_; }
    std::optional<BufferRegion> region() const { return region_; }

    ShardDataTransfer& shard_coord(const MeshCoordinate& shard_coord) {
        shard_coord_ = shard_coord;
        return *this;
    }
    ShardDataTransfer& host_data(void* host_data) {
        host_data_ = host_data;
        return *this;
    }
    ShardDataTransfer& region(std::optional<BufferRegion> region) {
        region_ = region;
        return *this;
    }
};

}  // namespace tt::tt_metal::distributed
