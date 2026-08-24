// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/shard_data_transfer.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {
class HostTensor;
class IDevice;
class MemoryConfig;
class MeshTensor;
class SystemMemoryManager;
class WorkerConfigBufferMgr;
namespace distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace distributed
struct ProgramCommandSequence;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {

class MeshEvent;
class MeshTraceDescriptor;
struct MeshBufferReadDescriptor;
struct MeshReadEventDescriptor;
struct MeshCoreDataReadDescriptor;

using MeshCompletionReaderVariant =
    std::variant<MeshBufferReadDescriptor, MeshReadEventDescriptor, MeshCoreDataReadDescriptor>;

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
    virtual WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) = 0;
    virtual void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) = 0;

    // MeshBuffer Write APIs
    virtual void enqueue_write_shard_to_sub_grid(
        const MeshBuffer& buffer,
        const void* host_data,
        const MeshCoordinateRange& device_range,
        bool blocking,
        std::optional<BufferRegion> region = std::nullopt) = 0;
    virtual void enqueue_write_mesh_buffer(
        const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) = 0;
    // If PinnedMemory is attached to a HostBuffer used within the enqueue_write, the contents of the memory must not be
    // modified until the enqueue_write has completed on the device. This may be checked by any of
    // * calling lock() on the PinnedMemory
    // * setting the blocking parameter to true
    // * calling finish() on the MeshCommandQueue
    // * calling enqueue_record_event_to_host() and then waiting for the event to complete on the host.
    virtual void enqueue_write(
        const std::shared_ptr<MeshBuffer>& mesh_buffer, const DistributedHostBuffer& host_buffer, bool blocking) = 0;
    // If PinnedMemory is set on a ShardDataTransfer, the contents of the memory must not be modified until the
    // enqueue_write has completed on the device. This may be checked by any of
    // * calling lock() on the PinnedMemory
    // * setting the blocking parameter to true
    // * calling finish() on the MeshCommandQueue
    // * calling enqueue_record_event_to_host() and then waiting for the event to complete on the host.
    virtual void enqueue_write_shards(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        bool blocking) = 0;

    // MeshBuffer Read APIs
    virtual void enqueue_read_mesh_buffer(
        void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) = 0;
    virtual void enqueue_read_shards(
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        bool blocking) = 0;
    // TODO: does "enqueue" make sense anymore? Return the object by value instead.
    virtual void enqueue_read(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        DistributedHostBuffer& host_buffer,
        const std::optional<std::unordered_set<MeshCoordinate>>& shards,
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
    virtual void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) = 0;
    virtual void record_end() = 0;
    virtual void enqueue_trace(const MeshTraceId& trace_id, bool blocking) = 0;
};

}  // namespace tt::tt_metal::distributed
