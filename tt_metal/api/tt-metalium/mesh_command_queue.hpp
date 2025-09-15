// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/command_queue_interface.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/vector_aligned.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
class SystemMemoryManager;
class WorkerConfigBufferMgr;
namespace distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace distributed
struct ProgramCommandSequence;
}  // namespace tt_metal
}  // namespace tt

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
    // Currently only supports dispatching workloads and relies on the
    // tt::tt_metal::CommandQueue.
    // Additional support for Reads and Writes to be added
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

    // Specifies host data to be written to or read from a MeshBuffer shard.
    struct ShardDataTransfer {
        MeshCoordinate shard_coord;
        void* host_data = nullptr;
        std::optional<BufferRegion> region;
    };

    // MeshBuffer Write APIs
    virtual void enqueue_write_shard_to_sub_grid(
        const MeshBuffer& buffer,
        const void* host_data,
        const MeshCoordinateRange& device_range,
        bool blocking,
        std::optional<BufferRegion> region = std::nullopt) = 0;
    virtual void enqueue_write_mesh_buffer(
        const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) = 0;
    virtual void enqueue_write_shards(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        bool blocking) = 0;
    virtual void enqueue_write(
        const std::shared_ptr<MeshBuffer>& mesh_buffer, const DistributedHostBuffer& host_buffer, bool blocking) = 0;

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

    virtual MeshEvent enqueue_record_event(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual MeshEvent enqueue_record_event_to_host(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual void enqueue_wait_for_event(const MeshEvent& sync_event) = 0;
    virtual void finish(tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_aligned<uint32_t>& go_signal_noc_data,
        const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping) = 0;
    virtual void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) = 0;
    virtual void record_end() = 0;
    virtual void enqueue_trace(const MeshTraceId& trace_id, bool blocking) = 0;
};

// Make the specified MeshCommandQueue record an event.
// Host is not notified when this event completes.
// Can be used for CQ to CQ synchronization.
MeshEvent EnqueueRecordEvent(
    MeshCommandQueue& mesh_cq,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {},
    const std::optional<MeshCoordinateRange>& device_range = std::nullopt);

// Make the specified MeshCommandQueue record an event and notify the host when it completes.
// Can be used for CQ to CQ and host to CQ synchronization.
MeshEvent EnqueueRecordEventToHost(
    MeshCommandQueue& mesh_cq,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {},
    const std::optional<MeshCoordinateRange>& device_range = std::nullopt);

// Make the specified MeshCommandQueue wait for the completion of an event.
// This operation is non-blocking on host, however the specified command queue
// will stall until the event is recorded.
void EnqueueWaitForEvent(MeshCommandQueue& mesh_cq, const MeshEvent& event);

// Make the current thread block until the event is recorded by the associated MeshCommandQueue.
void EventSynchronize(const MeshEvent& event);

// Query the status of an event tied to a MeshCommandQueue.
// Returns true if the CQ has completed recording the event, false otherwise.
bool EventQuery(const MeshEvent& event);

// Trace capture and replay helpers
MeshTraceId BeginTraceCapture(MeshDevice* device, uint8_t cq_id);
void EndTraceCapture(MeshDevice* device, uint8_t cq_id, const MeshTraceId& trace_id);
void ReplayTrace(MeshDevice* device, uint8_t cq_id, const MeshTraceId& trace_id, bool blocking);
void ReleaseTrace(MeshDevice* device, const MeshTraceId& trace_id);

// Synchronization helpers
void Synchronize(
    MeshDevice* device, std::optional<uint8_t> cq_id, tt::stl::Span<const SubDeviceId> sub_device_ids = {});
void Finish(MeshCommandQueue& mesh_cq, tt::stl::Span<const SubDeviceId> sub_device_ids = {});

// Returns true if the distributed environment is initialized and world_size > 1.
bool UsingDistributedEnvironment();

// MeshBuffer read/write helper templates
template <typename DType>
void WriteShard(
    MeshCommandQueue& mesh_cq,
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    std::vector<DType>& src,
    const MeshCoordinate& coord,
    bool blocking = false) {
    std::vector<MeshCommandQueue::ShardDataTransfer> shard_data_transfers = {{
        .shard_coord = coord,
        .host_data = src.data(),
        .region = std::nullopt,
    }};
    mesh_cq.enqueue_write_shards(mesh_buffer, shard_data_transfers, blocking);
}

template <typename DType>
void ReadShard(
    MeshCommandQueue& mesh_cq,
    std::vector<DType>& dst,
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    const MeshCoordinate& coord,
    bool blocking = true) {
    auto mesh_device = mesh_cq.device();
    if (!mesh_device->is_local(coord)) {
        return;
    }

    auto shard = mesh_buffer->get_device_buffer(coord);
    dst.resize(shard->page_size() * shard->num_pages() / sizeof(DType));
    std::vector<MeshCommandQueue::ShardDataTransfer> shard_data_transfers = {{
        .shard_coord = coord,
        .host_data = dst.data(),
        .region = std::nullopt,
    }};
    mesh_cq.enqueue_read_shards(shard_data_transfers, mesh_buffer, blocking);
}

template <typename DType>
void EnqueueWriteMeshBuffer(
    MeshCommandQueue& mesh_cq,
    std::shared_ptr<MeshBuffer>& mesh_buffer,
    const std::vector<DType>& src,
    bool blocking = false) {
    mesh_cq.enqueue_write_mesh_buffer(mesh_buffer, src.data(), blocking);
}

template <typename DType>
void EnqueueReadMeshBuffer(
    MeshCommandQueue& mesh_cq,
    std::vector<DType>& dst,
    std::shared_ptr<MeshBuffer>& mesh_buffer,
    bool blocking = true) {
    TT_FATAL(
        mesh_buffer->global_layout() == MeshBufferLayout::SHARDED,
        "Can only read a Sharded MeshBuffer from a MeshDevice.");
    dst.resize(mesh_buffer->global_shard_spec().global_size / sizeof(DType));
    mesh_cq.enqueue_read_mesh_buffer(dst.data(), mesh_buffer, blocking);
}

}  // namespace tt::tt_metal::distributed
