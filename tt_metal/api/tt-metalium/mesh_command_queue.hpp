// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <queue>

#include "buffer.hpp"
#include "command_queue.hpp"
#include "command_queue_interface.hpp"
#include "multi_producer_single_consumer_queue.hpp"

#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include "mesh_workload.hpp"
#include "mesh_trace.hpp"
#include "mesh_trace_id.hpp"

namespace tt::tt_metal {

class ThreadPool;

namespace distributed {

class MeshEvent;
struct MeshReadEventDescriptor;
struct MeshBufferReadDescriptor;

using MeshCompletionReaderVariant = std::variant<MeshBufferReadDescriptor, MeshReadEventDescriptor>;

class MeshCommandQueue {
    // Main interface to dispatch data and workloads to a MeshDevice
    // Currently only supports dispatching workloads and relies on the
    // tt::tt_metal::CommandQueue.
    // Additional support for Reads and Writes to be added
protected:
    MeshDevice* mesh_device_ = nullptr;
    uint32_t id_ = 0;
    std::shared_ptr<ThreadPool>
        dispatch_thread_pool_;  // Thread pool used to dispatch to the Mesh (used by main thread)

    MeshCommandQueue(MeshDevice* mesh_device, uint32_t id, std::shared_ptr<ThreadPool> dispatch_thread_pool) :
        mesh_device_(mesh_device), id_(id), dispatch_thread_pool_(std::move(dispatch_thread_pool)) {}

    // Helper functions for reading and writing individual shards
    virtual void write_shard_to_device(
        Buffer* shard_view,
        const void* src,
        const BufferRegion& region,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual void read_shard_from_device(
        Buffer* shard_view,
        void* dst,
        const BufferRegion& region,
        std::unordered_map<IDevice*, uint32_t>& num_txns_per_device,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual void submit_memcpy_request(std::unordered_map<IDevice*, uint32_t>& num_txns_per_device, bool blocking) = 0;

    // Helper functions for read and write entire Sharded-MeshBuffers
    void write_sharded_buffer(const MeshBuffer& buffer, const void* src);
    void read_sharded_buffer(MeshBuffer& buffer, void* dst);

public:
    MeshCommandQueue(const MeshCommandQueue& other) = delete;
    MeshCommandQueue& operator=(const MeshCommandQueue& other) = delete;

    virtual ~MeshCommandQueue() = default;

    MeshDevice* device() const { return mesh_device_; }
    uint32_t id() const { return id_; }
    virtual WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) = 0;
    virtual void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) = 0;

    // Specifies host data to be written to or read from a MeshBuffer shard.
    struct ShardDataTransfer {
        MeshCoordinate shard_coord;
        void* host_data = nullptr;
        std::optional<BufferRegion> region;
    };

    // MeshBuffer Write APIs
    void enqueue_write_shard_to_sub_grid(
        const MeshBuffer& buffer,
        const void* host_data,
        const MeshCoordinateRange& device_range,
        bool blocking,
        std::optional<BufferRegion> region = std::nullopt);
    void enqueue_write_mesh_buffer(const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking);
    void enqueue_write_shards(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        bool blocking);

    // MeshBuffer Read APIs
    void enqueue_read_mesh_buffer(void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking);
    void enqueue_read_shards(
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        bool blocking);

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
        const vector_memcpy_aligned<uint32_t>& go_signal_noc_data) = 0;
    virtual void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) = 0;
    virtual void record_end() = 0;
    virtual void enqueue_trace(const MeshTraceId& trace_id, bool blocking) = 0;
};

}  // namespace distributed

}  // namespace tt::tt_metal
