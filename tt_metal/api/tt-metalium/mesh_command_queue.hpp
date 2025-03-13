// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <queue>

#include "mesh_device.hpp"
#include "buffer.hpp"
#include "command_queue.hpp"
#include "command_queue_interface.hpp"

#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include "mesh_workload.hpp"
#include "mesh_trace.hpp"
#include "mesh_trace_id.hpp"

namespace tt::tt_metal::distributed {

class MeshEvent;
struct MeshReadEventDescriptor;
struct MeshBufferReadDescriptor;

/**
 * @brief Main interface to dispatch data and workloads to a MeshDevice
 *
 * Currently only supports dispatching workloads and relies on the
 * tt::tt_metal::CommandQueue.
 * Additional support for Reads and Writes to be added
 */
class MeshCommandQueue {
public:
    MeshCommandQueue();
    virtual ~MeshCommandQueue() = default;

    MeshCommandQueue(const MeshCommandQueue& other) = delete;
    MeshCommandQueue& operator=(const MeshCommandQueue& other) = delete;

    virtual MeshDevice* device() const = 0;
    virtual uint32_t id() const = 0;

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
    ;

    virtual void enqueue_write_mesh_buffer(
        const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) = 0;
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

    virtual MeshEvent enqueue_record_event(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual MeshEvent enqueue_record_event_to_host(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual void enqueue_wait_for_event(const MeshEvent& sync_event) = 0;

    virtual void drain_events_from_completion_queue() = 0;

    virtual void verify_reported_events_after_draining(const MeshEvent& event) = 0;

    virtual void finish(tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;

    virtual void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_memcpy_aligned<uint32_t>& go_signal_noc_data) = 0;

    virtual void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) = 0;
    virtual void record_end() = 0;

    virtual void enqueue_trace(const MeshTraceId& trace_id, bool blocking) = 0;

    // Main function (event loop) for the Completion Queue Reader
    virtual void read_completion_queue() = 0;
    // Helper function - read events from Completion Queue

    virtual void read_completion_queue_event(MeshReadEventDescriptor& read_event_descriptor) = 0;

    // Helper function - read buffer data from Completion Queue
    virtual void copy_buffer_data_to_user_space(MeshBufferReadDescriptor& read_buffer_descriptor) = 0;
};

}  // namespace tt::tt_metal::distributed
