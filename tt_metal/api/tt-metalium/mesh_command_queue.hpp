// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <queue>

#include "buffer.hpp"
#include "command_queue_interface.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include "mesh_workload.hpp"

namespace tt::tt_metal::distributed {

class MeshEvent;
struct MeshReadEventDescriptor;

class MeshCommandQueue {
    // Main interface to dispatch data and workloads to a MeshDevice
    // Currently only supports dispatching workloads and relies on the
    // tt::tt_metal::CommandQueue.
    // Additional support for Reads and Writes to be added
private:
    void populate_virtual_program_dispatch_core();
    void populate_dispatch_core_type();
    CoreCoord virtual_program_dispatch_core() const;
    CoreType dispatch_core_type() const;

    // Helper functions for reading and writing individual shards
    void write_shard_to_device(
        std::shared_ptr<Buffer>& shard_view,
        const void* src,
        const BufferRegion& region,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    void read_shard_from_device(
        std::shared_ptr<Buffer>& shard_view,
        void* dst,
        const BufferRegion& region,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});

    // Helper functions for read and write entire Sharded-MeshBuffers
    void write_sharded_buffer(const MeshBuffer& buffer, const void* src);
    void read_sharded_buffer(MeshBuffer& buffer, void* dst);
    void enqueue_record_event_helper(
        const std::shared_ptr<MeshEvent>& event,
        tt::stl::Span<const SubDeviceId> sub_device_ids,
        bool notify_host,
        const std::optional<LogicalDeviceRange>& device_range = std::nullopt);
    std::array<tt::tt_metal::WorkerConfigBufferMgr, DispatchSettings::DISPATCH_MESSAGE_ENTRIES> config_buffer_mgr_;
    std::array<uint32_t, DispatchSettings::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed_;
    MeshDevice* mesh_device_ = nullptr;
    uint32_t id_ = 0;
    CoreCoord dispatch_core_;
    CoreType dispatch_core_type_ = CoreType::WORKER;
    std::queue<std::shared_ptr<MeshReadEventDescriptor>> event_descriptors_;

public:
    MeshCommandQueue(MeshDevice* mesh_device, uint32_t id);
    MeshDevice* device() const { return mesh_device_; }
    uint32_t id() const { return id_; }
    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) { return config_buffer_mgr_[index]; };
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking);

    // Specifies host data to be written to or read from a MeshBuffer shard.
    struct ShardDataTransfer {
        MeshCoordinate shard_coord;
        void* host_data = nullptr;
        std::optional<BufferRegion> region;
    };

    // MeshBuffer Write APIs
    void enqueue_write_shard_to_sub_grid(
        const MeshBuffer& buffer, const void* host_data, const LogicalDeviceRange& device_range, bool blocking);
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

    void enqueue_record_event(
        const std::shared_ptr<MeshEvent>& event,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<LogicalDeviceRange>& device_range = std::nullopt);
    void enqueue_record_event_to_host(
        const std::shared_ptr<MeshEvent>& event,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<LogicalDeviceRange>& device_range = std::nullopt);
    void enqueue_wait_for_event(const std::shared_ptr<MeshEvent>& sync_event);
    void drain_events_from_completion_queue();
    void verify_reported_events_after_draining(const std::shared_ptr<MeshEvent>& event);
    void finish(tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_memcpy_aligned<uint32_t>& go_signal_noc_data);
};

}  // namespace tt::tt_metal::distributed
