// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "command_queue_interface.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include "mesh_workload.hpp"

namespace tt::tt_metal::distributed {

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
        std::shared_ptr<Buffer>& shard_view, const void* src, tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    void read_shard_from_device(
        std::shared_ptr<Buffer>& shard_view, void* dst, tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    // Helper functions for read and write entire Sharded-MeshBuffers
    void write_sharded_buffer(const MeshBuffer& buffer, const void* src);
    void read_sharded_buffer(MeshBuffer& buffer, void* dst);
    std::array<tt::tt_metal::WorkerConfigBufferMgr, DispatchSettings::DISPATCH_MESSAGE_ENTRIES> config_buffer_mgr_;
    std::array<uint32_t, DispatchSettings::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed_;
    MeshDevice* mesh_device_;
    uint32_t id_;
    CoreCoord dispatch_core_;
    CoreType dispatch_core_type_;

public:
    MeshCommandQueue(MeshDevice* mesh_device, uint32_t id);
    MeshDevice* device() const { return mesh_device_; }
    uint32_t id() const { return id_; }
    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) { return config_buffer_mgr_[index]; };
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking);
    // MeshBuffer Write APIs
    void enqueue_write_shard(
        std::shared_ptr<MeshBuffer>& mesh_buffer, const void* host_data, const Coordinate& coord, bool blocking);
    void enqueue_write_shard_to_sub_grid(
        const MeshBuffer& buffer, const void* host_data, const LogicalDeviceRange& device_range, bool blocking);
    void enqueue_write_mesh_buffer(const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking);
    // MeshBuffer Read APIs
    void enqueue_read_shard(
        void* host_data, const std::shared_ptr<MeshBuffer>& mesh_buffer, const Coordinate& coord, bool blocking);
    void enqueue_read_mesh_buffer(void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking);
    void finish();
    void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_memcpy_aligned<uint32_t>& go_signal_noc_data);
};

}  // namespace tt::tt_metal::distributed
