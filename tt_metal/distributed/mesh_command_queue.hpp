// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mesh_device.hpp>

#include "tt_metal/distributed/mesh_workload.hpp"

namespace tt::tt_metal::distributed {

class MeshCommandQueue {
    // Main interface to dispatch data and workloads to a MeshDevice
    // Currently only supports dispatching workloads and relies on the
    // tt::tt_metal::CommandQueue.
    // Additional support for Reads and Writes to be added
private:
    uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id);
    void populate_virtual_program_dispatch_core();
    void populate_dispatch_core_type();
    CoreCoord virtual_program_dispatch_core() const;
    CoreType dispatch_core_type() const;
    tt::tt_metal::WorkerConfigBufferMgr config_buffer_mgr_;
    LaunchMessageRingBufferState worker_launch_message_buffer_state_;
    uint32_t expected_num_workers_completed_ = 0;
    MeshDevice* mesh_device_;
    uint32_t id_;
    CoreCoord dispatch_core_;
    CoreType dispatch_core_type_;

public:
    MeshCommandQueue(MeshDevice* mesh_device, uint32_t id);
    MeshDevice* device() const { return mesh_device_; }
    uint32_t id() const { return id_; }
    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) { return config_buffer_mgr_; };
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking);
    void finish();
};

}  // namespace tt::tt_metal::distributed
