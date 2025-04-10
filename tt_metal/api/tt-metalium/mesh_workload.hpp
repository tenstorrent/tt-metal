// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>

namespace tt::tt_metal::distributed {
//using RuntimeArgsPerCore = std::vector<std::vector<RuntimeArgsData>>;

//class MeshCommandQueue;
//void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);

class MeshWorkloadImpl;

class MeshWorkload {
    // A MeshWorkload can be fully described using a set of programs mapped to different Logical Device Regions
    // in a Mesh + configurable runtime Args
    // The current iteration supports the following compute paradigms:
    //  - Single Program Multi Device (Completely Homogenous MeshWorkload)
    //  - Multi Program Multi Device (Completely Heterogeneous MeshWorkload)
    // Support for configurable runtime arguments will be added in future versions.
private:
    std::unique_ptr<MeshWorkloadImpl> impl_;

public:
    // Main User-Facing API building blocks
    MeshWorkload();
    ~MeshWorkload();
    MeshWorkload(MeshWorkload&& other) noexcept;
    MeshWorkload& operator=(MeshWorkload&& other) noexcept;

    void add_program(const MeshCoordinateRange& device_range, Program&& program);
    std::unordered_map<MeshCoordinateRange, Program>& get_programs();

    // For testing purposes only
    void set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq);
    MeshCommandQueue* get_last_used_command_queue() const;
    uint32_t get_sem_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_sem_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);

    // Internal API
    MeshWorkloadImpl* get_impl() { return impl_.get(); }
    const MeshWorkloadImpl* get_impl() const { return impl_.get(); }
};
}  // namespace tt::tt_metal::distributed
