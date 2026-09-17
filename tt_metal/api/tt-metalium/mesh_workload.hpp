// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal::distributed {

/// Program-memory use established by non-dispatch workload preparation.
struct MeshWorkloadProgramCapacity {
    uint32_t max_program_config_size_bytes = 0;
    uint32_t max_kernel_binary_size_bytes = 0;
};

class MeshWorkload;
class MeshWorkloadImpl;

class MeshCommandQueue;
class FDMeshCommandQueue;
void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);

class MeshWorkload {
    // A MeshWorkload can be fully described using a set of programs mapped to different Logical Device Regions
    // in a Mesh + configurable runtime Args
    // The current iteration supports the following compute paradigms:
    //  - Single Program Multi Device (Completely Homogeneous MeshWorkload)
    //  - Multi Program Multi Device (Completely Heterogeneous MeshWorkload)
    // Support for configurable runtime arguments will be added in future versions.
public:
    // Main User-Facing API building blocks
    MeshWorkload();
    ~MeshWorkload();
    MeshWorkload(MeshWorkload&& other) noexcept;
    MeshWorkload& operator=(MeshWorkload&& other) noexcept;
    // Requirement: In one MeshWorkload, device ranges for different programs must not share devices; each device can
    // belong to only one program. Not all devices need a program—leaving some devices unassigned is allowed.
    void add_program(const MeshCoordinateRange& device_range, Program&& program);
    std::unordered_map<MeshCoordinateRange, Program>& get_programs();
    const std::unordered_map<MeshCoordinateRange, Program>& get_programs() const;

    /// Compiles kernels, finalizes program offsets, and validates capacity without dispatching.
    MeshWorkloadProgramCapacity prepare(MeshDevice* mesh_device);

    // For testing purposes only
    void set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq);
    MeshCommandQueue* get_last_used_command_queue() const;
    uint32_t get_sem_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_sem_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    MeshWorkloadImpl& impl() { return *pimpl_; }

private:
    std::unique_ptr<MeshWorkloadImpl> pimpl_;

    friend void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);
    friend FDMeshCommandQueue;
};
}  // namespace tt::tt_metal::distributed
