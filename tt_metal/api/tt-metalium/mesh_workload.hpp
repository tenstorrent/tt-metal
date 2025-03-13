// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "host_api.hpp"
#include "mesh_device.hpp"
#include "mesh_buffer.hpp"

namespace tt::tt_metal::distributed {
using RuntimeArgsPerCore = std::vector<std::vector<RuntimeArgsData>>;

class MeshCommandQueue;
void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);

/**
 * @brief A MeshWorkload is a collection of programs mapped to different Logical Device Regions in a Mesh + configurable
 * runtime Args
 *
 * The current iteration supports the following compute paradigms:
 *  - Single Program Multi Device (Completely Homogenous MeshWorkload)
 *  - Multi Program Multi Device (Completely Heterogeneous MeshWorkload)
 *
 * Support for configurable runtime arguments will be added in future versions.
 */
class MeshWorkload {
public:
    MeshWorkload();
    ~MeshWorkload();

    MeshWorkload(const MeshWorkload&) = delete;
    MeshWorkload& operator=(const MeshWorkload&) = delete;

    MeshWorkload(MeshWorkload&&) noexcept;
    MeshWorkload& operator=(MeshWorkload&&) noexcept;

    void add_program(const MeshCoordinateRange& device_range, Program&& program);
    std::unordered_map<MeshCoordinateRange, Program>& get_programs();

    // For testing purposes only
    void set_last_used_command_queue_for_testing(MeshCommandQueue* mesh_cq);
    MeshCommandQueue* get_last_used_command_queue() const;
    uint32_t get_sem_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_sem_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_base_addr(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_size(std::shared_ptr<MeshDevice>& mesh_device, CoreCoord logical_core, CoreType core_type);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};
}  // namespace tt::tt_metal::distributed
