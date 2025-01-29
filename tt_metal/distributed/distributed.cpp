// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal::distributed {

MeshWorkload CreateMeshWorkload() { return MeshWorkload(); }

void AddProgramToMeshWorkload(
    MeshWorkload& mesh_workload, Program& program, const LogicalDeviceRange& device_range) {
    mesh_workload.add_program(device_range, std::move(program));
}

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    mesh_workload.compile(mesh_cq.device());
    mesh_workload.load_binaries(mesh_cq);
    mesh_workload.generate_dispatch_commands(mesh_cq);
    mesh_cq.enqueue_mesh_workload(mesh_workload, blocking);
}

void Finish(MeshCommandQueue& mesh_cq) { mesh_cq.finish(); }

}  // namespace tt::tt_metal::distributed
