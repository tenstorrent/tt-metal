// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <utility>

#include "device.hpp"
#include "mesh_device.hpp"
#include "mesh_trace.hpp"
#include "mesh_workload_impl.hpp"
#include "tt-metalium/program.hpp"
#include "dispatch/system_memory_manager.hpp"

namespace tt::tt_metal::distributed {

// TODO: EnqueueMeshWorkload has complex compilation logic that needs careful migration.
// This function should eventually be moved as a method or handled differently.
void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        mesh_workload.impl().compile(mesh_cq.device());
        mesh_workload.impl().load_binaries(mesh_cq);
        mesh_workload.impl().generate_dispatch_commands(mesh_cq);
    }
    mesh_cq.enqueue_mesh_workload(mesh_workload, blocking);
}

// TODO: This utility function is used by Python bindings.
// Consider moving to a more appropriate location.
bool UsingDistributedEnvironment() {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    return distributed_context.is_initialized() && *(distributed_context.size()) > 1;
}

}  // namespace tt::tt_metal::distributed