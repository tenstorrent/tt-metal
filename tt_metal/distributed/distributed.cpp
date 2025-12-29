// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    auto* mesh_device = mesh_cq.device();
    if (mesh_device->is_parent_mesh()) {
        auto submeshes = mesh_device->get_submeshes();
        // Only route to submeshes that have programs in the workload
        for (auto& submesh : submeshes) {
            // Check if this submesh has any programs in the workload
            // Use get_submesh_for_coordinate to check if coordinates belong to this submesh
            bool has_programs_for_submesh = false;
            for (const auto& [device_range, program] : mesh_workload.get_programs()) {
                for (const auto& coord : device_range) {
                    auto submesh_for_coord = mesh_device->get_submesh_for_coordinate(coord);
                    if (submesh_for_coord && submesh_for_coord.get() == submesh.get()) {
                        has_programs_for_submesh = true;
                        break;
                    }
                }
                if (has_programs_for_submesh) {
                    break;
                }
            }

            // Only route to submeshes that have matching programs
            if (has_programs_for_submesh) {
                for (uint8_t cq_id = 0; cq_id < submesh->num_hw_cqs(); ++cq_id) {
                    auto& submesh_cq = submesh->mesh_command_queue(cq_id);
                    EnqueueMeshWorkload(submesh_cq, mesh_workload, blocking);
                }
            }
        }
        return;
    }
    // Check if this submesh has any programs in the workload
    bool has_programs_for_device = false;
    if (mesh_device->get_parent_mesh()) {
        // This is a submesh - check if any programs belong to it
        auto* parent_mesh = mesh_device->get_parent_mesh().get();
        for (const auto& [device_range, program] : mesh_workload.get_programs()) {
            for (const auto& coord : device_range) {
                auto submesh_for_coord = parent_mesh->get_submesh_for_coordinate(coord);
                if (submesh_for_coord && submesh_for_coord.get() == mesh_device) {
                    has_programs_for_device = true;
                    break;
                }
            }
            if (has_programs_for_device) {
                break;
            }
        }
    } else {
        // Parent mesh - always has programs
        has_programs_for_device = !mesh_workload.get_programs().empty();
    }

    // Only process if there are programs for this device
    if (!has_programs_for_device) {
        return;
    }

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        mesh_workload.impl().compile(mesh_cq.device());
        mesh_workload.impl().load_binaries(mesh_cq);
        mesh_workload.impl().generate_dispatch_commands(mesh_cq);
    }
    mesh_cq.enqueue_mesh_workload(mesh_workload, blocking);
}

void EventSynchronize(const MeshEvent& event) {
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        return;
    }
    for (const auto& coord : event.device_range()) {
        auto* physical_device = event.device()->get_device(coord);
        while (physical_device->sysmem_manager().get_last_completed_event(event.mesh_cq_id()) < event.id()) {
            ;
        }
    }
}

bool EventQuery(const MeshEvent& event) {
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        return true;
    }
    bool event_completed = true;
    for (const auto& coord : event.device_range()) {
        auto* physical_device = event.device()->get_device(coord);
        event_completed &= physical_device->sysmem_manager().get_last_completed_event(event.mesh_cq_id()) >= event.id();
    }
    return event_completed;
}

MeshTraceId BeginTraceCapture(MeshDevice* device, uint8_t cq_id) {
    auto trace_id = MeshTrace::next_id();
    device->begin_mesh_trace(cq_id, trace_id);
    return trace_id;
}

void Synchronize(MeshDevice* device, std::optional<uint8_t> cq_id, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    if (!device->is_initialized()) {
        return;
    }
    if (cq_id.has_value()) {
        device->mesh_command_queue(cq_id).finish(sub_device_ids);
    } else {
        for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); ++cq_id) {
            device->mesh_command_queue(cq_id).finish(sub_device_ids);
        }
    }
}

void Finish(MeshCommandQueue& mesh_cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    mesh_cq.finish(sub_device_ids);
}

bool UsingDistributedEnvironment() {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    return distributed_context.is_initialized() && *(distributed_context.size()) > 1;
}

}  // namespace tt::tt_metal::distributed
