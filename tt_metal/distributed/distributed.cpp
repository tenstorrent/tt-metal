// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <utility>

#include "device.hpp"
#include "mesh_device.hpp"
#include "mesh_trace.hpp"
#include "mesh_workload_impl.hpp"
#include "mesh_workload_utils.hpp"
#include "tt-metalium/program.hpp"
#include "dispatch/system_memory_manager.hpp"

namespace tt::tt_metal::distributed {

namespace {

// Helper to compute the offset of a submesh relative to its parent mesh.
// The offset is the parent coordinate of the submesh's origin (0,0,...).
MeshCoordinate compute_submesh_offset(MeshDevice* parent_mesh, MeshDevice* submesh) {
    auto zero_coord = MeshCoordinate::zero_coordinate(submesh->shape().dims());
    auto* submesh_origin_device = submesh->get_device(zero_coord);
    return parent_mesh->get_view().find_device(submesh_origin_device->id());
}

// Helper to check if a parent coordinate falls within a submesh when translated by offset.
// Returns true if (parent_coord - offset) is a valid coordinate in mesh_shape.
bool is_coord_in_mesh_with_offset(
    const MeshCoordinate& parent_coord, const MeshCoordinate& offset, const MeshShape& mesh_shape) {
    if (parent_coord.dims() != offset.dims() || parent_coord.dims() != mesh_shape.dims()) {
        return false;
    }
    for (size_t i = 0; i < parent_coord.dims(); i++) {
        if (parent_coord[i] < offset[i]) {
            return false;
        }
        auto translated = parent_coord[i] - offset[i];
        if (translated >= mesh_shape[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

// Internal function that handles coordinate translation when routing to submeshes.
// The offset represents the submesh's origin in the parent mesh's coordinate system.
// Seems inefficient because this is called for every CQ in a submesh, so the check will be the same for every CQ.
void EnqueueMeshWorkloadWithOffset(
    MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking, const MeshCoordinate& offset) {
    auto* mesh_device = mesh_cq.device();

    bool has_programs_for_device = false;
    for (const auto& [device_range, program] : mesh_workload.get_programs()) {
        for (const auto& parent_coord : device_range) {
            if (is_coord_in_mesh_with_offset(parent_coord, offset, mesh_device->shape())) {
                has_programs_for_device = true;
                break;
            }
        }
        if (has_programs_for_device) {
            break;
        }
    }

    if (!has_programs_for_device) {
        return;
    }

    if (mesh_device->is_parent_mesh()) {
        auto submeshes = mesh_device->get_submeshes();

        std::unordered_set<int> all_submesh_devices;
        for (const auto& submesh : submeshes) {
            for (auto device_id : submesh->get_device_ids()) {
                all_submesh_devices.insert(device_id);
            }
        }

        size_t num_program_devices = 0;
        size_t num_program_devices_in_submeshes = 0;
        for (const auto& [device_range, program] : mesh_workload.get_programs()) {
            for (const auto& coord : device_range) {
                // Skip coordinates that don't fall within this mesh
                if (!is_coord_in_mesh_with_offset(coord, offset, mesh_device->shape())) {
                    continue;
                }
                num_program_devices++;
                auto translated_coord = coord.translate(offset, false);
                if (all_submesh_devices.contains(mesh_device->get_device(translated_coord)->id())) {
                    num_program_devices_in_submeshes++;
                }
            }
        }

        TT_FATAL(
            num_program_devices == num_program_devices_in_submeshes,
            "Program targets {} devices but only {} are covered by submeshes. "
            "Some devices in the program's device range are not in any submesh.",
            num_program_devices,
            num_program_devices_in_submeshes);

        // Route to submeshes, passing the offset for coordinate translation
        for (const auto& submesh : submeshes) {
            auto submesh_offset = offset.translate(compute_submesh_offset(mesh_device, submesh.get()));
            for (uint8_t i = 0; i < submesh->num_hw_cqs(); ++i) {
                auto& submesh_cq = submesh->mesh_command_queue(i);
                EnqueueMeshWorkloadWithOffset(submesh_cq, mesh_workload, blocking, submesh_offset);
            }
        }

        return;
    }

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        mesh_workload.impl().compile(mesh_cq.device());
        mesh_workload.impl().load_binaries(mesh_cq, offset);
        mesh_workload.impl().generate_dispatch_commands(mesh_cq);
    }
    mesh_cq.enqueue_mesh_workload(mesh_workload, blocking, offset);
}

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    EnqueueMeshWorkloadWithOffset(
        mesh_cq, mesh_workload, blocking, MeshCoordinate::zero_coordinate(mesh_cq.device()->shape().dims()));
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
