// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt_stl/tt_pause.hpp>
#include <tt-metalium/distributed.hpp>
#include <utility>

#include "device.hpp"
#include "mesh_device.hpp"
#include "mesh_device_impl.hpp"
#include "mesh_trace.hpp"
#include "mesh_workload_impl.hpp"
#include "tt-metalium/program.hpp"
#include "dispatch/system_memory_manager.hpp"

namespace tt::tt_metal::distributed {

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    // Short-circuit for inactive MeshDevices (no-op)
    if (mesh_cq.device()->get_view().get_devices().empty()) {
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
        // In multi-host meshes, event.device_range() spans all coordinates including remote
        // ranks. Remote devices are not accessible from this host — the remote rank runs its
        // own EventSynchronize and manages its local CQs. Skip non-local coordinates to
        // avoid TT_FATAL("Cannot get device for remote device") in MeshDeviceViewImpl::get_device.
        if (!event.device()->impl().is_local(coord)) {
            continue;
        }
        auto* physical_device = event.device()->impl().get_device(coord);
        auto& sysmem = physical_device->sysmem_manager();
        const auto cq_id = event.mesh_cq_id();
        const auto target_id = event.id();
        // If the device has been quiesced since this event was recorded, all in-flight
        // events are implicitly complete (finish_and_reset_in_use drained the CQ).
        if (sysmem.is_quiesced(cq_id)) {
            continue;
        }
        ttsl::nice_spin_until([&sysmem, cq_id, target_id] {
            return sysmem.is_quiesced(cq_id) || sysmem.get_last_completed_event(cq_id) >= target_id;
        });
    }
}

bool EventQuery(const MeshEvent& event) {
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        return true;
    }
    bool event_completed = true;
    for (const auto& coord : event.device_range()) {
        // Skip remote coordinates — see comment in EventSynchronize above.
        if (!event.device()->impl().is_local(coord)) {
            continue;
        }
        auto* physical_device = event.device()->impl().get_device(coord);
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
    using multihost::DistributedContext;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    return DistributedContext::is_initialized() && *(distributed_context.size()) > 1;
}

}  // namespace tt::tt_metal::distributed
