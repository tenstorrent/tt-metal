// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>
#include <tt-metalium/distributed.hpp>
#include <utility>

#include "device.hpp"
#include "mesh_device.hpp"
#include "mesh_device_impl.hpp"
#include "mesh_trace.hpp"
#include "mesh_workload_impl.hpp"
#include "tt-metalium/program.hpp"
#include "dispatch/system_memory_manager.hpp"
#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal::distributed {

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    // Short-circuit for inactive MeshDevices (no-op)
    if (mesh_cq.device()->get_view().get_devices().empty()) {
        return;
    }

    // Route programs targeting claimed service cores to the SD path. Done here, not in add_program,
    // because the physical device needed to device-scope the service-core check is only known at
    // enqueue. Common case - no service claimed so skip entirely.
    auto& svc = internal::ServiceCoreManager::get();
    auto& programs = mesh_workload.impl().get_programs();
    auto& service_programs = mesh_workload.impl().get_service_programs();
    if (svc.has_any_claims()) {
        for (auto it = programs.begin(); it != programs.end();) {
            bool targets_service_core = false;
            for (const auto& coord : it->first) {
                auto* device = mesh_cq.device()->impl().get_device(coord);
                if (device == nullptr) {
                    continue;
                }
                for (const auto& per_type : it->second.impl().logical_cores()) {
                    for (const auto& core : per_type) {
                        targets_service_core |= svc.is_service_core(device->id(), core);
                    }
                }
            }
            if (targets_service_core) {
                service_programs[it->first] = std::move(it->second);
                it = programs.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Service programs are dispatched via SD, independently of the FD command queue.
    for (auto& [device_range, program] : service_programs) {
        for (const auto& coord : device_range) {
            auto* device = mesh_cq.device()->impl().get_device(coord);
            TT_FATAL(
                device != nullptr,
                "EnqueueMeshWorkload: service program targets mesh coordinate {} with no local device",
                coord);
            for (const auto& per_type : program.impl().logical_cores()) {
                for (const auto& core : per_type) {
                    // Every core must be claimed on the device it launches on, else an SD service
                    // kernel would collide with FD using that core on this device.
                    TT_FATAL(
                        svc.is_service_core(device->id(), core),
                        "Service program targets core {} on device {} where it is not claimed. "
                        "Claim service cores on every device in the workload's range.",
                        core,
                        device->id());
                    svc.mark_launched(device->id(), core);  // launch-once
                }
            }
            tt::tt_metal::detail::LaunchProgram(device, program, false, true);
        }
    }

    if (programs.empty()) {
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
        auto* physical_device = event.device()->impl().get_device(coord);
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
