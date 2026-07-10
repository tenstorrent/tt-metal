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
#include "impl/internal/service/service_core_manager_impl.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::distributed {

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    // Short-circuit for inactive MeshDevices (no-op)
    if (mesh_cq.device()->get_view().get_devices().empty()) {
        return;
    }

    // Route service workloads to the SD path. Done here, not in add_program, because the physical
    // device needed to device-scope the service-core check is only known at enqueue. Common case -
    // no service claimed so skip entirely.
    //
    // No-mixing contract: a workload is entirely a service workload (every program on claimed service
    // cores) or entirely a normal one.
    auto& svc = tt::tt_metal::MetalContext::instance().get_service_core_manager();
    auto& programs = mesh_workload.impl().get_programs();
    if (svc.impl().has_any_claims()) {
        // Classify the workload as service vs normal once and cache it on the workload, so steady-state
        // re-enqueues skip the O(programs*coords*cores) scan (see is_service_workload_ for why
        // classify-once stays correct across re-enqueues).
        auto& is_service_workload = mesh_workload.impl().is_service_workload_;
        if (!is_service_workload.has_value()) {
            bool saw_service = false;
            bool saw_normal = false;
            for (auto& [device_range, program] : programs) {
                const auto logical_cores = program.impl().logical_cores();
                size_t service_cores = 0;
                size_t total_cores = 0;
                for (const auto& coord : device_range) {
                    auto* device = mesh_cq.device()->impl().get_device(coord);
                    if (device == nullptr) {
                        continue;
                    }
                    for (const auto& per_type : logical_cores) {
                        for (const auto& core : per_type) {
                            ++total_cores;
                            if (svc.impl().is_service_core(device->id(), core)) {
                                ++service_cores;
                            }
                        }
                    }
                }
                // Level 1: a program targets only claimed service cores or only worker-grid cores, never a
                // mix. This also catches a core claimed on some devices in the range but not others.
                TT_FATAL(
                    service_cores == 0 || service_cores == total_cores,
                    "MeshWorkload program spans both service and worker-grid cores ({}/{} placements are claimed "
                    "service cores). A program must target only claimed service cores (on every device in its "
                    "range) or only worker-grid cores.",
                    service_cores,
                    total_cores);
                const bool program_is_service = service_cores > 0;
                // Level 2: the workload is all-service or all-normal, not a mix of the two.
                saw_service |= program_is_service;
                saw_normal |= !program_is_service;
                TT_FATAL(
                    !(saw_service && saw_normal),
                    "MeshWorkload mixes service and normal programs. A workload must be entirely service (all "
                    "programs on claimed service cores) or entirely normal (all on the worker grid).");
            }
            is_service_workload = saw_service;
        }

        if (is_service_workload.value()) {
            // Service workload: dispatch each program via SD, bypassing FD. Re-confirm each core is still
            // a claimed service core (this loop runs every enqueue, unlike the cached scan): guards the
            // cached classification against a core released between enqueues, failing loudly instead of
            // SD-launching onto an unclaimed core.
            for (auto& [device_range, program] : programs) {
                for (const auto& coord : device_range) {
                    auto* device = mesh_cq.device()->impl().get_device(coord);
                    TT_FATAL(
                        device != nullptr,
                        "EnqueueMeshWorkload: service program targets mesh coordinate {} with no local device",
                        coord);
                    for (const auto& per_type : program.impl().logical_cores()) {
                        for (const auto& core : per_type) {
                            TT_FATAL(
                                svc.impl().is_service_core(device->id(), core),
                                "EnqueueMeshWorkload: service workload targets core {} on device {} that is not a "
                                "claimed service core (released since a prior enqueue?).",
                                core,
                                device->id());
                            svc.impl().mark_launched(device->id(), core);  // launch-once
                        }
                    }
                    tt::tt_metal::detail::LaunchProgram(device, program, false, true);
                }
            }
            return;
        }
    }

    auto& ctx = tt::tt_metal::MetalContext::instance();
    if (ctx.rtoptions().get_fast_dispatch()) {
        mesh_workload.impl().compile(mesh_cq.device());
        mesh_workload.impl().load_binaries(mesh_cq);
        mesh_workload.impl().generate_dispatch_commands(mesh_cq);
    } else if (ctx.get_cluster().is_mock_or_emulated()) {
        // Slow dispatch normally JIT-compiles inside LaunchProgram, but the SD mesh CQ
        // short-circuits for mock devices (no hardware to dispatch to). Compile here so
        // kernel artifacts are still produced; the SD CQ then no-ops the skipped dispatch.
        mesh_workload.impl().compile(mesh_cq.device());
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

void Synchronize(MeshDevice* device, std::optional<uint8_t> cq_id, ttsl::Span<const SubDeviceId> sub_device_ids) {
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

void Finish(MeshCommandQueue& mesh_cq, ttsl::Span<const SubDeviceId> sub_device_ids) {
    mesh_cq.finish(sub_device_ids);
}

bool UsingDistributedEnvironment() {
    using multihost::DistributedContext;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    return DistributedContext::is_initialized() && *(distributed_context.size()) > 1;
}

}  // namespace tt::tt_metal::distributed
