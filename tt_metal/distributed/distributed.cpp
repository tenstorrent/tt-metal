// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt-metalium/distributed.hpp>
#include <utility>

#include "device.hpp"
#include "mesh_device.hpp"
#include "mesh_device_impl.hpp"
#include "mesh_trace.hpp"
#include "mesh_workload_impl.hpp"
#include <tt-metalium/internal/service/service_core_manager.hpp>
#include "tt-metalium/program.hpp"
#include "dispatch/system_memory_manager.hpp"
#include <tt-metalium/internal/service/service_core_manager.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal::distributed {

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    // Short-circuit for inactive MeshDevices (no-op)
    if (mesh_cq.device()->get_view().get_devices().empty()) {
        return;
    }

    // Service programs target FD-idle dispatch-column cores and are dispatched via SD MMIO,
    // independently of the FD command queue. Common case: no service programs, skip entirely.
    auto& service_programs = mesh_workload.impl().get_service_programs();
    if (!service_programs.empty()) {
        // Per-device claim-existence assert: every core that the program targets must be
        // claimed via ServiceCoreManager on each device in the program's device_range.
        // MeshWorkloadImpl::add_program already enforced the "no mix of service + worker
        // cores in one program" invariant using a cross-device check; here we additionally
        // verify that the claim is actually present on every device the program will land
        // on, catching cases where a CoreCoord is claimed on one device but not another.
        const auto& claims = tt::tt_metal::internal::ServiceCoreManager::get();
        for (auto& [device_range, program] : service_programs) {
            for (const auto& coord : device_range) {
                auto* device = mesh_cq.device()->impl().get_device(coord);
                const auto& dev_claimed = claims.claimed_cores(device->id());
                for (const auto& per_type : program.impl().logical_cores()) {
                    for (const auto& core : per_type) {
                        TT_FATAL(
                            dev_claimed.count(core) > 0,
                            "Service program targets core {} on device coord {} (id {}), but that "
                            "core is not claimed via ServiceCoreManager on that device. "
                            "Either claim it before submitting the workload or remove the kernel.",
                            core,
                            coord,
                            device->id());
                    }
                }
                tt::tt_metal::detail::LaunchProgram(device, program, false, true);
            }
        }
    }

    if (mesh_workload.impl().get_programs().empty()) {
        return;
    }

    // Catch the "forgot to claim" footgun before FD compile/dispatch: any program in
    // the worker-grid bucket must target only cores inside the compute-with-storage
    // grid on every device it will run on. A core outside that grid means the caller
    // either targeted a service core but didn't claim it (so MeshWorkload::add_program
    // saw `has_any_claims()==false` and routed to programs_ silently), or claimed it
    // AFTER add_program (routing decision is made at add time). Either way, fail loudly
    // here instead of letting FD compile try to handle an off-grid kernel.
    {
        const auto& claims = tt::tt_metal::internal::ServiceCoreManager::get();
        const auto& programs = mesh_workload.impl().get_programs();
        for (const auto& [device_range, program] : programs) {
            for (const auto& coord : device_range) {
                auto* device = mesh_cq.device()->impl().get_device(coord);
                const auto grid = device->compute_with_storage_grid_size();
                for (const auto& per_type : program.impl().logical_cores()) {
                    for (const auto& core : per_type) {
                        if (core.x < grid.x && core.y < grid.y) {
                            continue;
                        }
                        const bool claimed_now = claims.claimed_cores(device->id()).count(core) > 0;
                        TT_FATAL(
                            false,
                            "Worker-grid program targets core {} on device {} (coord {}), which is "
                            "outside the compute-with-storage grid ({}x{}). {} If this core should "
                            "be a service core, call ServiceCoreManager::claim() on it BEFORE "
                            "MeshWorkload::add_program — the routing decision is made at add time "
                            "and won't reconsider later.",
                            core,
                            device->id(),
                            coord,
                            grid.x,
                            grid.y,
                            claimed_now ? "It IS currently claimed via ServiceCoreManager, but the program was "
                                          "added before the claim, so it ended up in the worker-grid bucket."
                                        : "It is NOT claimed via ServiceCoreManager.");
                    }
                }
            }
        }
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
