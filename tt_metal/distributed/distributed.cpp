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

MeshWorkload CreateMeshWorkload() { return MeshWorkload(); }

void AddProgramToMeshWorkload(MeshWorkload& mesh_workload, Program&& program, const MeshCoordinateRange& device_range) {
    mesh_workload.add_program(device_range, std::move(program));
}

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    if (mesh_cq.device()->using_fast_dispatch()) {
        mesh_workload.impl().compile(mesh_cq.device());
        mesh_workload.impl().load_binaries(mesh_cq);
        mesh_workload.impl().generate_dispatch_commands(mesh_cq);
    }
    mesh_cq.enqueue_mesh_workload(mesh_workload, blocking);
}

MeshEvent EnqueueRecordEvent(
    MeshCommandQueue& mesh_cq,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const std::optional<MeshCoordinateRange>& device_range) {
    return mesh_cq.enqueue_record_event(sub_device_ids, device_range);
}

MeshEvent EnqueueRecordEventToHost(
    MeshCommandQueue& mesh_cq,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const std::optional<MeshCoordinateRange>& device_range) {
    return mesh_cq.enqueue_record_event_to_host(sub_device_ids, device_range);
}

void EnqueueWaitForEvent(MeshCommandQueue& mesh_cq, const MeshEvent& event) { mesh_cq.enqueue_wait_for_event(event); }

void EventSynchronize(const MeshEvent& event) {
    if (event.device()->using_slow_dispatch()) {
        return;
    }
    for (const auto& coord : event.device_range()) {
        auto physical_device = event.device()->get_device(coord);
        while (physical_device->sysmem_manager().get_last_completed_event(event.mesh_cq_id()) < event.id());
    }
}

bool EventQuery(const MeshEvent& event) {
    const auto& mesh_device = event.device();
    const auto& device_range = event.device_range();
    auto& sysmem_manager = mesh_device->get_device(*(device_range.begin()))->sysmem_manager();
    bool event_completed = sysmem_manager.get_last_completed_event(event.mesh_cq_id()) >= event.id();
    return event_completed;
}

MeshTraceId BeginTraceCapture(MeshDevice* device, uint8_t cq_id) {
    auto trace_id = MeshTrace::next_id();
    device->begin_mesh_trace(cq_id, trace_id);
    return trace_id;
}

void EndTraceCapture(MeshDevice* device, uint8_t cq_id, const MeshTraceId& trace_id) {
    device->end_mesh_trace(cq_id, trace_id);
}

void ReplayTrace(MeshDevice* device, uint8_t cq_id, const MeshTraceId& trace_id, bool blocking) {
    device->replay_mesh_trace(cq_id, trace_id, blocking);
}

void ReleaseTrace(MeshDevice* device, const MeshTraceId& trace_id) { device->release_mesh_trace(trace_id); }

void Synchronize(MeshDevice* device, std::optional<uint8_t> cq_id, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    if (!device->is_initialized()) {
        return;
    }
    if (cq_id.has_value()) {
        device->mesh_command_queue(*cq_id).finish(sub_device_ids);
    } else {
        for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); ++cq_id) {
            device->mesh_command_queue(cq_id).finish(sub_device_ids);
        }
    }
}

void Finish(MeshCommandQueue& mesh_cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    mesh_cq.finish(sub_device_ids);
}

}  // namespace tt::tt_metal::distributed
