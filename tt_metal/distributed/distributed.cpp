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

void EnqueueRecordEvent(
    MeshCommandQueue& mesh_cq,
    const std::shared_ptr<MeshEvent>& event,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const std::optional<LogicalDeviceRange>& device_range) {
    mesh_cq.enqueue_record_event(event, sub_device_ids, device_range);
}

void EnqueueRecordEventToHost(
    MeshCommandQueue& mesh_cq,
    const std::shared_ptr<MeshEvent>& event,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const std::optional<LogicalDeviceRange>& device_range) {
    mesh_cq.enqueue_record_event_to_host(event, sub_device_ids, device_range);
}

void EnqueueWaitForEvent(MeshCommandQueue& mesh_cq, const std::shared_ptr<MeshEvent>& event) {
    mesh_cq.enqueue_wait_for_event(event);
}

void EventSynchronize(const std::shared_ptr<MeshEvent>& event) {
    auto& mesh_cq = event->device->mesh_command_queue(event->cq_id);
    mesh_cq.drain_events_from_completion_queue();
    mesh_cq.verify_reported_events_after_draining(event);
}

void Finish(MeshCommandQueue& mesh_cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    mesh_cq.finish(sub_device_ids);
}

}  // namespace tt::tt_metal::distributed
