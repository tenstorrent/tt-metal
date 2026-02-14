// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dummy_mesh_command_queue.hpp"
#include "tt_metal/common/thread_pool.hpp"
#include <mesh_device.hpp>
#include <mesh_event.hpp>
#include <mesh_workload.hpp>

namespace tt::tt_metal::distributed {

DummyMeshCommandQueue::DummyMeshCommandQueue(
    MeshDevice* mesh_device, uint32_t id, std::function<std::lock_guard<std::mutex>()> lock_api_function) :
    MeshCommandQueueBase(mesh_device, id, create_passthrough_thread_pool(), std::move(lock_api_function)) {}

std::optional<MeshTraceId> DummyMeshCommandQueue::trace_id() const { return std::nullopt; }

WorkerConfigBufferMgr& DummyMeshCommandQueue::get_config_buffer_mgr(uint32_t /*index*/) {
    TT_THROW("get_config_buffer_mgr() not supported for DummyMeshCommandQueue (inactive rank)");
}

bool DummyMeshCommandQueue::write_shard_to_device(
    const MeshBuffer& /*buffer*/,
    const MeshCoordinate& /*device_coord*/,
    const void* /*src*/,
    const std::optional<BufferRegion>& /*region*/,
    tt::stl::Span<const SubDeviceId> /*sub_device_ids*/,
    std::shared_ptr<experimental::PinnedMemory> /*pinned_memory*/) {
    // No-op for inactive rank; no pinned memory used
    return false;
}

void DummyMeshCommandQueue::read_shard_from_device(
    const MeshBuffer& /*buffer*/,
    const MeshCoordinate& /*device_coord*/,
    void* /*dst*/,
    std::shared_ptr<experimental::PinnedMemory> /*pinned_memory*/,
    const std::optional<BufferRegion>& /*region*/,
    std::unordered_map<IDevice*, uint32_t>& /*num_txns_per_device*/,
    tt::stl::Span<const SubDeviceId> /*sub_device_ids*/) {
    // No-op for inactive rank
}

void DummyMeshCommandQueue::submit_memcpy_request(
    std::unordered_map<IDevice*, uint32_t>& /*num_txns_per_device*/, bool /*blocking*/) {
    // No-op for inactive rank
}

void DummyMeshCommandQueue::finish_nolock(tt::stl::Span<const SubDeviceId> /*sub_device_ids*/) {
    // No-op for inactive rank
}

MeshEvent DummyMeshCommandQueue::enqueue_record_event_to_host_nolock(
    tt::stl::Span<const SubDeviceId> /*sub_device_ids*/, const std::optional<MeshCoordinateRange>& device_range) {
    // Return dummy event for inactive rank
    return MeshEvent(0, mesh_device_, id_, device_range.value_or(MeshCoordinateRange(mesh_device_->shape())));
}

void DummyMeshCommandQueue::enqueue_mesh_workload(MeshWorkload& /*mesh_workload*/, bool /*blocking*/) {
    // No-op for inactive rank
}

MeshEvent DummyMeshCommandQueue::enqueue_record_event(
    tt::stl::Span<const SubDeviceId> /*sub_device_ids*/, const std::optional<MeshCoordinateRange>& device_range) {
    // Return dummy event for inactive rank
    return MeshEvent(0, mesh_device_, id_, device_range.value_or(MeshCoordinateRange(mesh_device_->shape())));
}

MeshEvent DummyMeshCommandQueue::enqueue_record_event_to_host(
    tt::stl::Span<const SubDeviceId> sub_device_ids, const std::optional<MeshCoordinateRange>& device_range) {
    // Call the no-lock version since we don't need synchronization for inactive rank
    return this->enqueue_record_event_to_host_nolock(sub_device_ids, device_range);
}

void DummyMeshCommandQueue::enqueue_wait_for_event(const MeshEvent& /*sync_event*/) {
    // No-op for inactive rank
}

void DummyMeshCommandQueue::finish(tt::stl::Span<const SubDeviceId> /*sub_device_ids*/) {
    // No-op for inactive rank
}

void DummyMeshCommandQueue::reset_worker_state(
    bool /*reset_launch_msg_state*/,
    uint32_t /*num_sub_devices*/,
    const vector_aligned<uint32_t>& /*go_signal_noc_data*/,
    const std::vector<std::pair<CoreRangeSet, uint32_t>>& /*core_go_message_mapping*/) {
    // No-op for inactive rank
}

void DummyMeshCommandQueue::record_begin(
    const MeshTraceId& /*trace_id*/, const std::shared_ptr<MeshTraceDescriptor>& /*ctx*/) {
    TT_THROW("Trace operations not supported for DummyMeshCommandQueue (inactive rank)");
}

void DummyMeshCommandQueue::record_end() {
    TT_THROW("Trace operations not supported for DummyMeshCommandQueue (inactive rank)");
}

void DummyMeshCommandQueue::enqueue_trace(const MeshTraceId& /*trace_id*/, bool /*blocking*/) {
    TT_THROW("Trace operations not supported for DummyMeshCommandQueue (inactive rank)");
}

}  // namespace tt::tt_metal::distributed
