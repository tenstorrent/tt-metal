// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sd_mesh_command_queue.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/common/thread_pool.hpp"
#include "tt_metal/impl/program/program_impl.hpp"
#include <mesh_device.hpp>
#include <mesh_event.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <utility>
#include <llrt/tt_cluster.hpp>
#include <llrt/llrt.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace {

bool logical_cores_intersect(
    const std::vector<std::vector<tt::tt_metal::CoreCoord>>& previous_cores,
    const std::vector<std::vector<tt::tt_metal::CoreCoord>>& current_cores) {
    std::unordered_set<tt::tt_metal::CoreCoord> previous_cores_set;
    std::unordered_set<tt::tt_metal::CoreCoord> current_cores_set;

    for (const auto& previous_core_group : previous_cores) {
        for (const auto& previous_core : previous_core_group) {
            previous_cores_set.insert(previous_core);
        }
    }
    for (const auto& current_core_group : current_cores) {
        for (const auto& current_core : current_core_group) {
            current_cores_set.insert(current_core);
        }
    }

    for (const auto& core : current_cores_set) {
        if (previous_cores_set.contains(core)) {
            return true;
        }
    }
    return false;
}

}  // namespace

namespace tt::tt_metal::distributed {

SDMeshCommandQueue::SDMeshCommandQueue(
    MeshDevice* mesh_device, uint32_t id, std::function<std::lock_guard<std::mutex>()> lock_api_function) :
    MeshCommandQueueBase(mesh_device, id, create_passthrough_thread_pool(), std::move(lock_api_function)) {}

std::optional<MeshTraceId> SDMeshCommandQueue::trace_id() const {
    TT_THROW("Trace not supported for slow dispatch");
    return std::nullopt;
}

bool SDMeshCommandQueue::write_shard_to_device(
    const MeshBuffer& buffer,
    const MeshCoordinate& device_coord,
    const void* src,
    const std::optional<BufferRegion>& region,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    std::shared_ptr<experimental::PinnedMemory> /* pinned_memory */) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return false;  // Skip hardware write for mock devices
    }
    // Wait for idle here to ensure that a previous program potentially using this address space
    // is complete.
    wait_for_cores_idle();

    auto* device_buffer = buffer.get_device_buffer(device_coord);
    auto region_value = region.value_or(BufferRegion(0, device_buffer->size()));
    auto shard_view = device_buffer->view(region_value);

    TT_FATAL(sub_device_ids.empty(), "Sub-device IDs are not supported for slow dispatch");
    if (tt::tt_metal::GraphTracker::instance().hook_write_to_device(&buffer)) {
        return false;
    }

    tt::tt_metal::detail::WriteToBuffer(
        *shard_view,
        tt::stl::Span<const uint8_t>(static_cast<const uint8_t*>(src) + region_value.offset, region_value.size));
    return false;  // Slow dispatch doesn't support pinned memory
}

void SDMeshCommandQueue::read_shard_from_device(
    const MeshBuffer& buffer,
    const MeshCoordinate& device_coord,
    void* dst,
    std::shared_ptr<experimental::PinnedMemory> /* pinned_memory */,
    const std::optional<BufferRegion>& region,
    std::unordered_map<IDevice*, uint32_t>&,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return;  // Skip hardware read for mock devices
    }
    // Wait for idle here to ensure that programs emitting this data are complete.
    wait_for_cores_idle();
    auto* device_buffer = buffer.get_device_buffer(device_coord);
    auto shard_view = device_buffer->view(region.value_or(BufferRegion(0, device_buffer->size())));

    TT_FATAL(sub_device_ids.empty(), "Sub-device IDs are not supported for slow dispatch");
    if (tt::tt_metal::GraphTracker::instance().hook_read_from_device(&buffer)) {
        return;
    }

    tt::tt_metal::detail::ReadFromBuffer(*shard_view, static_cast<uint8_t*>(dst));
}

void SDMeshCommandQueue::submit_memcpy_request(std::unordered_map<IDevice*, uint32_t>&, bool) {}

WorkerConfigBufferMgr& SDMeshCommandQueue::get_config_buffer_mgr(uint32_t /*index*/) {
    TT_THROW("Not supported for slow dispatch");
}

void SDMeshCommandQueue::wait_for_cores_idle() {
    if (!logical_cores_for_previous_workload_.empty()) {
        for (const auto& [device_id, logical_cores] : logical_cores_for_previous_workload_) {
            tt::llrt::internal_::wait_for_idle(device_id, logical_cores);
        }
        logical_cores_for_previous_workload_.clear();
    }
}

void SDMeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return;  // Skip workload execution for mock devices
    }

    auto lock = lock_api_function_();

    if (!asynchronous_slow_dispatch_enabled_) {
        wait_for_cores_idle();
    }

    for (auto& [coord_range, program] : mesh_workload.get_programs()) {
        const auto& program_cores = program.impl().logical_cores();
        for (const auto& coord : coord_range) {
            if (!mesh_device_->impl().is_local(coord)) {
                continue;
            }
            auto* device = mesh_device_->impl().get_device(coord);
            if (asynchronous_slow_dispatch_enabled_) {
                auto it = logical_cores_for_previous_workload_.find(device->id());
                if (it != logical_cores_for_previous_workload_.end()) {
                    const auto& previous_cores = it->second;
                    // Only block before launching the current program if the previous program used the same cores
                    if (logical_cores_intersect(previous_cores, program_cores)) {
                        tt::llrt::internal_::wait_for_idle(device->id(), previous_cores);
                        // Clear the active cores in use for this device, since we blocked
                        // on them
                        logical_cores_for_previous_workload_.erase(device->id());
                    }
                }
            }

            tt_metal::detail::LaunchProgram(device, program, false);
        }
    }

    for (auto& [coord_range, program] : mesh_workload.get_programs()) {
        for (const auto& coord : coord_range) {
            if (mesh_device_->impl().is_local(coord)) {
                auto* device = mesh_device_->impl().get_device(coord);
                if (blocking) {
                    tt_metal::detail::WaitProgramDone(device, program);
                } else {
                    if (!(asynchronous_slow_dispatch_enabled_ and
                          logical_cores_for_previous_workload_.contains(device->id()))) {
                        // Device had no active cores until this program was launched
                        logical_cores_for_previous_workload_[device->id()] = program.impl().logical_cores();
                    } else {
                        // Device had active cores before this program was launched
                        // Merge the active cores from the previous program with the active cores from the current
                        // program
                        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
                        auto program_cores = program.impl().logical_cores();
                        for (uint32_t core_type_index = 0; core_type_index < hal.get_programmable_core_type_count();
                             core_type_index++) {
                            auto& active_cores = logical_cores_for_previous_workload_[device->id()][core_type_index];
                            auto curr_active_cores = program_cores[core_type_index];
                            active_cores.insert(active_cores.end(), curr_active_cores.begin(), curr_active_cores.end());
                        }
                    }
                }
            }
        }
    }
}

MeshEvent SDMeshCommandQueue::enqueue_record_event(
    tt::stl::Span<const SubDeviceId>, const std::optional<MeshCoordinateRange>& device_range) {
    // No synchronization is needed for slow dispatch, returning a dummy value
    return MeshEvent(0, mesh_device_, id_, device_range.value_or(MeshCoordinateRange(mesh_device_->shape())));
}

MeshEvent SDMeshCommandQueue::enqueue_record_event_to_host_nolock(
    tt::stl::Span<const SubDeviceId>, const std::optional<MeshCoordinateRange>& device_range) {
    // No synchronization is needed for slow dispatch, returning a dummy value
    return MeshEvent(0, mesh_device_, id_, device_range.value_or(MeshCoordinateRange(mesh_device_->shape())));
}

MeshEvent SDMeshCommandQueue::enqueue_record_event_to_host(
    tt::stl::Span<const SubDeviceId> sub_device_ids, const std::optional<MeshCoordinateRange>& device_range) {
    // No synchronization is needed for slow dispatch, so we can call the non-locking version.
    return this->enqueue_record_event_to_host_nolock(sub_device_ids, device_range);
}

void SDMeshCommandQueue::enqueue_wait_for_event(const MeshEvent&) { wait_for_cores_idle(); }

void SDMeshCommandQueue::finish(tt::stl::Span<const SubDeviceId>) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }
    wait_for_cores_idle();
    for (const auto& device : mesh_device_->get_devices()) {
        tt::tt_metal::MetalContext::instance().get_cluster().dram_barrier(device->id());
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
    }

    // Barrier across all hosts of the mesh
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_distributed_context(
        mesh_device_->get_view().mesh_id());
    distributed_context->barrier();
}

void SDMeshCommandQueue::finish_nolock(tt::stl::Span<const SubDeviceId>) {}

void SDMeshCommandQueue::reset_worker_state(
    bool, uint32_t, const vector_aligned<uint32_t>&, const std::vector<std::pair<CoreRangeSet, uint32_t>>&) {}

void SDMeshCommandQueue::record_begin(const MeshTraceId&, const std::shared_ptr<MeshTraceDescriptor>&) {
    TT_THROW("Not supported for slow dispatch");
}

void SDMeshCommandQueue::record_end() { TT_THROW("Not supported for slow dispatch"); }

void SDMeshCommandQueue::enqueue_trace(const MeshTraceId&, bool) { TT_THROW("Not supported for slow dispatch"); }

void SDMeshCommandQueue::enable_asynchronous_slow_dispatch() { asynchronous_slow_dispatch_enabled_ = true; }

void SDMeshCommandQueue::disable_asynchronous_slow_dispatch() {
    wait_for_cores_idle();
    asynchronous_slow_dispatch_enabled_ = false;
}

}  // namespace tt::tt_metal::distributed
