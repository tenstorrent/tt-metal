// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
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
#include <unordered_set>
#include <llrt/tt_cluster.hpp>
#include <llrt/llrt.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace {

bool logical_cores_intersect(
    const std::vector<std::vector<tt::tt_metal::CoreCoord>>& previous_cores,
    const std::vector<std::vector<tt::tt_metal::CoreCoord>>& current_cores) {
    // Build a set from previous_cores only, then probe with current_cores directly.
    std::unordered_set<tt::tt_metal::CoreCoord> previous_cores_set;
    for (const auto& core_group : previous_cores) {
        previous_cores_set.insert(core_group.begin(), core_group.end());
    }
    for (const auto& core_group : current_cores) {
        for (const auto& core : core_group) {
            if (previous_cores_set.contains(core)) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

namespace tt::tt_metal::distributed {

SDMeshCommandQueue::SDMeshCommandQueue(
    MeshDevice* mesh_device,
    uint32_t id,
    std::function<std::lock_guard<std::mutex>()> lock_api_function,
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context) :
    MeshCommandQueueBase(
        mesh_device,
        id,
        create_passthrough_thread_pool(mesh_device->impl().get_context_id()),
        std::move(lock_api_function)),
    active_distributed_context_(std::move(distributed_context)) {
    // Init thread pool with all local devices for parallel dispatch.
    // One thread per device enables NUMA-aware CPU binding.
    auto local_devices = mesh_device_->get_devices();
    if (local_devices.size() > 1) {
        launch_thread_pool_ = create_device_bound_thread_pool(mesh_device_->impl().get_context_id(), local_devices);
    }
}

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
    if (this->get_target_device_type() == tt::TargetDevice::Mock) {
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
    if (this->get_target_device_type() == tt::TargetDevice::Mock) {
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

void SDMeshCommandQueue::dispatch_program(const MeshCoordinateRange& coord_range, Program& program, bool blocking) {
    const auto& program_cores = program.impl().logical_cores();

    // Collect local devices for this program, handling async idle checks
    std::vector<IDevice*> local_devices;
    for (const auto& coord : coord_range) {
        if (!mesh_device_->impl().is_local(coord)) {
            continue;
        }
        auto* device = mesh_device_->impl().get_device(coord);
        bool need_wait = false;
        std::vector<std::vector<CoreCoord>> cores_to_wait;
        ChipId device_id = 0;
        {
            std::lock_guard<std::mutex> guard(logical_cores_mutex_);
            if (asynchronous_slow_dispatch_enabled_) {
                auto it = logical_cores_for_previous_workload_.find(device->id());
                if (it != logical_cores_for_previous_workload_.end()) {
                    const auto& previous_cores = it->second;
                    if (logical_cores_intersect(previous_cores, program_cores)) {
                        // Store the data so the thread does waiting after exiting
                        // the critical section
                        need_wait = true;
                        cores_to_wait = previous_cores;
                        device_id = device->id();
                        logical_cores_for_previous_workload_.erase(device_id);
                    }
                }
            }
        }

        if (need_wait) {
            tt::llrt::internal_::wait_for_idle(device_id, cores_to_wait);
        }

        local_devices.push_back(device);
    }

    if (local_devices.empty()) {
        return;
    }

    // First device: full LaunchProgram (compiles, finalizes, allocates CBs, dispatches)
    tt_metal::detail::LaunchProgram(local_devices[0], program, false);

    // Remaining devices: dispatch pre-compiled binary only.
    // TODO: This loop can be parallelized with a inner thread loop
    // since 1 program can span multiple devices on different PCIe links.
    // For 1:1 program-to-device mapping, this loop is empty.
    for (size_t i = 1; i < local_devices.size(); i++) {
        tt_metal::detail::DispatchCompiledProgramToDevice(local_devices[i], program);
    }

    if (blocking) {
        // Can be parallelized: wait across all devices
        for (auto* device : local_devices) {
            tt_metal::detail::WaitProgramDone(device, program);
        }
    } else {
        {
            std::lock_guard<std::mutex> guard(logical_cores_mutex_);
            for (auto* device : local_devices) {
                if (!asynchronous_slow_dispatch_enabled_ ||
                    !logical_cores_for_previous_workload_.contains(device->id())) {
                    logical_cores_for_previous_workload_[device->id()] = program_cores;
                } else {
                    // Device had active cores before this program was launched
                    // Merge the active cores from the previous program with the active cores from the current
                    // program
                    const auto& hal = tt::tt_metal::MetalContext::instance(mesh_device_->impl().get_context_id()).hal();
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

void SDMeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    if (this->get_target_device_type() == tt::TargetDevice::Mock) {
        return;  // Skip workload execution for mock devices
    }

    auto lock = lock_api_function_();

    if (!asynchronous_slow_dispatch_enabled_) {
        wait_for_cores_idle();
    }

    auto& range_program_map = mesh_workload.get_programs();

    if (launch_thread_pool_) {
        // Dispatch programs in parallel
        for (auto& [coord_range, program] : range_program_map) {
            // Find first local device for thread binding
            IDevice* device = nullptr;
            for (const auto& coord : coord_range) {
                if (mesh_device_->impl().is_local(coord)) {
                    device = mesh_device_->impl().get_device(coord);
                    break;
                }
            }
            if (!device) {
                continue;  // No local work for this host
            }
            auto* program_ptr = &program;
            launch_thread_pool_->enqueue(
                [this, coord_range, program_ptr, blocking]() { dispatch_program(coord_range, *program_ptr, blocking); },
                device->id());
        }
        launch_thread_pool_->wait();
    } else {
        // Single device: sequential launch
        for (auto& [coord_range, program] : range_program_map) {
            dispatch_program(coord_range, program, blocking);
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

void SDMeshCommandQueue::enqueue_wait_for_event(const MeshEvent&) {
    auto lock = lock_api_function_();
    wait_for_cores_idle();
}

void SDMeshCommandQueue::finish(tt::stl::Span<const SubDeviceId>) {
    if (this->get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }
    auto lock = lock_api_function_();
    wait_for_cores_idle();
    for (const auto& device : mesh_device_->get_devices()) {
        tt::tt_metal::MetalContext::instance(mesh_device_->impl().get_context_id())
            .get_cluster()
            .dram_barrier(device->id());
        tt::tt_metal::MetalContext::instance(mesh_device_->impl().get_context_id())
            .get_cluster()
            .l1_barrier(device->id());
    }

    // Barrier across all active hosts of the mesh
    active_distributed_context_->barrier();
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
    auto lock = lock_api_function_();
    wait_for_cores_idle();
    asynchronous_slow_dispatch_enabled_ = false;
}

}  // namespace tt::tt_metal::distributed
