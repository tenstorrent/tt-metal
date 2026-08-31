// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/mesh_trace.hpp>

#include <stdexcept>
#include <utility>

namespace tt::tt_metal::distributed::experimental {

class MeshTraceBuilder::Impl {};

MeshTraceBuilder::MeshTraceBuilder(MeshDevice& /*device*/) : impl_(std::make_unique<Impl>()) {}

MeshTraceBuilder::MeshTraceBuilder(MeshTraceBuilder&&) noexcept = default;

MeshTraceBuilder& MeshTraceBuilder::operator=(MeshTraceBuilder&&) noexcept = default;

MeshTraceBuilder::~MeshTraceBuilder() = default;

void MeshTraceBuilder::add(MeshWorkload& /*workload*/, const TraceParameters& /*parameters*/) {}

MeshTrace MeshTraceBuilder::build(MeshCommandQueue& /*cq*/) const {
    throw std::logic_error("MeshTraceBuilder::build is not implemented");
}

MeshDevice& MeshTraceBuilder::device() const {
    throw std::logic_error("MeshTraceBuilder::device is not implemented");
}

void MeshTraceBuilder::deallocate() {}

class MeshTrace::Impl {};

MeshTrace::MeshTrace(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

MeshTrace::MeshTrace(MeshTrace&&) noexcept = default;

MeshTrace& MeshTrace::operator=(MeshTrace&&) noexcept = default;

MeshTrace::~MeshTrace() = default;

void MeshTrace::update_args(const TraceArgPatch& /*patch*/) {}

MeshDevice& MeshTrace::device() const {
    throw std::logic_error("MeshTrace::device is not implemented");
}

uint8_t MeshTrace::cq_id() const {
    throw std::logic_error("MeshTrace::cq_id is not implemented");
}

void MeshTrace::deallocate() {}

void EnqueueMeshTrace(MeshCommandQueue& /*mesh_cq*/, MeshTrace& /*mesh_trace*/, bool /*blocking*/) {}

}  // namespace tt::tt_metal::distributed::experimental


#if 0

// All the trace code from everywhere else

// ============================================================================
// tt_metal/distributed/mesh_trace.hpp
// ============================================================================

// MeshTrace capture consists of 3 steps:
// 1. Staging: Workload dispatch commands are recorded into MeshTraceNodes.
// 2. Assembly: On trace end, dispatch commands are generated for all MeshTraceNodes and stored in a
// MeshTraceDescriptor.
// 3. Commit to Mesh: Write assembled trace to DRAM buffer.

struct MeshTraceData {
    MeshCoordinateRange device_range = MeshCoordinateRange(MeshShape(0, 0));
    std::vector<uint32_t> data;
};

class MeshTraceDescriptor {
public:
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> descriptors;
    std::vector<SubDeviceId> sub_device_ids;
    std::vector<MeshTraceData> ordered_trace_data;
    uint32_t total_trace_size = 0;
};

struct MeshTraceBuffer {
    std::shared_ptr<MeshTraceDescriptor> desc = nullptr;
    std::shared_ptr<MeshBuffer> mesh_buffer = nullptr;
    DeviceAddr dram_high_water_mark = 0;

    ~MeshTraceBuffer();
};

// ============================================================================
// tt_metal/distributed/mesh_trace.cpp
// ============================================================================

MeshTraceId MeshTrace::next_id() {
    static std::atomic<uint32_t> global_trace_id{0};
    return MeshTraceId(global_trace_id++);
}

std::shared_ptr<MeshTraceBuffer> MeshTrace::create_empty_mesh_trace_buffer() {
    return std::make_shared<MeshTraceBuffer>(std::make_shared<MeshTraceDescriptor>(), nullptr);
}

void MeshTrace::populate_mesh_buffer(
    MeshCommandQueue& mesh_cq,
    std::shared_ptr<MeshTraceBuffer>& trace_buffer,
    DeviceAddr dram_allocation_high_water_mark,
    DeviceAddr dram_deletion_high_water_mark,
    DeviceAddr max_live_trace_high_water_mark) {
    uint64_t unpadded_size = trace_buffer->desc->total_trace_size;
    size_t page_size = trace_dispatch::compute_interleaved_trace_buf_page_size(
        unpadded_size, mesh_cq.device()->allocator()->get_num_banks(BufferType::DRAM));
    size_t padded_size = round_up(unpadded_size, page_size);

    const auto current_trace_buffers_size = mesh_cq.device()->get_trace_buffers_size();
    mesh_cq.device()->set_trace_buffers_size(current_trace_buffers_size + padded_size);
    auto trace_region_size = mesh_cq.device()->allocator_impl()->get_config().trace_region_size;

    BufferType buffer_type = BufferType::TRACE;
    std::optional<bool> bottom_up = std::nullopt;

    if (trace_region_size == 0) {
        buffer_type = BufferType::DRAM;
        bottom_up = false;
    } else {
        TT_FATAL(
            mesh_cq.device()->get_trace_buffers_size() <= trace_region_size,
            "Creating trace buffers of size {}B on MeshDevice {}, but only {}B is allocated for trace region.",
            mesh_cq.device()->get_trace_buffers_size(),
            mesh_cq.device()->id(),
            trace_region_size);
    }

    DeviceLocalBufferConfig device_local_trace_buf_config = {
        .page_size = page_size,
        .buffer_type = buffer_type,
        .bottom_up = bottom_up,
    };

    ReplicatedBufferConfig global_trace_buf_config = {
        .size = padded_size,
    };

    {
        auto trace_storage_context = tt::tt_metal::make_allocation_context_guard("trace_storage");
        trace_buffer->mesh_buffer =
            MeshBuffer::create(global_trace_buf_config, device_local_trace_buf_config, mesh_cq.device());
    }

    trace_buffer->dram_high_water_mark = std::max(dram_allocation_high_water_mark, dram_deletion_high_water_mark);
    const DeviceAddr effective_high_water_mark =
        std::max({trace_buffer->dram_high_water_mark, max_live_trace_high_water_mark});
    if (trace_region_size == 0 && effective_high_water_mark > 0) {
        DeviceAddr trace_buffer_address = trace_buffer->mesh_buffer->address();
        if (trace_buffer_address < effective_high_water_mark) {
            bool allocation_overlap =
                dram_allocation_high_water_mark > 0 && trace_buffer_address < dram_allocation_high_water_mark;
            bool deletion_overlap =
                dram_deletion_high_water_mark > 0 && trace_buffer_address < dram_deletion_high_water_mark;

            if (allocation_overlap && deletion_overlap) {
                TT_FATAL(
                    false,
                    "Trace buffer at address {} overlaps with DRAM activity during trace capture. "
                    "Allocation high water mark: {}, Deletion high water mark: {}. "
                    "Reduce allocations during trace capture or set a non-zero trace_region_size.",
                    trace_buffer_address,
                    dram_allocation_high_water_mark,
                    dram_deletion_high_water_mark);
            } else if (deletion_overlap) {
                TT_FATAL(
                    false,
                    "Trace buffer at address {} overlaps with buffers deallocated during trace capture "
                    "(deletion high water mark: {}). "
                    "Avoid deallocating DRAM buffers during trace capture or set a non-zero trace_region_size.",
                    trace_buffer_address,
                    dram_deletion_high_water_mark);
            } else if (allocation_overlap) {
                TT_FATAL(
                    false,
                    "Trace buffer at address {} overlaps with buffers allocated during trace capture "
                    "(allocation high water mark: {}). "
                    "Reduce allocations during trace capture or set a non-zero trace_region_size.",
                    trace_buffer_address,
                    dram_allocation_high_water_mark);
            } else {
                TT_FATAL(
                    false,
                    "Trace buffer at address {} overlaps with DRAM activity captured by another live trace "
                    "(maximum live trace high water mark: {}). "
                    "Release the existing trace before capturing this trace or set a non-zero trace_region_size.",
                    trace_buffer_address,
                    max_live_trace_high_water_mark);
            }
        }
    }

    std::unordered_map<MeshCoordinateRange, uint32_t> write_offset_per_device_range = {};
    for (auto& mesh_trace_data : trace_buffer->desc->ordered_trace_data) {
        auto& device_range = mesh_trace_data.device_range;
        if (!write_offset_per_device_range.contains(device_range)) {
            write_offset_per_device_range.insert({device_range, 0});
        }
        std::vector<uint32_t> write_data = mesh_trace_data.data;
        auto unpadded_data_size = write_data.size() * sizeof(uint32_t);
        auto padded_data_size = round_up(unpadded_data_size, page_size);
        size_t numel_padding = (padded_data_size - unpadded_data_size) / sizeof(uint32_t);
        if (numel_padding > 0) {
            write_data.resize(write_data.size() + numel_padding, 0);
        }
        auto write_region =
            BufferRegion(write_offset_per_device_range.at(device_range), write_data.size() * sizeof(uint32_t));
        mesh_cq.enqueue_write_shard_to_sub_grid(
            *(trace_buffer->mesh_buffer), write_data.data(), device_range, true, write_region);
        write_offset_per_device_range.at(device_range) += mesh_trace_data.data.size() * sizeof(uint32_t);
    }
}

MeshTraceBuffer::~MeshTraceBuffer() {
    if (this->mesh_buffer && this->mesh_buffer->is_allocated() && this->mesh_buffer->device()) {
        auto current_trace_buffers_size = this->mesh_buffer->device()->get_trace_buffers_size();
        this->mesh_buffer->device()->set_trace_buffers_size(current_trace_buffers_size - this->mesh_buffer->size());
    }
}

// ============================================================================
// tt_metal/impl/sub_device/sub_device_manager.cpp
// ============================================================================

std::shared_ptr<MeshTraceBuffer>& SubDeviceManager::create_trace(const MeshTraceId& trace_id) {
    auto [trace, emplaced] =
        trace_buffer_pool_.emplace(trace_id, distributed::MeshTrace::create_empty_mesh_trace_buffer());
    TT_ASSERT(emplaced, "Trace buffer with trace id {} already exists", trace_id);
    return trace->second;
}

void SubDeviceManager::release_trace(const MeshTraceId& trace_id) { trace_buffer_pool_.erase(trace_id); }

std::shared_ptr<MeshTraceBuffer> SubDeviceManager::get_trace(const MeshTraceId& trace_id) {
    auto trace = trace_buffer_pool_.find(trace_id);
    if (trace != trace_buffer_pool_.end()) {
        return trace->second;
    }
    return nullptr;
}

DeviceAddr SubDeviceManager::get_max_trace_high_water_mark() const {
    DeviceAddr max_high_water_mark = 0;
    for (const auto& trace_entry : trace_buffer_pool_) {
        const auto& trace_buffer = trace_entry.second;
        max_high_water_mark = std::max(max_high_water_mark, trace_buffer->dram_high_water_mark);
    }
    return max_high_water_mark;
}

// tt_metal/impl/sub_device/sub_device_manager_tracker.cpp
DeviceAddr SubDeviceManagerTracker::get_max_trace_high_water_mark() const {
    DeviceAddr max_high_water_mark = 0;
    for (const auto& entry : sub_device_managers_) {
        max_high_water_mark = std::max(max_high_water_mark, entry.second->get_max_trace_high_water_mark());
    }
    return max_high_water_mark;
}

// ============================================================================
// tt_metal/distributed/mesh_device.cpp -- MeshDeviceImpl trace lifecycle
// ============================================================================

void MeshDeviceImpl::release_mesh_trace(const MeshTraceId& trace_id) {
    TracyTTMetalReleaseMeshTrace(this->get_device_ids(), *trace_id);

    validate_sub_device_manager_tracker();
    sub_device_manager_tracker_->get_active_sub_device_manager()->release_trace(trace_id);

    if (tensor_prefetcher_) {
        tensor_prefetcher_->release_trace(trace_id);
    }

    tt::tt_metal::experimental::inspector::ReleaseTraceDebugEntries(trace_id);

    this->unregister_active_trace(trace_id);
}

std::shared_ptr<MeshTraceBuffer> MeshDeviceImpl::get_mesh_trace(const MeshTraceId& trace_id) {
    validate_sub_device_manager_tracker();
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_trace(trace_id);
}

MeshTraceId MeshDeviceImpl::begin_mesh_trace(uint8_t cq_id) {
    auto trace_id = MeshTrace::next_id();
    this->begin_mesh_trace(cq_id, trace_id);
    return trace_id;
}

void MeshDeviceImpl::begin_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) {
    TracyTTMetalBeginMeshTrace(this->get_device_ids(), *trace_id);
    TT_FATAL(
        !this->mesh_command_queues_[cq_id]->trace_id().has_value(),
        "CQ {} is already being used for tracing tid {}",
        (uint32_t)cq_id,
        *trace_id);
    auto trace_region_size = this->allocator_impl()->get_config().trace_region_size;
    if (trace_region_size == 0) {
        this->allocator_impl()->begin_dram_high_water_mark_tracking();
    }

    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    TT_FATAL(
        active_sub_device_manager->get_trace(trace_id) == nullptr,
        "Trace already exists for tid {} on device {}'s active sub-device manager {}",
        *trace_id,
        this->mesh_id_,
        active_sub_device_manager->id());
    auto& trace_buffer = active_sub_device_manager->create_trace(trace_id);
    this->mesh_command_queues_[cq_id]->record_begin(trace_id, trace_buffer->desc);
}

void MeshDeviceImpl::end_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) {
    TracyTTMetalEndMeshTrace(this->get_device_ids(), *trace_id);

    auto register_trace_on_exit =
        ttsl::make_cleanup([this, trace_id]() { this->register_active_trace(trace_id); });

    TT_FATAL(
        this->mesh_command_queues_[cq_id]->trace_id() == trace_id,
        "CQ {} is not being used for tracing tid {}",
        (uint32_t)cq_id,
        trace_id);
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    auto trace_buffer = active_sub_device_manager->get_trace(trace_id);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on device {}'s active sub-device manager {}",
        *trace_id,
        this->mesh_id_,
        active_sub_device_manager->id());
    this->mesh_command_queues_[cq_id]->record_end();

    auto trace_region_size = this->allocator_impl()->get_config().trace_region_size;
    DeviceAddr dram_allocation_high_water_mark = 0;
    DeviceAddr dram_deletion_high_water_mark = 0;
    if (trace_region_size == 0) {
        this->allocator_impl()->end_dram_high_water_mark_tracking();
        dram_allocation_high_water_mark = this->allocator_impl()->get_dram_allocation_high_water_mark();
        dram_deletion_high_water_mark = this->allocator_impl()->get_dram_deletion_high_water_mark();
    }

    const DeviceAddr max_live_trace_high_water_mark = sub_device_manager_tracker_->get_max_trace_high_water_mark();
    MeshTrace::populate_mesh_buffer(
        *(mesh_command_queues_[cq_id]),
        trace_buffer,
        dram_allocation_high_water_mark,
        dram_deletion_high_water_mark,
        max_live_trace_high_water_mark);
}

void MeshDeviceImpl::replay_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id, bool blocking) {
    TTZoneScopedD(DISPATCH);
    TracyTTMetalReplayMeshTrace(this->get_device_ids(), *trace_id);
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    const auto& trace_buffer = active_sub_device_manager->get_trace(trace_id);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on Mesh device {}'s active sub-device manager {}",
        *trace_id,
        this->mesh_id_,
        *(active_sub_device_manager->id()));
    if (tensor_prefetcher_) {
        tensor_prefetcher_->replay_trace(trace_id);
    }
    mesh_command_queues_[cq_id]->enqueue_trace(trace_id, blocking);
}

uint32_t MeshDeviceImpl::get_trace_buffers_size() const { return trace_buffers_size_; }
void MeshDeviceImpl::set_trace_buffers_size(uint32_t size) { trace_buffers_size_ = size; }

// ============================================================================
// tt_metal/distributed/mesh_device.cpp -- public MeshDevice wrappers
// ============================================================================

MeshTraceId MeshDevice::begin_mesh_trace(MeshCommandQueue& cq) {
    TT_FATAL(cq.device() == this, "MeshCommandQueue belongs to a different MeshDevice");
    return pimpl_->begin_mesh_trace(static_cast<uint8_t>(cq.id()));
}
void MeshDevice::begin_mesh_trace(MeshCommandQueue& cq, const MeshTraceId& trace_id) {
    TT_FATAL(cq.device() == this, "MeshCommandQueue belongs to a different MeshDevice");
    pimpl_->begin_mesh_trace(static_cast<uint8_t>(cq.id()), trace_id);
}
void MeshDevice::end_mesh_trace(MeshCommandQueue& cq, const MeshTraceId& trace_id) {
    TT_FATAL(cq.device() == this, "MeshCommandQueue belongs to a different MeshDevice");
    pimpl_->end_mesh_trace(static_cast<uint8_t>(cq.id()), trace_id);
}
void MeshDevice::replay_mesh_trace(MeshCommandQueue& cq, const MeshTraceId& trace_id, bool blocking) {
    TT_FATAL(cq.device() == this, "MeshCommandQueue belongs to a different MeshDevice");
    pimpl_->replay_mesh_trace(static_cast<uint8_t>(cq.id()), trace_id, blocking);
}
MeshTraceId MeshDevice::begin_mesh_trace(uint8_t cq_id) { return pimpl_->begin_mesh_trace(cq_id); }
void MeshDevice::begin_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) {
    pimpl_->begin_mesh_trace(cq_id, trace_id);
}
void MeshDevice::end_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id) { pimpl_->end_mesh_trace(cq_id, trace_id); }
void MeshDevice::replay_mesh_trace(uint8_t cq_id, const MeshTraceId& trace_id, bool blocking) {
    pimpl_->replay_mesh_trace(cq_id, trace_id, blocking);
}
void MeshDevice::release_mesh_trace(const MeshTraceId& trace_id) { pimpl_->release_mesh_trace(trace_id); }
std::shared_ptr<MeshTraceBuffer> MeshDevice::get_mesh_trace(const MeshTraceId& trace_id) {
    return pimpl_->get_mesh_trace(trace_id);
}
uint32_t MeshDevice::get_trace_buffers_size() const { return pimpl_->get_trace_buffers_size(); }
void MeshDevice::set_trace_buffers_size(uint32_t size) { pimpl_->set_trace_buffers_size(size); }

// ============================================================================
// tt_metal/distributed/distributed.cpp -- EnqueueMeshWorkload trace path
// ============================================================================

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking) {
    if (mesh_cq.device()->get_view().get_devices().empty()) {
        return;
    }

    auto& ctx = tt::tt_metal::MetalContext::instance();
    if (ctx.rtoptions().get_fast_dispatch()) {
        mesh_workload.impl().compile(mesh_cq.device());
        mesh_workload.impl().load_binaries(mesh_cq);
        mesh_workload.impl().generate_dispatch_commands(mesh_cq);
    } else if (ctx.get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        mesh_workload.impl().compile(mesh_cq.device());
    }
    mesh_cq.enqueue_mesh_workload(mesh_workload, blocking);
}

MeshTraceId BeginTraceCapture(MeshDevice* device, uint8_t cq_id) {
    return device->begin_mesh_trace(device->mesh_command_queue(cq_id));
}

// tt_metal/distributed/mesh_workload.cpp -- load_binaries trace guard
if (max_kernel_bin_buf_size) {
    const bool is_capturing_trace = mesh_cq.trace_id().has_value();
    TT_FATAL(
        !is_capturing_trace,
        "Cannot load new binaries during trace capture."
        "This program is not yet in program cache. Warm up before capturing a trace."
        "See the operation's hash signature for arguments that must match.");
}

// ============================================================================
// tt_metal/distributed/fd_mesh_command_queue.hpp -- staged trace state
// ============================================================================

struct MeshTraceNode {
    std::vector<std::pair<MeshCoordinateRange, TraceNode>> trace_nodes;
    bool multicast_go_signals{false};
    bool unicast_go_signals{false};
    SubDeviceId sub_device_id;
};

std::optional<MeshTraceId> trace_id_;
std::shared_ptr<MeshTraceDescriptor> trace_ctx_;
std::vector<MeshTraceNode> trace_nodes_;

// ============================================================================
// tt_metal/distributed/fd_mesh_command_queue.cpp -- staging in enqueue_mesh_workload
// ============================================================================

TracyTTMetalEnqueueMeshWorkloadTrace(mesh_device_, mesh_workload, this->trace_id());

if (sysmem_manager.get_bypass_mode()) {
    TT_FATAL(!blocking, "Blocking is not supported when recording a trace.");
    trace_nodes_.push_back(MeshTraceNode{});
    auto& trace_node = trace_nodes_.back();
    bool use_prefetcher_cache = mesh_workload.impl().max_program_kernels_sizeB_ <= this->prefetcher_cache_sizeB_;
    for (auto& [device_range, program] : mesh_workload.get_programs()) {
        trace_node.trace_nodes.push_back(std::pair<MeshCoordinateRange, TraceNode>(
            device_range,
            program_dispatch::create_trace_node(program.impl(), mesh_device_, num_workers, use_prefetcher_cache)));
    }
    trace_node.multicast_go_signals = mcast_go_signals;
    trace_node.unicast_go_signals = unicast_go_signals;
    trace_node.sub_device_id = sub_device_id;
    return;
}

// ============================================================================
// tt_metal/distributed/fd_mesh_command_queue.cpp -- replay
// ============================================================================

void FDMeshCommandQueue::enqueue_trace(const MeshTraceId& trace_id, bool blocking) {
    auto lock = lock_api_function_();
    in_use_ = true;
    auto trace_inst = mesh_device_->get_mesh_trace(trace_id);
    auto descriptor = trace_inst->desc;
    auto buffer = trace_inst->mesh_buffer;
    uint32_t num_sub_devices = descriptor->sub_device_ids.size();
    auto& sub_device_cq_owner = cq_shared_state_->sub_device_cq_owner;
    for (auto sub_device_id : descriptor->sub_device_ids) {
        auto& sub_device = sub_device_cq_owner[*sub_device_id];
        sub_device.take_ownership(sub_device_id, this->id_);
    }

    auto cmd_sequence_sizeB = trace_dispatch::compute_trace_cmd_size(num_sub_devices);

    trace_dispatch::TraceDispatchMetadata dispatch_md(
        cmd_sequence_sizeB,
        descriptor->descriptors,
        descriptor->sub_device_ids,
        buffer->page_size(),
        buffer->num_pages(),
        buffer->address());

    for (auto* device : mesh_device_->get_devices()) {
        trace_dispatch::issue_trace_commands(
            mesh_device_, device->sysmem_manager(), dispatch_md, id_, expected_num_workers_completed_, dispatch_core_);
    }

    this->reset_prefetcher_cache_manager();

    trace_dispatch::update_worker_state_post_trace_execution(
        trace_inst->desc->descriptors,
        cq_shared_state_->worker_launch_message_buffer_state,
        config_buffer_mgr_,
        expected_num_workers_completed_);

    if (blocking) {
        this->finish_nolock();
    }
}

// ============================================================================
// tt_metal/distributed/fd_mesh_command_queue.cpp -- capture setup
// ============================================================================

void FDMeshCommandQueue::record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) {
    auto lock = lock_api_function_();
    trace_dispatch::reset_host_dispatch_state_for_trace(
        mesh_device_->num_sub_devices(),
        cq_shared_state_->worker_launch_message_buffer_state,
        expected_num_workers_completed_,
        config_buffer_mgr_,
        worker_launch_message_buffer_state_reset_,
        expected_num_workers_completed_reset_,
        config_buffer_mgr_reset_);

    trace_id_ = trace_id;
    trace_ctx_ = ctx;
    for (auto* device : mesh_device_->get_devices()) {
        device->sysmem_manager().set_bypass_mode(/*enable*/ true, /*clear*/ true);
    }

    swap(this->dummy_prefetcher_cache_manager_, this->prefetcher_cache_manager_);
}



// ============================================================================
// tt_metal/distributed/fd_mesh_command_queue.cpp -- trace assembly
// ============================================================================

template <typename VecIt, typename IndexIt>
static VecIt remove_by_index(VecIt begin, VecIt end, IndexIt index_begin, IndexIt index_end) {
    if (index_begin == index_end) {
        return end;
    }
    return std::remove_if(std::next(begin, *index_begin), end, [&](auto& value) {
        if (index_begin == index_end) {
            return false;
        }
        if (*index_begin == (&value - &*begin)) {
            ++index_begin;
            return true;
        }
        return false;
    });
}

void FDMeshCommandQueue::record_end() {
    const auto& hal = MetalContext::instance().hal();

    auto local_mesh_range = mesh_device_->get_view().get_local_mesh_coord_range();
    std::vector<MeshCoordinateRange> device_ranges{local_mesh_range};
    for (auto& trace_node : trace_nodes_) {
        for (auto& [device_range, program] : trace_node.trace_nodes) {
            auto local_device_range = local_mesh_range.intersection(device_range);
            if (!local_device_range.has_value()) {
                continue;
            }
            bool intersection_found = false;
            std::vector<size_t> device_range_idxs_to_invalidate;
            for (size_t i = 0; i < device_ranges.size(); i++) {
                auto& existing_range = device_ranges[i];
                TT_FATAL(
                    existing_range.dims() == local_device_range->dims(),
                    "Invalid mismatching dimensions for existing {} vs device range {}",
                    existing_range.dims(),
                    local_device_range->dims());
                if (existing_range.intersects(*local_device_range)) {
                    intersection_found = true;
                    auto intersection = *existing_range.intersection(*local_device_range);
                    if (intersection != existing_range) {
                        auto complement = subtract(existing_range, intersection);
                        device_range_idxs_to_invalidate.push_back(i);
                        for (const auto& complement_range : complement.ranges()) {
                            device_ranges.push_back(complement_range);
                        }
                        device_ranges.push_back(intersection);
                    }
                }
            }
            if (intersection_found) {
                if (!device_range_idxs_to_invalidate.empty()) {
                    device_ranges.erase(
                        remove_by_index(
                            device_ranges.begin(),
                            device_ranges.end(),
                            device_range_idxs_to_invalidate.begin(),
                            device_range_idxs_to_invalidate.end()),
                        device_ranges.end());
                }
            } else {
                device_ranges.push_back(*local_device_range);
            }
        }
    }
    std::vector<uint32_t> exec_buf_end = {};

    DeviceCommand command_sequence(MetalContext::instance().hal().get_alignment(HalMemType::HOST));
    command_sequence.add_prefetch_exec_buf_end();

    exec_buf_end.reserve(command_sequence.size_bytes() / sizeof(uint32_t));
    for (int i = 0; i < command_sequence.size_bytes() / sizeof(uint32_t); i++) {
        exec_buf_end.push_back(static_cast<uint32_t*>(command_sequence.data())[i]);
    }
    size_t max_trace_size = 0;
    std::set<SubDeviceId> sub_device_ids;
    std::optional<std::unordered_map<SubDeviceId, TraceWorkerDescriptor>> overall_trace_worker_descriptors;
    for (const auto& range : device_ranges) {
        std::vector<TraceNode*> trace_nodes;
        trace_nodes.reserve(trace_nodes_.size());
        std::vector<MeshTraceNode*> mesh_trace_nodes;
        mesh_trace_nodes.reserve(trace_nodes_.size());
        struct UnusedNodeData {
            uint32_t unused_nodes_both_multicast_and_unicast = 0;
            uint32_t unused_nodes_multicast = 0;
            uint32_t unused_nodes_unicast = 0;
        };
        DispatchArray<UnusedNodeData> unused_nodes;

        for (auto& mesh_node : trace_nodes_) {
            bool used = false;
            for (auto& [device_range, node] : mesh_node.trace_nodes) {
                if (!device_range.intersects(range)) {
                    continue;
                }
                TT_ASSERT(range == *device_range.intersection(range));
                trace_nodes.push_back(&node);
                mesh_trace_nodes.push_back(&mesh_node);

                used = true;
                break;
            }
            if (!used) {
                auto& unused_node = unused_nodes[*mesh_node.sub_device_id];
                if (mesh_node.multicast_go_signals && mesh_node.unicast_go_signals) {
                    unused_node.unused_nodes_both_multicast_and_unicast++;
                } else if (mesh_node.multicast_go_signals) {
                    unused_node.unused_nodes_multicast++;
                } else if (mesh_node.unicast_go_signals) {
                    unused_node.unused_nodes_unicast++;
                }
            }
        }
        std::vector<SimpleTraceAllocator::RingbufferConfig> ringbuffer_configs;
        ringbuffer_configs.reserve(hal.get_programmable_core_type_count());
        for (uint32_t idx = 0; idx < hal.get_programmable_core_type_count(); idx++) {
            auto core_type = hal.get_programmable_core_type(idx);
            uint32_t start = hal.get_dev_addr(core_type, tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG);
            uint32_t size;
            if (core_type == HalProgrammableCoreType::TENSIX) {
                size = mesh_device_->allocator_impl()->get_config().l1_unreserved_base - start;
            } else {
                size = hal.get_dev_size(core_type, tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG);
            }
            ringbuffer_configs.push_back({start, size});
        }
        SimpleTraceAllocator allocator{ringbuffer_configs};
        allocator.allocate_trace_programs(hal, trace_nodes);

        this->reset_prefetcher_cache_manager();

        auto& sysmem_manager_for_trace = mesh_device_->get_device(range.start_coord())->sysmem_manager();
        auto& worker_launch_message_buffer_state = cq_shared_state_->worker_launch_message_buffer_state;
        for (uint32_t sub_device_id = 0; sub_device_id < mesh_device_->num_sub_devices(); sub_device_id++) {
            worker_launch_message_buffer_state[sub_device_id].reset();
        }
        std::unordered_map<SubDeviceId, TraceWorkerDescriptor> trace_worker_descriptors;
        for (uint32_t sub_device_id = 0; sub_device_id < mesh_device_->num_sub_devices(); sub_device_id++) {
            for (uint32_t i = 0; i < unused_nodes[sub_device_id].unused_nodes_both_multicast_and_unicast +
                                         unused_nodes[sub_device_id].unused_nodes_multicast +
                                         unused_nodes[sub_device_id].unused_nodes_unicast;
                 i++) {
                bool multicast = i < unused_nodes[sub_device_id].unused_nodes_both_multicast_and_unicast +
                                         unused_nodes[sub_device_id].unused_nodes_multicast;
                bool unicast = i < unused_nodes[sub_device_id].unused_nodes_both_multicast_and_unicast || !multicast;
                SubDeviceId sub_device{static_cast<uint8_t>(sub_device_id)};
                auto& trace_worker_descriptor = trace_worker_descriptors[sub_device];
                program_dispatch::ProgramDispatchMetadata go_signal_md;
                go_signal_md.prefetcher_cache_info.is_cached = true;
                write_go_signal_sequence(
                    this->id_,
                    this->mesh_device_,
                    sub_device,
                    sysmem_manager_for_trace,
                    trace_worker_descriptor.num_completion_worker_cores,
                    this->virtual_program_dispatch_core(),
                    multicast,
                    unicast,
                    go_signal_md,
                    std::nullopt);

                auto& worker_launch_msg_state = worker_launch_message_buffer_state[sub_device_id];
                if (multicast) {
                    trace_worker_descriptor.num_completion_worker_cores +=
                        mesh_device_->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device);
                    worker_launch_msg_state.inc_mcast_wptr(1);
                    trace_worker_descriptor.num_traced_programs_needing_go_signal_multicast++;
                }
                if (unicast) {
                    trace_worker_descriptor.num_completion_worker_cores +=
                        mesh_device_->impl().num_virtual_eth_cores(sub_device);
                    worker_launch_msg_state.inc_unicast_wptr(1);
                    trace_worker_descriptor.num_traced_programs_needing_go_signal_unicast++;
                }
            }
        }
        DispatchArray<uint32_t> starting_workers_completed{};
        for (auto& [sub_device_id, trace_worker_descriptor] : trace_worker_descriptors) {
            starting_workers_completed[*sub_device_id] = trace_worker_descriptor.num_completion_worker_cores;
        }

        for (uint32_t node_idx = 0; node_idx < trace_nodes.size(); node_idx++) {
            auto& node = *trace_nodes[node_idx];
            auto sub_device_id = node.sub_device_id;
            auto& program = *node.program;
            auto& mesh_node = *mesh_trace_nodes[node_idx];

            sub_device_ids.insert(sub_device_id);
            uint32_t num_workers = node.num_workers;
            uint32_t num_virtual_eth_cores = 0;

            if (mesh_node.unicast_go_signals) {
                num_virtual_eth_cores = mesh_device_->impl().num_virtual_eth_cores(sub_device_id);
            }

            uint64_t command_hash = *mesh_device_->get_active_sub_device_manager_id();
            auto& cached_program_command_sequence =
                program.get_trace_cached_program_command_sequences().at(command_hash);

            if (node.dispatch_metadata.send_binary && cached_program_command_sequence.prefetcher_cache_used) {
                auto cache_result = prefetcher_cache_manager_->get_cache_offset(
                    program.get_id(), cached_program_command_sequence.kernel_bins_sizeB);
                TT_ASSERT(
                    cache_result.has_value(),
                    "Prefetcher cache query failed for program {} with size {} in trace recording",
                    program.get_id(),
                    cached_program_command_sequence.kernel_bins_sizeB);
                node.dispatch_metadata.prefetcher_cache_info = {
                    .mesh_max_program_kernels_sizeB = cached_program_command_sequence.kernel_bins_sizeB,
                    .is_cached = cache_result->is_cached,
                    .offset = cache_result->offset * prefetcher_dram_aligned_block_size_};
            } else if (node.dispatch_metadata.send_binary) {
                this->reset_prefetcher_cache_manager();
            }

            auto& worker_launch_msg_state = worker_launch_message_buffer_state[*sub_device_id];
            node.dispatch_metadata.sync_count += starting_workers_completed[*sub_device_id];

            program_dispatch::update_traced_program_dispatch_commands(
                node,
                cached_program_command_sequence,
                worker_launch_msg_state.get_mcast_wptr(),
                worker_launch_msg_state.get_unicast_wptr(),
                trace_worker_descriptors[sub_device_id].num_completion_worker_cores,
                this->virtual_program_dispatch_core(),
                sub_device_id,
                ProgramBinaryStatus::Committed,
                std::pair<bool, int>(mesh_node.unicast_go_signals, num_virtual_eth_cores),
                static_cast<uint8_t>(this->id()));

            record_program_sub_device_for_range(mesh_device_, range, node.program_runtime_id, sub_device_id);

            program_dispatch::write_program_command_sequence(
                cached_program_command_sequence,
                sysmem_manager_for_trace,
                this->id_,
                node.dispatch_metadata.stall_first,
                node.dispatch_metadata.stall_before_program,
                node.dispatch_metadata.send_binary);

            if (mesh_node.multicast_go_signals) {
                worker_launch_msg_state.inc_mcast_wptr(1);
                trace_worker_descriptors[sub_device_id].num_traced_programs_needing_go_signal_multicast++;
            }
            if (mesh_node.unicast_go_signals) {
                worker_launch_msg_state.inc_unicast_wptr(1);
                trace_worker_descriptors[sub_device_id].num_traced_programs_needing_go_signal_unicast++;
            }
            trace_worker_descriptors[sub_device_id].num_completion_worker_cores += num_workers;
        }

        auto& bypass_data = sysmem_manager_for_trace.get_bypass_data();
        bypass_data.insert(bypass_data.end(), exec_buf_end.begin(), exec_buf_end.end());

        max_trace_size = std::max(max_trace_size, bypass_data.size());

        trace_ctx_->ordered_trace_data.push_back(MeshTraceData{range, std::move(bypass_data)});

        if (!overall_trace_worker_descriptors) {
            overall_trace_worker_descriptors = trace_worker_descriptors;
        } else {
            TT_FATAL(
                overall_trace_worker_descriptors == trace_worker_descriptors,
                "All device ranges must produce identical TraceWorkerDescriptors after dummy GO equalization");
        }
    }
    trace_ctx_->total_trace_size = max_trace_size * sizeof(uint32_t);

    trace_ctx_->sub_device_ids.reserve(sub_device_ids.size());
    if (overall_trace_worker_descriptors) {
        trace_ctx_->descriptors = overall_trace_worker_descriptors.value();
    }

    for (auto& [sub_device_id, trace_worker_descriptor] : trace_ctx_->descriptors) {
        trace_ctx_->sub_device_ids.push_back(sub_device_id);
    }

    trace_nodes_.clear();

    trace_id_ = std::nullopt;
    trace_ctx_ = nullptr;

    trace_dispatch::load_host_dispatch_state(
        mesh_device_->num_sub_devices(),
        cq_shared_state_->worker_launch_message_buffer_state,
        expected_num_workers_completed_,
        config_buffer_mgr_,
        worker_launch_message_buffer_state_reset_,
        expected_num_workers_completed_reset_,
        config_buffer_mgr_reset_);

    for (auto* device : mesh_device_->get_devices()) {
        device->sysmem_manager().set_bypass_mode(/*enable*/ false, /*clear*/ true);
    }

    this->reset_prefetcher_cache_manager();
    swap(this->dummy_prefetcher_cache_manager_, this->prefetcher_cache_manager_);
}

#endif