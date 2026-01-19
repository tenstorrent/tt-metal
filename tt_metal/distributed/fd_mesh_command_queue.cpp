// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fd_mesh_command_queue.hpp"

#include <tracy/Tracy.hpp>

#include <mesh_device.hpp>
#include <mesh_event.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

#include <tt_stl/assert.hpp>
#include "buffer.hpp"
#include "buffer_types.hpp"
#include "device.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "event/dispatch.hpp"
#include "hal_types.hpp"
#include "mesh_config.hpp"
#include "mesh_coord.hpp"
#include "mesh_workload.hpp"
#include "mesh_workload_impl.hpp"
#include "sub_device/sub_device_manager_tracker.hpp"
#include "tt-metalium/program.hpp"
#include "shape2d.hpp"
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "trace/trace_buffer.hpp"
#include "tt_metal/common/thread_pool.hpp"
#include "tt_metal/common/multi_producer_single_consumer_queue.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include "tt_metal/impl/buffers/dispatch.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/tools/profiler/tt_metal_tracy.hpp"
#include "tt_metal/impl/device/dispatch.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt_stl/overloaded.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {
struct ProgramCommandSequence;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {

namespace {

// Don't use std::forward since we are in a loop.
// NOLINTBEGIN(cppcoreguidelines-missing-std-forward)
template <typename Container, typename Func>
void for_each_local(MeshDevice* mesh_device, const Container& container, Func&& func) {
    std::for_each(std::cbegin(container), std::cend(container), [&](const auto& coord) {
        if (mesh_device->impl().is_local(coord)) {
            std::invoke(func, coord);
        }
    });
}
// NOLINTEND(cppcoreguidelines-missing-std-forward)

[[maybe_unused]] MeshCoordinate get_local_start_coord(MeshDevice* mesh_device, const MeshCoordinateRange& range) {
    for (const auto& coord : range) {
        if (mesh_device->impl().is_local(coord)) {
            return coord;
        }
    }
    TT_THROW("No local device found for range");
}

}  // namespace

struct MeshReadEventDescriptor {
    ReadEventDescriptor single_device_descriptor;
    MeshCoordinateRange device_range;
};

struct MeshBufferReadDescriptor {
    std::unordered_map<IDevice*, uint32_t> num_reads_per_dev;
};

struct MeshCoreDataReadDescriptor {
    ReadCoreDataDescriptor single_core_descriptor;
    MeshCoordinate device_coord;
};

FDMeshCommandQueue::FDMeshCommandQueue(
    MeshDevice* mesh_device,
    uint32_t id,
    std::shared_ptr<ThreadPool>& dispatch_thread_pool,
    std::shared_ptr<ThreadPool>& reader_thread_pool,
    std::shared_ptr<CQSharedState>& cq_shared_state,
    std::function<std::lock_guard<std::mutex>()> lock_api_function) :
    MeshCommandQueueBase(mesh_device, id, dispatch_thread_pool, std::move(lock_api_function)),
    cq_shared_state_(cq_shared_state),
    dispatch_core_type_(MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type()),
    reader_thread_pool_(reader_thread_pool),
    prefetcher_dram_aligned_block_size_(MetalContext::instance().hal().get_alignment(HalMemType::DRAM)),
    prefetcher_cache_sizeB_(MetalContext::instance().dispatch_mem_map(this->dispatch_core_type_).ringbuffer_size()),
    prefetcher_dram_aligned_num_blocks_(prefetcher_cache_sizeB_ / prefetcher_dram_aligned_block_size_),
    prefetcher_cache_manager_size_(
        1 << (std::bit_width(std::min(1024u, std::max(2u, prefetcher_dram_aligned_num_blocks_ >> 4))) - 1)),
    prefetcher_cache_manager_(std::make_unique<RingbufferCacheManager>(
        prefetcher_dram_aligned_block_size_, prefetcher_dram_aligned_num_blocks_, prefetcher_cache_manager_size_)),
    dummy_prefetcher_cache_manager_(std::make_unique<RingbufferCacheManager>(
        prefetcher_dram_aligned_block_size_, prefetcher_dram_aligned_num_blocks_, prefetcher_cache_manager_size_)) {
    program_dispatch::reset_config_buf_mgrs_and_expected_workers(
        config_buffer_mgr_,
        expected_num_workers_completed_,
        DispatchSettings::DISPATCH_MESSAGE_ENTRIES,
        mesh_device_->allocator_impl()->get_config().l1_unreserved_base);
    this->populate_virtual_program_dispatch_core();
    this->populate_read_descriptor_queue();
    completion_queue_reader_thread_ = std::thread(&FDMeshCommandQueue::read_completion_queue, this);
}

FDMeshCommandQueue::~FDMeshCommandQueue() {
    if (in_use_) {
        // If the FDMeshCommandQueue is being used, have it clear worker state
        // before going out of scope. This is a blocking operation - it waits
        // for all queued up work to complete.
        // This allows physical device close to proceed correctly, since we still
        // rely on single device CQs during this step. Not needed for functionality
        // once single device CQs are removed, however this is still good practice.
        this->clear_expected_num_workers_completed();
    }

    TT_FATAL(completion_queue_reads_.empty(), "The completion reader queue must be empty when closing devices.");

    for (auto& queue : read_descriptors_) {
        TT_FATAL(queue.second->empty(), "No buffer read requests should be outstanding when closing devices.");
    }

    TT_FATAL(
        num_outstanding_reads_ == 0,
        "Mismatch between num_outstanding reads and number of entries in completion reader queue.");

    {
        std::lock_guard lock(reader_thread_cv_mutex_);
        reader_thread_cv_.notify_one();
        exit_condition_ = true;
    }
    completion_queue_reader_thread_.join();
}

void FDMeshCommandQueue::populate_read_descriptor_queue() {
    for (auto& device : mesh_device_->get_devices()) {
        read_descriptors_.emplace(
            device->id(), std::make_unique<MultiProducerSingleConsumerQueue<CompletionReaderVariant>>());
    }
}

void FDMeshCommandQueue::populate_virtual_program_dispatch_core() {
    int device_idx = 0;
    for (auto* device : mesh_device_->get_devices()) {
        if (device_idx) {
            TT_FATAL(
                this->dispatch_core_ == device->virtual_program_dispatch_core(this->id_),
                "Expected Dispatch Cores to match across devices in a Mesh");
        } else {
            this->dispatch_core_ = device->virtual_program_dispatch_core(this->id_);
        }
        device_idx++;
    }
}

CoreCoord FDMeshCommandQueue::virtual_program_dispatch_core() const { return this->dispatch_core_; }

CoreType FDMeshCommandQueue::dispatch_core_type() const { return this->dispatch_core_type_; }

void FDMeshCommandQueue::clear_expected_num_workers_completed() {
    auto sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, {});
    auto& sysmem_manager = this->reference_sysmem_manager();
    auto event =
        MeshEvent(sysmem_manager.get_next_event(id_), mesh_device_, id_, MeshCoordinateRange(mesh_device_->shape()));

    // Issue commands to clear expected_num_workers_completed counter(s) on the dispatcher
    for (auto* device : mesh_device_->get_devices()) {
        event_dispatch::issue_record_event_commands(
            mesh_device_,
            device->id(),
            event.id(),
            id_,
            mesh_device_->num_hw_cqs(),
            device->sysmem_manager(),
            sub_device_ids,
            expected_num_workers_completed_,
            true, /* notify_host */
            true /* clear_count */);
    }
    // Clear counter(s) on host to reflect update device state
    for (auto sub_device_id : sub_device_ids) {
        expected_num_workers_completed_[*sub_device_id] = 0;
    }

    // Block after clearing counter(s) on dispatcher
    completion_queue_reads_.push(std::make_shared<MeshCompletionReaderVariant>(
        std::in_place_type<MeshReadEventDescriptor>, ReadEventDescriptor(event.id()), event.device_range()));
    this->increment_num_entries_in_completion_queue();
    std::unique_lock<std::mutex> lock(reads_processed_cv_mutex_);
    reads_processed_cv_.wait(
        lock, [this] { return num_outstanding_reads_.load() == 0 || thread_exception_state_.load(); });
}

void FDMeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    auto lock = lock_api_function_();
    in_use_ = true;
    uint64_t command_hash = *mesh_device_->get_active_sub_device_manager_id();
    std::unordered_set<SubDeviceId> sub_device_ids = mesh_workload.impl().determine_sub_device_ids(mesh_device_);
    TT_FATAL(sub_device_ids.size() == 1, "Programs must be executed on a single sub-device");
    SubDeviceId sub_device_id = *(sub_device_ids.begin());
    auto mesh_device_id = mesh_device_->id();
    auto& sysmem_manager = this->reference_sysmem_manager();
    auto dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);
    if (!sysmem_manager.get_bypass_mode()) {
        auto& sub_device_cq_owner = cq_shared_state_->sub_device_cq_owner;
        auto& sub_device = sub_device_cq_owner[*sub_device_id];
        sub_device.take_ownership(sub_device_id, this->id_);
    }

    TT_FATAL(
        mesh_workload.impl().get_program_binary_status(mesh_device_id) != ProgramBinaryStatus::NotSent,
        "Expected program binaries to be written to the MeshDevice.");

    // Compute number of workers being used for this workload.
    uint32_t num_workers = 0;
    bool unicast_go_signals = mesh_workload.impl().runs_on_noc_unicast_only_cores();
    bool mcast_go_signals = mesh_workload.impl().runs_on_noc_multicast_only_cores();

    uint32_t num_virtual_eth_cores = 0;

    if (mcast_go_signals) {
        num_workers += mesh_device_->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
    }
    if (unicast_go_signals) {
        // Issue #19729: Running MeshWorkloads on Active Eth cores is supported through multiple workarounds
        // in the dispatch infra. This support should eventually be deprecated.
        // This function currently assumes a uniform number of ethernet cores across all physical devices in the mesh
        // through the num_virtual_eth_cores() function.
        // The physical device itself may have less ethernet cores than what is queried here and will dispatch
        // accordingly.
        num_virtual_eth_cores = mesh_device_->num_virtual_eth_cores(sub_device_id);
        num_workers += num_virtual_eth_cores;
    }

    program_dispatch::ProgramDispatchMetadata dispatch_metadata;
    // Expected number of workers from the previous run
    uint32_t expected_num_workers_completed = sysmem_manager.get_bypass_mode()
                                                  ? trace_ctx_->descriptors[sub_device_id].num_completion_worker_cores
                                                  : expected_num_workers_completed_[*sub_device_id];
    const auto updated_worker_counts =
        program_dispatch::get_expected_num_workers_completed_updates(expected_num_workers_completed, num_workers);

    // Need to stall and reset counters if host wraps
    if (updated_worker_counts.wrapped) [[unlikely]] {
        get_config_buffer_mgr(*sub_device_id).mark_completely_full(0);
        if (sysmem_manager.get_bypass_mode()) {
            capture_expected_worker_count_reset_cmd(expected_num_workers_completed, sub_device_id);
        } else {
            for (auto* device : mesh_device_->get_devices()) {
                program_dispatch::reset_expected_num_workers_completed_on_device(
                    device, sub_device_id, expected_num_workers_completed, id());
            }
        }
    }

    if (sysmem_manager.get_bypass_mode()) {
        if (mcast_go_signals) {
            // The workload contains programs that required a go signal mcast. Capture this here
            // to accurately update the launch msg ring buffer state post trace execution on all
            // mcast cores.
            trace_ctx_->descriptors[sub_device_id].num_traced_programs_needing_go_signal_multicast++;
        }
        if (unicast_go_signals) {
            trace_ctx_->descriptors[sub_device_id].num_traced_programs_needing_go_signal_unicast++;
        }
        // Update the expected number of workers dispatch must wait on
        trace_ctx_->descriptors[sub_device_id].num_completion_worker_cores = updated_worker_counts.current;
    } else {
        expected_num_workers_completed_[*sub_device_id] = updated_worker_counts.current;
    }
    expected_num_workers_completed = updated_worker_counts.previous;

    // Reserve space in the L1 Kernel Config Ring Buffer for this workload.
    program_dispatch::reserve_space_in_kernel_config_buffer(
        this->get_config_buffer_mgr(*sub_device_id),
        mesh_workload.impl().get_program_config_sizes(),
        mesh_workload.impl().get_program_binary_status(mesh_device_id),
        num_workers,
        expected_num_workers_completed,
        dispatch_metadata);

    std::unordered_set<uint32_t> chip_ids_in_workload = {};
    std::vector<MeshCoordinateRange> active_sub_grids = {};

    auto max_program_kernels_sizeB = mesh_workload.impl().max_program_kernels_sizeB_;
    bool use_prefetcher_cache = mesh_workload.impl().use_prefetcher_cache_;
    if (use_prefetcher_cache) {
        bool is_cached;
        uint32_t cache_offset;
        std::tie(is_cached, cache_offset) =
            this->query_prefetcher_cache(mesh_workload.impl().get_id(), max_program_kernels_sizeB);
        TT_ASSERT(
            cache_offset + max_program_kernels_sizeB <= this->prefetcher_cache_sizeB_,
            "Prefetcher cache offset: {}, max_program_kernels_sizeB: {}, prefetcher_cache_sizeB: {}",
            cache_offset,
            max_program_kernels_sizeB,
            this->prefetcher_cache_sizeB_);
        dispatch_metadata.prefetcher_cache_info.is_cached = is_cached;
        dispatch_metadata.prefetcher_cache_info.offset = cache_offset;
        dispatch_metadata.prefetcher_cache_info.mesh_max_program_kernels_sizeB = max_program_kernels_sizeB;
    } else {
        // prefetcher cache will be overwritten, reset for next workload
        this->reset_prefetcher_cache_manager();
    }
    // Iterate over all programs. Update dispatch commands per program to reflect
    // current device state. Write the finalized program command sequence to each
    // physical device tied to the program.
    TracyTTMetalEnqueueMeshWorkloadTrace(mesh_device_, mesh_workload, this->trace_id());
    for (auto& [device_range, program] : mesh_workload.get_programs()) {
        auto& program_cmd_seq = mesh_workload.impl().get_dispatch_cmds_for_program(program, command_hash);
        TT_ASSERT(
            use_prefetcher_cache == program_cmd_seq.prefetcher_cache_used,
            "use_prefetcher_cache: {}, program_cmd_seq.prefetcher_cache_used: {}",
            use_prefetcher_cache,
            program_cmd_seq.prefetcher_cache_used);
        program_dispatch::update_program_dispatch_commands(
            program.impl(),
            program_cmd_seq,
            cq_shared_state_->worker_launch_message_buffer_state[*sub_device_id].get_mcast_wptr(),
            cq_shared_state_->worker_launch_message_buffer_state[*sub_device_id].get_unicast_wptr(),
            expected_num_workers_completed,
            this->virtual_program_dispatch_core(),
            dispatch_core_type,
            sub_device_id,
            dispatch_metadata,
            mesh_workload.impl().get_program_binary_status(mesh_device_id),
            std::pair<bool, int>(unicast_go_signals, num_virtual_eth_cores));

        if (sysmem_manager.get_bypass_mode()) {
            auto local_mesh_range = mesh_device_->get_view().get_local_mesh_coord_range();
            auto local_device_range = local_mesh_range.intersection(device_range);
            if (local_device_range.has_value()) {
                this->capture_program_trace_on_subgrid(
                    local_device_range.value(),
                    program_cmd_seq,
                    dispatch_metadata.stall_first,
                    dispatch_metadata.stall_before_program,
                    program.get_runtime_id());
                active_sub_grids.push_back(local_device_range.value());
            }
        } else {
            this->write_program_cmds_to_subgrid(
                device_range,
                program_cmd_seq,
                dispatch_metadata.stall_first,
                dispatch_metadata.stall_before_program,
                chip_ids_in_workload,
                program.get_runtime_id());
        }
    }
    // Send go signals to devices not running a program to ensure consistent global state
    if (not sysmem_manager.get_bypass_mode()) {
        this->write_go_signal_to_unused_sub_grids(
            chip_ids_in_workload,
            sub_device_id,
            expected_num_workers_completed,
            mcast_go_signals,
            unicast_go_signals,
            dispatch_metadata);
    } else {
        MeshCoordinateRangeSet active_sub_grids_set;
        for (const auto& sub_grid : active_sub_grids) {
            active_sub_grids_set.merge(sub_grid);
        }
        this->capture_go_signal_trace_on_unused_subgrids(
            active_sub_grids_set,
            sub_device_id,
            expected_num_workers_completed,
            mcast_go_signals,
            unicast_go_signals,
            dispatch_metadata);
    }
    // Increment Launch Message Buffer Write Pointers
    if (mcast_go_signals) {
        cq_shared_state_->worker_launch_message_buffer_state[*sub_device_id].inc_mcast_wptr(1);
    }
    if (unicast_go_signals) {
        cq_shared_state_->worker_launch_message_buffer_state[*sub_device_id].inc_unicast_wptr(1);
    }

    // From the dispatcher's perspective, binaries are now committed to DRAM
    mesh_workload.impl().set_program_binary_status(mesh_device_id, ProgramBinaryStatus::Committed);
    mesh_workload.set_last_used_command_queue_for_testing(this);

    if (blocking) {
        this->finish_nolock({{sub_device_id}});
    }
}

void FDMeshCommandQueue::enqueue_write_shard_to_core(
    DeviceMemoryAddress address,
    const void* src,
    uint32_t size_bytes,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScoped;

    auto lock = lock_api_function_();
    if (!mesh_device_->impl().is_local(address.device_coord)) {
        return;
    }

    in_use_ = true;
    TT_FATAL(!trace_id_.has_value(), "Writes are not supported during trace capture.");

    IDevice* device = mesh_device_->impl().get_device(address.device_coord);
    address.address = device_dispatch::add_bank_offset_to_address(device, address.virtual_core_coord, address.address);

    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);

    device_dispatch::write_to_core(
        device,
        address.virtual_core_coord,
        src,
        address.address,
        size_bytes,
        id_,
        expected_num_workers_completed_,
        sub_device_ids);

    if (blocking) {
        this->finish_nolock(sub_device_ids);
    }
}

void FDMeshCommandQueue::enqueue_read_shard_from_core(
    DeviceMemoryAddress address,
    void* dst,
    uint32_t size_bytes,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScoped;
    auto lock = lock_api_function_();
    if (!mesh_device_->impl().is_local(address.device_coord)) {
        return;
    }

    in_use_ = true;
    TT_FATAL(!trace_id_.has_value(), "Reads are not supported during trace capture.");

    IDevice* device = mesh_device_->impl().get_device(address.device_coord);
    address.address = device_dispatch::add_bank_offset_to_address(device, address.virtual_core_coord, address.address);

    device_dispatch::validate_core_read_write_bounds(device, address.virtual_core_coord, address.address, size_bytes);

    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);

    if (size_bytes > 0) {
        this->reset_prefetcher_cache_manager();
        device_dispatch::CoreReadDispatchParams dispatch_params{
            address.virtual_core_coord,
            address.address,
            size_bytes,
            device,
            id_,
            dispatch_core_type_,
            expected_num_workers_completed_,
            sub_device_ids};
        device_dispatch::issue_core_read_command_sequence(dispatch_params);
    }

    this->submit_core_data_memcpy_request(
        ReadCoreDataDescriptor(dst, size_bytes), address.device_coord, blocking, sub_device_ids);
}

void FDMeshCommandQueue::finish_nolock(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("FDMeshCommandQueue::finish_nolock");
    auto event = this->enqueue_record_event_to_host_nolock(sub_device_ids);

    std::unique_lock<std::mutex> lock(reads_processed_cv_mutex_);
    reads_processed_cv_.wait(
        lock, [this] { return num_outstanding_reads_.load() == 0 || thread_exception_state_.load(); });
    auto& sub_device_cq_owner = cq_shared_state_->sub_device_cq_owner;
    for (const auto& sub_device_id : buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids)) {
        sub_device_cq_owner[*sub_device_id].finished(this->id_);
    }

    if (should_handle_exception_.load()) {
        std::lock_guard<std::mutex> exception_lock(exception_mutex_);
        if (auto exception_ptr = thread_exception_ptr_) {
            thread_exception_ptr_ = nullptr;
            should_handle_exception_.store(false);
            num_outstanding_reads_.store(0);
            reads_processed_cv_.notify_all();
            lock.unlock();
            std::rethrow_exception(exception_ptr);
        }
    }
}

void FDMeshCommandQueue::finish(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto lock = lock_api_function_();
    this->finish_nolock(sub_device_ids);

    // Barrier across all hosts of the mesh
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_distributed_context(
        mesh_device_->get_view().mesh_id());
    distributed_context->barrier();
}

void FDMeshCommandQueue::write_shard_to_device(
    const MeshBuffer& buffer,
    const MeshCoordinate& device_coord,
    const void* src,
    const std::optional<BufferRegion>& region,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    if (!mesh_device_->impl().is_local(device_coord)) {
        return;
    }

    if (tt::tt_metal::GraphTracker::instance().hook_write_to_device(&buffer)) {
        return;
    }

    in_use_ = true;
    TT_FATAL(!trace_id_.has_value(), "Writes are not supported during trace capture. trace id: {}", trace_id_.value());

    auto* device_buffer = buffer.get_device_buffer(device_coord);
    auto shard_view = device_buffer->view(region.value_or(BufferRegion(0, device_buffer->size())));

    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);
    buffer_dispatch::write_to_device_buffer(
        src, *shard_view, id_, expected_num_workers_completed_, this->dispatch_core_type(), sub_device_ids);
}

void FDMeshCommandQueue::read_shard_from_device(
    const MeshBuffer& buffer,
    const MeshCoordinate& device_coord,
    void* dst,
    std::shared_ptr<experimental::PinnedMemory> pinned_memory,
    const std::optional<BufferRegion>& region,
    std::unordered_map<IDevice*, uint32_t>& num_txns_per_device,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    if (!mesh_device_->impl().is_local(device_coord)) {
        return;
    }

    if (tt::tt_metal::GraphTracker::instance().hook_read_from_device(&buffer)) {
        return;
    }

    in_use_ = true;
    TT_FATAL(!trace_id_.has_value(), "Reads are not supported during trace capture.");

    auto* device_buffer = buffer.get_device_buffer(device_coord);
    auto shard_view = device_buffer->view(region.value_or(BufferRegion(0, device_buffer->size())));

    auto* device = shard_view->device();
    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);
    // Reading from device would clobber prefetcher cache, so reset it now
    this->reset_prefetcher_cache_manager();

    if (is_sharded(shard_view->buffer_layout())) {
        auto dispatch_params = buffer_dispatch::initialize_sharded_buf_read_dispatch_params(
            *shard_view, id_, expected_num_workers_completed_);
        const auto& cores = dispatch_params.buffer_page_mapping->all_cores;
        for (uint32_t core_id = 0; core_id < shard_view->num_cores(); ++core_id) {
            for (const auto& core_page_mapping : dispatch_params.buffer_page_mapping->core_page_mappings[core_id]) {
                buffer_dispatch::copy_sharded_buffer_from_core_to_completion_queue(
                    core_id,
                    core_page_mapping,
                    *shard_view,
                    dispatch_params,
                    sub_device_ids,
                    cores[core_id],
                    this->dispatch_core_type());
                if (dispatch_params.pages_per_txn > 0) {
                    num_txns_per_device[device]++;
                    auto& read_descriptor_queue = this->get_read_descriptor_queue(device);
                    read_descriptor_queue.push(
                        buffer_dispatch::generate_sharded_buffer_read_descriptor(dst, dispatch_params, *shard_view));
                }
            }
        }
    } else {
        buffer_dispatch::BufferReadDispatchParams dispatch_params =
            buffer_dispatch::initialize_interleaved_buf_read_dispatch_params(
                *shard_view, id_, expected_num_workers_completed_);

        buffer_dispatch::copy_interleaved_buffer_to_completion_queue(
            dispatch_params, *shard_view, sub_device_ids, this->dispatch_core_type(), dst, pinned_memory);
        if ((dispatch_params.pages_per_txn > 0) && (dispatch_params.requires_completion_read)) {
            num_txns_per_device[device]++;
            auto& read_descriptor_queue = this->get_read_descriptor_queue(device);
            read_descriptor_queue.push(
                buffer_dispatch::generate_interleaved_buffer_read_descriptor(dst, dispatch_params, *shard_view));
        }
    }
}

void FDMeshCommandQueue::increment_num_entries_in_completion_queue() {
    {
        std::lock_guard lock(reader_thread_cv_mutex_);
        num_outstanding_reads_++;
        reader_thread_cv_.notify_one();
    }
}

void FDMeshCommandQueue::submit_memcpy_request(
    std::unordered_map<IDevice*, uint32_t>& num_txns_per_device, bool blocking) {
    completion_queue_reads_.push(std::make_shared<MeshCompletionReaderVariant>(
        std::in_place_type<MeshBufferReadDescriptor>, std::move(num_txns_per_device)));

    this->increment_num_entries_in_completion_queue();

    if (blocking) {
        this->finish_nolock();
    }
}

void FDMeshCommandQueue::submit_core_data_memcpy_request(
    const ReadCoreDataDescriptor& read_descriptor,
    const MeshCoordinate& device_coord,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    completion_queue_reads_.push(std::make_shared<MeshCompletionReaderVariant>(
        std::in_place_type<MeshCoreDataReadDescriptor>, read_descriptor, device_coord));
    this->increment_num_entries_in_completion_queue();

    if (blocking) {
        this->finish_nolock(sub_device_ids);
    }
}

MeshEvent FDMeshCommandQueue::enqueue_record_event_helper(
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    bool notify_host,
    const std::optional<MeshCoordinateRange>& device_range) {
    in_use_ = true;
    TT_FATAL(!trace_id_.has_value(), "Event Synchronization is not supported during trace capture.");

    auto& sysmem_manager = this->reference_sysmem_manager();
    auto event = MeshEvent(
        sysmem_manager.get_next_event(id_),
        mesh_device_,
        id_,
        device_range.value_or(MeshCoordinateRange(mesh_device_->shape())));

    sub_device_ids = buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids);
    auto dispatch_lambda = [this, &event, &sub_device_ids, notify_host](const MeshCoordinate& coord) {
        event_dispatch::issue_record_event_commands(
            mesh_device_,
            mesh_device_->impl().get_device(coord)->id(),
            event.id(),
            id_,
            mesh_device_->num_hw_cqs(),
            mesh_device_->impl().get_device(coord)->sysmem_manager(),
            sub_device_ids,
            expected_num_workers_completed_,
            notify_host);
    };

    for_each_local(mesh_device_, event.device_range(), [&](const auto& coord) {
        dispatch_thread_pool_->enqueue(
            [&dispatch_lambda, coord]() { dispatch_lambda(coord); }, mesh_device_->impl().get_device(coord)->id());
    });
    dispatch_thread_pool_->wait();
    return event;
}

MeshEvent FDMeshCommandQueue::enqueue_record_event(
    tt::stl::Span<const SubDeviceId> sub_device_ids, const std::optional<MeshCoordinateRange>& device_range) {
    auto lock = lock_api_function_();
    auto& sub_device_cq_owner = cq_shared_state_->sub_device_cq_owner;

    MeshEvent event = this->enqueue_record_event_helper(sub_device_ids, /*notify_host=*/false, device_range);
    for (const auto& sub_device_id : buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids)) {
        auto& sub_device_entry = sub_device_cq_owner[*sub_device_id];
        sub_device_entry.recorded_event(event.id(), event.mesh_cq_id());
    }
    return event;
}

MeshEvent FDMeshCommandQueue::enqueue_record_event_to_host_nolock(
    tt::stl::Span<const SubDeviceId> sub_device_ids, const std::optional<MeshCoordinateRange>& device_range) {
    auto event = this->enqueue_record_event_helper(sub_device_ids, /*notify_host=*/true, device_range);
    completion_queue_reads_.push(std::make_shared<MeshCompletionReaderVariant>(
        std::in_place_type<MeshReadEventDescriptor>, ReadEventDescriptor(event.id()), event.device_range()));
    this->increment_num_entries_in_completion_queue();
    auto& sub_device_cq_owner = cq_shared_state_->sub_device_cq_owner;
    for (const auto& sub_device_id : buffer_dispatch::select_sub_device_ids(mesh_device_, sub_device_ids)) {
        auto& sub_device_entry = sub_device_cq_owner[*sub_device_id];
        sub_device_entry.recorded_event(event.id(), event.mesh_cq_id());
    }
    return event;
}

MeshEvent FDMeshCommandQueue::enqueue_record_event_to_host(
    tt::stl::Span<const SubDeviceId> sub_device_ids, const std::optional<MeshCoordinateRange>& device_range) {
    auto lock = lock_api_function_();
    return this->enqueue_record_event_to_host_nolock(sub_device_ids, device_range);
}

void FDMeshCommandQueue::enqueue_wait_for_event(const MeshEvent& sync_event) {
    auto lock = lock_api_function_();
    in_use_ = true;
    TT_FATAL(!trace_id_.has_value(), "Event Synchronization is not supported during trace capture.");
    for_each_local(mesh_device_, sync_event.device_range(), [&](const auto& coord) {
        event_dispatch::issue_wait_for_event_commands(
            id_, sync_event.mesh_cq_id(), mesh_device_->impl().get_device(coord)->sysmem_manager(), sync_event.id());
    });
    auto& sub_device_cq_owner = cq_shared_state_->sub_device_cq_owner;
    for (auto& sub_device_entry : sub_device_cq_owner) {
        sub_device_entry.waited_for_event(sync_event.id(), sync_event.mesh_cq_id(), this->id_);
    }
}

void FDMeshCommandQueue::read_completion_queue() {
    while (!thread_exception_state_.load()) {
        try {
            {
                std::unique_lock<std::mutex> lock(reader_thread_cv_mutex_);
                reader_thread_cv_.wait(lock, [this] { return num_outstanding_reads_ or exit_condition_; });
            }
            if (exit_condition_) {
                return;
            }

            uint32_t num_reads = num_outstanding_reads_.load();
            for (uint32_t i = 0; i < num_reads; i++) {
                auto mesh_read_descriptor = *(completion_queue_reads_.pop());
                std::visit(
                    [&](auto&& mesh_read_descriptor) {
                        using T = std::decay_t<decltype(mesh_read_descriptor)>;
                        if constexpr (std::is_same_v<T, MeshBufferReadDescriptor>) {
                            this->copy_buffer_data_to_user_space(mesh_read_descriptor);
                        } else if constexpr (std::is_same_v<T, MeshReadEventDescriptor>) {
                            this->read_completion_queue_event(mesh_read_descriptor);
                        } else {
                            this->read_l1_data_from_completion_queue(mesh_read_descriptor);
                        }
                    },
                    mesh_read_descriptor);
            }
            std::unique_lock<std::mutex> lock(reads_processed_cv_mutex_);
            num_outstanding_reads_.fetch_sub(num_reads);
            if (num_outstanding_reads_ == 0) {
                reads_processed_cv_.notify_one();
            }
        } catch (const std::runtime_error& e) {
            // Just to clarify, this is a weird case and it is an unrecoverable error.
            // If we are here, its likely the device is hung, meaning that the whole program is stuck.
            // We don't have a recovery mechanism for this, so we just need to clean up the state and let the main
            // thread handle it.
            {
                std::lock_guard<std::mutex> exception_lock(exception_mutex_);
                thread_exception_ptr_ = std::current_exception();
                should_handle_exception_.store(true);
            }

            thread_exception_state_.store(true);
            exit_condition_.store(true);
            num_outstanding_reads_.store(0);
            reads_processed_cv_.notify_all();
            completion_queue_reads_.clear();
            reader_thread_cv_.notify_all();
            return;
        }
    }
}

MultiProducerSingleConsumerQueue<CompletionReaderVariant>& FDMeshCommandQueue::get_read_descriptor_queue(
    IDevice* device) {
    return *(read_descriptors_[device->id()]);
}

void FDMeshCommandQueue::copy_buffer_data_to_user_space(MeshBufferReadDescriptor& read_buffer_descriptor) {
    auto reader_lambda = [this](IDevice* device, uint32_t num_reads) {
        auto& read_descriptor_queue = this->get_read_descriptor_queue(device);
        ChipId mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());

        for (int i = 0; i < num_reads; i++) {
            buffer_dispatch::copy_completion_queue_data_into_user_space(
                std::get<ReadBufferDescriptor>(*(read_descriptor_queue.pop())),
                mmio_device_id,
                channel,
                id_,
                device->sysmem_manager(),
                exit_condition_);
        }
    };

    {
        // The reader_thread_pool is a shared resource between command queues.
        // It must be used inside a critical section. Performance-wise, this is
        // okay, since workloads don't require issuing simultaneous reads on different
        // MeshCQs. If such an access pattern is used, the reads will be serialized
        // on host across MeshCQs. This is still better than independent reader threads
        // per MeshCQ (since we run out of host resources with 2 reader threads per
        // physical device).
        std::lock_guard<std::mutex> lock(reader_thread_pool_mutex_);
        for (auto& metadata : read_buffer_descriptor.num_reads_per_dev) {
            reader_thread_pool_->enqueue(
                [&reader_lambda, device = metadata.first, num_reads = metadata.second]() {
                    reader_lambda(device, num_reads);
                },
                metadata.first->id());
        }
        reader_thread_pool_->wait();
    }
}

void FDMeshCommandQueue::read_completion_queue_event(MeshReadEventDescriptor& read_event_descriptor) {
    auto& device_range = read_event_descriptor.device_range;
    for_each_local(mesh_device_, device_range, [&](const auto& coord) {
        auto device = mesh_device_->impl().get_device(coord);
        ChipId mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
        uint16_t channel =
            tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());
        device->sysmem_manager().completion_queue_wait_front(id_, exit_condition_);

        event_dispatch::read_events_from_completion_queue(
            read_event_descriptor.single_device_descriptor,
            mmio_device_id,
            device->id(),
            channel,
            id_,
            device->sysmem_manager());
    });
}

void FDMeshCommandQueue::read_l1_data_from_completion_queue(MeshCoreDataReadDescriptor& read_l1_data_descriptor) {
    if (!mesh_device_->impl().is_local(read_l1_data_descriptor.device_coord)) {
        return;
    }
    IDevice* device = mesh_device_->impl().get_device(read_l1_data_descriptor.device_coord);
    const ChipId mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device->id());
    const uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device->id());
    device_dispatch::read_core_data_from_completion_queue(
        read_l1_data_descriptor.single_core_descriptor,
        mmio_device_id,
        channel,
        id_,
        device->sysmem_manager(),
        exit_condition_);
}

void FDMeshCommandQueue::reset_worker_state(
    bool reset_launch_msg_state,
    uint32_t num_sub_devices,
    const vector_aligned<uint32_t>& go_signal_noc_data,
    const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping) {
    for (auto* device : mesh_device_->get_devices()) {
        TT_FATAL(!device->sysmem_manager().get_bypass_mode(), "Cannot reset worker state during trace capture");
    }
    cq_shared_state_->sub_device_cq_owner.clear();
    cq_shared_state_->sub_device_cq_owner.resize(num_sub_devices);
    in_use_ = true;
    for (auto* device : mesh_device_->get_devices()) {
        program_dispatch::reset_worker_dispatch_state_on_device(
            mesh_device_,
            device->sysmem_manager(),
            id_,
            this->virtual_program_dispatch_core(),
            expected_num_workers_completed_,
            reset_launch_msg_state);
        program_dispatch::set_num_worker_sems_on_dispatch(mesh_device_, device->sysmem_manager(), id_, num_sub_devices);
        program_dispatch::set_go_signal_noc_data_on_dispatch(
            mesh_device_, go_signal_noc_data, device->sysmem_manager(), id_);
        if (reset_launch_msg_state) {
            program_dispatch::set_core_go_message_mapping_on_device(
                device, core_go_message_mapping, device->sysmem_manager(), id_);
        }
    }
    program_dispatch::reset_config_buf_mgrs_and_expected_workers(
        config_buffer_mgr_,
        expected_num_workers_completed_,
        mesh_device_->num_sub_devices(),
        mesh_device_->allocator_impl()->get_config().l1_unreserved_base);
    if (reset_launch_msg_state) {
        std::for_each(
            this->cq_shared_state_->worker_launch_message_buffer_state.begin(),
            this->cq_shared_state_->worker_launch_message_buffer_state.begin() + num_sub_devices,
            std::mem_fn(&LaunchMessageRingBufferState::reset));
    }
}

void FDMeshCommandQueue::write_program_cmds_to_subgrid(
    const MeshCoordinateRange& sub_grid,
    ProgramCommandSequence& program_cmd_seq,
    bool stall_first,
    bool stall_before_program,
    std::unordered_set<uint32_t>& chip_ids_in_workload,
    uint32_t program_runtime_id) {
    auto dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);
    for_each_local(mesh_device_, sub_grid, [&](const auto& coord) {
        auto device = mesh_device_->impl().get_device(coord);
        this->update_launch_messages_for_device_profiler(program_cmd_seq, program_runtime_id, device);
        program_dispatch::write_program_command_sequence(
            program_cmd_seq, device->sysmem_manager(), id_, dispatch_core_type, stall_first, stall_before_program);
        chip_ids_in_workload.insert(device->id());
    });
}

void FDMeshCommandQueue::write_go_signal_to_unused_sub_grids(
    std::unordered_set<uint32_t>& chip_ids_in_workload,
    const SubDeviceId& sub_device_id,
    uint32_t expected_num_workers_completed,
    bool mcast_go_signals,
    bool unicast_go_signals,
    const program_dispatch::ProgramDispatchMetadata& dispatch_md) {
    for (auto& device : mesh_device_->get_devices()) {
        if (!chip_ids_in_workload.contains(device->id())) {
            write_go_signal(
                id_,
                mesh_device_,
                sub_device_id,
                device->sysmem_manager(),
                expected_num_workers_completed,
                this->virtual_program_dispatch_core(),
                mcast_go_signals,
                unicast_go_signals,
                dispatch_md);
        }
    }
}

void FDMeshCommandQueue::capture_program_trace_on_subgrid(
    const MeshCoordinateRange& sub_grid,
    ProgramCommandSequence& program_cmd_seq,
    bool stall_first,
    bool stall_before_program,
    uint32_t program_runtime_id) {
    auto dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled()) {
        // Host Memory Intensive Path (when profiler is enabled): The launch messages across devices are unique, since
        // the host_assigned_field in the launch_msg contains the physical device id (required by the performance
        // profiler). Hence the trace per device must be uniquely captured.
        for_each_local(mesh_device_, sub_grid, [&](const auto& coord) {
            auto& sysmem_manager_for_trace = mesh_device_->impl().get_device(coord)->sysmem_manager();
            uint32_t sysmem_manager_offset = sysmem_manager_for_trace.get_issue_queue_write_ptr(id_);

            auto device = mesh_device_->impl().get_device(coord);
            this->update_launch_messages_for_device_profiler(program_cmd_seq, program_runtime_id, device);
            program_dispatch::write_program_command_sequence(
                program_cmd_seq, sysmem_manager_for_trace, id_, dispatch_core_type, stall_first, stall_before_program);
            auto mesh_trace_md = MeshTraceStagingMetadata{
                MeshCoordinateRange(coord, coord),
                coord,
                sysmem_manager_offset,
                sysmem_manager_for_trace.get_issue_queue_write_ptr(id_) - sysmem_manager_offset};
            ordered_mesh_trace_md_.push_back(mesh_trace_md);
        });
    } else {
        // Optimized Path (generic use-cases): Program dispatch commands across the entire sub-grid are identical.
        // Capture once.
        auto local_start_coord = get_local_start_coord(mesh_device_, sub_grid);
        auto& sysmem_manager_for_trace = mesh_device_->impl().get_device(local_start_coord)->sysmem_manager();
        uint32_t sysmem_manager_offset = sysmem_manager_for_trace.get_issue_queue_write_ptr(id_);

        program_dispatch::write_program_command_sequence(
            program_cmd_seq, sysmem_manager_for_trace, id_, dispatch_core_type, stall_first, stall_before_program);
        auto mesh_trace_md = MeshTraceStagingMetadata{
            sub_grid,
            local_start_coord,
            sysmem_manager_offset,
            sysmem_manager_for_trace.get_issue_queue_write_ptr(id_) - sysmem_manager_offset};
        ordered_mesh_trace_md_.push_back(mesh_trace_md);
    }
}

void FDMeshCommandQueue::capture_go_signal_trace_on_unused_subgrids(
    const MeshCoordinateRangeSet& active_grids_set,
    const SubDeviceId& sub_device_id,
    uint32_t expected_num_workers_completed,
    bool mcast_go_signals,
    bool unicast_go_signals,
    const program_dispatch::ProgramDispatchMetadata& dispatch_md) {
    MeshCoordinateRange full_grid(mesh_device_->get_view().get_local_mesh_coord_range());
    MeshCoordinateRangeSet unused_grids_set(full_grid);

    // Subtract each active grid from the unused grids set to handle non-convex grids
    for (const auto& active_grid : active_grids_set.ranges()) {
        MeshCoordinateRangeSet new_unused_set;
        for (const auto& unused_range : unused_grids_set.ranges()) {
            auto subtracted_ranges = subtract(unused_range, active_grid);
            for (const auto& range : subtracted_ranges.ranges()) {
                new_unused_set.merge(range);
            }
        }
        unused_grids_set = new_unused_set;
    }

    for (const auto& unused_grid : unused_grids_set.ranges()) {
        if (!mesh_device_->impl().is_local(unused_grid.start_coord())) {
            continue;
        }
        auto& sysmem_manager_for_trace = mesh_device_->impl().get_device(unused_grid.start_coord())->sysmem_manager();
        uint32_t sysmem_manager_offset = sysmem_manager_for_trace.get_issue_queue_write_ptr(id_);
        write_go_signal(
            id_,
            mesh_device_,
            sub_device_id,
            sysmem_manager_for_trace,
            expected_num_workers_completed,
            this->virtual_program_dispatch_core(),
            mcast_go_signals,
            unicast_go_signals,
            dispatch_md);
        auto mesh_trace_md = MeshTraceStagingMetadata{
            unused_grid,
            unused_grid.start_coord(),
            sysmem_manager_offset,
            sysmem_manager_for_trace.get_issue_queue_write_ptr(id_) - sysmem_manager_offset};
        ordered_mesh_trace_md_.push_back(mesh_trace_md);
    }
}

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

    // Reset the prefetcher cache manager, since trace capture modifies the state on host for subsequent non-trace
    // programs
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

void FDMeshCommandQueue::record_end() {
    trace_ctx_->assemble_dispatch_commands(this->device(), ordered_mesh_trace_md_);
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

    ordered_mesh_trace_md_.clear();
    for (auto* device : mesh_device_->get_devices()) {
        device->sysmem_manager().set_bypass_mode(/*enable*/ false, /*clear*/ true);
    }

    // Trace has modified the prefetcher cache manager so reset it first and then swap to restore the state as before
    // the recording
    this->reset_prefetcher_cache_manager();
    swap(this->dummy_prefetcher_cache_manager_, this->prefetcher_cache_manager_);
}

SystemMemoryManager& FDMeshCommandQueue::reference_sysmem_manager() {
    auto local_devices = mesh_device_->get_devices();
    return local_devices.at(0)->sysmem_manager();
}

void FDMeshCommandQueue::update_launch_messages_for_device_profiler(
    ProgramCommandSequence& program_cmd_seq, uint32_t program_runtime_id, IDevice* device) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled()) {
        for (auto& [is_multicast, original_launch_msg, launch_msg] : program_cmd_seq.launch_messages) {
            launch_msg.kernel_config().host_assigned_id() =
                tt_metal::detail::EncodePerDeviceProgramID(program_runtime_id, device->id());
        }
    }
}

std::pair<bool, size_t> FDMeshCommandQueue::query_prefetcher_cache(uint64_t workload_id, uint32_t lengthB) {
    auto result = prefetcher_cache_manager_->get_cache_offset(workload_id, lengthB);
    TT_FATAL(
        result.has_value(),
        "Prefetcher cache query failed. Cache size: {}, requested: {}",
        this->prefetcher_cache_manager_->get_cache_sizeB(),
        lengthB);
    return std::make_pair(result.value().is_cached, result.value().offset * this->prefetcher_dram_aligned_block_size_);
}

void FDMeshCommandQueue::reset_prefetcher_cache_manager() { prefetcher_cache_manager_->reset(); }

int FDMeshCommandQueue::get_prefetcher_cache_sizeB() const {
    return this->prefetcher_cache_manager_->get_cache_sizeB();
}

void FDMeshCommandQueue::capture_expected_worker_count_reset_cmd(
    uint32_t previous_expected_workers, SubDeviceId sub_device) {
    for (auto* device : mesh_device_->get_devices()) {
        auto& sysmem_manager = device->sysmem_manager();
        uint32_t sysmem_manager_offset = sysmem_manager.get_issue_queue_write_ptr(id_);
        program_dispatch::reset_expected_num_workers_completed_on_device(
            device, sub_device, previous_expected_workers, id());

        // Find the coordinate for this device by iterating over all coordinates
        MeshCoordinate device_coord{0xffffffff};
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            if (mesh_device_->impl().get_device(coord) == device) {
                device_coord = coord;
                break;
            }
        }
        TT_ASSERT(device_coord != MeshCoordinate{0xffffffff});
        auto mesh_trace_md = MeshTraceStagingMetadata{
            MeshCoordinateRange{device_coord},
            device_coord,
            sysmem_manager_offset,
            sysmem_manager.get_issue_queue_write_ptr(id_) - sysmem_manager_offset};
        ordered_mesh_trace_md_.push_back(mesh_trace_md);
    }
}

void FDMeshCommandQueue::wait_for_completion(bool reset_launch_msg_state) {
    if (in_use_) {
        size_t num_sub_devices = mesh_device_->num_sub_devices();
        for (auto* device : mesh_device_->get_devices()) {
            TT_FATAL(!device->sysmem_manager().get_bypass_mode(), "Cannot reset worker state during trace capture");
        }
        cq_shared_state_->sub_device_cq_owner.clear();
        cq_shared_state_->sub_device_cq_owner.resize(num_sub_devices);
        for (auto* device : mesh_device_->get_devices()) {
            program_dispatch::reset_worker_dispatch_state_on_device(
                mesh_device_,
                device->sysmem_manager(),
                id_,
                this->virtual_program_dispatch_core(),
                expected_num_workers_completed_,
                reset_launch_msg_state);
        }
        program_dispatch::reset_config_buf_mgrs_and_expected_workers(
            config_buffer_mgr_,
            expected_num_workers_completed_,
            mesh_device_->num_sub_devices(),
            mesh_device_->allocator_impl()->get_config().l1_unreserved_base);
        if (reset_launch_msg_state) {
            std::for_each(
                this->cq_shared_state_->worker_launch_message_buffer_state.begin(),
                this->cq_shared_state_->worker_launch_message_buffer_state.begin() + num_sub_devices,
                std::mem_fn(&LaunchMessageRingBufferState::reset));
        }
        finish();
    }
}

void FDMeshCommandQueue::finish_and_reset_in_use() {
    if (in_use_) {
        auto lock = lock_api_function_();
        uint32_t current_event = reference_sysmem_manager().get_current_event(id_);
        for (auto* device : mesh_device_->get_devices()) {
            TT_ASSERT(
                device->sysmem_manager().get_last_completed_event(id_) == current_event,
                "Current event must be equal to last completed event");
            bool is_reference_cq = &device->sysmem_manager() == &reference_sysmem_manager();
            // Ensure the next command will be recorded as event 0
            device->sysmem_manager().set_current_and_last_completed_event(
                id_, is_reference_cq ? UINT32_MAX : 0, UINT32_MAX);
        }
        finish_nolock({});

        in_use_ = false;
    }
}

}  // namespace tt::tt_metal::distributed
