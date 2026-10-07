// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/command_list.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>

#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/distributed/mesh_coord_utils.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/distributed/mesh_workload_impl.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/impl/context/metal_env_impl.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/dispatch_mem_map.hpp"
#include "tt_metal/impl/dispatch/dispatch_settings.hpp"
#include "tt_metal/impl/dispatch/launch_message_ring_buffer_state.hpp"
#include "tt_metal/impl/dispatch/ringbuffer_cache.hpp"
#include "tt_metal/impl/dispatch/simple_trace_allocator.hpp"
#include "tt_metal/impl/internal/service/service_core_manager_impl.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"
#include "tt_metal/impl/program/program_impl.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "tt_metal/impl/trace/trace_node.hpp"
#include <internal/service/service_core_manager.hpp>

namespace tt::tt_metal::experimental {
using namespace tt::tt_metal::distributed;

namespace detail {

struct CommandListData {
    MeshCoordinateRange device_range = MeshCoordinateRange(MeshShape(0, 0));
    std::vector<uint32_t> data;
};

class CommandListDescriptor {
public:
    CommandListDescriptor() = default;
    CommandListDescriptor(
        const DispatchArray<std::optional<TraceWorkerDescriptor>>& worker_descriptors,
        uint32_t max_command_stream_bytes) :
        max_command_stream_bytes_(max_command_stream_bytes) {
        for (uint32_t i = 0; i < worker_descriptors.size(); ++i) {
            if (worker_descriptors[i]) {
                const SubDeviceId sub_device{static_cast<uint8_t>(i)};
                worker_descriptors_.emplace(sub_device, *worker_descriptors[i]);
                sub_device_ids_.push_back(sub_device);
            }
        }
    }

    const std::unordered_map<SubDeviceId, TraceWorkerDescriptor>& worker_descriptors() const {
        return worker_descriptors_;
    }
    const std::vector<SubDeviceId>& sub_device_ids() const { return sub_device_ids_; }
    uint32_t max_command_stream_bytes() const { return max_command_stream_bytes_; }

private:
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> worker_descriptors_;
    // Keys of worker_descriptors_, cached because replay passes them as a vector on every call.
    std::vector<SubDeviceId> sub_device_ids_;
    uint32_t max_command_stream_bytes_ = 0;
};

struct CapturedProgram {
    MeshCoordinateRange device_range;
    TraceNode trace_node;
};

struct StagedCommandListNode {
    std::vector<CapturedProgram> programs;
    bool multicast_go_signals = false;
    bool unicast_go_signals = false;
    SubDeviceId sub_device_id;
    uint32_t num_workers = 0;
};

struct CommandListAssembly {
    CommandListDescriptor descriptor;
    std::vector<CommandListData> serialized_ranges;
};

}  // namespace detail

namespace {

void append_command_bytes(std::vector<uint32_t>& output, const void* data, uint32_t size_bytes) {
    TT_ASSERT(size_bytes % sizeof(uint32_t) == 0);
    const size_t old_size = output.size();
    output.resize(old_size + size_bytes / sizeof(uint32_t));
    std::memcpy(output.data() + old_size, data, size_bytes);
}

void append_go_signal_sequence(
    std::vector<uint32_t>& output,
    uint8_t cq_id,
    MeshDevice& mesh_device,
    SubDeviceId sub_device,
    uint32_t expected_workers,
    CoreCoord dispatch_core,
    bool send_multicast,
    bool send_unicasts) {
    program_dispatch::ProgramDispatchMetadata dispatch_metadata;
    // GO-signal-only sequences carry no kernel binary; marking them cached skips the prefetcher ring-buffer offset
    // command.
    dispatch_metadata.prefetcher_cache_info.is_cached = true;
    HostMemDeviceCommand commands = build_go_signal_sequence(
        cq_id,
        &mesh_device,
        sub_device,
        expected_workers,
        dispatch_core,
        send_multicast,
        send_unicasts,
        dispatch_metadata,
        std::nullopt);
    append_command_bytes(output, commands.data(), commands.size_bytes());
}

FDMeshCommandQueue& as_fd_queue(MeshCommandQueue& cq) {
    auto* fd_cq = dynamic_cast<FDMeshCommandQueue*>(&cq);
    TT_FATAL(fd_cq != nullptr, "CommandList only supports fast-dispatch mesh command queues");
    return *fd_cq;
}

void validate_no_service_cores(MeshWorkload& workload, MeshDevice& mesh_device) {
    auto& service_core_manager = mesh_device.impl().metal_context().get_service_core_manager();
    if (!service_core_manager.impl().has_any_claims()) {
        return;
    }
    for (auto& [device_range, program] : workload.get_programs()) {
        const auto logical_cores = program.impl().logical_cores();
        for (const auto& coord : device_range) {
            auto* device = mesh_device.impl().get_device(coord);
            if (device == nullptr) {
                continue;
            }
            for (const auto& per_type : logical_cores) {
                for (const auto& core : per_type) {
                    TT_FATAL(
                        !service_core_manager.impl().is_service_core(device->id(), core),
                        "Command lists do not support workloads targeting service core {} on device {}",
                        core,
                        device->id());
                }
            }
        }
    }
}

}  // namespace

namespace detail {

class CommandListBuilderImpl {
public:
    explicit CommandListBuilderImpl(MeshDevice& mesh_device);
    ~CommandListBuilderImpl();

    void add(MeshWorkload& workload);
    CommandList build(MeshCommandQueue& cq) const;
    MeshDevice& device() const;
    void clear();
    void deallocate();

private:
    struct OfflineDispatchState;

    static std::vector<MeshCoordinateRange> compute_device_ranges(
        const std::vector<StagedCommandListNode>& staged_nodes, const MeshCoordinateRange& local_mesh_range);
    // One entry per staged node: the program it runs on range, or nullptr if it has none there.
    static std::vector<CapturedProgram*> select_programs_for_range(
        std::vector<StagedCommandListNode>& staged_nodes, const MeshCoordinateRange& range);
    static CommandListData serialize_range(
        MeshDevice& mesh_device,
        OfflineDispatchState& dispatch_state,
        std::vector<StagedCommandListNode>& staged_nodes,
        const MeshCoordinateRange& range,
        const std::vector<uint32_t>& exec_buf_end);
    static CommandListAssembly assemble(
        MeshCommandQueue& cq,
        const std::vector<StagedCommandListNode>& staged_nodes,
        const DispatchArray<std::optional<TraceWorkerDescriptor>>& worker_descriptors);
    std::shared_ptr<MeshBuffer> allocate_and_commit(
        MeshCommandQueue& cq,
        const CommandListDescriptor& descriptor,
        const std::vector<CommandListData>& serialized_ranges) const;
    uint32_t get_num_workers(bool multicast, bool unicast, SubDeviceId sub_device) const;

    MeshDevice& mesh_device;
    SubDeviceManagerId sub_device_manager_id;
    std::vector<StagedCommandListNode> staged_nodes;
    // Indexed by sub-device. Every staged node adds the same workers and GO signals on every device range, through its
    // program or a dummy GO signal, so these are the same for the whole mesh.
    DispatchArray<std::optional<TraceWorkerDescriptor>> worker_descriptors;
    std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers;
    bool valid = true;
    bool lock_held = false;
};

}  // namespace detail

class CommandList::Impl {
public:
    Impl(
        MeshDevice& mesh_device,
        detail::CommandListDescriptor descriptor,
        std::shared_ptr<MeshBuffer> command_buffer,
        std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers,
        uint8_t cq_id,
        SubDeviceManagerId sub_device_manager_id);
    ~Impl();

    void deallocate();
    void replay(bool blocking) const;
    MeshDevice& get_device() const;
    uint8_t get_cq_id() const;

private:
    void validate() const;
    void release_resources() noexcept;

    MeshDevice* mesh_device = nullptr;
    detail::CommandListDescriptor descriptor;
    std::shared_ptr<MeshBuffer> command_buffer;
    // Pins kernel-binary MeshBuffers referenced by the serialized commands.
    std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers;
    uint8_t bound_cq_id = 0;
    SubDeviceManagerId sub_device_manager_id;
    bool valid = true;
};

CommandList::Impl::Impl(
    MeshDevice& mesh_device,
    detail::CommandListDescriptor descriptor,
    std::shared_ptr<MeshBuffer> command_buffer,
    std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers,
    uint8_t cq_id,
    SubDeviceManagerId sub_device_manager_id) :
    mesh_device(&mesh_device),
    descriptor(std::move(descriptor)),
    command_buffer(std::move(command_buffer)),
    retained_binary_buffers(std::move(retained_binary_buffers)),
    bound_cq_id(cq_id),
    sub_device_manager_id(sub_device_manager_id) {}

CommandList::Impl::~Impl() {
    try {
        deallocate();
    } catch (...) {
        release_resources();
    }
}

void CommandList::Impl::validate() const {
    TT_FATAL(valid, "CommandList has been deallocated");
    TT_FATAL(mesh_device != nullptr, "CommandList has no MeshDevice");
}

void CommandList::Impl::release_resources() noexcept {
    if (!valid) {
        return;
    }
    if (command_buffer && command_buffer->is_allocated()) {
        const auto current_size = mesh_device->get_trace_buffers_size();
        mesh_device->set_trace_buffers_size(current_size - command_buffer->size());
    }
    command_buffer.reset();
    descriptor = {};
    retained_binary_buffers.clear();
    valid = false;
}

void CommandList::Impl::deallocate() {
    if (!valid) {
        return;
    }
    as_fd_queue(mesh_device->mesh_command_queue(bound_cq_id)).drain_device_work();
    release_resources();
}

void CommandList::Impl::replay(bool blocking) const {
    validate();
    as_fd_queue(mesh_device->mesh_command_queue(bound_cq_id))
        .enqueue_command_list(
            descriptor.worker_descriptors(),
            descriptor.sub_device_ids(),
            *command_buffer,
            sub_device_manager_id,
            blocking);
}

MeshDevice& CommandList::Impl::get_device() const {
    validate();
    return *mesh_device;
}

uint8_t CommandList::Impl::get_cq_id() const {
    validate();
    return bound_cq_id;
}

namespace detail {

CommandListBuilderImpl::CommandListBuilderImpl(MeshDevice& mesh_device) :
    mesh_device(mesh_device),
    sub_device_manager_id(mesh_device.impl().acquire_command_list_builder()),
    lock_held(true) {}

CommandListBuilderImpl::~CommandListBuilderImpl() {
    if (lock_held) {
        mesh_device.impl().release_command_list_builder();
    }
}

struct CommandListBuilderImpl::OfflineDispatchState {
    // Prefetcher cache settings must match FDMeshCommandQueue's: the cache offsets baked into the commands are replayed
    // against the real prefetcher.
    OfflineDispatchState(MeshDevice& mesh_device, uint8_t cq_id) :
        cq_id(cq_id),
        dispatch_core(mesh_device.virtual_program_dispatch_core(cq_id)),
        prefetcher_block_size(mesh_device.impl().metal_env().get_hal().get_alignment(HalMemType::DRAM)),
        prefetcher_cache_size(mesh_device.impl().metal_context().dispatch_mem_map().ringbuffer_size()),
        prefetcher_num_blocks(prefetcher_cache_size / prefetcher_block_size),
        prefetcher_manager_size(1 << (std::bit_width(std::min(1024u, std::max(2u, prefetcher_num_blocks >> 4))) - 1)),
        prefetcher_cache(prefetcher_block_size, prefetcher_num_blocks, prefetcher_manager_size) {}

    void reset(uint32_t num_sub_devices) {
        prefetcher_cache.reset();
        for (uint32_t i = 0; i < num_sub_devices; ++i) {
            launch_state[i].reset();
        }
    }

    uint8_t cq_id;
    CoreCoord dispatch_core;
    uint32_t prefetcher_block_size;
    uint64_t prefetcher_cache_size;
    uint32_t prefetcher_num_blocks;
    uint32_t prefetcher_manager_size;
    RingbufferCacheManager prefetcher_cache;
    DispatchArray<LaunchMessageRingBufferState> launch_state;
};

std::vector<MeshCoordinateRange> CommandListBuilderImpl::compute_device_ranges(
    const std::vector<StagedCommandListNode>& staged_nodes, const MeshCoordinateRange& local_mesh_range) {
    std::vector<MeshCoordinateRange> device_ranges{local_mesh_range};
    for (const auto& staged_node : staged_nodes) {
        for (const auto& program : staged_node.programs) {
            auto local_device_range = local_mesh_range.intersection(program.device_range);
            if (!local_device_range.has_value()) {
                continue;
            }
            partition_mesh_coordinate_ranges(device_ranges, *local_device_range);
        }
    }
    return device_ranges;
}

std::vector<CapturedProgram*> CommandListBuilderImpl::select_programs_for_range(
    std::vector<StagedCommandListNode>& staged_nodes, const MeshCoordinateRange& range) {
    std::vector<CapturedProgram*> programs;
    programs.reserve(staged_nodes.size());
    for (auto& staged_node : staged_nodes) {
        auto it = std::ranges::find_if(staged_node.programs, [&](const CapturedProgram& program) {
            return program.device_range.intersects(range);
        });
        if (it == staged_node.programs.end()) {
            programs.push_back(nullptr);
            continue;
        }
        TT_ASSERT(range == *it->device_range.intersection(range));
        programs.push_back(&*it);
    }
    return programs;
}

CommandListData CommandListBuilderImpl::serialize_range(
    MeshDevice& mesh_device,
    OfflineDispatchState& dispatch_state,
    std::vector<StagedCommandListNode>& staged_nodes,
    const MeshCoordinateRange& range,
    const std::vector<uint32_t>& exec_buf_end) {
    const auto& hal = mesh_device.impl().metal_env().get_hal();
    const auto programs = select_programs_for_range(staged_nodes, range);

    std::vector<TraceNode*> trace_nodes;
    trace_nodes.reserve(programs.size());
    for (auto* program : programs) {
        if (program != nullptr) {
            trace_nodes.push_back(&program->trace_node);
        }
    }

    std::vector<SimpleTraceAllocator::RingbufferConfig> ringbuffer_configs;
    ringbuffer_configs.reserve(hal.get_programmable_core_type_count());
    for (uint32_t i = 0; i < hal.get_programmable_core_type_count(); ++i) {
        const auto core_type = hal.get_programmable_core_type(i);
        const uint32_t start = hal.get_dev_addr(core_type, HalL1MemAddrType::KERNEL_CONFIG);
        const uint32_t size = core_type == HalProgrammableCoreType::TENSIX
                                  ? mesh_device.allocator_impl()->get_config().l1_unreserved_base - start
                                  : hal.get_dev_size(core_type, HalL1MemAddrType::KERNEL_CONFIG);
        ringbuffer_configs.push_back({start, size});
    }
    SimpleTraceAllocator{ringbuffer_configs}.allocate_trace_programs(hal, trace_nodes);

    // Each range is an independent stream, and replay resets worker launch-message write pointers to 0, so every range
    // is serialized from a clean prefetcher cache and launch state.
    dispatch_state.reset(mesh_device.num_sub_devices());
    std::vector<uint32_t> bytes;
    auto& launch_state = dispatch_state.launch_state;
    DispatchArray<uint32_t> expected_workers{};

    // Each staged node with no program on this range becomes a dummy GO signal at the start of the stream, so
    // launch-message write pointers and expected worker counts match across all devices. They go at the start because
    // the last program may still be running when the stream ends.
    for (uint32_t sub_device_idx = 0; sub_device_idx < mesh_device.num_sub_devices(); ++sub_device_idx) {
        const SubDeviceId sub_device{static_cast<uint8_t>(sub_device_idx)};
        // Multicast + unicast first, then multicast-only, then unicast-only.
        for (const auto& [multicast, unicast] :
             {std::pair{true, true}, std::pair{true, false}, std::pair{false, true}}) {
            for (size_t i = 0; i < staged_nodes.size(); ++i) {
                const auto& staged_node = staged_nodes[i];
                if (programs[i] != nullptr || *staged_node.sub_device_id != sub_device_idx ||
                    staged_node.multicast_go_signals != multicast || staged_node.unicast_go_signals != unicast) {
                    continue;
                }
                append_go_signal_sequence(
                    bytes,
                    dispatch_state.cq_id,
                    mesh_device,
                    sub_device,
                    expected_workers[sub_device_idx],
                    dispatch_state.dispatch_core,
                    multicast,
                    unicast);
                if (multicast) {
                    launch_state[sub_device_idx].inc_mcast_wptr(1);
                }
                if (unicast) {
                    launch_state[sub_device_idx].inc_unicast_wptr(1);
                }
                expected_workers[sub_device_idx] += staged_node.num_workers;
            }
        }
    }

    // SimpleTraceAllocator computes sync_count from 0, but the dummy GO signals above already add to the completion
    // counter, so each program's sync_count is offset by this amount.
    const DispatchArray<uint32_t> starting_workers = expected_workers;

    for (size_t i = 0; i < staged_nodes.size(); ++i) {
        if (programs[i] == nullptr) {
            continue;
        }
        const auto& staged_node = staged_nodes[i];
        auto& node = programs[i]->trace_node;
        const auto sub_device = node.sub_device_id;

        auto& command_sequence = node.program->get_trace_cached_program_command_sequences().at(
            *mesh_device.get_active_sub_device_manager_id());
        // Binaries already resident in worker SRAM (send_binary == false) bypass the prefetcher cache. Binaries that
        // don't fit in the cache reset it so later programs reload it from scratch.
        if (node.dispatch_metadata.send_binary && command_sequence.prefetcher_cache_used) {
            const auto cache = dispatch_state.prefetcher_cache.get_cache_offset(
                node.program->get_id(), command_sequence.kernel_bins_sizeB);
            TT_ASSERT(cache.has_value(), "Command-list prefetcher cache query failed");
            node.dispatch_metadata.prefetcher_cache_info = {
                .mesh_max_program_kernels_sizeB = command_sequence.kernel_bins_sizeB,
                .is_cached = cache->is_cached,
                .offset = cache->offset * dispatch_state.prefetcher_block_size};
        } else if (node.dispatch_metadata.send_binary) {
            dispatch_state.prefetcher_cache.reset();
        }

        auto& worker_launch_state = launch_state[*sub_device];
        node.dispatch_metadata.sync_count += starting_workers[*sub_device];
        const uint32_t virtual_eth_cores =
            staged_node.unicast_go_signals ? mesh_device.impl().num_virtual_eth_cores(sub_device) : 0;
        program_dispatch::update_traced_program_dispatch_commands(
            node,
            command_sequence,
            worker_launch_state.get_mcast_wptr(),
            worker_launch_state.get_unicast_wptr(),
            expected_workers[*sub_device],
            dispatch_state.dispatch_core,
            sub_device,
            ProgramBinaryStatus::Committed,
            {staged_node.unicast_go_signals, virtual_eth_cores},
            dispatch_state.cq_id);

        program_dispatch::for_each_program_command_sequence_chunk(
            command_sequence,
            node.dispatch_metadata.stall_first,
            node.dispatch_metadata.stall_before_program,
            node.dispatch_metadata.send_binary,
            [&](const void* chunk, uint32_t chunk_size) { append_command_bytes(bytes, chunk, chunk_size); });

        if (staged_node.multicast_go_signals) {
            worker_launch_state.inc_mcast_wptr(1);
        }
        if (staged_node.unicast_go_signals) {
            worker_launch_state.inc_unicast_wptr(1);
        }
        expected_workers[*sub_device] += node.num_workers;
    }

    // Returns the prefetcher from the exec buffer to the issue queue.
    bytes.insert(bytes.end(), exec_buf_end.begin(), exec_buf_end.end());
    return {.device_range = range, .data = std::move(bytes)};
}

CommandListAssembly CommandListBuilderImpl::assemble(
    MeshCommandQueue& cq,
    const std::vector<StagedCommandListNode>& staged_nodes,
    const DispatchArray<std::optional<TraceWorkerDescriptor>>& worker_descriptors) {
    auto staged_nodes_copy = staged_nodes;
    auto& mesh_device = *cq.device();
    OfflineDispatchState dispatch_state(mesh_device, static_cast<uint8_t>(cq.id()));

    CommandListAssembly assembly;

    DeviceCommand end_command(
        mesh_device.impl().metal_context(), mesh_device.impl().metal_env().get_hal().get_alignment(HalMemType::HOST));
    end_command.add_prefetch_exec_buf_end();
    std::vector<uint32_t> exec_buf_end(end_command.size_bytes() / sizeof(uint32_t));
    std::memcpy(exec_buf_end.data(), end_command.data(), end_command.size_bytes());

    size_t max_command_list_size = 0;

    const auto device_ranges =
        compute_device_ranges(staged_nodes_copy, mesh_device.get_view().get_local_mesh_coord_range());
    for (const auto& range : device_ranges) {
        auto serialized = serialize_range(mesh_device, dispatch_state, staged_nodes_copy, range, exec_buf_end);
        max_command_list_size = std::max(max_command_list_size, serialized.data.size());
        assembly.serialized_ranges.push_back(std::move(serialized));
    }

    assembly.descriptor =
        CommandListDescriptor(worker_descriptors, static_cast<uint32_t>(max_command_list_size * sizeof(uint32_t)));
    return assembly;
}

std::shared_ptr<MeshBuffer> CommandListBuilderImpl::allocate_and_commit(
    MeshCommandQueue& cq,
    const CommandListDescriptor& descriptor,
    const std::vector<CommandListData>& serialized_ranges) const {
    const size_t page_size = trace_dispatch::compute_interleaved_trace_buf_page_size(
        descriptor.max_command_stream_bytes(), mesh_device.allocator()->get_num_banks(BufferType::DRAM));
    const size_t padded_size = round_up(descriptor.max_command_stream_bytes(), page_size);
    const auto trace_region_size = mesh_device.allocator_impl()->get_config().trace_region_size;
    const BufferType buffer_type = trace_region_size == 0 ? BufferType::DRAM : BufferType::TRACE;
    const std::optional<bool> bottom_up = trace_region_size == 0 ? std::optional<bool>{false} : std::nullopt;

    // Command lists share the trace region's budget with traces.
    const auto current_size = mesh_device.get_trace_buffers_size();
    TT_FATAL(
        trace_region_size == 0 || current_size + padded_size <= trace_region_size,
        "Command-list buffers exceed the configured trace-region capacity");
    mesh_device.set_trace_buffers_size(current_size + padded_size);

    std::shared_ptr<MeshBuffer> buffer;
    try {
        {
            // The trace allocation tracker recognizes this context and excludes trace storage from unsafe-allocation
            // accounting.
            auto allocation_context = tt::tt_metal::make_allocation_context_guard("trace_storage");
            buffer = MeshBuffer::create(
                ReplicatedBufferConfig{.size = padded_size},
                DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type, .bottom_up = bottom_up},
                &mesh_device);
        }
        for (const auto& data : serialized_ranges) {
            // Shard writes cover whole pages, so each range's stream is zero-padded to a page boundary.
            std::vector<uint32_t> padded = data.data;
            padded.resize(round_up(padded.size() * sizeof(uint32_t), page_size) / sizeof(uint32_t), 0);
            cq.enqueue_write_shard_to_sub_grid(
                *buffer, padded.data(), data.device_range, true, BufferRegion(0, padded.size() * sizeof(uint32_t)));
        }
    } catch (...) {
        mesh_device.set_trace_buffers_size(current_size);
        throw;
    }
    return buffer;
}

uint32_t CommandListBuilderImpl::get_num_workers(bool multicast, bool unicast, SubDeviceId sub_device) const {
    uint32_t workers = 0;
    if (multicast) {
        workers += mesh_device.num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device);
    }
    if (unicast) {
        workers += mesh_device.impl().num_virtual_eth_cores(sub_device);
    }
    return workers;
}

void CommandListBuilderImpl::add(MeshWorkload& workload) {
    TT_FATAL(valid, "CommandListBuilder has been deallocated");
    TT_FATAL(
        mesh_device.get_active_sub_device_manager_id() == sub_device_manager_id,
        "The active sub-device manager changed while recording a command list");
    validate_no_service_cores(workload, mesh_device);

    auto& binary_load_cq = mesh_device.mesh_command_queue();
    auto binary_buffer = workload.impl().prepare_for_command_list(binary_load_cq);

    StagedCommandListNode staged_node;
    staged_node.unicast_go_signals = workload.impl().runs_on_noc_unicast_only_cores();
    staged_node.multicast_go_signals = workload.impl().runs_on_noc_multicast_only_cores();
    const auto sub_devices = workload.impl().determine_sub_device_ids(&mesh_device);
    TT_FATAL(sub_devices.size() == 1, "A command-list workload must execute on one sub-device");
    staged_node.sub_device_id = *sub_devices.begin();
    staged_node.num_workers =
        get_num_workers(staged_node.multicast_go_signals, staged_node.unicast_go_signals, staged_node.sub_device_id);
    const uint32_t cache_size = mesh_device.impl().metal_context().dispatch_mem_map().ringbuffer_size();
    const uint32_t max_program_kernels_size = workload.impl().max_program_kernels_size();
    const bool use_prefetcher_cache = max_program_kernels_size != 0 && max_program_kernels_size <= cache_size;

    for (auto& [device_range, program] : workload.get_programs()) {
        staged_node.programs.push_back(
            {device_range,
             program_dispatch::create_trace_node(
                 program.impl(), &mesh_device, staged_node.num_workers, use_prefetcher_cache)});
    }

    auto& worker = worker_descriptors[*staged_node.sub_device_id];
    if (!worker) {
        worker.emplace();
    }
    worker->num_completion_worker_cores += staged_node.num_workers;
    if (staged_node.multicast_go_signals) {
        ++worker->num_traced_programs_needing_go_signal_multicast;
    }
    if (staged_node.unicast_go_signals) {
        ++worker->num_traced_programs_needing_go_signal_unicast;
    }
    staged_nodes.push_back(std::move(staged_node));
    if (binary_buffer) {
        retained_binary_buffers.push_back(std::move(binary_buffer));
    }
}

CommandList CommandListBuilderImpl::build(MeshCommandQueue& cq) const {
    TT_FATAL(valid, "CommandListBuilder has been deallocated");
    TT_FATAL(cq.device() == &mesh_device, "Command queue belongs to a different MeshDevice");
    TT_FATAL(!staged_nodes.empty(), "Cannot build an empty CommandList");
    TT_FATAL(
        mesh_device.get_active_sub_device_manager_id() == sub_device_manager_id,
        "The active sub-device manager changed while building the command list");

    (void)as_fd_queue(cq);
    auto assembly = assemble(cq, staged_nodes, worker_descriptors);
    auto command_buffer = allocate_and_commit(cq, assembly.descriptor, assembly.serialized_ranges);
    return CommandList(std::make_unique<CommandList::Impl>(
        mesh_device,
        std::move(assembly.descriptor),
        std::move(command_buffer),
        retained_binary_buffers,
        static_cast<uint8_t>(cq.id()),
        sub_device_manager_id));
}

MeshDevice& CommandListBuilderImpl::device() const {
    TT_FATAL(valid, "CommandListBuilder has been deallocated");
    return mesh_device;
}

void CommandListBuilderImpl::clear() {
    TT_FATAL(valid, "CommandListBuilder has been deallocated");
    staged_nodes.clear();
    worker_descriptors = {};
    retained_binary_buffers.clear();
}

void CommandListBuilderImpl::deallocate() {
    staged_nodes.clear();
    worker_descriptors = {};
    retained_binary_buffers.clear();
    valid = false;
    if (lock_held) {
        mesh_device.impl().release_command_list_builder();
        lock_held = false;
    }
}

}  // namespace detail

CommandListBuilder::CommandListBuilder(MeshDevice& device) :
    impl_(std::make_unique<detail::CommandListBuilderImpl>(device)) {}
CommandListBuilder::CommandListBuilder(CommandListBuilder&&) noexcept = default;
CommandListBuilder& CommandListBuilder::operator=(CommandListBuilder&&) noexcept = default;
CommandListBuilder::~CommandListBuilder() = default;

void CommandListBuilder::add(MeshWorkload& workload) {
    TT_FATAL(impl_ != nullptr, "CommandListBuilder has been moved from");
    impl_->add(workload);
}

CommandList CommandListBuilder::build(MeshCommandQueue& cq) const {
    TT_FATAL(impl_ != nullptr, "CommandListBuilder has been moved from");
    return impl_->build(cq);
}

MeshDevice& CommandListBuilder::device() const {
    TT_FATAL(impl_ != nullptr, "CommandListBuilder has been moved from");
    return impl_->device();
}

void CommandListBuilder::clear() {
    TT_FATAL(impl_ != nullptr, "CommandListBuilder has been moved from");
    impl_->clear();
}

void CommandListBuilder::deallocate() {
    if (impl_) {
        impl_->deallocate();
    }
}

CommandList::CommandList(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CommandList::CommandList(CommandList&&) noexcept = default;
CommandList& CommandList::operator=(CommandList&&) noexcept = default;
CommandList::~CommandList() = default;

void CommandList::replay(bool blocking) const {
    TT_FATAL(impl_ != nullptr, "CommandList has been moved from");
    impl_->replay(blocking);
}

MeshDevice& CommandList::device() const {
    TT_FATAL(impl_ != nullptr, "CommandList has been moved from");
    return impl_->get_device();
}

uint8_t CommandList::cq_id() const {
    TT_FATAL(impl_ != nullptr, "CommandList has been moved from");
    return impl_->get_cq_id();
}

void CommandList::deallocate() {
    if (impl_) {
        impl_->deallocate();
    }
}

void EnqueueCommandList(MeshCommandQueue& cq, CommandList& command_list, bool blocking) {
    TT_FATAL(&command_list.device() == cq.device(), "CommandList belongs to a different MeshDevice");
    TT_FATAL(command_list.cq_id() == cq.id(), "CommandList was built for a different command queue");
    command_list.replay(blocking);
}

}  // namespace tt::tt_metal::experimental
