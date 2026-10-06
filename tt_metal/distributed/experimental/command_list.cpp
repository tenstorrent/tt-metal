// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/command_list.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>

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
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/metal2_host_api/tensor_binding_crtas.hpp"
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

// A parameter's words inside one entry of a traced command sequence's rta_updates.
struct CapturedArgLocation {
    size_t rta_update_index = 0;
    uint32_t byte_offset = 0;
    uint32_t num_words = 1;
};

struct TensorPatchBinding {
    TensorBindingHandle binding;
    TensorSpec expected_spec;
    TensorSpecRelaxations relaxations;
};

struct CapturedRuntimePatch {
    CmdListRuntimeArgName name;
    CapturedArgLocation location;
};

struct CapturedCommonRuntimePatch {
    CmdListCommonRuntimeArgName name;
    CapturedArgLocation location;
};

struct CapturedTensorPatch {
    CmdListTensorArgName name;
    CapturedArgLocation location;
    std::shared_ptr<const TensorPatchBinding> binding;
};

struct CapturedProgram {
    MeshCoordinateRange device_range;
    TraceNode trace_node;
    std::vector<CapturedRuntimePatch> runtime_patches;
    std::vector<CapturedCommonRuntimePatch> common_runtime_patches;
    std::vector<CapturedTensorPatch> tensor_patches;
};

struct StagedCommandListNode {
    std::vector<CapturedProgram> programs;
    bool multicast_go_signals = false;
    bool unicast_go_signals = false;
    SubDeviceId sub_device_id;
    uint32_t num_workers = 0;
};

// A parameter's words in the command stream of one device range.
struct PatchTarget {
    size_t range_index = 0;
    uint32_t stream_offset = 0;
    uint32_t num_words = 1;
};

struct TensorPatchTarget {
    PatchTarget target;
    std::shared_ptr<const TensorPatchBinding> binding;
};

// Patch locations of one command list. Device writes must be DRAM-aligned, so the registry keeps the current contents
// of each aligned window that holds a parameter, but not the rest of the command stream.
struct CommandListParameterRegistry {
    std::unordered_map<CmdListRuntimeArgName, std::vector<PatchTarget>> runtime_targets;
    std::unordered_map<CmdListCommonRuntimeArgName, std::vector<PatchTarget>> common_runtime_targets;
    std::unordered_map<CmdListTensorArgName, std::vector<TensorPatchTarget>> tensor_targets;
    std::vector<MeshCoordinateRange> device_ranges;
    // windows[range_index][offset] holds the current contents of one window of that range's command stream.
    // range_index indexes device_ranges, like PatchTarget::range_index. offset is the byte offset of the window's
    // start in the stream, a multiple of window_size. Each value is window_size / sizeof(uint32_t) words.
    std::vector<std::unordered_map<uint32_t, std::vector<uint32_t>>> windows;
    uint32_t window_size = 0;
};

struct CommandListAssembly {
    CommandListDescriptor descriptor;
    std::vector<CommandListData> serialized_ranges;
    CommandListParameterRegistry registry;
};

}  // namespace detail

namespace {

// Returns the byte offset of the appended bytes in output.
uint32_t append_command_bytes(std::vector<uint32_t>& output, const void* data, uint32_t size_bytes) {
    TT_ASSERT(size_bytes % sizeof(uint32_t) == 0);
    const size_t old_size = output.size();
    output.resize(old_size + size_bytes / sizeof(uint32_t));
    std::memcpy(output.data() + old_size, data, size_bytes);
    return static_cast<uint32_t>(old_size * sizeof(uint32_t));
}

// Finds every rta_updates entry that copies args[word_offset, word_offset + num_words) into the command sequence.
std::vector<detail::CapturedArgLocation> resolve_arg_locations(
    const RuntimeArgsData& args,
    uint32_t word_offset,
    uint32_t num_words,
    const ProgramCommandSequence& command_sequence) {
    TT_FATAL(
        word_offset + num_words <= args.size(),
        "Command list parameter at word {} exceeds its {}-word runtime argument buffer",
        word_offset,
        args.size());
    const auto target = reinterpret_cast<uintptr_t>(args.data() + word_offset);
    const uint32_t size_bytes = num_words * sizeof(uint32_t);
    std::vector<detail::CapturedArgLocation> locations;
    for (size_t i = 0; i < command_sequence.rta_updates.size(); ++i) {
        const auto& update = command_sequence.rta_updates[i];
        const auto begin = reinterpret_cast<uintptr_t>(update.src);
        if (target >= begin && target + size_bytes <= begin + update.size) {
            locations.push_back(
                {.rta_update_index = i, .byte_offset = static_cast<uint32_t>(target - begin), .num_words = num_words});
        }
    }
    TT_FATAL(!locations.empty(), "Command list parameter is not part of its program's dispatch commands");
    return locations;
}

// Records the stream offset of every patch of program that lies in chunk. Returns how many it recorded.
size_t record_patch_targets(
    const detail::CapturedProgram& program,
    const ProgramCommandSequence& command_sequence,
    const void* chunk,
    uint32_t chunk_size,
    uint32_t chunk_offset,
    size_t range_index,
    detail::CommandListParameterRegistry& registry) {
    const auto chunk_begin = reinterpret_cast<uintptr_t>(chunk);
    auto locate = [&](const detail::CapturedArgLocation& location) -> std::optional<detail::PatchTarget> {
        const auto& update = command_sequence.rta_updates.at(location.rta_update_index);
        const auto target = reinterpret_cast<uintptr_t>(update.dst) + location.byte_offset;
        if (target < chunk_begin || target + location.num_words * sizeof(uint32_t) > chunk_begin + chunk_size) {
            return std::nullopt;
        }
        return detail::PatchTarget{
            .range_index = range_index,
            .stream_offset = chunk_offset + static_cast<uint32_t>(target - chunk_begin),
            .num_words = location.num_words};
    };

    size_t num_recorded = 0;
    for (const auto& patch : program.runtime_patches) {
        if (auto target = locate(patch.location)) {
            registry.runtime_targets[patch.name].push_back(*target);
            ++num_recorded;
        }
    }
    for (const auto& patch : program.common_runtime_patches) {
        if (auto target = locate(patch.location)) {
            registry.common_runtime_targets[patch.name].push_back(*target);
            ++num_recorded;
        }
    }
    for (const auto& patch : program.tensor_patches) {
        if (auto target = locate(patch.location)) {
            registry.tensor_targets[patch.name].push_back({.target = *target, .binding = patch.binding});
            ++num_recorded;
        }
    }
    return num_recorded;
}

// Copies every window that holds a patch target out of the serialized streams. Bytes past the end of a stream are
// the zero padding written by allocate_and_commit.
void capture_patch_windows(
    detail::CommandListParameterRegistry& registry, const std::vector<detail::CommandListData>& serialized_ranges) {
    const uint32_t window_words = registry.window_size / sizeof(uint32_t);
    registry.windows.resize(serialized_ranges.size());
    auto capture = [&](const detail::PatchTarget& target) {
        const auto& stream = serialized_ranges[target.range_index].data;
        const uint32_t end = target.stream_offset + target.num_words * sizeof(uint32_t);
        for (uint32_t offset = target.stream_offset - target.stream_offset % registry.window_size; offset < end;
             offset += registry.window_size) {
            auto [it, inserted] = registry.windows[target.range_index].try_emplace(offset);
            if (!inserted) {
                continue;
            }
            it->second.assign(window_words, 0);
            const size_t first_word = offset / sizeof(uint32_t);
            const size_t num_words = std::min<size_t>(window_words, stream.size() - first_word);
            std::copy_n(stream.begin() + first_word, num_words, it->second.begin());
        }
    };
    for (const auto& [_, targets] : registry.runtime_targets) {
        std::ranges::for_each(targets, capture);
    }
    for (const auto& [_, targets] : registry.common_runtime_targets) {
        std::ranges::for_each(targets, capture);
    }
    for (const auto& [_, targets] : registry.tensor_targets) {
        for (const auto& tensor_target : targets) {
            capture(tensor_target.target);
        }
    }
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

    void add(MeshWorkload& workload, const CmdListParameters& parameters);
    CommandList build(MeshCommandQueue& cq) const;
    MeshDevice& device() const;
    void clear();
    void deallocate();

private:
    struct OfflineDispatchState;

    // Appends the location of each parameter to the runtime_patches, common_runtime_patches, or tensor_patches of
    // the CapturedProgram in staged_node that it belongs to.
    void resolve_parameters(StagedCommandListNode& staged_node, const CmdListParameters& parameters) const;
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
        const std::vector<uint32_t>& exec_buf_end,
        size_t range_index,
        CommandListParameterRegistry& registry);
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
        detail::CommandListParameterRegistry registry,
        uint8_t cq_id,
        SubDeviceManagerId sub_device_manager_id);
    ~Impl();

    void deallocate();
    void replay(bool blocking) const;
    void update_args(const CmdListArgPatch& patch, bool blocking);
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
    detail::CommandListParameterRegistry registry;
    uint8_t bound_cq_id = 0;
    SubDeviceManagerId sub_device_manager_id;
    bool valid = true;
};

CommandList::Impl::Impl(
    MeshDevice& mesh_device,
    detail::CommandListDescriptor descriptor,
    std::shared_ptr<MeshBuffer> command_buffer,
    std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers,
    detail::CommandListParameterRegistry registry,
    uint8_t cq_id,
    SubDeviceManagerId sub_device_manager_id) :
    mesh_device(&mesh_device),
    descriptor(std::move(descriptor)),
    command_buffer(std::move(command_buffer)),
    retained_binary_buffers(std::move(retained_binary_buffers)),
    registry(std::move(registry)),
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
    registry = {};
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

void CommandList::Impl::update_args(const CmdListArgPatch& patch, bool blocking) {
    validate();

    // Words are applied to copies of their windows. The registry keeps the old contents until the patch is enqueued,
    // so a rejected patch leaves the command list unchanged.
    std::map<std::pair<size_t, uint32_t>, std::vector<uint32_t>> staged_windows;
    auto stage = [&](const detail::PatchTarget& target, ttsl::Span<const uint32_t> words) {
        for (uint32_t i = 0; i < words.size(); ++i) {
            const uint32_t offset = target.stream_offset + i * sizeof(uint32_t);
            const uint32_t window_offset = offset - offset % registry.window_size;
            auto [window, inserted] = staged_windows.try_emplace({target.range_index, window_offset});
            if (inserted) {
                window->second = registry.windows[target.range_index].at(window_offset);
            }
            window->second[(offset - window_offset) / sizeof(uint32_t)] = words[i];
        }
    };

    for (const auto& [name, value] : patch.runtime_args) {
        const auto it = registry.runtime_targets.find(name);
        TT_FATAL(it != registry.runtime_targets.end(), "Unknown command-list runtime parameter '{}'", *name);
        for (const auto& target : it->second) {
            stage(target, ttsl::Span<const uint32_t>(&value, 1));
        }
    }
    for (const auto& [name, value] : patch.common_runtime_args) {
        const auto it = registry.common_runtime_targets.find(name);
        TT_FATAL(
            it != registry.common_runtime_targets.end(), "Unknown command-list common-runtime parameter '{}'", *name);
        for (const auto& target : it->second) {
            stage(target, ttsl::Span<const uint32_t>(&value, 1));
        }
    }
    std::vector<uint32_t> tensor_words;
    for (const auto& [name, argument] : patch.tensor_args) {
        const auto it = registry.tensor_targets.find(name);
        TT_FATAL(it != registry.tensor_targets.end(), "Unknown command-list tensor parameter '{}'", *name);
        const auto& tensor = mesh_tensor_of(argument);
        TT_FATAL(
            &tensor.device() == mesh_device,
            "Command-list tensor parameter '{}' belongs to a different MeshDevice",
            *name);
        for (const auto& [target, binding] : it->second) {
            TT_FATAL(
                tensorspecs_match_with_relaxation(tensor.tensor_spec(), binding->expected_spec, binding->relaxations),
                "Command-list tensor parameter '{}' does not match its declared TensorSpec",
                *name);
            tensor_words.clear();
            EmitBindingCrtaValues(binding->binding, tensor, [&](uint32_t word) { tensor_words.push_back(word); });
            TT_FATAL(tensor_words.size() == target.num_words, "Command-list tensor parameter '{}' changed size", *name);
            stage(target, tensor_words);
        }
    }

    std::vector<MeshBufferPatch> patches;
    patches.reserve(staged_windows.size());
    for (const auto& [key, words] : staged_windows) {
        patches.push_back({.device_range = registry.device_ranges[key.first], .offset = key.second, .data = words});
    }
    auto& cq = mesh_device->mesh_command_queue(bound_cq_id);
    as_fd_queue(cq).enqueue_command_list_patch(*command_buffer, patches);
    for (auto& [key, words] : staged_windows) {
        registry.windows[key.first][key.second] = std::move(words);
    }
    if (blocking) {
        cq.finish();
    }
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
    const std::vector<uint32_t>& exec_buf_end,
    size_t range_index,
    CommandListParameterRegistry& registry) {
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

        const auto& captured = *programs[i];
        size_t num_recorded = 0;
        program_dispatch::for_each_program_command_sequence_chunk(
            command_sequence,
            node.dispatch_metadata.stall_first,
            node.dispatch_metadata.stall_before_program,
            node.dispatch_metadata.send_binary,
            [&](const void* chunk, uint32_t chunk_size) {
                const uint32_t chunk_offset = append_command_bytes(bytes, chunk, chunk_size);
                num_recorded += record_patch_targets(
                    captured, command_sequence, chunk, chunk_size, chunk_offset, range_index, registry);
            });
        TT_FATAL(
            num_recorded == captured.runtime_patches.size() + captured.common_runtime_patches.size() +
                                captured.tensor_patches.size(),
            "Failed to locate every command list parameter in the serialized command stream");

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
    auto& registry = assembly.registry;
    registry.device_ranges = device_ranges;
    registry.window_size = mesh_device.impl().metal_env().get_hal().get_alignment(HalMemType::DRAM);
    for (size_t range_index = 0; range_index < device_ranges.size(); ++range_index) {
        auto serialized = serialize_range(
            mesh_device,
            dispatch_state,
            staged_nodes_copy,
            device_ranges[range_index],
            exec_buf_end,
            range_index,
            registry);
        max_command_list_size = std::max(max_command_list_size, serialized.data.size());
        assembly.serialized_ranges.push_back(std::move(serialized));
    }
    capture_patch_windows(registry, assembly.serialized_ranges);

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

void CommandListBuilderImpl::resolve_parameters(
    StagedCommandListNode& staged_node, const CmdListParameters& parameters) const {
    auto find_captured_program = [&](const Program& program) -> CapturedProgram& {
        for (auto& captured : staged_node.programs) {
            if (captured.trace_node.program.get() == &program.impl()) {
                return captured;
            }
        }
        TT_THROW("Command list parameter references a Program that is not in the recorded MeshWorkload");
    };
    auto command_sequence_of = [&](CapturedProgram& captured) -> const ProgramCommandSequence& {
        return captured.trace_node.program->get_trace_cached_program_command_sequences().at(*sub_device_manager_id);
    };
    auto rta_schema_of = [](CapturedProgram& captured, const KernelSpecName& kernel_name) {
        const auto* schema = captured.trace_node.program->get_kernel_rta_schema(*kernel_name);
        TT_FATAL(schema != nullptr, "Kernel '{}' has no runtime argument schema", kernel_name);
        return schema;
    };

    for (const auto& [name, infos] : parameters.runtime_parameters) {
        for (const auto& info : infos) {
            TT_FATAL(!info.nodes.empty(), "Command list runtime parameter '{}' must name at least one node", *name);
            auto& captured = find_captured_program(info.program.get());
            const auto* schema = rta_schema_of(captured, info.kernel_name);
            const auto slot = schema->runtime_arg_name_to_slot.find(info.arg_name);
            TT_FATAL(
                slot != schema->runtime_arg_name_to_slot.end(),
                "Runtime argument '{}' is not declared by kernel '{}'",
                info.arg_name,
                info.kernel_name);
            auto kernel = captured.trace_node.program->get_kernel_by_spec_name(*info.kernel_name);
            for (const auto& node : info.nodes) {
                for (const auto& location : resolve_arg_locations(
                         kernel->runtime_args_data(node), slot->second, 1, command_sequence_of(captured))) {
                    captured.runtime_patches.push_back({.name = name, .location = location});
                }
            }
        }
    }

    for (const auto& [name, infos] : parameters.common_runtime_parameters) {
        for (const auto& info : infos) {
            auto& captured = find_captured_program(info.program.get());
            const auto* schema = rta_schema_of(captured, info.kernel_name);
            const auto slot = schema->common_runtime_arg_name_to_slot.find(info.arg_name);
            TT_FATAL(
                slot != schema->common_runtime_arg_name_to_slot.end(),
                "Common runtime argument '{}' is not declared by kernel '{}'",
                info.arg_name,
                info.kernel_name);
            auto kernel = captured.trace_node.program->get_kernel_by_spec_name(*info.kernel_name);
            for (const auto& location : resolve_arg_locations(
                     kernel->common_runtime_args_data(), slot->second, 1, command_sequence_of(captured))) {
                captured.common_runtime_patches.push_back({.name = name, .location = location});
            }
        }
    }

    for (const auto& [name, infos] : parameters.tensor_parameters) {
        for (const auto& info : infos) {
            auto& captured = find_captured_program(info.program.get());
            auto& program = *captured.trace_node.program;
            const auto* expected_spec = program.get_tensor_parameter_layout(*info.param_name);
            TT_FATAL(expected_spec != nullptr, "TensorParameter '{}' is not declared by the program", info.param_name);
            const auto relaxations = program.get_tensor_parameter_relaxations(*info.param_name);
            bool bound = false;
            for (const auto& kernel_name : program.get_registered_kernel_names()) {
                auto kernel = program.get_kernel_by_spec_name(kernel_name);
                for (const auto& handle : kernel->tensor_binding_handles()) {
                    if (handle.tensor_parameter_name != *info.param_name) {
                        continue;
                    }
                    bound = true;
                    auto binding = std::make_shared<const TensorPatchBinding>(TensorPatchBinding{
                        .binding = handle, .expected_spec = *expected_spec, .relaxations = relaxations});
                    for (const auto& location : resolve_arg_locations(
                             kernel->common_runtime_args_data(),
                             handle.addr_crta_offset / sizeof(uint32_t),
                             1 + handle.num_runtime_field_crta_words,
                             command_sequence_of(captured))) {
                        captured.tensor_patches.push_back({.name = name, .location = location, .binding = binding});
                    }
                }
            }
            TT_FATAL(bound, "TensorParameter '{}' is not bound by any kernel", info.param_name);
        }
    }
}

void CommandListBuilderImpl::add(MeshWorkload& workload, const CmdListParameters& parameters) {
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
    resolve_parameters(staged_node, parameters);

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
        std::move(assembly.registry),
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

void CommandListBuilder::add(MeshWorkload& workload, const CmdListParameters& parameters) {
    TT_FATAL(impl_ != nullptr, "CommandListBuilder has been moved from");
    impl_->add(workload, parameters);
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

void CommandList::update_args(const CmdListArgPatch& patch, bool blocking) {
    TT_FATAL(impl_ != nullptr, "CommandList has been moved from");
    impl_->update_args(patch, blocking);
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
