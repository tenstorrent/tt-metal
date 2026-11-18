// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/command_list.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>

#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/workload/specification/tensor_spec_relaxations.hpp>
#include <tt_stl/fmt.hpp>
#include <tt-logger/tt-logger.hpp>

#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/distributed/mesh_coord_utils.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/distributed/mesh_workload_impl.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/buffers/tensor_prefetcher_manager.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/impl/context/metal_env_impl.hpp"
#include "tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "tt_metal/impl/dataflow_buffer/prefetcher_pipe.hpp"
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
#include "hostdev/remote_dfb_constants.h"
#include <internal/service/service_core_manager.hpp>

namespace tt::tt_metal {

using distributed::build_go_signal_sequence;
using distributed::coordinate_index_in_range;
using distributed::DeviceLocalBufferConfig;
using distributed::FDMeshCommandQueue;
using distributed::MeshBuffer;
using distributed::MeshBufferPatch;
using distributed::MeshCommandQueue;
using distributed::MeshCoordinateRange;
using distributed::MeshDevice;
using distributed::MeshShape;
using distributed::MeshWorkload;
using distributed::partition_mesh_coordinate_ranges;
using distributed::PerDeviceRuntimeArgs;
using distributed::ReplicatedBufferConfig;
using distributed::resolve_per_device_runtime_args;
using distributed::ResolvedPerDeviceRuntimeArgsField;
using distributed::snapshot_per_device_runtime_args;

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

using PrefetcherPipeBindingKey = ::tt::tt_metal::detail::ProgramImpl::PrefetcherPipeBindingKey;

// A recorded (program, PrefetcherPipeParameter) and the pipe it currently runs.
struct PrefetcherPipeBindingSnapshot {
    CmdListPrefetcherPipeArgName name;
    experimental::PrefetcherPipeParamName param_name;
    const MeshDevice* device = nullptr;
    experimental::PrefetcherPipeIdentity identity{0};
    PrefetcherPipeBindingKey key;
};

enum class PrefetcherPipeAddress : uint8_t { Config, Buffer };

// A word that holds (address + offset) >> shift, for address one of its pipes' config page or ring addresses.
struct PrefetcherPipeWord {
    PrefetcherPipeAddress address = PrefetcherPipeAddress::Config;
    uint32_t offset = 0;
    uint8_t shift = 0;
    // Indices into the recorded PrefetcherPipeBindingSnapshots of the parameters whose pipes the word serves.
    std::vector<size_t> owners;
    // The word also serves a pipe bound to a parameter that is not a command-list parameter, so it cannot change.
    bool has_unlisted_owner = false;
};

struct CapturedPrefetcherPipePatch {
    const uint8_t* dst = nullptr;
    PrefetcherPipeWord word;
};

struct CapturedProgram {
    MeshCoordinateRange device_range;
    TraceNode trace_node;
    // The program's per-device runtime args as recorded (null when it has none). A program with them is serialized
    // once per device, each stream carrying that device's values.
    std::shared_ptr<const PerDeviceRuntimeArgs> per_device_runtime_args;
    std::vector<CapturedRuntimePatch> runtime_patches;
    std::vector<CapturedCommonRuntimePatch> common_runtime_patches;
    std::vector<CapturedTensorPatch> tensor_patches;
    std::vector<CapturedPrefetcherPipePatch> prefetcher_pipe_patches;
};

struct StagedCommandListNode {
    std::vector<CapturedProgram> programs;
    bool multicast_go_signals = false;
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

struct PrefetcherPipePatchTarget {
    PatchTarget target;
    PrefetcherPipeWord word;
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
    // Footprint of every recorded (program, TensorParameter) binding. Only DRAM footprints are checked.
    std::vector<BoundTensorFootprint> tensor_regions;
    // Indices into tensor_regions that each command-list tensor parameter patches.
    std::unordered_map<CmdListTensorArgName, std::vector<size_t>> tensor_region_slots;
    std::vector<PrefetcherPipeBindingSnapshot> prefetcher_pipe_bindings;
    // Indices into prefetcher_pipe_bindings that each command-list PrefetcherPipe parameter patches.
    std::unordered_map<CmdListPrefetcherPipeArgName, std::vector<size_t>> prefetcher_pipe_binding_slots;
    std::vector<PrefetcherPipePatchTarget> prefetcher_pipe_targets;
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
    auto locate_words = [&](const void* dst, uint32_t num_words) -> std::optional<detail::PatchTarget> {
        const auto target = reinterpret_cast<uintptr_t>(dst);
        if (target < chunk_begin || target + num_words * sizeof(uint32_t) > chunk_begin + chunk_size) {
            return std::nullopt;
        }
        return detail::PatchTarget{
            .range_index = range_index,
            .stream_offset = chunk_offset + static_cast<uint32_t>(target - chunk_begin),
            .num_words = num_words};
    };
    auto locate = [&](const detail::CapturedArgLocation& location) {
        const auto& update = command_sequence.rta_updates.at(location.rta_update_index);
        return locate_words(static_cast<const uint8_t*>(update.dst) + location.byte_offset, location.num_words);
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
    for (const auto& patch : program.prefetcher_pipe_patches) {
        if (auto target = locate_words(patch.dst, 1)) {
            registry.prefetcher_pipe_targets.push_back({.target = *target, .word = patch.word});
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
    for (const auto& pipe_target : registry.prefetcher_pipe_targets) {
        capture(pipe_target.target);
    }
}

uint32_t read_word(const uint8_t* src) {
    uint32_t word = 0;
    std::memcpy(&word, src, sizeof(word));
    return word;
}

// Appends to captured.prefetcher_pipe_patches every word of its commands that holds an address of a pipe bound to one
// of bindings, each paired with its index among the recorded PrefetcherPipeBindingSnapshots: the config page address
// in each copy of the PrefetcherPipe index, and the ring addresses in each relay DFB config.
void record_prefetcher_pipe_words(
    detail::CapturedProgram& captured,
    const ProgramCommandSequence& command_sequence,
    const Hal& hal,
    const std::vector<std::pair<size_t, const detail::PrefetcherPipeBindingSnapshot*>>& bindings) {
    const auto& program = *captured.trace_node.program;
    const auto& per_core_pipes = program.get_per_core_prefetcher_pipes();
    auto identities_on = [&](const CoreRange& cores, uint8_t prefetcher_pipe_id) {
        std::set<experimental::PrefetcherPipeIdentity> identities;
        for (const CoreCoord& core : cores) {
            const auto it = per_core_pipes.find(core);
            if (it == per_core_pipes.end()) {
                continue;
            }
            for (const auto& participant : it->second) {
                if (participant.prefetcher_pipe_id == prefetcher_pipe_id && participant.is_bound()) {
                    identities.insert(participant.pipe_identity);
                }
            }
        }
        return identities;
    };
    // The word serving the pipes bound to slot prefetcher_pipe_id on cores, or nullopt if none is one of bindings.
    auto make_word = [&](const CoreRange& cores, uint8_t prefetcher_pipe_id, detail::PrefetcherPipeAddress address)
        -> std::optional<std::pair<detail::PrefetcherPipeWord, const detail::PrefetcherPipeBindingSnapshot*>> {
        const auto identities = identities_on(cores, prefetcher_pipe_id);
        detail::PrefetcherPipeWord word{.address = address};
        const detail::PrefetcherPipeBindingSnapshot* owner = nullptr;
        std::set<experimental::PrefetcherPipeIdentity> listed;
        for (const auto& [index, binding] : bindings) {
            if (identities.contains(binding->identity)) {
                word.owners.push_back(index);
                listed.insert(binding->identity);
                owner = binding;
            }
        }
        if (owner == nullptr) {
            return std::nullopt;
        }
        word.has_unlisted_owner = listed.size() != identities.size();
        return std::pair{std::move(word), owner};
    };

    const uint8_t num_slots = program.num_prefetcher_pipe_slots();
    for (const auto& location : command_sequence.prefetcher_pipe_index_locations) {
        for (uint8_t prefetcher_pipe_id = 0; prefetcher_pipe_id < num_slots; ++prefetcher_pipe_id) {
            auto word = make_word(location.logical_cores, prefetcher_pipe_id, detail::PrefetcherPipeAddress::Config);
            if (!word) {
                continue;
            }
            const uint32_t byte_offset =
                (REMOTE_DFB_REGION_HEADER_WORDS + prefetcher_pipe_id * UINT32_WORDS_PER_REMOTE_DFB_CONFIG) *
                sizeof(uint32_t);
            TT_FATAL(
                byte_offset + sizeof(uint32_t) <= location.size &&
                    read_word(location.dst + byte_offset) == word->second->key.config_address,
                "Internal error: PrefetcherPipe slot {} is not at its expected place in the PrefetcherPipe index",
                prefetcher_pipe_id);
            captured.prefetcher_pipe_patches.push_back(
                {.dst = location.dst + byte_offset, .word = std::move(word->first)});
        }
    }

    const uint32_t dfb_size =
        program.get_program_config(hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)).dfb_size;
    const auto dfb_ranges = program.dataflow_buffers_unique_coreranges();
    for (size_t i = 0; i < command_sequence.dfb_config_updates.size(); ++i) {
        const auto& update = command_sequence.dfb_config_updates[i];
        const auto& recorded = captured.trace_node.dfb_configs_payloads.at(i);
        const auto range = std::ranges::find(dfb_ranges, update.logical_core, &CoreRange::start_coord);
        TT_FATAL(
            range != dfb_ranges.end(), "Internal error: no DFB core range starts at {}", update.logical_core.str());
        std::optional<std::vector<experimental::dfb::detail::DfbConfigAddressWord>> address_words;
        for (uint8_t prefetcher_pipe_id = 0; prefetcher_pipe_id < num_slots; ++prefetcher_pipe_id) {
            const auto& slot = program.get_prefetcher_pipe_slot(prefetcher_pipe_id);
            if (!slot.relay_dfb_host_id.has_value()) {
                continue;
            }
            const auto relay = program.get_dataflow_buffer(*slot.relay_dfb_host_id);
            if (std::ranges::find(update.dataflow_buffers, relay) == update.dataflow_buffers.end()) {
                continue;
            }
            auto word = make_word(*range, prefetcher_pipe_id, detail::PrefetcherPipeAddress::Buffer);
            if (!word) {
                continue;
            }
            TT_FATAL(
                !relay->alias_primary_id.has_value(),
                "PrefetcherPipe relay DFB {} is an alias of DFB {}, which command lists cannot patch",
                relay->id,
                *relay->alias_primary_id);
            if (!address_words.has_value()) {
                address_words.emplace();
                std::vector<uint8_t> scratch(dfb_size, 0);
                experimental::dfb::detail::serialize_dfb_config_for_core(
                    update.logical_core, update.dataflow_buffers, scratch, &*address_words);
            }
            const uint32_t ring_address = word->second->key.buffer_address;
            const uint32_t ring_size = word->second->key.ring_size;
            for (const auto& address_word : *address_words) {
                if (address_word.dfb_id != relay->id &&
                    std::ranges::find(relay->alias_secondary_ids, address_word.dfb_id) ==
                        relay->alias_secondary_ids.end()) {
                    continue;
                }
                TT_FATAL(
                    address_word.address >= ring_address && address_word.address - ring_address <= ring_size &&
                        address_word.byte_offset + sizeof(uint32_t) <= recorded.size() &&
                        read_word(recorded.data() + address_word.byte_offset) ==
                            address_word.address >> address_word.shift,
                    "Internal error: relay DFB {} of PrefetcherPipe slot {} does not hold the ring of its pipe",
                    relay->id,
                    prefetcher_pipe_id);
                auto relay_word = word->first;
                relay_word.offset = address_word.address - ring_address;
                relay_word.shift = address_word.shift;
                captured.prefetcher_pipe_patches.push_back(
                    {.dst = update.dst + address_word.byte_offset, .word = std::move(relay_word)});
            }
        }
    }
}

void append_go_signal_sequence(
    std::vector<uint32_t>& output,
    MeshDevice& mesh_device,
    SubDeviceId sub_device,
    uint32_t expected_workers,
    uint32_t go_ring_slot) {
    program_dispatch::ProgramDispatchMetadata dispatch_metadata;
    // GO-signal-only sequences carry no kernel binary; marking them cached skips the prefetcher ring-buffer offset
    // command.
    dispatch_metadata.prefetcher_cache_info.is_cached = true;
    HostMemDeviceCommand commands = build_go_signal_sequence(
        &mesh_device,
        sub_device,
        expected_workers,
        /*send_mcast=*/true,
        go_ring_slot,
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

    void add(MeshCommandQueue& binary_upload_cq, MeshWorkload& workload, const CmdListParameters& parameters);
    void add_tensor_prefetcher_request(
        const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes,
        ttsl::optional_reference<const distributed::MeshCoordinateRangeSet> device_subset,
        const std::vector<experimental::TensorPrefetcherInput>& input_tensors);
    CommandList build(MeshCommandQueue& cq) const;
    MeshDevice& device() const;
    void clear();
    void deallocate();

private:
    struct OfflineDispatchState;

    // Appends the location of each parameter to the runtime_patches, common_runtime_patches, tensor_patches, or
    // prefetcher_pipe_patches of the CapturedProgram in staged_node that it belongs to. Appends a snapshot of each
    // PrefetcherPipe parameter to new_pipe_bindings; patches index the snapshots as if appended to
    // prefetcher_pipe_bindings.
    void resolve_parameters(
        StagedCommandListNode& staged_node,
        const CmdListParameters& parameters,
        std::vector<PrefetcherPipeBindingSnapshot>& new_pipe_bindings) const;
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
    MeshDevice& mesh_device;
    SubDeviceManagerId sub_device_manager_id;
    std::vector<StagedCommandListNode> staged_nodes;
    // Indexed by sub-device. Every staged node adds the same workers and GO signals on every device range, through its
    // program or a dummy GO signal, so these are the same for the whole mesh.
    DispatchArray<std::optional<TraceWorkerDescriptor>> worker_descriptors;
    std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers;
    // Become CommandListParameterRegistry::tensor_regions and tensor_region_slots at build().
    std::vector<BoundTensorFootprint> tensor_regions;
    std::unordered_map<CmdListTensorArgName, std::vector<size_t>> tensor_region_slots;
    // Become CommandListParameterRegistry::prefetcher_pipe_bindings and prefetcher_pipe_binding_slots at build().
    std::vector<PrefetcherPipeBindingSnapshot> prefetcher_pipe_bindings;
    std::unordered_map<CmdListPrefetcherPipeArgName, std::vector<size_t>> prefetcher_pipe_binding_slots;
    distributed::TensorPrefetcherManager::CapturedRequests tensor_prefetcher_requests;
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
        distributed::TensorPrefetcherManager::CapturedRequests tensor_prefetcher_requests,
        uint8_t cq_id,
        SubDeviceManagerId sub_device_manager_id);
    ~Impl();

    void deallocate();
    void replay(bool blocking, bool check_dram_overlap) const;
    void update_args(const CmdListArgPatch& patch, bool blocking);
    MeshDevice& get_device() const;
    uint8_t get_cq_id() const;

private:
    void validate() const;
    // Fails if a live DRAM allocation overlaps a tensor region without exactly matching its address and size.
    void verify_no_dram_overlap() const;
    void release_resources() noexcept;

    MeshDevice* mesh_device = nullptr;
    detail::CommandListDescriptor descriptor;
    std::shared_ptr<MeshBuffer> command_buffer;
    // Pins kernel-binary MeshBuffers referenced by the serialized commands.
    std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers;
    detail::CommandListParameterRegistry registry;
    // Re-queued on the Tensor prefetcher before each replay's commands.
    distributed::TensorPrefetcherManager::CapturedRequests tensor_prefetcher_requests;
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
    distributed::TensorPrefetcherManager::CapturedRequests tensor_prefetcher_requests,
    uint8_t cq_id,
    SubDeviceManagerId sub_device_manager_id) :
    mesh_device(&mesh_device),
    descriptor(std::move(descriptor)),
    command_buffer(std::move(command_buffer)),
    retained_binary_buffers(std::move(retained_binary_buffers)),
    registry(std::move(registry)),
    tensor_prefetcher_requests(std::move(tensor_prefetcher_requests)),
    bound_cq_id(cq_id),
    sub_device_manager_id(sub_device_manager_id) {
    mesh_device.impl().register_command_list();
}

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
    tensor_prefetcher_requests = {};
    valid = false;
    mesh_device->impl().unregister_command_list();
}

void CommandList::Impl::deallocate() {
    if (!valid) {
        return;
    }
    as_fd_queue(mesh_device->mesh_command_queue(bound_cq_id)).drain_device_work();
    release_resources();
}

void CommandList::Impl::verify_no_dram_overlap() const {
    const auto& allocator = mesh_device->allocator_impl();
    const auto allocations = allocator->get_dram_allocations();
    for (const auto& region : registry.tensor_regions) {
        if (region.buffer_type != BufferType::DRAM || region.size_per_bank == 0) {
            continue;
        }
        const DeviceAddr end = region.address + region.size_per_bank;
        // Allocations are disjoint and sorted, so their ends are sorted too. An exact match is the only allocation
        // overlapping the range.
        const auto first_overlap = std::partition_point(
            allocations.begin(), allocations.end(), [&](const auto& a) { return a.second <= region.address; });
        if (first_overlap == allocations.end() || first_overlap->first >= end) {
            continue;
        }
        const auto& [alloc_begin, alloc_end] = *first_overlap;
        TT_FATAL(
            alloc_begin == region.address && alloc_end == end,
            "Command-list DRAM range [{:#x}, {:#x}) overlaps a live allocation at [{:#x}, {:#x}) that does not "
            "exactly match it",
            region.address,
            end,
            alloc_begin,
            alloc_end);
        if (region.buffer_id.has_value() && allocator->is_buffer_allocated_at(*region.buffer_id, region.address)) {
            continue;
        }
        log_warning(
            tt::LogMetal,
            "Command-list DRAM range [{:#x}, {:#x}) is exactly occupied by a live allocation other than the tensor "
            "bound to it. Replay will read or overwrite that allocation. This is expected for a same-size replacement "
            "tensor, but not for any other buffer, such as a command-list command buffer.",
            region.address,
            end);
    }
}

void CommandList::Impl::replay(bool blocking, bool check_dram_overlap) const {
    validate();
    if (check_dram_overlap) {
        verify_no_dram_overlap();
    }
    // Queued ahead of the commands, as trace replay does, so the prefetcher can start filling the pipes while the
    // consumers are dispatched. Queued under the same API lock as the commands, after the list is accepted, so a
    // rejected list sends nothing and concurrent replays reach the pipes in the order their consumers run.
    std::function<void()> queue_prefetcher_requests;
    if (!tensor_prefetcher_requests.requests.empty()) {
        queue_prefetcher_requests = [this] {
            mesh_device->impl().tensor_prefetcher(mesh_device).replay_captured_nolock(tensor_prefetcher_requests);
        };
    }
    as_fd_queue(mesh_device->mesh_command_queue(bound_cq_id))
        .enqueue_command_list(
            descriptor.worker_descriptors(),
            descriptor.sub_device_ids(),
            *command_buffer,
            sub_device_manager_id,
            blocking,
            queue_prefetcher_requests);
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
        const uint32_t word = value.to_uint32();
        for (const auto& target : it->second) {
            stage(target, ttsl::Span<const uint32_t>(&word, 1));
        }
    }
    for (const auto& [name, value] : patch.common_runtime_args) {
        const auto it = registry.common_runtime_targets.find(name);
        TT_FATAL(
            it != registry.common_runtime_targets.end(), "Unknown command-list common-runtime parameter '{}'", *name);
        const uint32_t word = value.to_uint32();
        for (const auto& target : it->second) {
            stage(target, ttsl::Span<const uint32_t>(&word, 1));
        }
    }
    std::vector<uint32_t> tensor_words;
    std::vector<std::pair<size_t, detail::BoundTensorFootprint>> staged_regions;
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
        const auto footprint = ::tt::tt_metal::detail::make_bound_tensor_footprint(tensor);
        for (const size_t slot : registry.tensor_region_slots.at(name)) {
            staged_regions.emplace_back(slot, footprint);
        }
    }

    // The new pipe of each patched entry of registry.prefetcher_pipe_bindings.
    std::map<size_t, const experimental::PrefetcherPipeImpl*> staged_pipes;
    // The new pipe of each replaced pipe, or null if two parameters bound to it get different ones.
    std::map<experimental::PrefetcherPipeIdentity, const experimental::PrefetcherPipe*> replacements;
    for (const auto& [name, argument] : patch.prefetcher_pipe_args) {
        const auto it = registry.prefetcher_pipe_binding_slots.find(name);
        TT_FATAL(
            it != registry.prefetcher_pipe_binding_slots.end(),
            "Unknown command-list PrefetcherPipe parameter '{}'",
            *name);
        const auto& pipe = argument.get().impl();
        for (const size_t slot : it->second) {
            const auto [replacement, inserted] =
                replacements.try_emplace(registry.prefetcher_pipe_bindings[slot].identity, &argument.get());
            if (!inserted && replacement->second != &argument.get()) {
                replacement->second = nullptr;
            }
            const auto& binding = registry.prefetcher_pipe_bindings[slot];
            TT_FATAL(
                pipe.get_device() == binding.device,
                "Command-list PrefetcherPipe parameter '{}' supplies a pipe on a different MeshDevice than the Program "
                "of PrefetcherPipeParameter '{}'",
                *name,
                binding.param_name);
            TT_FATAL(
                binding.key.same_geometry(::tt::tt_metal::detail::ProgramImpl::prefetcher_pipe_binding_key(pipe)),
                "Command-list PrefetcherPipe parameter '{}' supplies a pipe whose sender, receivers, ring size, entry "
                "size or credit lanes differ from those of the pipe PrefetcherPipeParameter '{}' was recorded with",
                *name,
                binding.param_name);
            staged_pipes.emplace(slot, &pipe);
        }
    }
    for (const auto& [target, word] : registry.prefetcher_pipe_targets) {
        const auto patched =
            std::ranges::find_if(word.owners, [&](size_t owner) { return staged_pipes.contains(owner); });
        if (patched == word.owners.end()) {
            continue;
        }
        const auto& patched_name = *registry.prefetcher_pipe_bindings[*patched].name;
        const char* address_name = word.address == detail::PrefetcherPipeAddress::Config ? "config page" : "ring";
        TT_FATAL(
            !word.has_unlisted_owner,
            "Command-list PrefetcherPipe parameter '{}' shares its {} address with a pipe whose parameter is not a "
            "command-list parameter, so it cannot be patched",
            patched_name,
            address_name);
        std::optional<uint32_t> address;
        for (const size_t owner : word.owners) {
            const auto staged = staged_pipes.find(owner);
            const auto& owner_name = *registry.prefetcher_pipe_bindings[owner].name;
            TT_FATAL(
                staged != staged_pipes.end(),
                "Command-list PrefetcherPipe parameters '{}' and '{}' share a {} address. Patch them in one "
                "update_args() call",
                patched_name,
                owner_name,
                address_name);
            const uint32_t owner_address = word.address == detail::PrefetcherPipeAddress::Config
                                               ? staged->second->config_address()
                                               : staged->second->buffer_address();
            TT_FATAL(
                !address.has_value() || *address == owner_address,
                "Command-list PrefetcherPipe parameters '{}' and '{}' share a {} address, so their new pipes must too",
                patched_name,
                owner_name,
                address_name);
            address = owner_address;
        }
        const uint32_t value = (*address + word.offset) >> word.shift;
        stage(target, ttsl::Span<const uint32_t>(&value, 1));
    }
    std::optional<distributed::TensorPrefetcherManager::CapturedRequests> retargeted_requests;
    if (!replacements.empty() && !tensor_prefetcher_requests.requests.empty()) {
        retargeted_requests = mesh_device->impl()
                                  .tensor_prefetcher(mesh_device)
                                  .retarget_captured(tensor_prefetcher_requests, replacements);
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
    for (auto& [slot, region] : staged_regions) {
        registry.tensor_regions[slot] = region;
    }
    for (const auto& [slot, pipe] : staged_pipes) {
        auto& binding = registry.prefetcher_pipe_bindings[slot];
        binding.identity = pipe->identity();
        binding.key = ::tt::tt_metal::detail::ProgramImpl::prefetcher_pipe_binding_key(*pipe);
    }
    if (retargeted_requests.has_value()) {
        tensor_prefetcher_requests = std::move(*retargeted_requests);
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
    explicit OfflineDispatchState(MeshDevice& mesh_device) :
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
            if (program.per_device_runtime_args != nullptr) {
                for (const auto& coord : *local_device_range) {
                    partition_mesh_coordinate_ranges(device_ranges, MeshCoordinateRange(coord));
                }
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
    // (stream byte offset, size) of every per-device runtime argument written into this range's stream
    std::vector<std::pair<uint32_t, uint32_t>> per_device_words;
    auto& launch_state = dispatch_state.launch_state;
    DispatchArray<uint32_t> expected_workers{};

    // Each staged node with no program on this range becomes a dummy GO signal at the start of the stream, so
    // launch-message write pointers and expected worker counts match across all devices. They go at the start because
    // the last program may still be running when the stream ends.
    for (size_t i = 0; i < staged_nodes.size(); ++i) {
        const auto& staged_node = staged_nodes[i];
        if (programs[i] != nullptr || !staged_node.multicast_go_signals) {
            continue;
        }
        const auto sub_device = staged_node.sub_device_id;
        append_go_signal_sequence(
            bytes, mesh_device, sub_device, expected_workers[*sub_device], launch_state[*sub_device].get_mcast_wptr());
        launch_state[*sub_device].inc_mcast_wptr(1);
        expected_workers[*sub_device] += staged_node.num_workers;
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
        program_dispatch::update_traced_program_dispatch_commands(
            node,
            command_sequence,
            worker_launch_state.get_mcast_wptr(),
            expected_workers[*sub_device],
            ProgramBinaryStatus::Committed);

        const auto& captured = *programs[i];
        // This device's values of the program's per-device runtime args (the range is one device; see
        // compute_device_ranges), written over the serialized words.
        std::vector<ResolvedPerDeviceRuntimeArgsField> per_device_fields;
        uint32_t coord_index = 0;
        if (captured.per_device_runtime_args != nullptr) {
            TT_ASSERT(range.shape().mesh_size() == 1);
            per_device_fields = resolve_per_device_runtime_args(*captured.per_device_runtime_args, command_sequence);
            coord_index = coordinate_index_in_range(captured.device_range, range.start_coord());
        }
        size_t num_recorded = 0;
        size_t num_fields_written = 0;
        program_dispatch::for_each_program_command_sequence_chunk(
            command_sequence,
            node.dispatch_metadata.stall_first,
            node.dispatch_metadata.stall_before_program,
            node.dispatch_metadata.send_binary,
            [&](const void* chunk, uint32_t chunk_size) {
                const uint32_t chunk_offset = append_command_bytes(bytes, chunk, chunk_size);
                const auto* chunk_begin = static_cast<const uint8_t*>(chunk);
                for (const auto& [location, field] : per_device_fields) {
                    const auto* loc = reinterpret_cast<const uint8_t*>(location);
                    const uint32_t field_bytes = field->num_words * sizeof(uint32_t);
                    if (loc < chunk_begin || loc + field_bytes > chunk_begin + chunk_size) {
                        continue;
                    }
                    const uint32_t stream_offset = chunk_offset + static_cast<uint32_t>(loc - chunk_begin);
                    // A device without a value keeps the program's own words, which a parameter may patch.
                    if (field->present[coord_index] != 0) {
                        std::memcpy(
                            reinterpret_cast<uint8_t*>(bytes.data()) + stream_offset,
                            field->values.data() + static_cast<size_t>(coord_index) * field->num_words,
                            field_bytes);
                        per_device_words.push_back({stream_offset, field_bytes});
                    }
                    ++num_fields_written;
                }
                num_recorded += record_patch_targets(
                    captured, command_sequence, chunk, chunk_size, chunk_offset, range_index, registry);
            });
        TT_FATAL(
            num_fields_written == per_device_fields.size(),
            "Failed to locate every per-device runtime argument in the serialized command stream");
        TT_FATAL(
            num_recorded == captured.runtime_patches.size() + captured.common_runtime_patches.size() +
                                captured.tensor_patches.size() + captured.prefetcher_pipe_patches.size(),
            "Failed to locate every command list parameter in the serialized command stream");

        if (staged_node.multicast_go_signals) {
            worker_launch_state.inc_mcast_wptr(1);
        }
        expected_workers[*sub_device] += node.num_workers;
    }

    // A command-list parameter under a per-device runtime argument would overwrite that device's value with the one
    // every device gets on update_args.
    if (!per_device_words.empty()) {
        auto check = [&](const PatchTarget& target) {
            if (target.range_index != range_index) {
                return;
            }
            const uint32_t end = target.stream_offset + target.num_words * static_cast<uint32_t>(sizeof(uint32_t));
            for (const auto& [offset, size] : per_device_words) {
                TT_FATAL(
                    end <= offset || offset + size <= target.stream_offset,
                    "A command list parameter overlaps a per-device runtime argument");
            }
        };
        for (const auto& [_, targets] : registry.runtime_targets) {
            std::ranges::for_each(targets, check);
        }
        for (const auto& [_, targets] : registry.common_runtime_targets) {
            std::ranges::for_each(targets, check);
        }
        for (const auto& [_, targets] : registry.tensor_targets) {
            for (const auto& target : targets) {
                check(target.target);
            }
        }
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
    OfflineDispatchState dispatch_state(mesh_device);

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

    // Command lists are counted against the trace region the same way traces are. MeshDevice rejects a trace capture
    // while a command list exists, and a command-list builder while a trace exists.
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

void CommandListBuilderImpl::resolve_parameters(
    StagedCommandListNode& staged_node,
    const CmdListParameters& parameters,
    std::vector<PrefetcherPipeBindingSnapshot>& new_pipe_bindings) const {
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
            TT_FATAL(
                !schema->runtime_array_arg_slots.contains(info.arg_name),
                "Runtime argument '{}' of kernel '{}' is an array argument, which command list parameters do not "
                "support yet",
                info.arg_name,
                info.kernel_name);
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
            TT_FATAL(
                !schema->common_runtime_array_arg_slots.contains(info.arg_name),
                "Common runtime argument '{}' of kernel '{}' is an array argument, which command list parameters do "
                "not support yet",
                info.arg_name,
                info.kernel_name);
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
            for (const auto& binding : program.get_dfb_storage_bindings()) {
                const auto* tensor_name = std::get_if<tt::tt_metal::TensorParamName>(&binding.source);
                TT_FATAL(
                    tensor_name == nullptr || tensor_name->get() != *info.param_name,
                    "TensorParameter '{}' backs borrowed-memory DFB {}, which command lists cannot patch",
                    info.param_name,
                    binding.dfb_id);
            }
            const auto relaxations = program.get_tensor_parameter_relaxations(*info.param_name);
            bool bound = false;
            for (const auto& kernel_name : program.get_registered_kernel_names()) {
                auto kernel = program.get_kernel_by_spec_name(kernel_name);
                for (const auto& handle : kernel->tensor_binding_handles()) {
                    if (handle.tensor_parameter_name != *info.param_name) {
                        continue;
                    }
                    TT_FATAL(
                        !handle.is_per_node,
                        "TensorParameter '{}' is per-node allocated, which command lists cannot patch",
                        info.param_name);
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

    std::map<CapturedProgram*, std::vector<size_t>> pipe_bindings_of;
    std::set<std::pair<const ::tt::tt_metal::detail::ProgramImpl*, std::string>> listed_pipe_parameters;
    for (const auto& [name, infos] : parameters.prefetcher_pipe_parameters) {
        for (const auto& info : infos) {
            auto& captured = find_captured_program(info.program.get());
            const auto& program = *captured.trace_node.program;
            const auto* binding = program.get_prefetcher_pipe_parameter(*info.param_name);
            TT_FATAL(
                binding != nullptr, "PrefetcherPipeParameter '{}' is not declared by the program", info.param_name);
            TT_FATAL(
                binding->is_bound(),
                "PrefetcherPipeParameter '{}' has no bound pipe. Call SetProgramRunArgs before recording the workload",
                info.param_name);
            TT_FATAL(
                listed_pipe_parameters.emplace(&program, *info.param_name).second,
                "PrefetcherPipeParameter '{}' is listed more than once in the command-list parameters",
                info.param_name);
            pipe_bindings_of[&captured].push_back(new_pipe_bindings.size());
            new_pipe_bindings.push_back(
                {.name = name,
                 .param_name = info.param_name,
                 .device = binding->device,
                 .identity = binding->bound_pipe_identity,
                 .key = binding->bound_pipe_key});
        }
    }
    const auto& hal = mesh_device.impl().metal_env().get_hal();
    for (auto& [captured, local_indices] : pipe_bindings_of) {
        std::vector<std::pair<size_t, const PrefetcherPipeBindingSnapshot*>> bindings;
        for (const size_t local_index : local_indices) {
            bindings.emplace_back(prefetcher_pipe_bindings.size() + local_index, &new_pipe_bindings[local_index]);
        }
        record_prefetcher_pipe_words(*captured, command_sequence_of(*captured), hal, bindings);
    }
}

void CommandListBuilderImpl::add(
    MeshCommandQueue& binary_upload_cq, MeshWorkload& workload, const CmdListParameters& parameters) {
    TT_FATAL(valid, "CommandListBuilder has been deallocated");
    TT_FATAL(binary_upload_cq.device() == &mesh_device, "Command queue belongs to a different MeshDevice");
    TT_FATAL(
        mesh_device.get_active_sub_device_manager_id() == sub_device_manager_id,
        "The active sub-device manager changed while recording a command list");
    validate_no_service_cores(workload, mesh_device);

    auto binary_buffer = workload.impl().prepare_for_command_list(binary_upload_cq);

    StagedCommandListNode staged_node;
    staged_node.multicast_go_signals = workload.impl().runs_on_noc_multicast_only_cores();
    const auto sub_devices = workload.impl().determine_sub_device_ids(&mesh_device);
    TT_FATAL(sub_devices.size() == 1, "A command-list workload must execute on one sub-device");
    staged_node.sub_device_id = *sub_devices.begin();
    staged_node.num_workers =
        staged_node.multicast_go_signals
            ? mesh_device.num_worker_cores(HalProgrammableCoreType::TENSIX, staged_node.sub_device_id)
            : 0;
    const uint32_t cache_size = mesh_device.impl().metal_context().dispatch_mem_map().ringbuffer_size();
    const uint32_t max_program_kernels_size = workload.impl().max_program_kernels_size();
    const bool use_prefetcher_cache = max_program_kernels_size != 0 && max_program_kernels_size <= cache_size;

    for (auto& [device_range, program] : workload.get_programs()) {
        const auto* per_device = workload.impl().per_device_runtime_args(device_range);
        staged_node.programs.push_back(
            {.device_range = device_range,
             .trace_node = program_dispatch::create_trace_node(
                 program.impl(), &mesh_device, staged_node.num_workers, use_prefetcher_cache),
             .per_device_runtime_args =
                 per_device != nullptr ? snapshot_per_device_runtime_args(*per_device) : nullptr});
    }
    std::vector<PrefetcherPipeBindingSnapshot> new_pipe_bindings;
    resolve_parameters(staged_node, parameters, new_pipe_bindings);

    std::vector<BoundTensorFootprint> new_regions;
    std::map<std::pair<const ::tt::tt_metal::detail::ProgramImpl*, std::string>, size_t> region_slot_of;
    for (auto& [device_range, program] : workload.get_programs()) {
        const auto& program_impl = program.impl();
        for (const auto& param_name : program_impl.get_registered_tensor_parameter_names()) {
            const auto* footprint = program_impl.get_bound_tensor_footprint(param_name);
            TT_FATAL(
                footprint != nullptr,
                "TensorParameter '{}' has no bound tensor. Call SetProgramRunArgs before recording the workload",
                param_name);
            region_slot_of.emplace(std::pair{&program_impl, param_name}, tensor_regions.size() + new_regions.size());
            new_regions.push_back(*footprint);
        }
    }
    std::unordered_map<CmdListTensorArgName, std::vector<size_t>> new_region_slots;
    for (const auto& [name, infos] : parameters.tensor_parameters) {
        for (const auto& info : infos) {
            new_region_slots[name].push_back(region_slot_of.at({&info.program.get().impl(), *info.param_name}));
        }
    }

    auto& worker = worker_descriptors[*staged_node.sub_device_id];
    if (!worker) {
        worker.emplace();
    }
    worker->num_completion_worker_cores += staged_node.num_workers;
    if (staged_node.multicast_go_signals) {
        ++worker->num_traced_programs_needing_go_signal_multicast;
    }
    staged_nodes.push_back(std::move(staged_node));
    if (binary_buffer) {
        retained_binary_buffers.push_back(std::move(binary_buffer));
    }
    tensor_regions.insert(tensor_regions.end(), new_regions.begin(), new_regions.end());
    for (auto& [name, slots] : new_region_slots) {
        auto& all_slots = tensor_region_slots[name];
        all_slots.insert(all_slots.end(), slots.begin(), slots.end());
    }
    for (auto& binding : new_pipe_bindings) {
        prefetcher_pipe_binding_slots[binding.name].push_back(prefetcher_pipe_bindings.size());
        prefetcher_pipe_bindings.push_back(std::move(binding));
    }
}

void CommandListBuilderImpl::add_tensor_prefetcher_request(
    const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes,
    ttsl::optional_reference<const distributed::MeshCoordinateRangeSet> device_subset,
    const std::vector<experimental::TensorPrefetcherInput>& input_tensors) {
    TT_FATAL(valid, "CommandListBuilder has been deallocated");
    mesh_device.impl()
        .tensor_prefetcher(&mesh_device)
        .capture(tensor_prefetcher_requests, prefetcher_pipes, device_subset, input_tensors);
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
    assembly.registry.tensor_regions = tensor_regions;
    assembly.registry.tensor_region_slots = tensor_region_slots;
    assembly.registry.prefetcher_pipe_bindings = prefetcher_pipe_bindings;
    assembly.registry.prefetcher_pipe_binding_slots = prefetcher_pipe_binding_slots;
    auto command_buffer = allocate_and_commit(cq, assembly.descriptor, assembly.serialized_ranges);
    return CommandList(std::make_unique<CommandList::Impl>(
        mesh_device,
        std::move(assembly.descriptor),
        std::move(command_buffer),
        retained_binary_buffers,
        std::move(assembly.registry),
        tensor_prefetcher_requests,
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
    tensor_regions.clear();
    tensor_region_slots.clear();
    prefetcher_pipe_bindings.clear();
    prefetcher_pipe_binding_slots.clear();
    tensor_prefetcher_requests = {};
}

void CommandListBuilderImpl::deallocate() {
    staged_nodes.clear();
    worker_descriptors = {};
    retained_binary_buffers.clear();
    tensor_regions.clear();
    tensor_region_slots.clear();
    prefetcher_pipe_bindings.clear();
    prefetcher_pipe_binding_slots.clear();
    tensor_prefetcher_requests = {};
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

void CommandListBuilder::add(
    MeshCommandQueue& binary_upload_cq, MeshWorkload& workload, const CmdListParameters& parameters) {
    TT_FATAL(impl_ != nullptr, "CommandListBuilder has been moved from");
    impl_->add(binary_upload_cq, workload, parameters);
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

detail::CommandListBuilderImpl& CommandListBuilder::impl() {
    TT_FATAL(impl_ != nullptr, "CommandListBuilder has been moved from");
    return *impl_;
}

CommandList::CommandList(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CommandList::CommandList(CommandList&&) noexcept = default;
CommandList& CommandList::operator=(CommandList&&) noexcept = default;
CommandList::~CommandList() = default;

void CommandList::replay(bool blocking, bool check_dram_overlap) const {
    TT_FATAL(impl_ != nullptr, "CommandList has been moved from");
    impl_->replay(blocking, check_dram_overlap);
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

void EnqueueCommandList(MeshCommandQueue& cq, CommandList& command_list, bool blocking, bool check_dram_overlap) {
    TT_FATAL(&command_list.device() == cq.device(), "CommandList belongs to a different MeshDevice");
    TT_FATAL(command_list.cq_id() == cq.id(), "CommandList was built for a different command queue");
    command_list.replay(blocking, check_dram_overlap);
}

}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

void AddTensorPrefetcherRequest(
    CommandListBuilder& builder,
    const std::vector<std::reference_wrapper<const PrefetcherPipe>>& prefetcher_pipes,
    ttsl::optional_reference<const distributed::MeshCoordinateRangeSet> device_subset,
    const std::vector<TensorPrefetcherInput>& input_tensors) {
    builder.impl().add_tensor_prefetcher_request(prefetcher_pipes, device_subset, input_tensors);
}

}  // namespace tt::tt_metal::experimental
