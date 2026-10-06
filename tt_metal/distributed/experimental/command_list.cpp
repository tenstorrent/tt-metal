// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/command_list.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <iterator>
#include <limits>
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
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/dispatch_mem_map.hpp"
#include "tt_metal/impl/dispatch/dispatch_settings.hpp"
#include "tt_metal/impl/dispatch/launch_message_ring_buffer_state.hpp"
#include "tt_metal/impl/dispatch/ringbuffer_cache.hpp"
#include "tt_metal/impl/dispatch/simple_trace_allocator.hpp"
#include "tt_metal/impl/internal/service/service_core_manager_impl.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"
#include "tt_metal/impl/program/program_impl.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "tt_metal/impl/trace/trace_node.hpp"
#include <internal/service/service_core_manager.hpp>

namespace tt::tt_metal::experimental {
using namespace tt::tt_metal::distributed;

namespace {

struct CommandListData {
    MeshCoordinateRange device_range = MeshCoordinateRange(MeshShape(0, 0));
    std::vector<uint32_t> data;
};

struct CommandListDescriptor {
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> worker_descriptors;
    std::vector<SubDeviceId> sub_device_ids;
    std::vector<CommandListData> ordered_data;
    uint32_t total_size = 0;
};

struct CapturedArgLocation {
    size_t rta_update_index = 0;
    uint32_t byte_offset = 0;
    uint32_t size_bytes = sizeof(uint32_t);
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
    TensorBindingHandle binding;
    TensorSpec expected_spec;
    tt::tt_metal::experimental::TensorSpecRelaxations relaxations;
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
};

struct SelectedProgram {
    CapturedProgram* program;
    StagedCommandListNode* staged_node;
};

struct UnusedNodeData {
    uint32_t both = 0;
    uint32_t multicast = 0;
    uint32_t unicast = 0;
};

struct RangeSelection {
    std::vector<SelectedProgram> programs;
    DispatchArray<UnusedNodeData> unused_nodes;
};

struct PatchResolutionCounts {
    size_t runtime = 0;
    size_t common_runtime = 0;
    size_t tensor = 0;
};

struct SerializedRange {
    CommandListData data;
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> worker_descriptors;
};

struct ScalarPatchTarget {
    MeshCoordinateRange device_range;
    uint32_t byte_offset = 0;
};

struct TensorPatchTarget {
    MeshCoordinateRange device_range;
    uint32_t byte_offset = 0;
    TensorBindingHandle binding;
    TensorSpec expected_spec;
    tt::tt_metal::experimental::TensorSpecRelaxations relaxations;
};

struct CommandListPatchRegistry {
    std::unordered_map<CmdListTensorArgName, std::vector<TensorPatchTarget>> tensor_targets;
    std::unordered_map<CmdListRuntimeArgName, std::vector<ScalarPatchTarget>> runtime_targets;
    std::unordered_map<CmdListCommonRuntimeArgName, std::vector<ScalarPatchTarget>> common_runtime_targets;
};

struct CommandListAssembly {
    CommandListDescriptor descriptor;
    CommandListPatchRegistry patch_registry;
};

CapturedArgLocation resolve_arg_location(
    const RuntimeArgsData& args,
    uint32_t word_offset,
    uint32_t word_count,
    const ProgramCommandSequence& command_sequence) {
    TT_FATAL(word_offset + word_count <= args.size(), "Command-list argument location exceeds its RTA buffer");

    const auto* target = reinterpret_cast<const uint8_t*>(args.data() + word_offset);
    const uint32_t size_bytes = word_count * sizeof(uint32_t);
    for (size_t i = 0; i < command_sequence.rta_updates.size(); ++i) {
        const auto& update = command_sequence.rta_updates[i];
        const auto* begin = static_cast<const uint8_t*>(update.src);
        const auto* end = begin + update.size;
        if (target >= begin && target + size_bytes <= end) {
            return CapturedArgLocation{
                .rta_update_index = i, .byte_offset = static_cast<uint32_t>(target - begin), .size_bytes = size_bytes};
        }
    }
    TT_THROW("Could not resolve a Metal 2.0 argument into the traced program command sequence");
}

std::vector<uint32_t> serialize_tensor_binding(const TensorBindingHandle& handle, const MeshTensor& tensor) {
    std::vector<uint32_t> words;
    words.reserve(1 + handle.num_runtime_field_crta_words);

    const auto address = tensor.address();
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(),
        "Tensor argument for TensorParameter '{}' base address {} exceeds uint32_t max",
        handle.tensor_parameter_name,
        address);
    words.push_back(static_cast<uint32_t>(address));

    if (handle.num_runtime_field_crta_words == 0) {
        return words;
    }

    const Buffer* buffer = tensor.mesh_buffer().get_reference_buffer();
    TT_FATAL(
        buffer != nullptr,
        "Tensor argument for TensorParameter '{}' has runtime fields but no backing Buffer",
        handle.tensor_parameter_name);
    if (handle.runtime_field_is_page_size) {
        TT_FATAL(
            handle.num_runtime_field_crta_words == 1,
            "TensorParameter '{}' page-size binding must reserve exactly one runtime word",
            handle.tensor_parameter_name);
        words.push_back(static_cast<uint32_t>(buffer->aligned_page_size()));
        return words;
    }

    const auto& distribution = buffer->buffer_distribution_spec();
    TT_FATAL(
        distribution.has_value(),
        "Tensor argument for TensorParameter '{}' has no BufferDistributionSpec",
        handle.tensor_parameter_name);
    const auto& shape = distribution->tensor_shape_in_pages();
    TT_FATAL(
        shape.rank() == handle.num_runtime_field_crta_words,
        "Tensor argument for TensorParameter '{}' changed rank from {} to {}",
        handle.tensor_parameter_name,
        handle.num_runtime_field_crta_words,
        shape.rank());
    for (size_t i = 0; i < shape.rank(); ++i) {
        words.push_back(static_cast<uint32_t>(shape[i]));
    }
    return words;
}

uint32_t append_command_bytes(std::vector<uint32_t>& output, const void* data, uint32_t size_bytes) {
    TT_ASSERT(size_bytes % sizeof(uint32_t) == 0);
    const uint32_t byte_offset = static_cast<uint32_t>(output.size() * sizeof(uint32_t));
    const size_t old_size = output.size();
    output.resize(old_size + size_bytes / sizeof(uint32_t));
    std::memcpy(output.data() + old_size, data, size_bytes);
    return byte_offset;
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

template <typename VecIt, typename IndexIt>
VecIt remove_by_index(VecIt begin, VecIt end, IndexIt index_begin, IndexIt index_end) {
    if (index_begin == index_end) {
        return end;
    }
    return std::remove_if(std::next(begin, *index_begin), end, [&](auto& value) {
        if (index_begin == index_end) {
            return false;
        }
        if (*index_begin == static_cast<size_t>(&value - &*begin)) {
            ++index_begin;
            return true;
        }
        return false;
    });
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

class CommandListBuilder::Impl {
public:
    explicit Impl(MeshDevice& mesh_device) :
        mesh_device(mesh_device), sub_device_manager_id(mesh_device.impl().acquire_command_list_builder()) {
        lock_held = true;
    }
    ~Impl() {
        if (lock_held) {
            mesh_device.impl().release_command_list_builder();
        }
    }

    MeshDevice& mesh_device;
    SubDeviceManagerId sub_device_manager_id;
    std::vector<StagedCommandListNode> staged_nodes;
    std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers;
    bool valid = true;
    bool lock_held = false;

private:
    struct OfflineDispatchState {
        OfflineDispatchState(MeshDevice& mesh_device, uint8_t cq_id) :
            cq_id(cq_id),
            dispatch_core(mesh_device.virtual_program_dispatch_core(cq_id)),
            prefetcher_block_size(
                MetalContext::instance(mesh_device.impl().get_context_id()).hal().get_alignment(HalMemType::DRAM)),
            prefetcher_cache_size(
                MetalContext::instance(mesh_device.impl().get_context_id()).dispatch_mem_map().ringbuffer_size()),
            prefetcher_num_blocks(prefetcher_cache_size / prefetcher_block_size),
            prefetcher_manager_size(
                1 << (std::bit_width(std::min(1024u, std::max(2u, prefetcher_num_blocks >> 4))) - 1)),
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

    static std::vector<MeshCoordinateRange> compute_device_ranges(
        const std::vector<StagedCommandListNode>& staged_nodes, const MeshCoordinateRange& local_mesh_range) {
        std::vector<MeshCoordinateRange> device_ranges{local_mesh_range};
        for (const auto& staged_node : staged_nodes) {
            for (const auto& program : staged_node.programs) {
                auto local_device_range = local_mesh_range.intersection(program.device_range);
                if (!local_device_range.has_value()) {
                    continue;
                }
                bool intersection_found = false;
                std::vector<size_t> invalid_indices;
                for (size_t i = 0; i < device_ranges.size(); ++i) {
                    auto& existing = device_ranges[i];
                    TT_FATAL(
                        existing.dims() == local_device_range->dims(),
                        "Mismatching command-list mesh range dimensions");
                    if (!existing.intersects(*local_device_range)) {
                        continue;
                    }
                    intersection_found = true;
                    const auto intersection = *existing.intersection(*local_device_range);
                    if (intersection != existing) {
                        invalid_indices.push_back(i);
                        for (const auto& complement : subtract(existing, intersection).ranges()) {
                            device_ranges.push_back(complement);
                        }
                        device_ranges.push_back(intersection);
                    }
                }
                if (!intersection_found) {
                    device_ranges.push_back(*local_device_range);
                } else if (!invalid_indices.empty()) {
                    device_ranges.erase(
                        remove_by_index(
                            device_ranges.begin(), device_ranges.end(), invalid_indices.begin(), invalid_indices.end()),
                        device_ranges.end());
                }
            }
        }
        return device_ranges;
    }

    static RangeSelection select_programs_for_range(
        std::vector<StagedCommandListNode>& staged_nodes, const MeshCoordinateRange& range) {
        RangeSelection selection;
        for (auto& staged_node : staged_nodes) {
            bool used = false;
            for (auto& program : staged_node.programs) {
                if (!program.device_range.intersects(range)) {
                    continue;
                }
                TT_ASSERT(range == *program.device_range.intersection(range));
                selection.programs.push_back({&program, &staged_node});
                used = true;
                break;
            }
            if (!used) {
                auto& unused = selection.unused_nodes[*staged_node.sub_device_id];
                if (staged_node.multicast_go_signals && staged_node.unicast_go_signals) {
                    ++unused.both;
                } else if (staged_node.multicast_go_signals) {
                    ++unused.multicast;
                } else if (staged_node.unicast_go_signals) {
                    ++unused.unicast;
                }
            }
        }
        return selection;
    }

    static void resolve_patch_targets(
        const CapturedProgram& captured,
        const ProgramCommandSequence& command_sequence,
        const MeshCoordinateRange& range,
        const void* chunk,
        uint32_t chunk_size,
        uint32_t stream_offset,
        CommandListPatchRegistry& patch_registry,
        PatchResolutionCounts& counts) {
        const auto* chunk_begin = static_cast<const uint8_t*>(chunk);
        const auto* chunk_end = chunk_begin + chunk_size;
        auto resolve_offset = [&](const CapturedArgLocation& location) -> std::optional<uint32_t> {
            TT_ASSERT(location.rta_update_index < command_sequence.rta_updates.size());
            const auto& update = command_sequence.rta_updates[location.rta_update_index];
            const auto* target = static_cast<const uint8_t*>(update.dst) + location.byte_offset;
            if (target >= chunk_begin && target + location.size_bytes <= chunk_end) {
                return stream_offset + static_cast<uint32_t>(target - chunk_begin);
            }
            return std::nullopt;
        };

        for (const auto& patch : captured.runtime_patches) {
            if (auto offset = resolve_offset(patch.location)) {
                patch_registry.runtime_targets[patch.name].push_back({range, *offset});
                ++counts.runtime;
            }
        }
        for (const auto& patch : captured.common_runtime_patches) {
            if (auto offset = resolve_offset(patch.location)) {
                patch_registry.common_runtime_targets[patch.name].push_back({range, *offset});
                ++counts.common_runtime;
            }
        }
        for (const auto& patch : captured.tensor_patches) {
            if (auto offset = resolve_offset(patch.location)) {
                patch_registry.tensor_targets[patch.name].push_back(
                    {range, *offset, patch.binding, patch.expected_spec, patch.relaxations});
                ++counts.tensor;
            }
        }
    }

    static SerializedRange serialize_range(
        MeshDevice& mesh_device,
        OfflineDispatchState& dispatch_state,
        std::vector<StagedCommandListNode>& staged_nodes,
        const MeshCoordinateRange& range,
        const std::vector<uint32_t>& exec_buf_end,
        CommandListPatchRegistry& patch_registry) {
        const auto& hal = MetalContext::instance(mesh_device.impl().get_context_id()).hal();
        auto selection = select_programs_for_range(staged_nodes, range);

        std::vector<TraceNode*> trace_nodes;
        trace_nodes.reserve(selection.programs.size());
        for (auto& selected : selection.programs) {
            trace_nodes.push_back(&selected.program->trace_node);
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

        dispatch_state.reset(mesh_device.num_sub_devices());
        std::vector<uint32_t> bytes;
        auto& launch_state = dispatch_state.launch_state;

        std::unordered_map<SubDeviceId, TraceWorkerDescriptor> worker_descriptors;
        for (uint32_t sub_device_idx = 0; sub_device_idx < mesh_device.num_sub_devices(); ++sub_device_idx) {
            const auto& unused = selection.unused_nodes[sub_device_idx];
            const uint32_t count = unused.both + unused.multicast + unused.unicast;
            for (uint32_t i = 0; i < count; ++i) {
                const bool multicast = i < unused.both + unused.multicast;
                const bool unicast = i < unused.both || !multicast;
                SubDeviceId sub_device{static_cast<uint8_t>(sub_device_idx)};
                auto& worker = worker_descriptors[sub_device];
                append_go_signal_sequence(
                    bytes,
                    dispatch_state.cq_id,
                    mesh_device,
                    sub_device,
                    worker.num_completion_worker_cores,
                    dispatch_state.dispatch_core,
                    multicast,
                    unicast);
                if (multicast) {
                    worker.num_completion_worker_cores +=
                        mesh_device.num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device);
                    launch_state[sub_device_idx].inc_mcast_wptr(1);
                    ++worker.num_traced_programs_needing_go_signal_multicast;
                }
                if (unicast) {
                    worker.num_completion_worker_cores += mesh_device.impl().num_virtual_eth_cores(sub_device);
                    launch_state[sub_device_idx].inc_unicast_wptr(1);
                    ++worker.num_traced_programs_needing_go_signal_unicast;
                }
            }
        }

        DispatchArray<uint32_t> starting_workers{};
        for (const auto& [sub_device, worker] : worker_descriptors) {
            starting_workers[*sub_device] = worker.num_completion_worker_cores;
        }

        for (auto& selected : selection.programs) {
            auto& captured = *selected.program;
            auto& node = captured.trace_node;
            auto& staged_node = *selected.staged_node;
            const auto sub_device = node.sub_device_id;

            auto& command_sequence = node.program->get_trace_cached_program_command_sequences().at(
                *mesh_device.get_active_sub_device_manager_id());
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

            auto& worker = worker_descriptors[sub_device];
            auto& worker_launch_state = launch_state[*sub_device];
            node.dispatch_metadata.sync_count += starting_workers[*sub_device];
            const uint32_t virtual_eth_cores =
                staged_node.unicast_go_signals ? mesh_device.impl().num_virtual_eth_cores(sub_device) : 0;
            program_dispatch::update_traced_program_dispatch_commands(
                node,
                command_sequence,
                worker_launch_state.get_mcast_wptr(),
                worker_launch_state.get_unicast_wptr(),
                worker.num_completion_worker_cores,
                dispatch_state.dispatch_core,
                sub_device,
                ProgramBinaryStatus::Committed,
                {staged_node.unicast_go_signals, virtual_eth_cores},
                dispatch_state.cq_id);

            PatchResolutionCounts resolved;
            program_dispatch::for_each_program_command_sequence_chunk(
                command_sequence,
                node.dispatch_metadata.stall_first,
                node.dispatch_metadata.stall_before_program,
                node.dispatch_metadata.send_binary,
                [&](const void* chunk, uint32_t chunk_size) {
                    const uint32_t stream_offset = append_command_bytes(bytes, chunk, chunk_size);
                    resolve_patch_targets(
                        captured, command_sequence, range, chunk, chunk_size, stream_offset, patch_registry, resolved);
                });
            TT_FATAL(
                resolved.runtime == captured.runtime_patches.size() &&
                    resolved.common_runtime == captured.common_runtime_patches.size() &&
                    resolved.tensor == captured.tensor_patches.size(),
                "Failed to resolve every command-list parameter into the serialized command stream");

            if (staged_node.multicast_go_signals) {
                worker_launch_state.inc_mcast_wptr(1);
                ++worker.num_traced_programs_needing_go_signal_multicast;
            }
            if (staged_node.unicast_go_signals) {
                worker_launch_state.inc_unicast_wptr(1);
                ++worker.num_traced_programs_needing_go_signal_unicast;
            }
            worker.num_completion_worker_cores += node.num_workers;
        }

        bytes.insert(bytes.end(), exec_buf_end.begin(), exec_buf_end.end());
        return {
            .data = {.device_range = range, .data = std::move(bytes)},
            .worker_descriptors = std::move(worker_descriptors)};
    }

public:
    static CommandListAssembly assemble(MeshCommandQueue& cq, std::vector<StagedCommandListNode>& staged_nodes) {
        auto& mesh_device = *cq.device();
        OfflineDispatchState dispatch_state(mesh_device, static_cast<uint8_t>(cq.id()));

        CommandListDescriptor descriptor;
        CommandListPatchRegistry patch_registry;

        const auto local_mesh_range = mesh_device.get_view().get_local_mesh_coord_range();
        const auto device_ranges = compute_device_ranges(staged_nodes, local_mesh_range);

        auto& metal_context = MetalContext::instance(mesh_device.impl().get_context_id());
        DeviceCommand end_command(metal_context, metal_context.hal().get_alignment(HalMemType::HOST));
        end_command.add_prefetch_exec_buf_end();
        std::vector<uint32_t> exec_buf_end(end_command.size_bytes() / sizeof(uint32_t));
        std::memcpy(exec_buf_end.data(), end_command.data(), end_command.size_bytes());

        size_t max_command_list_size = 0;
        std::optional<std::unordered_map<SubDeviceId, TraceWorkerDescriptor>> overall_worker_descriptors;

        for (const auto& range : device_ranges) {
            auto serialized =
                serialize_range(mesh_device, dispatch_state, staged_nodes, range, exec_buf_end, patch_registry);
            max_command_list_size = std::max(max_command_list_size, serialized.data.data.size());
            descriptor.ordered_data.push_back(std::move(serialized.data));
            if (!overall_worker_descriptors) {
                overall_worker_descriptors = std::move(serialized.worker_descriptors);
            } else {
                TT_FATAL(
                    *overall_worker_descriptors == serialized.worker_descriptors,
                    "All command-list mesh ranges must produce identical worker descriptors");
            }
        }

        descriptor.total_size = static_cast<uint32_t>(max_command_list_size * sizeof(uint32_t));
        if (overall_worker_descriptors) {
            descriptor.worker_descriptors = std::move(*overall_worker_descriptors);
        }
        descriptor.sub_device_ids.reserve(descriptor.worker_descriptors.size());
        for (const auto& [sub_device_id, _] : descriptor.worker_descriptors) {
            descriptor.sub_device_ids.push_back(sub_device_id);
        }
        std::ranges::sort(descriptor.sub_device_ids, {}, [](SubDeviceId id) { return *id; });
        return {.descriptor = std::move(descriptor), .patch_registry = std::move(patch_registry)};
    }

    uint32_t get_num_workers(bool multicast, bool unicast, SubDeviceId sub_device) const {
        uint32_t workers = 0;
        if (multicast) {
            workers += mesh_device.num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device);
        }
        if (unicast) {
            workers += mesh_device.impl().num_virtual_eth_cores(sub_device);
        }
        return workers;
    }

    CapturedProgram* find_program(StagedCommandListNode& node, const Program& requested) {
        for (auto& captured : node.programs) {
            if (captured.trace_node.program.get() == &requested.impl()) {
                return &captured;
            }
        }
        TT_THROW("Command-list parameter references a Program that is not in the supplied MeshWorkload");
    }

    struct ResolvedKernelParameter {
        CapturedProgram& captured;
        std::shared_ptr<Kernel> kernel;
        const ::tt::tt_metal::detail::ProgramImpl::KernelRTASchema& schema;
        const ProgramCommandSequence& command_sequence;
    };

    ResolvedKernelParameter resolve_kernel_parameter(
        StagedCommandListNode& node,
        const Program& requested_program,
        const KernelSpecName& kernel_name,
        SubDeviceManagerId sub_device_manager_id) {
        auto& captured = *find_program(node, requested_program);
        auto& program = *captured.trace_node.program;
        auto kernel = program.get_kernel_by_spec_name(*kernel_name);
        const auto* schema = program.get_kernel_rta_schema(*kernel_name);
        TT_FATAL(schema != nullptr, "Kernel '{}' has no Metal 2.0 RTA schema", kernel_name);
        const auto& command_sequence = program.get_trace_cached_program_command_sequences().at(*sub_device_manager_id);
        return {
            .captured = captured, .kernel = std::move(kernel), .schema = *schema, .command_sequence = command_sequence};
    }

    void resolve_parameters(
        StagedCommandListNode& node, const CmdListParameters& parameters, SubDeviceManagerId sub_device_manager_id) {
        for (const auto& [name, infos] : parameters.runtime_parameters) {
            for (const auto& info : infos) {
                TT_FATAL(
                    !info.nodes.empty(), "Command-list runtime parameter '{}' must identify at least one node", *name);
                auto resolved =
                    resolve_kernel_parameter(node, info.program.get(), info.kernel_name, sub_device_manager_id);
                const auto slot = resolved.schema.runtime_arg_name_to_slot.find(info.arg_name);
                TT_FATAL(
                    slot != resolved.schema.runtime_arg_name_to_slot.end(),
                    "Runtime argument '{}' is not declared by kernel '{}'",
                    info.arg_name,
                    info.kernel_name);
                for (const auto& node_coord : info.nodes) {
                    resolved.captured.runtime_patches.push_back(
                        {name,
                         resolve_arg_location(
                             resolved.kernel->runtime_args_data(node_coord),
                             slot->second,
                             1,
                             resolved.command_sequence)});
                }
            }
        }

        for (const auto& [name, infos] : parameters.common_runtime_parameters) {
            for (const auto& info : infos) {
                auto resolved =
                    resolve_kernel_parameter(node, info.program.get(), info.kernel_name, sub_device_manager_id);
                const auto slot = resolved.schema.common_runtime_arg_name_to_slot.find(info.arg_name);
                TT_FATAL(
                    slot != resolved.schema.common_runtime_arg_name_to_slot.end(),
                    "Common runtime argument '{}' is not declared by kernel '{}'",
                    info.arg_name,
                    info.kernel_name);
                resolved.captured.common_runtime_patches.push_back(
                    {name,
                     resolve_arg_location(
                         resolved.kernel->common_runtime_args_data(), slot->second, 1, resolved.command_sequence)});
            }
        }

        for (const auto& [name, infos] : parameters.tensor_parameters) {
            for (const auto& info : infos) {
                auto* captured = find_program(node, info.program.get());
                auto& program = *captured->trace_node.program;
                const auto& sequence = program.get_trace_cached_program_command_sequences().at(*sub_device_manager_id);
                bool found = false;
                for (const auto& kernel_name : program.get_registered_kernel_names()) {
                    auto kernel = program.get_kernel_by_spec_name(kernel_name);
                    for (const auto& binding : kernel->tensor_binding_handles()) {
                        if (binding.tensor_parameter_name != *info.param_name) {
                            continue;
                        }
                        found = true;
                        captured->tensor_patches.push_back(
                            {name,
                             resolve_arg_location(
                                 kernel->common_runtime_args_data(),
                                 binding.addr_crta_offset / sizeof(uint32_t),
                                 1 + binding.num_runtime_field_crta_words,
                                 sequence),
                             binding,
                             *program.get_tensor_parameter_layout(*info.param_name),
                             program.get_tensor_parameter_relaxations(*info.param_name)});
                    }
                }
                TT_FATAL(found, "TensorParameter '{}' is not bound by any kernel", info.param_name);
            }
        }
    }

    void add(MeshWorkload& workload, const CmdListParameters& parameters) {
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
        const uint32_t workers = get_num_workers(
            staged_node.multicast_go_signals, staged_node.unicast_go_signals, staged_node.sub_device_id);
        const uint32_t cache_size =
            MetalContext::instance(mesh_device.impl().get_context_id()).dispatch_mem_map().ringbuffer_size();
        const uint32_t max_program_kernels_size = workload.impl().max_program_kernels_size();
        const bool use_prefetcher_cache = max_program_kernels_size != 0 && max_program_kernels_size <= cache_size;

        for (auto& [device_range, program] : workload.get_programs()) {
            staged_node.programs.push_back(
                {device_range,
                 program_dispatch::create_trace_node(program.impl(), &mesh_device, workers, use_prefetcher_cache)});
        }
        resolve_parameters(staged_node, parameters, sub_device_manager_id);
        staged_nodes.push_back(std::move(staged_node));
        if (binary_buffer) {
            retained_binary_buffers.push_back(std::move(binary_buffer));
        }
    }
};

class CommandList::Impl {
public:
    Impl(
        MeshDevice& mesh_device,
        CommandListDescriptor descriptor,
        std::shared_ptr<MeshBuffer> command_buffer,
        std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers,
        CommandListPatchRegistry patch_registry,
        uint8_t cq_id,
        SubDeviceManagerId sub_device_manager_id) :
        mesh_device(&mesh_device),
        descriptor(std::move(descriptor)),
        command_buffer(std::move(command_buffer)),
        retained_binary_buffers(std::move(retained_binary_buffers)),
        patch_registry(std::move(patch_registry)),
        bound_cq_id(cq_id),
        sub_device_manager_id(sub_device_manager_id) {
        mesh_device.impl().register_command_list();
    }
    ~Impl() {
        try {
            deallocate();
        } catch (...) {
            release_resources();
        }
    }

private:
    MeshDevice* mesh_device = nullptr;
    CommandListDescriptor descriptor;
    std::shared_ptr<MeshBuffer> command_buffer;
    // Pins kernel-binary MeshBuffers referenced by the serialized commands.
    std::vector<std::shared_ptr<MeshBuffer>> retained_binary_buffers;
    CommandListPatchRegistry patch_registry;
    uint8_t bound_cq_id = 0;
    SubDeviceManagerId sub_device_manager_id;
    bool valid = true;

    void validate() const {
        TT_FATAL(valid, "CommandList has been deallocated");
        TT_FATAL(mesh_device != nullptr, "CommandList has no MeshDevice");
    }

    void release_resources() noexcept {
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
        patch_registry = {};
        valid = false;
        mesh_device->impl().unregister_command_list();
    }

public:
    void deallocate() {
        if (!valid) {
            return;
        }
        as_fd_queue(mesh_device->mesh_command_queue(bound_cq_id)).drain_device_work();
        release_resources();
    }

    void replay(bool blocking) const {
        validate();
        as_fd_queue(mesh_device->mesh_command_queue(bound_cq_id))
            .enqueue_command_list(
                descriptor.worker_descriptors,
                descriptor.sub_device_ids,
                *command_buffer,
                sub_device_manager_id,
                blocking);
    }

    void update_args(const CmdListArgPatch& patch) {
        validate();

        struct Write {
            MeshCoordinateRange range;
            uint32_t offset;
            std::vector<uint32_t> words;
        };
        std::vector<Write> writes;

        for (const auto& [name, value] : patch.runtime_args) {
            const auto it = patch_registry.runtime_targets.find(name);
            TT_FATAL(it != patch_registry.runtime_targets.end(), "Unknown command-list runtime parameter '{}'", *name);
            for (const auto& target : it->second) {
                writes.push_back({target.device_range, target.byte_offset, {value}});
            }
        }
        for (const auto& [name, value] : patch.common_runtime_args) {
            const auto it = patch_registry.common_runtime_targets.find(name);
            TT_FATAL(
                it != patch_registry.common_runtime_targets.end(),
                "Unknown command-list common-runtime parameter '{}'",
                *name);
            for (const auto& target : it->second) {
                writes.push_back({target.device_range, target.byte_offset, {value}});
            }
        }
        for (const auto& [name, argument] : patch.tensor_args) {
            const auto it = patch_registry.tensor_targets.find(name);
            TT_FATAL(it != patch_registry.tensor_targets.end(), "Unknown command-list tensor parameter '{}'", *name);
            const auto& tensor = ::tt::tt_metal::experimental::mesh_tensor_of(argument);
            TT_FATAL(
                &tensor.device() == mesh_device,
                "Command-list tensor parameter '{}' belongs to a different MeshDevice",
                *name);
            for (const auto& target : it->second) {
                TT_FATAL(
                    tt::tt_metal::experimental::tensorspecs_match_with_relaxation(
                        tensor.tensor_spec(), target.expected_spec, target.relaxations),
                    "Command-list tensor parameter '{}' does not match its declared TensorSpec",
                    *name);
                writes.push_back(
                    {target.device_range, target.byte_offset, serialize_tensor_binding(target.binding, tensor)});
            }
        }

        std::vector<size_t> data_indices;
        data_indices.reserve(writes.size());
        for (const auto& write : writes) {
            const auto data_it = std::find_if(
                descriptor.ordered_data.begin(), descriptor.ordered_data.end(), [&](const CommandListData& data) {
                    return data.device_range == write.range;
                });
            TT_FATAL(data_it != descriptor.ordered_data.end(), "Command-list patch targets an unknown device range");
            TT_FATAL(
                write.offset + write.words.size() * sizeof(uint32_t) <= data_it->data.size() * sizeof(uint32_t),
                "Command-list patch exceeds the serialized command stream");
            data_indices.push_back(static_cast<size_t>(data_it - descriptor.ordered_data.begin()));
        }

        auto& cq = mesh_device->mesh_command_queue(bound_cq_id);
        cq.finish();
        std::set<size_t> dirty_data_indices;
        for (size_t i = 0; i < writes.size(); ++i) {
            const auto& write = writes[i];
            auto& data = descriptor.ordered_data[data_indices[i]];
            std::memcpy(
                reinterpret_cast<uint8_t*>(data.data.data()) + write.offset,
                write.words.data(),
                write.words.size() * sizeof(uint32_t));
            dirty_data_indices.insert(data_indices[i]);
        }

        const size_t page_size = command_buffer->page_size();
        for (const size_t data_index : dirty_data_indices) {
            const auto& data = descriptor.ordered_data[data_index];
            std::vector<uint32_t> padded = data.data;
            padded.resize(round_up(padded.size() * sizeof(uint32_t), page_size) / sizeof(uint32_t), 0);
            cq.enqueue_write_shard_to_sub_grid(
                *command_buffer,
                padded.data(),
                data.device_range,
                true,
                BufferRegion(0, padded.size() * sizeof(uint32_t)));
        }
    }

    MeshDevice& get_device() const {
        validate();
        return *mesh_device;
    }

    uint8_t get_cq_id() const {
        validate();
        return bound_cq_id;
    }
};

namespace {

std::shared_ptr<MeshBuffer> allocate_and_commit(
    MeshDevice& device, MeshCommandQueue& cq, const CommandListDescriptor& descriptor) {
    const size_t page_size = trace_dispatch::compute_interleaved_trace_buf_page_size(
        descriptor.total_size, device.allocator()->get_num_banks(BufferType::DRAM));
    const size_t padded_size = round_up(descriptor.total_size, page_size);
    const auto trace_region_size = device.allocator_impl()->get_config().trace_region_size;
    const BufferType buffer_type = trace_region_size == 0 ? BufferType::DRAM : BufferType::TRACE;
    const std::optional<bool> bottom_up = trace_region_size == 0 ? std::optional<bool>{false} : std::nullopt;

    const auto current_size = device.get_trace_buffers_size();
    TT_FATAL(
        trace_region_size == 0 || current_size + padded_size <= trace_region_size,
        "Command-list buffers exceed the configured trace-region capacity");
    device.set_trace_buffers_size(current_size + padded_size);

    std::shared_ptr<MeshBuffer> buffer;
    try {
        {
            auto allocation_context = tt::tt_metal::make_allocation_context_guard("trace_storage");
            buffer = MeshBuffer::create(
                ReplicatedBufferConfig{.size = padded_size},
                DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type, .bottom_up = bottom_up},
                &device);
        }
        for (const auto& data : descriptor.ordered_data) {
            std::vector<uint32_t> padded = data.data;
            padded.resize(round_up(padded.size() * sizeof(uint32_t), page_size) / sizeof(uint32_t), 0);
            cq.enqueue_write_shard_to_sub_grid(
                *buffer, padded.data(), data.device_range, true, BufferRegion(0, padded.size() * sizeof(uint32_t)));
        }
    } catch (...) {
        device.set_trace_buffers_size(current_size);
        throw;
    }
    return buffer;
}

}  // namespace

CommandListBuilder::CommandListBuilder(MeshDevice& device) : impl_(std::make_unique<Impl>(device)) {}
CommandListBuilder::CommandListBuilder(CommandListBuilder&&) noexcept = default;
CommandListBuilder& CommandListBuilder::operator=(CommandListBuilder&&) noexcept = default;
CommandListBuilder::~CommandListBuilder() = default;

void CommandListBuilder::add(MeshWorkload& workload, const CmdListParameters& parameters) {
    TT_FATAL(impl_ != nullptr, "CommandListBuilder has been moved from");
    impl_->add(workload, parameters);
}

CommandList CommandListBuilder::build(MeshCommandQueue& cq) const {
    TT_FATAL(impl_ != nullptr && impl_->valid, "CommandListBuilder has been deallocated");
    TT_FATAL(cq.device() == &impl_->mesh_device, "Command queue belongs to a different MeshDevice");
    TT_FATAL(!impl_->staged_nodes.empty(), "Cannot build an empty CommandList");
    TT_FATAL(
        impl_->mesh_device.get_active_sub_device_manager_id() == impl_->sub_device_manager_id,
        "The active sub-device manager changed while building the command list");

    (void)as_fd_queue(cq);
    auto staged_nodes = impl_->staged_nodes;
    auto assembly = CommandListBuilder::Impl::assemble(cq, staged_nodes);
    auto command_buffer = allocate_and_commit(impl_->mesh_device, cq, assembly.descriptor);
    return CommandList(std::make_unique<CommandList::Impl>(
        impl_->mesh_device,
        std::move(assembly.descriptor),
        std::move(command_buffer),
        impl_->retained_binary_buffers,
        std::move(assembly.patch_registry),
        static_cast<uint8_t>(cq.id()),
        impl_->sub_device_manager_id));
}

MeshDevice& CommandListBuilder::device() const {
    TT_FATAL(impl_ != nullptr && impl_->valid, "CommandListBuilder has been deallocated");
    return impl_->mesh_device;
}

void CommandListBuilder::clear() {
    TT_FATAL(impl_ != nullptr && impl_->valid, "CommandListBuilder has been deallocated");
    impl_->staged_nodes.clear();
    impl_->retained_binary_buffers.clear();
}

void CommandListBuilder::deallocate() {
    if (impl_) {
        impl_->staged_nodes.clear();
        impl_->retained_binary_buffers.clear();
        impl_->valid = false;
        if (impl_->lock_held) {
            impl_->mesh_device.impl().release_command_list_builder();
            impl_->lock_held = false;
        }
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

void CommandList::update_args(const CmdListArgPatch& patch) {
    TT_FATAL(impl_ != nullptr, "CommandList has been moved from");
    impl_->update_args(patch);
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
