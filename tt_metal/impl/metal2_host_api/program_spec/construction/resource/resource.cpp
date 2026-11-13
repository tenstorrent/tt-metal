// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <optional>

#include <tt-metalium/buffer_distribution_spec.hpp>

#include "impl/metal2_host_api/helpers.hpp"

namespace tt::tt_metal::experimental {

namespace {

// A node set as rectangles in NoC (virtual) coordinates, which the kernel stores in 8 bits each.
std::vector<NodeRange> ToVirtualNodeRanges(distributed::MeshDevice& mesh_device, const NodeRangeSet& nodes) {
    std::vector<NodeRange> virtual_ranges;
    virtual_ranges.reserve(nodes.ranges().size());
    for (const NodeRange& range : nodes.ranges()) {
        const NodeCoord start = mesh_device.virtual_core_from_logical_core(range.start_coord, CoreType::WORKER);
        const NodeCoord end = mesh_device.virtual_core_from_logical_core(range.end_coord, CoreType::WORKER);
        TT_FATAL(
            start.x <= 0xFF && start.y <= 0xFF && end.x <= 0xFF && end.y <= 0xFF,
            "Watcher stores NoC coordinates in 8 bits, but node range {} spans {} to {}",
            range.str(),
            start.str(),
            end.str());
        virtual_ranges.emplace_back(start, end);
    }
    return virtual_ranges;
}

WatcherNodeRanges MakeWatcherNodeRanges(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    MetalEnvImpl& metal_env) {
    WatcherNodeRanges ranges;
    const auto& rtoptions = metal_env.get_rtoptions();
    if (!rtoptions.get_watcher_enabled() || rtoptions.watcher_noc_sanitize_disabled()) {
        return ranges;
    }
    for (const auto& [dfb_name, nodes] : collected.dfb_node_set) {
        const std::optional<TensorParamName>& borrowed_from = collected.dfb_by_name.at(dfb_name)->borrowed_from;
        if (!borrowed_from.has_value()) {
            ranges.dfbs.emplace(dfb_name, ToVirtualNodeRanges(mesh_device, nodes));
            continue;
        }
        // A borrowed-memory DFB's address is reserved on its backing tensor's nodes, which may reach past the DFB's
        // own. An interleaved backing tensor carries no node set, so a DFB in it isn't checked.
        const TensorSpec& tensor_spec = collected.tensor_parameter_by_name.at(*borrowed_from)->spec;
        if (!tensor_spec.memory_config().is_sharded()) {
            continue;
        }
        if (const std::optional<BufferDistributionSpec> distribution =
                tensor_spec.compute_buffer_sharding_args().buffer_distribution_spec()) {
            ranges.dfbs.emplace(
                dfb_name, ToVirtualNodeRanges(mesh_device, NodeRangeSet(distribution->cores()).merge_ranges()));
        }
    }
    for (const auto& sem : spec.semaphores) {
        ranges.semaphores.emplace(sem.unique_id, ToVirtualNodeRanges(mesh_device, to_node_range_set(sem.target_nodes)));
    }
    return ranges;
}

}  // namespace

ProgramResources RegisterResources(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    const KernelRiscMaskMap& risc_masks,
    MetalEnvImpl& metal_env,
    detail::ProgramImpl& program_impl) {
    ProgramResources resources;
    for (const auto& parameter : spec.advanced_options.sram_arena_parameters) {
        program_impl.register_sram_arena_parameter(parameter.unique_id.get(), parameter.properties, mesh_device);
    }
    resources.tensor_parameters = RegisterTensorParameters(spec, mesh_device, program_impl);
    resources.dfbs = RegisterDataflowBuffers(spec, collected, risc_masks, program_impl);

    // Reserve PrefetcherPipe slots (one per kernel accessor group) from the spec geometry, register
    // relay DFBs against them, and record each parameter's placement for SetProgramRunArgs. Must
    // precede kernel creation: the slot id is baked into the kernel's `pipe::<accessor>` token and
    // a relay DFB's `dfb::` token.
    resources.prefetcher_pipes =
        ReservePrefetcherPipeSlots(mesh_device, spec, collected, program_impl, resources.dfbs.id);
    resources.dfbs.relay_pipe_id = RecordRelayPipeIds(resources.dfbs.id, program_impl);

    WireDFBAliases(spec, resources.dfbs.id, program_impl);
    resources.semaphores = RegisterSemaphores(spec, collected, metal_env, program_impl);
    resources.watcher_node_ranges = MakeWatcherNodeRanges(mesh_device, spec, collected, metal_env);
    return resources;
}

KernelResourceBindings ResolveKernelResourceBindings(
    const KernelSpec& kernel_spec, const CollectedSpecData& collected, const ProgramResources& resources) {
    KernelResourceBindings bindings{
        .dfbs = MakeDataflowBufferBindingHandles(
            kernel_spec,
            resources.dfbs.slot,
            resources.dfbs.is_relay,
            resources.dfbs.relay_pipe_id,
            collected.dfb_by_name,
            resources.dfbs.id,
            resources.dfbs.patterns),
        .semaphores = MakeSemaphoreBindingHandles(
            kernel_spec, collected.semaphore_binders, resources.semaphores.id, resources.semaphores.scope),
        .prefetcher_pipes = {}};
    if (auto pipe_it = resources.prefetcher_pipes.find(&kernel_spec); pipe_it != resources.prefetcher_pipes.end()) {
        bindings.prefetcher_pipes = pipe_it->second;
    }
    // Only data-movement kernels multicast, and a relay DFB's token carries no ranges, so only those bindings get them:
    // ranges anywhere else would only make the kernel's hash depend on its placement.
    if (!kernel_spec.is_data_movement_kernel()) {
        return bindings;
    }
    const WatcherNodeRanges& watcher_node_ranges = resources.watcher_node_ranges;
    for (const auto& dfb_binding : kernel_spec.dfb_bindings) {
        DataflowBufferBindingHandle& handle = bindings.dfbs.at(dfb_binding.accessor_name);
        if (auto it = watcher_node_ranges.dfbs.find(dfb_binding.dfb_spec_name);
            it != watcher_node_ranges.dfbs.end() && !handle.is_relay) {
            handle.watcher_node_ranges = it->second;
        }
    }
    for (const auto& semaphore_binding : kernel_spec.semaphore_bindings) {
        if (auto it = watcher_node_ranges.semaphores.find(semaphore_binding.semaphore_spec_name);
            it != watcher_node_ranges.semaphores.end()) {
            bindings.semaphores.at(semaphore_binding.accessor_name).watcher_node_ranges = it->second;
        }
    }
    return bindings;
}

}  // namespace tt::tt_metal::experimental
