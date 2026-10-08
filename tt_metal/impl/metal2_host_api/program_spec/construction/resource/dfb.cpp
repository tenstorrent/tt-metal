// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // fmt::formatter<tt::DataFormat> for TT_FATAL messages
#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental {

namespace {

std::optional<LLKMetadata> LLKMetadataFromDfb(const DataflowBufferSpec& spec) {
    if (!spec.data_format_metadata.has_value()) {
        TT_FATAL(
            !spec.tile_format_metadata.has_value(),
            "DFB '{}' need to have a configured data_format_metadata for it's tile_format_metadata to be respected",
            spec.unique_id);
        return std::nullopt;
    }
    const Tile tile = spec.tile_format_metadata.value_or(Tile{});
    return LLKMetadata{.format = *spec.data_format_metadata, .tile = tile};
}

}  // namespace

// Create map of local accessor name -> DFB device slot. This is the value baked into the kernel's
// dfb::<name> accessor, so it must be the device slot rather than the program-wide id.
// `dfb_name_to_is_relay` marks CrossNode/PrefetcherPipe relay locals so codegen emits
// RelayDFBBindingToken instead of DFBBindingToken.
// `dfb_name_to_prefetcher_pipe_id` carries the PrefetcherPipe slot for PrefetcherPipe relays (0xFF
// otherwise) so TRISC construction can align to the durable checkpoint.
tt::tt_metal::DataflowBufferBindingHandleMap MakeDataflowBufferBindingHandles(
    const KernelSpec& kernel_spec,
    const DFBNameToSlotMap& dfb_name_to_slot,
    const std::unordered_map<DFBSpecName, bool>& dfb_name_to_is_relay,
    const std::unordered_map<DFBSpecName, uint8_t>& dfb_name_to_prefetcher_pipe_id,
    const std::unordered_map<DFBSpecName, const DataflowBufferSpec*>& dfb_by_name,
    const DFBNameToIdMap& dfb_name_to_id) {
    tt::tt_metal::DataflowBufferBindingHandleMap out;
    out.reserve(kernel_spec.dfb_bindings.size());
    for (const auto& dfb_binding : kernel_spec.dfb_bindings) {
        const uint32_t slot = dfb_name_to_slot.at(dfb_binding.dfb_spec_name);
        TT_FATAL(
            slot <= std::numeric_limits<uint16_t>::max(),
            "Kernel '{}' DFB '{}' device slot {} does not fit uint16_t",
            kernel_spec.unique_id,
            dfb_binding.dfb_spec_name,
            slot);
        tt::tt_metal::DataflowBufferBindingHandle handle;
        handle.logical_dfb_id = static_cast<uint16_t>(slot);
        handle.is_relay = dfb_name_to_is_relay.at(dfb_binding.dfb_spec_name);
        handle.prefetcher_pipe_id = dfb_name_to_prefetcher_pipe_id.at(dfb_binding.dfb_spec_name);
        if (!handle.is_relay) {
            handle.llk_metadata = LLKMetadataFromDfb(*dfb_by_name.at(dfb_binding.dfb_spec_name));
        }
        // Borrowed-memory DFB: remember the tensor it is and this kernel's side, for op-to-op R/W inference.
        if (const auto& borrowed_from = dfb_by_name.at(dfb_binding.dfb_spec_name)->borrowed_from) {
            handle.borrowed_dfb_id = dfb_name_to_id.at(dfb_binding.dfb_spec_name);
            handle.borrowed_tensor_parameter_name = borrowed_from->get();
            handle.produces = dfb_binding.endpoint_type == DFBEndpointType::PRODUCER;
            handle.consumes = !handle.produces;
        }
        // A self-loop pair may bind one DFB as PRODUCER and as CONSUMER under the same accessor name; that is one
        // handle, on both sides.
        auto [it, inserted] = out.try_emplace(dfb_binding.accessor_name, handle);
        if (!inserted) {
            it->second.produces |= handle.produces;
            it->second.consumes |= handle.consumes;
        }
    }
    return out;
}

namespace {

// Create a DataflowBufferConfig from a DataflowBufferSpec and endpoint info.
experimental::dfb::DataflowBufferConfig MakeDataflowBufferConfig(
    const DataflowBufferSpec* dfb_spec,
    const CollectedSpecData::DFBEndpointInfo& dfb_endpoint_info,
    const KernelRiscMaskMap& kernel_to_risc_mask) {
    // With multi-binding, all same-role KernelSpecs share kind (DM/compute), access_pattern,
    // num_threads, and risc_mask. The first three are enforced in ValidateProgramSpec; the
    // fourth is solver-guaranteed on Gen2 (the coupling-group equivalence-class constraint)
    // and user-validated on Gen1 (see Step 2b in MakeProgramFromSpec). So any
    // representative producer/consumer gives the correct DFB config — we take the first.
    const KernelSpec* producer = dfb_endpoint_info.producers.front().kernel;
    const KernelSpec* consumer = dfb_endpoint_info.consumers.front().kernel;
    const DFBBinding* producer_binding = dfb_endpoint_info.producers.front().binding;
    const DFBBinding* consumer_binding = dfb_endpoint_info.consumers.front().binding;

    uint16_t producer_risc_mask = kernel_to_risc_mask.at(producer);
    uint16_t consumer_risc_mask = kernel_to_risc_mask.at(consumer);

    // Convert user-facing access pattern enum to hardware interface access pattern enum
    // (TODO: We should merge these enums; it's silly to have separate ones.)
    auto to_hw_access_pattern = [](DFBAccessPattern pattern) -> experimental::dfb::AccessPattern {
        switch (pattern) {
            case DFBAccessPattern::STRIDED: return experimental::dfb::AccessPattern::STRIDED;
            case DFBAccessPattern::ALL: return experimental::dfb::AccessPattern::ALL;
            case DFBAccessPattern::BLOCKED: TT_FATAL(false, "BLOCKED access pattern is not yet supported");
        }
        TT_FATAL(false, "Unknown DFBAccessPattern");
    };
    auto producer_access_pattern = to_hw_access_pattern(producer_binding->access_pattern);
    auto consumer_access_pattern = to_hw_access_pattern(consumer_binding->access_pattern);

    // A compute kernel that self-loops a DFB (binds it as both producer and consumer) lowers to the
    // intra-Tensix packer->unpacker flow, so the lower-layer DFB API needs TensixScope::INTRA. The
    // Metal 2.0 surface does not expose a scope option — INTRA is the only supported topology, applied
    // automatically here. Self-loop is detected as any overlap between the producer and consumer kernel
    // sets — under the multi-binding regime the first-record pointers may differ even when the kernel
    // sets are identical (the overlap is what matters, not vector ordering). Upstream validation
    // guarantees producer set == consumer set whenever any overlap exists, so reading from the first
    // producer is safe and representative. A DM self-loop (Gen1-only) needs no tensix_scope.
    const bool is_self_loop = [&] {
        for (const auto& p : dfb_endpoint_info.producers) {
            for (const auto& c : dfb_endpoint_info.consumers) {
                if (p.kernel == c.kernel) {
                    return true;
                }
            }
        }
        return false;
    }();
    std::optional<experimental::dfb::TensixScope> tensix_scope;
    if (is_self_loop && producer->is_compute_kernel()) {
        tensix_scope = experimental::dfb::TensixScope::INTRA;
    }

    // Compute the per-side implicit-sync value by polling the bound DM kernels' Gen2 votes.
    // Sides with no DM endpoints get implicit_sync=false (no DM endpoint to enable it for).
    // Validator guarantees per-side agreement among DM kernels, so any DM kernel's vote works.
    auto side_implicit_sync_enabled =
        [&](const std::vector<CollectedSpecData::DFBEndpointInfo::EndpointRecord>& endpoints) -> bool {
        bool any_dm = false;
        bool disabled = false;
        for (const auto& ep : endpoints) {
            if (!ep.kernel->is_data_movement_kernel()) {
                continue;
            }
            any_dm = true;
            const auto& dm_config = std::get<DataMovementHardwareConfig>(ep.kernel->hw_config);
            // config_2xx is unused on Gen1; implicit sync stays at the current default.
            if (DmKernelDisablesImplicitSync(dm_config, dfb_spec->unique_id)) {
                disabled = true;
            }
        }
        return any_dm && !disabled;
    };
    return experimental::dfb::DataflowBufferConfig{
        .entry_size = dfb_spec->entry_size,
        .num_entries = dfb_spec->num_entries,
        .producer_risc_mask = producer_risc_mask,
        .num_producers = static_cast<uint8_t>(producer->num_threads),
        .pap = producer_access_pattern,
        .consumer_risc_mask = consumer_risc_mask,
        .num_consumers = static_cast<uint8_t>(consumer->num_threads),
        .cap = consumer_access_pattern,
        .enable_producer_implicit_sync = side_implicit_sync_enabled(dfb_endpoint_info.producers),
        .enable_consumer_implicit_sync = side_implicit_sync_enabled(dfb_endpoint_info.consumers),
        .data_format = dfb_spec->data_format_metadata.value_or(tt::DataFormat::Invalid),
        .tile = dfb_spec->tile_format_metadata,
        .tensix_scope = tensix_scope,
        // DFB borrowed memory mode is declared at program creation time.
        // The actual backing memory L1 address is attached at runtime: from the borrowed
        // TensorParameter's MeshTensor, or (relay) from the PrefetcherPipe ring the relay aliases.
        .borrows_memory =
            dfb_spec->borrowed_from.has_value() || !dfb_spec->advanced_options.prefetcher_pipe_relays.empty(),
        // A PrefetcherPipe relay is lane-interleaved (producer h owns entries h, h+P, ...).
        .is_relay = !dfb_spec->advanced_options.prefetcher_pipe_relays.empty()};
}

}  // namespace

DataflowBufferHandles RegisterDataflowBuffers(
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    const KernelRiscMaskMap& kernel_to_risc_mask,
    detail::ProgramImpl& program_impl) {
    // Create DataflowBuffers and build name -> ID map.
    // NOTE: Iterate over spec.dataflow_buffers (not collected.dfb_endpoints) to ensure
    //       deterministic DFB ID assignment based on user-specified order.
    DFBNameToIdMap dfb_name_to_id;
    DFBNameToSlotMap dfb_name_to_slot;
    std::unordered_map<DFBSpecName, bool> dfb_name_to_is_relay;
    for (const auto& dfb_spec : spec.dataflow_buffers) {
        const DFBSpecName& dfb_name = dfb_spec.unique_id;
        const auto& dfb_endpoint_info = collected.dfb_endpoints.at(dfb_name);
        const experimental::dfb::DataflowBufferConfig config =
            MakeDataflowBufferConfig(&dfb_spec, dfb_endpoint_info, kernel_to_risc_mask);

        // Add the DFB to the ProgramImpl, and register the name -> handle mapping.
        // Allocation nodes are derived from binding kernels' WorkUnitSpec membership.
        // (For borrowed-memory DFBs, config.borrows_memory was set in MakeDataflowBufferConfig;
        // the device-side runtime uses that to skip regular L1 allocation.)
        uint32_t dfb_id = program_impl.add_dataflow_buffer(collected.dfb_node_set.at(dfb_name), config);
        program_impl.register_dfb_spec_name(dfb_name.get(), dfb_id);
        dfb_name_to_id[dfb_name] = dfb_id;
        const auto& created_config = program_impl.get_dataflow_buffer(dfb_id)->config;
        dfb_name_to_slot[dfb_name] = program_impl.get_dataflow_buffer(dfb_id)->device_slot;
        dfb_name_to_is_relay[dfb_name] = created_config.is_relay;

        // Borrowed-memory DFB: record the dfb_id ↔ TensorParamName binding so that
        // SetProgramRunArgs / UpdateTensorArgs can resolve and attach the actual L1 Buffer
        // at runtime (analog of dynamic CB's UpdateDynamicCircularBufferAddress).
        if (dfb_spec.borrowed_from.has_value()) {
            program_impl.register_dfb_borrowed_binding(dfb_id, dfb_spec.borrowed_from->get());
        }
    }
    return DataflowBufferHandles{
        .id = std::move(dfb_name_to_id),
        .slot = std::move(dfb_name_to_slot),
        .is_relay = std::move(dfb_name_to_is_relay),
        .relay_pipe_id = {}};
}

void WireDFBAliases(const ProgramSpec& spec, const DFBNameToIdMap& dfb_name_to_id, detail::ProgramImpl& program_impl) {
    // Wire alias groups: for each DFB that has alias_with entries, make the first
    // encountered DFB in the group the primary and call set_dfb_alias for each secondary.
    // handled_as_secondary prevents a DFB from being treated as a primary when it was
    // already registered as a secondary by an earlier DFB in the group. Soundness relies
    // on the strict-clique invariant enforced by ValidateProgramSpec: every group member
    // lists every other member, so the primary's alias_with covers the whole group.
    {
        std::unordered_set<DFBSpecName> handled_as_secondary;
        for (const auto& dfb_spec : spec.dataflow_buffers) {
            if (handled_as_secondary.contains(dfb_spec.unique_id)) {
                continue;
            }
            if (dfb_alias_with(dfb_spec).empty()) {
                continue;
            }
            const uint32_t primary_id = dfb_name_to_id.at(dfb_spec.unique_id);
            for (const auto& alias_name : dfb_alias_with(dfb_spec)) {
                if (handled_as_secondary.contains(alias_name)) {
                    continue;
                }
                const uint32_t secondary_id = dfb_name_to_id.at(alias_name);
                program_impl.set_dfb_alias(primary_id, secondary_id);
                handled_as_secondary.insert(alias_name);
            }
        }
    }
}

}  // namespace tt::tt_metal::experimental
