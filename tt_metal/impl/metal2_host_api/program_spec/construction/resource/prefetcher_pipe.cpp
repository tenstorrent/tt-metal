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

// ----------------------------------------------------------------------------
// ReservePrefetcherPipeSlots: PrefetcherPipeParameters -> Program slots
// ----------------------------------------------------------------------------
//
// One Program slot per accessor group (a KernelAdvancedOptions::PrefetcherPipeBinding), reserved on the
// kernel's nodes from spec geometry alone. The group's role (sender / receiver; validated exact
// by ValidateProgramSpec) decides the slot's receiver cores and credit lanes P (the receiver
// kernel's num_threads). A relay DFB whose prefetcher_pipe_relays equals the group's pipe set is
// registered against the slot (its base address is supplied when the pipe binds).
//
// Every parameter named by the group is recorded with the slot and the cores it owns inside the
// kernel's nodes (its sender node or its receivers -- the group tiles the nodes, so this is one
// pipe per node). SetProgramRunArgs later binds the supplied pipe object onto exactly those cores,
// so a multi-pipe accessor resolves per node on the host and the kernel binary sees one slot.

PrefetcherPipeHandlesByKernel ReservePrefetcherPipeSlots(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    detail::ProgramImpl& program_impl,
    const DFBNameToIdMap& dfb_name_to_id) {
    PrefetcherPipeHandlesByKernel handles;
    if (spec.advanced_options.prefetcher_pipe_parameters.empty()) {
        return handles;
    }

    // Per-parameter placement, accumulated across the accessor groups that name it.
    std::unordered_map<PrefetcherPipeParamName, detail::ProgramImpl::PrefetcherPipeParameterBinding> placements;
    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        placements[pipe.unique_id] = detail::ProgramImpl::PrefetcherPipeParameterBinding{
            .device = &mesh_device,
            .receivers = to_node_range_set(pipe.receivers),
            .ring_size = pipe.ring_size,
            .slots = {},
            .bound_pipe = nullptr};
    }

    // Relay DFBs keyed by their (sorted) relayed pipe set, so a group can find its relay.
    auto sorted_names = [](std::vector<PrefetcherPipeParamName> names) {
        std::sort(names.begin(), names.end());
        return names;
    };
    std::map<std::vector<PrefetcherPipeParamName>, const DataflowBufferSpec*> relay_by_pipe_set;
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.advanced_options.prefetcher_pipe_relays.empty()) {
            continue;
        }
        auto [it, inserted] =
            relay_by_pipe_set.try_emplace(sorted_names(dfb.advanced_options.prefetcher_pipe_relays), &dfb);
        TT_FATAL(
            inserted,
            "DFBs '{}' and '{}' both relay the same PrefetcherPipe set; a pipe set has at most one relay DFB",
            it->second->unique_id,
            dfb.unique_id);
    }
    std::unordered_set<const DataflowBufferSpec*> relays_registered;

    for (const KernelSpec& kernel : spec.kernels) {
        if (kernel.advanced_options.prefetcher_pipe_bindings.empty()) {
            continue;
        }
        const NodeRangeSet& nodes = collected.kernel_node_set.at(kernel.unique_id);
        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            const PrefetcherPipeParameter* first =
                collected.prefetcher_pipe_by_name.at(binding.pipe_parameter_names[0]);
            NodeRangeSet group_receivers;
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                const PrefetcherPipeParameter* pipe = collected.prefetcher_pipe_by_name.at(pipe_name);
                group_receivers = group_receivers.merge(to_node_range_set(pipe->receivers));
            }
            const bool is_sender_role =
                !is_prefetcher_pipe_receiver_role(nodes, group_receivers) &&
                is_prefetcher_pipe_sender_role(nodes, group_receivers, binding.pipe_parameter_names.size());

            const NodeRangeSet receiver_cores = is_sender_role ? NodeRangeSet() : nodes;
            const uint32_t num_credit_lanes = is_sender_role ? 1u : kernel.num_threads;
            const uint8_t prefetcher_pipe_id = program_impl.reserve_prefetcher_pipe_slot(
                nodes, receiver_cores, first->ring_size, first->entry_size, num_credit_lanes);
            handles[&kernel].push_back(
                {.accessor_name = binding.accessor_name, .prefetcher_pipe_id = prefetcher_pipe_id});

            // Sender role: the spec does not say which of the kernel's nodes hosts which pipe, so
            // every pipe's placement names all of them; the bind narrows it to the pipe's sender.
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                const PrefetcherPipeParameter* pipe = collected.prefetcher_pipe_by_name.at(pipe_name);
                placements.at(pipe_name).slots.push_back(
                    {.prefetcher_pipe_id = prefetcher_pipe_id,
                     .cores = is_sender_role ? nodes : to_node_range_set(pipe->receivers),
                     .sender_role = is_sender_role});
            }

            // A relay over exactly this group's pipes hangs off the receiver kernel's slot.
            if (!is_sender_role) {
                auto relay_it = relay_by_pipe_set.find(sorted_names(binding.pipe_parameter_names));
                if (relay_it != relay_by_pipe_set.end()) {
                    const DataflowBufferSpec* relay = relay_it->second;
                    TT_FATAL(
                        relays_registered.insert(relay).second,
                        "Relay DFB '{}' matches PrefetcherPipe accessor groups in more than one receiver kernel",
                        relay->unique_id);
                    program_impl.register_prefetcher_pipe_relay_dfb(
                        prefetcher_pipe_id, dfb_name_to_id.at(relay->unique_id));
                }
            }
        }
    }

    for (const auto& dfb : spec.dataflow_buffers) {
        if (!dfb.advanced_options.prefetcher_pipe_relays.empty()) {
            TT_FATAL(
                relays_registered.contains(&dfb),
                "Relay DFB '{}' has no data-movement kernel binding its relayed PrefetcherPipe set as receiver; the "
                "relay's PRODUCER must bind those pipes under one accessor",
                dfb.unique_id);
        }
    }

    for (auto& [pipe_name, placement] : placements) {
        program_impl.register_prefetcher_pipe_parameter(pipe_name.get(), std::move(placement));
    }
    return handles;
}

std::unordered_map<DFBSpecName, uint8_t> RecordRelayPipeIds(
    const DFBNameToIdMap& dfb_name_to_id, const detail::ProgramImpl& program_impl) {
    std::unordered_map<DFBSpecName, uint8_t> dfb_name_to_prefetcher_pipe_id;
    for (const auto& [dfb_name, dfb_id] : dfb_name_to_id) {
        dfb_name_to_prefetcher_pipe_id[dfb_name] = program_impl.get_prefetcher_pipe_id_for_relay(dfb_id).value_or(0xFF);
    }
    return dfb_name_to_prefetcher_pipe_id;
}

}  // namespace tt::tt_metal::experimental
