// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

// The spec never names a pipe's sender: that is the pipe object's (a consumer Program need not
// know it, and a DRAM-resident sender has no worker node to name). A Program that runs the sender
// kernel places it through a WorkUnitSpec; the supplied pipe's sender must be one of those nodes,
// which is checked when the pipe is bound.
//
// Rules per accessor group (one KernelAdvancedOptions::PrefetcherPipeBinding; its pipes share one device
// slot on every node the kernel runs on, so one binary serves them all):
//  2. The binding kernel is a data-movement kernel (compute reaches the ring via a relay DFB).
//  3. Tiling: the group's pipes agree on ring_size / entry_size and their receiver sets are
//     pairwise disjoint. The kernel's nodes equal EITHER the union of the receiver sets
//     (receiver role) OR avoid every receiver and number one per pipe (sender role); mixed or
//     partial coverage is rejected. Per pipe, at most one kernel plays sender and at most one
//     plays receiver, so exactly one kernel instance owns the credit counters on each node. Roles
//     may be split across Programs (sender op vs consumer op).
PrefetcherPipeRoles ValidatePrefetcherPipeRoles(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    PrefetcherPipeRoles roles;
    auto& pipe_receiver_set = roles.pipe_receiver_set;
    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        pipe_receiver_set.emplace(pipe.unique_id, to_node_range_set(pipe.receivers));
    }

    // Derive each group's role once, then record the sender / receiver kernel of every pipe in it.
    std::unordered_map<PrefetcherPipeParamName, const KernelSpec*> sender_kernel_of;
    auto& receiver_kernel_of = roles.receiver_kernel_of;
    for (const auto& kernel : spec.kernels) {
        if (kernel.advanced_options.prefetcher_pipe_bindings.empty()) {
            continue;
        }
        TT_FATAL(
            kernel.is_data_movement_kernel(),
            "Kernel '{}' binds PrefetcherPipeParameter(s) (accessor '{}') but is a compute kernel. Only "
            "data-movement kernels bind a pipe; compute consumes through a relay DFB "
            "(DFBAdvancedOptions::prefetcher_pipe_relays).",
            kernel.unique_id,
            kernel.advanced_options.prefetcher_pipe_bindings[0].accessor_name);
        const NodeRangeSet& nodes = collected.kernel_node_set.at(kernel.unique_id);
        TT_FATAL(
            nodes.num_cores() > 0,
            "Kernel '{}' binds PrefetcherPipeParameter(s) but its WorkUnitSpecs place it on no nodes",
            kernel.unique_id);

        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            const PrefetcherPipeParameter* first =
                collected.prefetcher_pipe_by_name.at(binding.pipe_parameter_names[0]);
            NodeRangeSet group_receivers;
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                const PrefetcherPipeParameter* pipe = collected.prefetcher_pipe_by_name.at(pipe_name);
                TT_FATAL(
                    pipe->ring_size == first->ring_size && pipe->entry_size == first->entry_size,
                    "Kernel '{}' accessor '{}' names PrefetcherPipeParameters '{}' (ring_size {}, entry_size {}) "
                    "and '{}' (ring_size {}, entry_size {}); pipes sharing an accessor must share ring_size and "
                    "entry_size (one compiled kernel, one geometry)",
                    kernel.unique_id,
                    binding.accessor_name,
                    first->unique_id,
                    first->ring_size,
                    first->entry_size,
                    pipe->unique_id,
                    pipe->ring_size,
                    pipe->entry_size);
                const NodeRangeSet& receivers = pipe_receiver_set.at(pipe_name);
                TT_FATAL(
                    !receivers.intersects(group_receivers),
                    "Kernel '{}' accessor '{}' names PrefetcherPipeParameter '{}' whose receiver nodes overlap "
                    "another pipe's in the same accessor; pipes sharing an accessor must occupy disjoint nodes (one "
                    "pipe per node, so the accessor resolves to exactly one pipe on every node)",
                    kernel.unique_id,
                    binding.accessor_name,
                    pipe_name);
                group_receivers = group_receivers.merge(receivers);
            }

            // Role: the kernel's nodes are exactly the group's receivers, or one sender node per
            // pipe, none of them a receiver.
            const size_t num_pipes = binding.pipe_parameter_names.size();
            const bool is_receiver_role = is_prefetcher_pipe_receiver_role(nodes, group_receivers);
            const bool is_sender_role =
                !is_receiver_role && is_prefetcher_pipe_sender_role(nodes, group_receivers, num_pipes);
            if (!is_sender_role && !is_receiver_role) {
                const uint32_t on_receivers = nodes.intersection(group_receivers).num_cores();
                TT_THROW(
                    "Kernel '{}' accessor '{}' ({} pipe(s)): the kernel's WorkUnitSpec nodes must equal either the "
                    "union of the group's receiver nodes (receiver role) or be {} node(s) outside them, one per pipe "
                    "(sender role). Kernel covers {} node(s): {} of the {} receiver node(s), {} outside the "
                    "receivers. A role cannot be partial, mixed with the other role, or spill onto extra nodes.",
                    kernel.unique_id,
                    binding.accessor_name,
                    num_pipes,
                    num_pipes,
                    nodes.num_cores(),
                    on_receivers,
                    group_receivers.num_cores(),
                    nodes.num_cores() - on_receivers);
            }

            auto& role_map = is_sender_role ? sender_kernel_of : receiver_kernel_of;
            for (const auto& pipe_name : binding.pipe_parameter_names) {
                auto [it, inserted] = role_map.try_emplace(pipe_name, &kernel);
                if (!inserted) {
                    TT_THROW(
                        "Kernels '{}' and '{}' both bind PrefetcherPipeParameter '{}' as its {}. Only one "
                        "data-movement kernel may own a pipe's {} credits.",
                        it->second->unique_id,
                        kernel.unique_id,
                        pipe_name,
                        is_sender_role ? "sender" : "receiver",
                        is_sender_role ? "sender" : "receiver");
                }
            }
        }
    }
    return roles;
}

}  // namespace tt::tt_metal::experimental
