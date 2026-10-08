// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <tt_stl/assert.hpp>

#include "hostdev/remote_dfb_config_layout.h"  // PREFETCHER_PIPE_MAX_CREDIT_LANES
#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

// Everything about a PrefetcherPipeParameter is decidable from the spec alone (geometry and kernel
// placement); the pipe object arrives later via ProgramRunArgs and is reconciled against this
// geometry then.
//
// Rule 1. Geometry: non-empty receivers, ring_size > 0, entry_size > 0, L1-aligned and <= ring_size.
// (Rules 2-6 are structural: roles.cpp and lanes_and_relays.cpp.)
void ValidatePrefetcherPipeParameter(const PrefetcherPipeParameter& pipe, uint32_t l1_alignment) {
    const NodeRangeSet receivers = to_node_range_set(pipe.receivers);
    TT_FATAL(receivers.num_cores() > 0, "PrefetcherPipeParameter '{}' has no receiver nodes", pipe.unique_id);
    TT_FATAL(pipe.ring_size > 0, "PrefetcherPipeParameter '{}' has ring_size = 0", pipe.unique_id);
    TT_FATAL(pipe.entry_size > 0, "PrefetcherPipeParameter '{}' has entry_size = 0", pipe.unique_id);
    TT_FATAL(
        pipe.entry_size % l1_alignment == 0,
        "PrefetcherPipeParameter '{}' entry_size {} must be a multiple of the L1 alignment ({})",
        pipe.unique_id,
        pipe.entry_size,
        l1_alignment);
    TT_FATAL(
        pipe.entry_size <= pipe.ring_size,
        "PrefetcherPipeParameter '{}' entry_size {} exceeds ring_size {}",
        pipe.unique_id,
        pipe.entry_size,
        pipe.ring_size);
}

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

// Rules per parameter:
//  4. Receiver-side credit lanes P: the receiver kernel's num_threads (and, with a relay, the
//     relay's PRODUCER kernels' num_threads) must agree, fit the architecture's lane capacity,
//     and, when P > 1, divide the ring's entry count.
// Rules per relay DFB:
//  5. Not also borrowed_from (checked in ValidateDFBSpec). Every relayed pipe shares
//     ring_size / entry_size; the DFB's entry_size divides that entry_size (the relay may page one pipe entry as
//     several pages, e.g. a K-block as tiles, or one entry per consumer; only with a single-threaded producer) and
//     entry_size * num_entries is the pipe's
//     whole entries: ring_size rounded down to a multiple of the pipe's entry_size (the DFB is
//     exactly the ring the pipe uses; the pipe skips any trailing gap at the wrap).
//  6. The relayed pipes' receiver sets are pairwise disjoint and their union equals the DFB's
//     node set; every PRODUCER kernel binds exactly the relayed pipe set under one accessor (so
//     it is those pipes' receiver kernel and can drive the protocol the relay depends on).
void ValidatePrefetcherPipeLanesAndRelays(
    const ValidationContext& ctx, const PrefetcherPipeRoles& roles, tt::ARCH arch) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;
    const uint32_t lane_capacity = is_gen2_arch(arch) ? PREFETCHER_PIPE_MAX_CREDIT_LANES : 1u;
    const auto& pipe_receiver_set = roles.pipe_receiver_set;
    const auto& receiver_kernel_of = roles.receiver_kernel_of;

    for (const auto& pipe : spec.advanced_options.prefetcher_pipe_parameters) {
        auto receiver_it = receiver_kernel_of.find(pipe.unique_id);
        const KernelSpec* receiver_kernel = receiver_it == receiver_kernel_of.end() ? nullptr : receiver_it->second;

        // Rule 4: receiver-side credit lanes. Sources: the receiver binding kernel and every
        // relay's PRODUCER kernels (uniform per role by ValidateDFBEndpoints).
        std::optional<uint32_t> lanes;
        const KernelSpec* lanes_source = nullptr;
        auto take_lanes = [&](const KernelSpec* kernel) {
            if (!lanes.has_value()) {
                lanes = kernel->num_threads;
                lanes_source = kernel;
                return;
            }
            TT_FATAL(
                *lanes == kernel->num_threads,
                "PrefetcherPipeParameter '{}' receiver-side kernels disagree on thread count: '{}' has {} "
                "threads, '{}' has {}. The receiver kernel and every relay DFB producer must use the same "
                "num_threads (this is the pipe's credit lane count).",
                pipe.unique_id,
                lanes_source->unique_id,
                *lanes,
                kernel->unique_id,
                kernel->num_threads);
        };
        if (receiver_kernel != nullptr) {
            take_lanes(receiver_kernel);
        }
        // A pipe bound only by kernels (no relay DFB) has no entry.
        if (const auto relays_it = collected.prefetcher_pipe_relays.find(pipe.unique_id);
            relays_it != collected.prefetcher_pipe_relays.end()) {
            for (const DataflowBufferSpec* relay : relays_it->second) {
                for (const auto& rec : collected.dfb_endpoints.at(relay->unique_id).producers) {
                    take_lanes(rec.kernel);
                }
            }
        }
        if (lanes.has_value() && *lanes > 1) {
            TT_FATAL(
                *lanes <= lane_capacity,
                "PrefetcherPipeParameter '{}' receiver kernel '{}' has {} threads, but a pipe supports at most {} "
                "credit lanes on this architecture",
                pipe.unique_id,
                lanes_source->unique_id,
                *lanes,
                lane_capacity);
            TT_FATAL(
                pipe.ring_size % pipe.entry_size == 0,
                "PrefetcherPipeParameter '{}' with {} credit lanes requires entry_size {} to divide ring_size {}",
                pipe.unique_id,
                *lanes,
                pipe.entry_size,
                pipe.ring_size);
            TT_FATAL(
                (pipe.ring_size / pipe.entry_size) % *lanes == 0,
                "PrefetcherPipeParameter '{}' ring holds {} entries of {} bytes, which is not a multiple of {} "
                "credit lanes (receiver kernel '{}' num_threads)",
                pipe.unique_id,
                pipe.ring_size / pipe.entry_size,
                pipe.entry_size,
                *lanes,
                lanes_source->unique_id);
        }
    }

    // Rules 5 and 6: relay DFBs.
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.advanced_options.prefetcher_pipe_relays.empty()) {
            continue;
        }

        const PrefetcherPipeParameter* first =
            collected.prefetcher_pipe_by_name.at(dfb.advanced_options.prefetcher_pipe_relays[0]);
        NodeRangeSet relayed_receivers;
        for (const auto& pipe_name : dfb.advanced_options.prefetcher_pipe_relays) {
            const PrefetcherPipeParameter* pipe = collected.prefetcher_pipe_by_name.at(pipe_name);
            TT_FATAL(
                pipe->ring_size == first->ring_size && pipe->entry_size == first->entry_size,
                "DFB '{}' relays PrefetcherPipeParameters '{}' (ring_size {}, entry_size {}) and '{}' (ring_size "
                "{}, entry_size {}); every pipe relayed by one DFB must share ring_size and entry_size",
                dfb.unique_id,
                first->unique_id,
                first->ring_size,
                first->entry_size,
                pipe->unique_id,
                pipe->ring_size,
                pipe->entry_size);
            const NodeRangeSet& receivers = pipe_receiver_set.at(pipe_name);
            TT_FATAL(
                !relayed_receivers.intersects(receivers),
                "DFB '{}' relays PrefetcherPipeParameter '{}' whose receiver nodes overlap another relayed pipe's; "
                "relayed pipes must have disjoint receivers",
                dfb.unique_id,
                pipe_name);
            relayed_receivers = relayed_receivers.merge(receivers);
        }

        TT_FATAL(
            dfb.entry_size != 0 && first->entry_size % dfb.entry_size == 0,
            "DFB '{}' entry_size {} must divide relayed PrefetcherPipeParameter '{}' entry_size {}: a relay DFB "
            "pages each pipe entry as a whole number of its own entries",
            dfb.unique_id,
            dfb.entry_size,
            first->unique_id,
            first->entry_size);
        if (dfb.entry_size != first->entry_size) {
            // Credit lanes stripe whole pipe entries over the relay's producer threads; a relay paged
            // finer than the pipe is only implemented for one producer thread.
            for (const auto& rec : collected.dfb_endpoints.at(dfb.unique_id).producers) {
                TT_FATAL(
                    rec.kernel->num_threads == 1,
                    "DFB '{}' pages relayed PrefetcherPipeParameter '{}' entry_size {} as entries of {} bytes, which "
                    "needs a single-threaded relay producer, but kernel '{}' has {} threads",
                    dfb.unique_id,
                    first->unique_id,
                    first->entry_size,
                    dfb.entry_size,
                    rec.kernel->unique_id,
                    rec.kernel->num_threads);
            }
        }
        const uint32_t usable_ring_size = first->ring_size - first->ring_size % first->entry_size;
        TT_FATAL(
            static_cast<uint64_t>(dfb.entry_size) * dfb.num_entries == usable_ring_size,
            "DFB '{}' (entry_size {} * num_entries {} = {} bytes) must exactly cover the {} bytes of whole entries in "
            "relayed PrefetcherPipeParameter '{}' (ring_size {}, entry_size {})",
            dfb.unique_id,
            dfb.entry_size,
            dfb.num_entries,
            static_cast<uint64_t>(dfb.entry_size) * dfb.num_entries,
            usable_ring_size,
            first->unique_id,
            first->ring_size,
            first->entry_size);

        const NodeRangeSet& dfb_nodes = collected.dfb_node_set.at(dfb.unique_id);
        TT_FATAL(
            same_node_set(dfb_nodes, relayed_receivers),
            "DFB '{}' relays PrefetcherPipe(s) whose receiver nodes do not match the DFB's node set (union of its "
            "bound kernels' WorkUnitSpec nodes). The relay must live on exactly the receiver nodes.",
            dfb.unique_id);

        // Every PRODUCER must be the relayed pipes' receiver kernel: it binds exactly this pipe
        // set under one accessor. (Binding implies data-movement by rule 2; the tiling rule then
        // makes its nodes the receiver union, i.e. the DFB's nodes.) Without the binding the
        // producer could not drive the pipe protocol the relay depends on.
        const std::unordered_set<PrefetcherPipeParamName> relayed_set(
            dfb.advanced_options.prefetcher_pipe_relays.begin(), dfb.advanced_options.prefetcher_pipe_relays.end());
        for (const auto& rec : collected.dfb_endpoints.at(dfb.unique_id).producers) {
            const bool binds_relayed_set = std::any_of(
                rec.kernel->advanced_options.prefetcher_pipe_bindings.begin(),
                rec.kernel->advanced_options.prefetcher_pipe_bindings.end(),
                [&](const KernelAdvancedOptions::PrefetcherPipeBinding& binding) {
                    return binding.pipe_parameter_names.size() == relayed_set.size() &&
                           std::all_of(
                               binding.pipe_parameter_names.begin(),
                               binding.pipe_parameter_names.end(),
                               [&](const PrefetcherPipeParamName& n) { return relayed_set.contains(n); });
                });
            TT_FATAL(
                binds_relayed_set,
                "Kernel '{}' is a PRODUCER of relay DFB '{}' but has no PrefetcherPipe accessor naming exactly the "
                "relayed pipe set ({} pipe(s), first '{}'). A relay's producer is the relayed pipes' receiver "
                "data-movement kernel; it must bind them (KernelAdvancedOptions::prefetcher_pipe_bindings) under one "
                "accessor.",
                rec.kernel->unique_id,
                dfb.unique_id,
                relayed_set.size(),
                first->unique_id);
        }
    }
}

// PrefetcherPipe bindings: accessor names, non-empty pipe lists, each pipe bound once
void ValidatePrefetcherPipeBindings(const KernelSpec& kernel) {
    std::unordered_set<std::string> accessor_names;
    // A kernel binds a given pipe at most once, within and across accessors: a second binding
    // would be a second device object over the same credit counters (two names for one pipe is a
    // handle alias, not a binding).
    std::unordered_set<PrefetcherPipeParamName> bound_pipes;
    for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
        auto [it, inserted] = accessor_names.insert(binding.accessor_name);
        TT_FATAL(
            inserted,
            "Kernel '{}' has duplicate PrefetcherPipe accessor_name '{}'",
            kernel.unique_id,
            binding.accessor_name);
        TT_FATAL(
            IsValidCppIdentifier(binding.accessor_name),
            "Kernel '{}' PrefetcherPipe accessor_name '{}' must be a valid C++ identifier",
            kernel.unique_id,
            binding.accessor_name);
        ValidateAccessorNameLength(kernel.unique_id, "PrefetcherPipe", binding.accessor_name);
        TT_FATAL(
            !binding.pipe_parameter_names.empty(),
            "Kernel '{}' PrefetcherPipe accessor '{}' names no PrefetcherPipeParameter",
            kernel.unique_id,
            binding.accessor_name);
        for (const auto& pipe_name : binding.pipe_parameter_names) {
            auto [pit, pinserted] = bound_pipes.insert(pipe_name);
            TT_FATAL(
                pinserted,
                "Kernel '{}' binds PrefetcherPipeParameter '{}' more than once (latest under accessor_name '{}'). "
                "A kernel may bind a given pipe at most once.",
                kernel.unique_id,
                pipe_name,
                binding.accessor_name);
        }
    }
}

void ValidatePrefetcherPipesUsed(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    // Every declared PrefetcherPipeParameter must be used by a kernel binding or a relay DFB.
    // (An unused pipe parameter would demand a run arg nothing reads.)
    std::unordered_set<PrefetcherPipeParamName> used_pipes;
    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            used_pipes.insert(binding.pipe_parameter_names.begin(), binding.pipe_parameter_names.end());
        }
    }
    for (const auto& [pipe_name, relays] : collected.prefetcher_pipe_relays) {
        used_pipes.insert(pipe_name);
    }
    for (const auto& pipe_parameter : spec.advanced_options.prefetcher_pipe_parameters) {
        TT_FATAL(
            used_pipes.contains(pipe_parameter.unique_id),
            "PrefetcherPipeParameter '{}' is defined but not bound by any kernel or relay DFB",
            pipe_parameter.unique_id);
    }
}

}  // namespace tt::tt_metal::experimental
