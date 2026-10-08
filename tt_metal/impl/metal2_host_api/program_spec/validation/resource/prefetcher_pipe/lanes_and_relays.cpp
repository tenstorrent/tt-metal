// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <optional>
#include <unordered_set>

#include <tt_stl/assert.hpp>

#include "hostdev/remote_dfb_config_layout.h"  // PREFETCHER_PIPE_MAX_CREDIT_LANES
#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

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

}  // namespace tt::tt_metal::experimental
