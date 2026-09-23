// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <tt_stl/assert.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <core_coord.hpp>

#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe_dram_sender_internal.hpp"
#include "mesh_device.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::experimental {

namespace {
std::atomic<uint64_t> next_tensor_prefetcher_factory_id{1};
}

std::vector<PrefetcherPipe> CreatePrefetcherPipesForTensorPrefetcher(
    PrefetcherPipeSpace& space,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    bool support_multi_receiver_shards) {
    TT_FATAL(!bank_to_receivers.empty(), "CreatePrefetcherPipesForTensorPrefetcher requires at least one DRAM bank");
    const auto* mesh_device = space.get_device();
    TT_FATAL(mesh_device != nullptr, "CreatePrefetcherPipesForTensorPrefetcher requires a live space");

    // Multi-receiver shards (the legacy interleaved layout) force one sender per bank; the
    // receiver-contiguous layout that disallows them is what lets a bank use two senders.
    const auto mapping = build_dram_sender_mapping(
        mesh_device,
        bank_to_receivers,
        support_multi_receiver_shards ? DramSenderSplit::OnePerBank : DramSenderSplit::TwoPerBank);
    // Capacity, receiver domain and claim state all live in the space, so its batch validator is
    // the one place they are checked; set_dram_sender_cores below rechecks that each selected core
    // really is a provisioned sender for its bank.
    space.impl().validate_dram_carves(mapping);

    // Each sender's bank-local slab base, taken from the mapping while the whole bank's receiver
    // split is still in one place. Handing it to the pipe is what frees the queue path from
    // re-deriving it from list position.
    const auto bases = recv_index_bases_per_sender(mapping);

    std::vector<CoreCoord> senders;
    senders.reserve(mapping.size());
    for (const auto& [sender, _receivers] : mapping) {
        senders.push_back(sender);
    }
    // Selection and every capacity/domain check complete before either DRISC reservations or
    // worker claims mutate the space.
    set_dram_sender_cores(space, senders);

    // One pipe per sender, in mapping order: build_dram_sender_mapping emits a bank's senders
    // adjacently, in role order, and keeps the banks in input order. The returned list keeps that
    // order as a convention consumers can rely on for pairing pipes with banks; it no longer
    // carries slab numbering, which each pipe now holds itself.
    std::vector<PrefetcherPipe> pipes;
    pipes.reserve(mapping.size());
    const uint64_t factory_id = next_tensor_prefetcher_factory_id.fetch_add(1, std::memory_order_relaxed);
    for (size_t s = 0; s < mapping.size(); ++s) {
        const auto& [sender_logical, receivers] = mapping[s];
        pipes.push_back(create_dram_sender_pipe(
            space, sender_logical, receivers, bases[s], factory_id, static_cast<uint32_t>(mapping.size())));
    }
    return pipes;
}

DeviceAddr sender_state_drisc_l1_base(const PrefetcherPipe& pipe) { return pipe.impl().sender_state_drisc_l1_base(); }

std::shared_ptr<DriscL1Allocation> sender_state_drisc_l1_allocation(const PrefetcherPipe& pipe) {
    return pipe.impl().sender_state_drisc_l1_allocation();
}

}  // namespace tt::tt_metal::experimental
