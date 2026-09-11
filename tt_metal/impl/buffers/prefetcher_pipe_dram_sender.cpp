// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <tt_stl/assert.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <buffer_types.hpp>
#include <core_coord.hpp>

#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/prefetcher_pipe_dram_sender_internal.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "mesh_device.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::experimental {

std::vector<std::shared_ptr<PrefetcherPipe>> CreatePrefetcherPipesForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type,
    bool support_multi_receiver_shards) {
    TT_FATAL(!bank_to_receivers.empty(), "CreatePrefetcherPipesForTensorPrefetcher requires at least one DRAM bank");
    TT_FATAL(entry_size > 0, "PrefetcherPipe entry_size must be > 0");
    TT_FATAL(num_entries > 0, "PrefetcherPipe num_entries must be > 0");
    // The ring is entry_size * num_entries and is sized in uint32. Catch the overflow here: a
    // wrapped product can still be a legal, allocatable ring, so it would surface much later as a
    // capacity error naming a size the caller never asked for.
    TT_FATAL(
        num_entries <= std::numeric_limits<uint32_t>::max() / entry_size,
        "PrefetcherPipe ring size overflows: {} entries of {} B exceeds the {} B a ring can be",
        num_entries,
        entry_size,
        std::numeric_limits<uint32_t>::max());

    // Multi-receiver shards (the legacy interleaved layout) force one sender per bank; the
    // receiver-contiguous layout that disallows them is what lets a bank use two senders.
    const auto mapping = build_dram_sender_mapping(
        &mesh_device,
        bank_to_receivers,
        support_multi_receiver_shards ? DramSenderSplit::OnePerBank : DramSenderSplit::TwoPerBank);
    validate_dram_senders_across_mesh(&mesh_device, mapping);

    // One pipe per sender, in mapping order: build_dram_sender_mapping emits a bank's senders
    // adjacently, in role order, and keeps the banks in input order. That is the order the returned
    // list must keep, since it is what the queue path accumulates bank-local slab bases in.
    std::vector<std::shared_ptr<PrefetcherPipe>> pipes;
    pipes.reserve(mapping.size());
    for (const auto& [sender_logical, receivers] : mapping) {
        pipes.push_back(prefetcher_pipe_dram_sender::PrefetcherPipeDramSenderInternals::make_dram_sender(
            &mesh_device, sender_logical, receivers, entry_size * num_entries, entry_size, buffer_type));
    }
    return pipes;
}

DeviceAddr sender_state_drisc_l1_base(const PrefetcherPipe& pipe) {
    return prefetcher_pipe_dram_sender::PrefetcherPipeDramSenderInternals::sender_state_drisc_l1_base(pipe);
}

}  // namespace tt::tt_metal::experimental
