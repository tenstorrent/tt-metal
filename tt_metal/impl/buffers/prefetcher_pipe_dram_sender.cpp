// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <tt_stl/assert.hpp>

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

namespace tt::tt_metal::experimental {

std::vector<TensorPrefetcherBankPipes> CreatePrefetcherPipesForTensorPrefetcher(
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

    // build_dram_sender_mapping emits a bank's senders adjacently and keeps the banks in input
    // order, so walking both lists in lockstep regroups the flat mapping without re-deriving the
    // split. A sender's DRAM-logical x is its bank id.
    std::vector<TensorPrefetcherBankPipes> bank_pipes;
    bank_pipes.reserve(bank_to_receivers.size());
    size_t sender = 0;
    for (const auto& [bank_id, _receivers] : bank_to_receivers) {
        TensorPrefetcherBankPipes group{.bank_id = bank_id, .pipes = {}};
        while (sender < mapping.size() && static_cast<uint32_t>(mapping[sender].first.x) == bank_id) {
            group.pipes.push_back(prefetcher_pipe_dram_sender::PrefetcherPipeDramSenderInternals::make_dram_sender(
                &mesh_device,
                mapping[sender].first,
                mapping[sender].second,
                entry_size * num_entries,
                entry_size,
                buffer_type));
            ++sender;
        }
        TT_FATAL(
            !group.pipes.empty(),
            "DRAM bank {} was placed no sender: build_dram_sender_mapping must emit each bank's senders adjacently "
            "and in bank_to_receivers order for the regrouping above to hold.",
            bank_id);
        bank_pipes.push_back(std::move(group));
    }
    return bank_pipes;
}

std::vector<std::pair<CoreCoord, CoreRangeSet>> prefetcher_pipe_sender_receiver_mapping(
    const std::vector<TensorPrefetcherBankPipes>& banks) {
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    size_t num_pipes = 0;
    for (const auto& bank : banks) {
        num_pipes += bank.pipes.size();
    }
    mapping.reserve(num_pipes);
    for (const auto& bank : banks) {
        for (const auto& pipe : bank.pipes) {
            TT_FATAL(pipe != nullptr, "PrefetcherPipe group for DRAM bank {} holds a null pipe", bank.bank_id);
            mapping.emplace_back(pipe->sender_core(), pipe->receiver_cores());
        }
    }
    return mapping;
}

CoreCoord prefetcher_pipe_sender_core(const PrefetcherPipe& pipe) { return pipe.sender_core(); }

const CoreRangeSet& prefetcher_pipe_receiver_cores(const PrefetcherPipe& pipe) { return pipe.receiver_cores(); }

SenderCoreType prefetcher_pipe_sender_core_type(const PrefetcherPipe& pipe) { return pipe.sender_core_type(); }

uint32_t prefetcher_pipe_config_address(const PrefetcherPipe& pipe) { return pipe.config_address(); }

DeviceAddr sender_state_drisc_l1_base(const PrefetcherPipe& pipe) {
    return prefetcher_pipe_dram_sender::PrefetcherPipeDramSenderInternals::sender_state_drisc_l1_base(pipe);
}

}  // namespace tt::tt_metal::experimental
