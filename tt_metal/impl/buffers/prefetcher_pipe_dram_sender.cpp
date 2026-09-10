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
    return prefetcher_pipe_sender_receiver_mapping(flatten_prefetcher_pipe_banks(banks));
}

std::vector<std::shared_ptr<PrefetcherPipe>> flatten_prefetcher_pipe_banks(
    const std::vector<TensorPrefetcherBankPipes>& banks) {
    std::vector<std::shared_ptr<PrefetcherPipe>> pipes;
    size_t num_pipes = 0;
    for (const auto& bank : banks) {
        num_pipes += bank.pipes.size();
    }
    pipes.reserve(num_pipes);
    for (const auto& bank : banks) {
        for (const auto& pipe : bank.pipes) {
            TT_FATAL(pipe != nullptr, "PrefetcherPipe group for DRAM bank {} holds a null pipe", bank.bank_id);
            pipes.push_back(pipe);
        }
    }
    return pipes;
}

std::vector<TensorPrefetcherBankPipes> group_prefetcher_pipes_by_bank(
    const std::vector<std::shared_ptr<PrefetcherPipe>>& pipes) {
    std::vector<TensorPrefetcherBankPipes> banks;
    for (const auto& pipe : pipes) {
        TT_FATAL(pipe != nullptr, "PrefetcherPipe list holds a null pipe at index {}", banks.size());
        const auto bank_id = static_cast<uint32_t>(pipe->sender_core().x);
        // A run, not a lookup: a bank that reappears after its run opens a second group with the
        // same id, which the queue path rejects. Folding it back into the first group would accept a
        // list whose banks interleave, and slab bases come from adjacency.
        if (banks.empty() || banks.back().bank_id != bank_id) {
            banks.push_back(TensorPrefetcherBankPipes{.bank_id = bank_id, .pipes = {}});
        }
        banks.back().pipes.push_back(pipe);
    }
    for (const auto& bank : banks) {
        TT_FATAL(
            bank.pipes.size() <= 2,
            "DRAM bank {} is driven by {} PrefetcherPipes; a bank has two DRISC sender cores, so it can hold at most "
            "two",
            bank.bank_id,
            bank.pipes.size());
        // Which of the bank's two DRISC cores a pipe sends from is what orders the pair: the first
        // role's pipe owns the bank's leading receivers, and slab bases are accumulated in list
        // order, so a swapped pair would hand the trailing receivers base 0. Receiver counts cannot
        // tell the two apart -- an even split gives both the same count.
        auto* mesh_device = bank.pipes.front()->get_device();
        // Any device of the mesh answers this: the roles are logical coords naming endpoint roles,
        // which a well-formed descriptor set resolves the same way mesh-wide (only the physical
        // subchannel behind a role moves with a device's DRAM harvest).
        const std::vector<CoreCoord> roles =
            mesh_device->impl().dram_sender_logical_cores(mesh_device->get_devices().front(), bank.bank_id);
        size_t previous_role = 0;
        for (size_t p = 0; p < bank.pipes.size(); ++p) {
            const CoreCoord sender = bank.pipes[p]->sender_core();
            const auto role = std::find(roles.begin(), roles.end(), sender);
            TT_FATAL(
                role != roles.end(),
                "PrefetcherPipe {} sends from {}, which is not one of DRAM bank {}'s sender cores",
                p,
                sender.str(),
                bank.bank_id);
            const auto role_index = static_cast<size_t>(std::distance(roles.begin(), role));
            TT_FATAL(
                p == 0 || role_index > previous_role,
                "DRAM bank {}'s PrefetcherPipes are not in sender order: pipe {} sends from {}, its bank's sender "
                "{}, after a pipe that sends from its sender {}. A sender's bank-local slab base is accumulated in "
                "list order, so pass the pipes as CreatePrefetcherPipesForTensorPrefetcher returned them",
                bank.bank_id,
                p,
                sender.str(),
                role_index,
                previous_role);
            previous_role = role_index;
        }
    }
    return banks;
}

DeviceAddr sender_state_drisc_l1_base(const PrefetcherPipe& pipe) {
    return prefetcher_pipe_dram_sender::PrefetcherPipeDramSenderInternals::sender_state_drisc_l1_base(pipe);
}

}  // namespace tt::tt_metal::experimental
