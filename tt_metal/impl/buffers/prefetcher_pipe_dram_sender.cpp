// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <tt_stl/assert.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <buffer_types.hpp>
#include <core_coord.hpp>

#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/prefetcher_pipe_dram_sender_internal.hpp"
#include "impl/buffers/prefetcher_pipe_dram_sender_state.hpp"
#include "impl/buffers/prefetcher_pipe_internal.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "mesh_device.hpp"

namespace tt::tt_metal::experimental {

TensorPrefetcherPipes::TensorPrefetcherPipes(
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping,
    CoreRangeSet receiver_cores,
    std::shared_ptr<DriscL1Allocation> drisc_sender_state_alloc,
    std::vector<std::shared_ptr<PrefetcherPipe>> pipes,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t max_num_receivers) :
    mapping_(std::move(mapping)),
    receiver_cores_(std::move(receiver_cores)),
    drisc_sender_state_alloc_(std::move(drisc_sender_state_alloc)),
    pipes_(std::move(pipes)),
    entry_size_(entry_size),
    num_entries_(num_entries),
    ring_size_(entry_size * num_entries),
    max_num_receivers_(max_num_receivers) {}

TensorPrefetcherPipes::~TensorPrefetcherPipes() = default;

PrefetcherPipe& TensorPrefetcherPipes::pipe(size_t index) {
    TT_FATAL(
        index < pipes_.size(), "TensorPrefetcherPipes has {} pipes; index {} is out of range", pipes_.size(), index);
    return *pipes_[index];
}

DeviceAddr sender_state_drisc_l1_base(const TensorPrefetcherPipes& pipes) {
    return pipes.drisc_sender_state_alloc_->addr();
}

std::shared_ptr<TensorPrefetcherPipes> CreatePrefetcherPipesForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type,
    bool support_multi_receiver_shards) {
    TT_FATAL(entry_size > 0, "TensorPrefetcherPipes entry_size must be > 0");
    TT_FATAL(num_entries > 0, "TensorPrefetcherPipes num_entries must be > 0");

    // Multi-receiver shards (legacy interleaved layout) force one sender per bank; the
    // receiver-contiguous layout that disallows them is what lets a bank use two senders. Shared
    // with the GlobalCircularBuffer factory so both transports place senders identically.
    const auto mapping = build_dram_sender_mapping(
        &mesh_device, bank_to_receivers, /*dual_senders_per_bank=*/!support_multi_receiver_shards);
    validate_dram_senders_across_mesh(&mesh_device, mapping);

    CoreRangeSet receiver_cores;
    uint32_t total_receivers = 0;
    uint32_t max_num_receivers = 0;
    for (const auto& [_sender, receivers] : mapping) {
        total_receivers += receivers.num_cores();
        max_num_receivers = std::max(max_num_receivers, receivers.num_cores());
        receiver_cores = receiver_cores.merge(receivers);
    }
    TT_FATAL(
        receiver_cores.num_cores() == total_receivers,
        "Receiver sets must be disjoint across DRAM senders: {} receivers were listed but only {} are distinct",
        total_receivers,
        receiver_cores.num_cores());

    const auto context_id = mesh_device.impl().get_context_id();
    const uint32_t l1_alignment = MetalContext::instance(context_id).hal().get_alignment(HalMemType::L1);

    // One arena block for the whole set: every sender plants its prefix + config page at the same
    // DRISC L1 offset, so the block is sized for the largest sender and the shorter ones leave the
    // tail unused. Allocating per pipe would consume the small DRISC zone once per sender.
    const uint32_t block_bytes = kPrefetcherPipeSenderPrefixBytes +
                                 compute_prefetcher_pipe_config_page_layout(max_num_receivers, l1_alignment).page_size;
    auto drisc_alloc = mesh_device.impl().drisc_l1_arena().allocate(block_bytes, l1_alignment);

    const std::vector<uint32_t> recv_index_bases = recv_index_bases_per_sender(mapping);
    const uint32_t ring_size = entry_size * num_entries;

    std::vector<std::shared_ptr<PrefetcherPipe>> pipes;
    pipes.reserve(mapping.size());
    for (size_t s = 0; s < mapping.size(); ++s) {
        const prefetcher_pipe_dram_sender::DramSenderPlacement placement{
            .sender_logical = mapping[s].first,
            .drisc_block_addr = drisc_alloc->addr(),
            .recv_index_base = recv_index_bases[s],
            .max_num_receivers = max_num_receivers,
        };
        pipes.push_back(prefetcher_pipe_dram_sender::PrefetcherPipeDramSenderInternals::make_dram_sender(
            &mesh_device, mapping[s].second, ring_size, entry_size, placement, buffer_type));
    }

    return std::shared_ptr<TensorPrefetcherPipes>(new TensorPrefetcherPipes(
        mapping,
        std::move(receiver_cores),
        std::move(drisc_alloc),
        std::move(pipes),
        entry_size,
        num_entries,
        max_num_receivers));
}

std::vector<uint8_t> AttachTensorPrefetcherPipes(Program& program, TensorPrefetcherPipes& pipes) {
    std::vector<uint8_t> pipe_ids;
    pipe_ids.reserve(pipes.num_pipes());
    for (size_t s = 0; s < pipes.num_pipes(); ++s) {
        pipe_ids.push_back(AttachPrefetcherPipe(
            program, pipes.pipe(s), pipes.sender_receiver_core_mapping()[s].second, pipes.entry_size()));
    }
    return pipe_ids;
}

}  // namespace tt::tt_metal::experimental
