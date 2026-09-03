// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <tt_stl/assert.hpp>

#include <cstdint>
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

std::vector<std::pair<CoreCoord, CoreRangeSet>> BuildTensorPrefetcherSenderMapping(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    bool dual_senders_per_bank) {
    auto mapping = build_dram_sender_mapping(&mesh_device, bank_to_receivers, dual_senders_per_bank);
    validate_dram_senders_across_mesh(&mesh_device, mapping);
    return mapping;
}

std::shared_ptr<PrefetcherPipe> CreatePrefetcherPipeForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    CoreCoord dram_sender_logical,
    const CoreRangeSet& receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type) {
    TT_FATAL(entry_size > 0, "PrefetcherPipe entry_size must be > 0");
    TT_FATAL(num_entries > 0, "PrefetcherPipe num_entries must be > 0");
    return prefetcher_pipe_dram_sender::PrefetcherPipeDramSenderInternals::make_dram_sender(
        &mesh_device, dram_sender_logical, receivers, entry_size * num_entries, entry_size, buffer_type);
}

CoreCoord prefetcher_pipe_sender_core(const PrefetcherPipe& pipe) { return pipe.sender_core(); }

const CoreRangeSet& prefetcher_pipe_receiver_cores(const PrefetcherPipe& pipe) { return pipe.receiver_cores(); }

uint32_t prefetcher_pipe_ring_size(const PrefetcherPipe& pipe) { return pipe.ring_size(); }

uint32_t prefetcher_pipe_fixed_entry_size(const PrefetcherPipe& pipe) { return pipe.fixed_entry_size(); }

SenderCoreType prefetcher_pipe_sender_core_type(const PrefetcherPipe& pipe) { return pipe.sender_core_type(); }

DeviceAddr sender_state_drisc_l1_base(const PrefetcherPipe& pipe) {
    return prefetcher_pipe_dram_sender::PrefetcherPipeDramSenderInternals::sender_state_drisc_l1_base(pipe);
}

}  // namespace tt::tt_metal::experimental
