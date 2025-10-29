// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_channel_allocator.hpp"
#include <tt_stl/assert.hpp>
#include <algorithm>
#include <numeric>

namespace tt::tt_fabric {

// Base FabricChannelAllocator implementation
FabricChannelAllocator::FabricChannelAllocator(
    tt::tt_fabric::Topology topology,
    const tt::tt_fabric::FabricEriscDatamoverOptions& options,
    const std::vector<MemoryRegion>& memory_regions) :
    topology_(topology), options_(options), memory_regions_(memory_regions) {
    TT_FATAL(!memory_regions_.empty(), "At least one memory region must be provided");

    // Validate that regions don't overlap
    for (size_t i = 0; i < memory_regions_.size(); ++i) {
        for (size_t j = i + 1; j < memory_regions_.size(); ++j) {
            const auto& region1 = memory_regions_[i];
            const auto& region2 = memory_regions_[j];

            bool overlap = (region1.start_address < region2.get_end_address()) &&
                           (region2.start_address < region1.get_end_address());
            TT_FATAL(!overlap, "Memory regions {} and {} overlap", i, j);
        }
    }
}

size_t FabricChannelAllocator::get_total_available_memory() const {
    return std::accumulate(
        memory_regions_.begin(), memory_regions_.end(), size_t{0}, [](size_t sum, const MemoryRegion& region) {
            return sum + region.get_size();
        });
}

// ElasticChannelsAllocator implementation
ElasticChannelsAllocator::ElasticChannelsAllocator(
    tt::tt_fabric::Topology topology,
    const tt::tt_fabric::FabricEriscDatamoverOptions& options,
    const std::vector<MemoryRegion>& memory_regions,
    size_t buffer_slot_size_bytes,
    size_t min_buffers_per_chunk,
    size_t max_buffers_per_chunk) :
    FabricChannelAllocator(topology, options, memory_regions) {
    TT_THROW("Not implemented");
}

void ElasticChannelsAllocator::emit_ct_args(std::vector<uint32_t>& ct_args, size_t num_fwd_paths, size_t num_used_sender_channels, size_t num_used_receiver_channels) const { TT_THROW("Not implemented"); }

}  // namespace tt::tt_fabric
