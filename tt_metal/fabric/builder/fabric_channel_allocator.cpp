// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_channel_allocator.hpp"
#include "fabric_router_recipe.hpp"

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

            bool overlap = (region1.start_address < region2.start_address &&
                            region1.start_address + region1.size > region2.start_address) ||
                           (region2.start_address < region1.start_address &&
                            region2.start_address + region2.size > region1.start_address);
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
    const tt::tt_fabric::ChannelPoolDefinition& /*pool_definition*/,
    const std::vector<MemoryRegion>& memory_regions,
    size_t buffer_slot_size_bytes,
    size_t num_slots_per_chunk) :
    FabricChannelAllocator(topology, options, memory_regions), num_slots_per_chunk_(num_slots_per_chunk) {
    size_t num_entries = std::accumulate(
        memory_regions.begin(),
        memory_regions.end(),
        size_t{0},
        [buffer_slot_size_bytes](size_t sum, const MemoryRegion& region) {
            return sum + (region.get_size() / buffer_slot_size_bytes);
        });

    chunk_addresses_.reserve(num_entries);
    size_t chunk_size = buffer_slot_size_bytes * num_slots_per_chunk_;
    for (const auto& memory_region : memory_regions) {
        size_t current_address = memory_region.get_start_address();
        size_t next_address = current_address + chunk_size;
        while (next_address < memory_region.get_end_address()) {
            chunk_addresses_.push_back(current_address);
            current_address = next_address;
            next_address += chunk_size;
        }
    }

    auto regions_overlap = [](size_t address1, size_t address2, size_t size) -> bool {
        bool overlap =
            ((address1 < address2 && address1 + size > address2) ||
             (address2 < address1 && address2 + size > address1));
        return overlap;
    };

    // validate that the chunk addresses are not overlapping
    for (size_t i = 0; i < chunk_addresses_.size(); ++i) {
        for (size_t j = i + 1; j < chunk_addresses_.size(); ++j) {
            TT_FATAL(
                !regions_overlap(chunk_addresses_[i], chunk_addresses_[j], chunk_size),
                "Chunk addresses {} and {} overlap",
                i,
                j);
        }
    }
}

void ElasticChannelsAllocator::emit_ct_args(
    std::vector<uint32_t>& ct_args,
    size_t /*num_fwd_paths*/,
    size_t num_used_sender_channels,
    size_t num_used_receiver_channels) const {
    ct_args.reserve(ct_args.size() + 2 + chunk_addresses_.size());

    ct_args.push_back(static_cast<uint32_t>(chunk_addresses_.size()));
    ct_args.push_back(static_cast<uint32_t>(num_slots_per_chunk_));
    std::for_each(chunk_addresses_.begin(), chunk_addresses_.end(), [&ct_args](size_t chunk_address) {
        ct_args.push_back(static_cast<uint32_t>(chunk_address));
    });
    log_info(tt::LogMetal, "\tElasticChannelsAllocator::emit_ct_args");
    log_info(tt::LogMetal, "\t\tnum_used_sender_channels={}", num_used_sender_channels);
    log_info(tt::LogMetal, "\t\tnum_used_receiver_channels={}", num_used_receiver_channels);
    log_info(tt::LogMetal, "\t\tchunk_addresses={}", chunk_addresses_);
}

}  // namespace tt::tt_fabric
