// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "global_pool_allocator.hpp"
#include "fabric_channel_allocator.hpp"
#include "fabric_static_sized_channels_allocator.hpp"
#include "fabric_router_recipe.hpp"

#include <algorithm>

namespace tt::tt_fabric {

std::shared_ptr<tt::tt_fabric::FabricChannelAllocator> create_elastic_channel_allocator(
    tt::tt_fabric::Topology topology,
    const tt::tt_fabric::FabricEriscDatamoverOptions& options,
    const std::vector<ChannelPoolDefinition>& pool_definitions,
    const std::shared_ptr<tt::tt_fabric::FabricStaticSizedChannelsAllocator>& static_allocator,
    size_t num_used_sender_channels,
    size_t num_used_receiver_channels,
    size_t channel_buffer_size_bytes,
    // The memory regions, including the parts already allocated to the static allocator
    const std::vector<MemoryRegion>& router_memory_regions) {
    TT_FATAL(
        pool_definitions.size() == 2,
        "When elastic channels are enabled, only two channel definitions are supported, currently: one for elastic and "
        "one for static channels.");
    const auto& elastic_channel_definition =
        pool_definitions[0].type == FabricChannelPoolType::ELASTIC ? pool_definitions[0] : pool_definitions[1];

    // Current support for elastic channels is very limited. When enabled, only VC0 forwarded sender
    // channels are supported in elastic mode (purely due to the limitations of the builder here).

    // Identify forwarded channels based on topology: Ring (channels 1,2), Torus (channels 1,2,3,4) - VC0 only
    auto my_direction = options.direction;
    std::vector<size_t> forwarded_channel_ids;
    if (topology == tt::tt_fabric::Topology::Linear) {
        forwarded_channel_ids = {1};
    } else if (topology == tt::tt_fabric::Topology::Ring) {
        forwarded_channel_ids = {1, 2};
    } else if (topology == tt::tt_fabric::Topology::Mesh) {
        forwarded_channel_ids = {1, 2, 3, 4};
        forwarded_channel_ids.erase(forwarded_channel_ids.begin() + my_direction);
    } else if (topology == tt::tt_fabric::Topology::Torus) {
        forwarded_channel_ids = {1, 2, 3, 4};
        forwarded_channel_ids.erase(forwarded_channel_ids.begin() + my_direction);
    } else {
        TT_FATAL(false, "Elastic channels are only supported for Ring and Torus topologies");
    }

    // Remove forwarded channels in descending order to maintain channel_id correctness
    for (auto it = forwarded_channel_ids.rbegin(); it != forwarded_channel_ids.rend(); ++it) {
        static_allocator->remove_sender_channel(*it);
    }

    // Get consumed memory regions and defragment remaining memory
    auto consumed = static_allocator->get_consumed_memory_regions();
    auto remaining_memory_regions = subtract_memory_regions(router_memory_regions, consumed);

    size_t num_slots_per_chunk = 2;
    if (getenv("TT_FABRIC_SLOTS_PER_CHUNK")) {
        // TEMPORARY: MOVE OUT BEFORE MERGE
        num_slots_per_chunk = std::stoi(getenv("TT_FABRIC_SLOTS_PER_CHUNK"));
    }

    auto elastic_channels_allocator = std::make_shared<ElasticChannelsAllocator>(
        topology,
        options,
        elastic_channel_definition,
        remaining_memory_regions,
        channel_buffer_size_bytes,
        num_slots_per_chunk);

    return elastic_channels_allocator;
}

/*
 * The current strategy, to preserve performance and to simplify apples-to-apples comparisons
 * when enabling elastic channels vs static channels, is to first create all the static channels
 * and then inspect the `ChannelPoolDefinition`s to find which ones are elastic channels.
 * "Compress" out the allocated static channels that are actually supposed to be elastic channels.
 * Defragment the memory regions to make the static channels contiguous so the elastic channel regions
 * are as contiguous as possible.
 */
GlobalPoolAllocator create_global_pool_allocators(
    tt::tt_fabric::Topology topology,
    const tt::tt_fabric::FabricEriscDatamoverOptions& options,
    const std::vector<ChannelPoolDefinition>& pool_definitions,
    size_t num_used_sender_channels,
    size_t num_used_receiver_channels,
    size_t channel_buffer_size_bytes,
    const std::vector<MemoryRegion>& available_buffer_memory_regions) {
    GlobalPoolAllocator global_pool_allocator;
    std::vector<std::shared_ptr<tt::tt_fabric::FabricChannelAllocator>> pool_allocators;
    std::vector<tt::tt_fabric::FabricChannelPoolType> pool_types;

    // Create the static channel pool allocator first
    auto static_allocator = std::make_shared<tt::tt_fabric::FabricStaticSizedChannelsAllocator>(
        topology,
        options,
        pool_definitions[0].type == FabricChannelPoolType::STATIC ? pool_definitions[0] : pool_definitions[1],
        num_used_sender_channels,
        num_used_receiver_channels,
        channel_buffer_size_bytes,
        std::accumulate(
            available_buffer_memory_regions.begin(),
            available_buffer_memory_regions.end(),
            size_t{0},
            [](size_t sum, const MemoryRegion& region) { return sum + region.get_size(); }),
        available_buffer_memory_regions);

    pool_allocators.push_back(static_allocator);
    pool_types.push_back(tt::tt_fabric::FabricChannelPoolType::STATIC);

    // Check for elastic channels in pool definitions
    bool elastic_channels_found =
        std::any_of(pool_definitions.begin(), pool_definitions.end(), [](const ChannelPoolDefinition& pool_definition) {
            return pool_definition.type == FabricChannelPoolType::ELASTIC;
        });

    if (elastic_channels_found) {
        // Create elastic channel allocator by compressing out forwarded channels
        auto elastic_allocator = create_elastic_channel_allocator(
            topology,
            options,
            pool_definitions,
            static_allocator,
            num_used_sender_channels,
            num_used_receiver_channels,
            channel_buffer_size_bytes,
            available_buffer_memory_regions);

        pool_allocators.push_back(elastic_allocator);
        pool_types.push_back(tt::tt_fabric::FabricChannelPoolType::ELASTIC);
    }

    // Populate the global pool allocator result
    global_pool_allocator.local_pool_allocators = pool_allocators;
    global_pool_allocator.local_pool_types = pool_types;

    // For now, remote allocators mirror local ones (this may need refinement)
    global_pool_allocator.remote_pool_allocators = pool_allocators;
    global_pool_allocator.remote_pool_types = pool_types;

    return global_pool_allocator;
}

}  // namespace tt::tt_fabric
