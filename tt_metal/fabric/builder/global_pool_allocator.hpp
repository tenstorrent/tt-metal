// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains a global allocator strategy for the fabric router

#include "fabric_channel_allocator.hpp"
#include "fabric_router_recipe.hpp"
namespace tt::tt_fabric {
struct ChannelPoolDefinition;
struct GlobalPoolAllocator {
    std::vector<std::shared_ptr<FabricChannelAllocator>> local_pool_allocators;
    std::vector<tt::tt_fabric::FabricChannelPoolType> local_pool_types;
    std::vector<std::shared_ptr<FabricChannelAllocator>> remote_pool_allocators;
    std::vector<tt::tt_fabric::FabricChannelPoolType> remote_pool_types;
};

GlobalPoolAllocator create_global_pool_allocators(
    tt::tt_fabric::Topology topology,
    const tt::tt_fabric::FabricEriscDatamoverOptions& options,
    const std::vector<ChannelPoolDefinition>& pool_definitions,
    size_t num_used_sender_channels,
    size_t num_used_receiver_channels,
    size_t channel_buffer_size_bytes,
    const std::vector<MemoryRegion>& available_buffer_memory_regions);

}  // namespace tt::tt_fabric
