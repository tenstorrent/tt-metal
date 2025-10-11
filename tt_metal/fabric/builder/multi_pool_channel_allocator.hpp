// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_channel_allocator.hpp"
#include "fabric_router_recipe.hpp"
#include "fabric_static_sized_channels_allocator.hpp"
#include <memory>
#include <vector>

namespace tt::tt_fabric {

/**
 * Multi-pool channel allocator coordinator/manager.
 *
 * The `MultiPoolChannelAllocator` manages multiple channel allocators (pools), where each
 * pool can be either a static or elastic allocator. This design centralizes memory
 * management across pools, enabling future optimizations like memory balancing across pools.
 *
 * Key Design Principles:
 * - Single owner of all pool allocators
 * - Supports heterogeneous pool types (static and elastic)
 * - Enables future memory balancing across pools from a central location
 * - Coordinator pattern: doesn't inherit from FabricChannelAllocator, but manages multiple instances
 *
 * The allocator emits compile-time arguments in the following format:
 * 1. Special tag (0xabcd1234)
 * 2. Number of pools
 * 3. Array of pool types (FabricChannelPoolType for each pool)
 * 4. Individual pool CT args (each pool's allocator emits its own args)
 *
 * Note: The channel-to-pool mappings (steps 5-8 of the schema) are emitted separately
 * by the ChannelToPoolMapping class, not by this allocator.
 */
class MultiPoolChannelAllocator {
public:
    /**
     * Construct a multi-pool allocator with the given pool allocators.
     *
     * @param pool_allocators Vector of individual pool allocators (static or elastic)
     * @param pool_types Types of each pool (must match allocators)
     */
    MultiPoolChannelAllocator(
        std::vector<std::shared_ptr<FabricChannelAllocator>> pool_allocators,
        std::vector<FabricChannelPoolType> pool_types);

    /**
     * Emit compile-time arguments for all pools.
     *
     * Format:
     * 1. num_pools (uint32_t)
     * 2. pool_types[] (array of FabricChannelPoolType as uint32_t)
     * 3. For each pool: pool-specific CT args via pool->emit_ct_args()
     *
     * @param ct_args Vector to append compile-time arguments to
     * @param num_fwd_paths Number of forwarding paths
     * @param num_used_sender_channels Number of sender channels in use
     * @param num_used_receiver_channels Number of receiver channels in use
     */
    void emit_ct_args(
        std::vector<uint32_t>& ct_args,
        size_t num_fwd_paths,
        size_t num_used_sender_channels,
        size_t num_used_receiver_channels) const;

    /**
     * Get the number of pools managed by this allocator.
     */
    size_t get_num_pools() const { return pool_allocators_.size(); }

    /**
     * Get a specific pool allocator by index.
     *
     * @param pool_index Index of the pool
     * @return Shared pointer to the pool allocator
     */
    std::shared_ptr<FabricChannelAllocator> get_pool(size_t pool_index) const;

    /**
     * Get the type of a specific pool.
     *
     * @param pool_index Index of the pool
     * @return Type of the pool
     */
    FabricChannelPoolType get_pool_type(size_t pool_index) const;

    /**
     * Get a pool allocator cast to a specific type.
     * This is a helper for backward compatibility with code that expects specific allocator types.
     *
     * @tparam AllocatorType The type to cast the pool to
     * @param pool_index Index of the pool
     * @return Pointer to the pool allocator cast to the specified type, or nullptr if the cast fails
     */
    template <typename AllocatorType>
    AllocatorType* get_pool_as(size_t pool_index) const {
        auto pool = get_pool(pool_index);
        return dynamic_cast<AllocatorType*>(pool.get());
    }

private:
    std::vector<std::shared_ptr<FabricChannelAllocator>> pool_allocators_;
    std::vector<FabricChannelPoolType> pool_types_;
};

}  // namespace tt::tt_fabric
