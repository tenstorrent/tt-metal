// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_router_recipe.hpp"
#include <cstdint>
#include <vector>

namespace tt::tt_fabric {

/**
 * Encapsulates the channel-to-pool mappings for both sender and receiver channels.
 *
 * This class is responsible for:
 * 1. Storing the mapping of channels to pool indices
 * 2. Storing the mapping of channels to pool types
 * 3. Emitting these mappings as compile-time arguments
 *
 * The compile-time argument schema (steps 5-8 from the fabric router CT args schema):
 * 5) Sender channel to pool index mapping
 * 6) Sender channel to pool type mapping
 * 7) Receiver channel to pool index mapping
 * 8) Receiver channel to pool type mapping
 */
class ChannelToPoolMapping {
public:
    /**
     * Construct from a FabricRouterRecipe.
     *
     * @param recipe The recipe defining the pool configuration and mappings
     */
    explicit ChannelToPoolMapping(const FabricRouterRecipe& recipe);

    /**
     * Emit the channel-to-pool mappings as compile-time arguments.
     * This appends the mapping data to the provided vector.
     *
     * @param ct_args Vector to append compile-time arguments to
     */
    void emit_ct_args(std::vector<uint32_t>& ct_args) const;

    // Accessors for validation and debugging
    const std::vector<size_t>& get_sender_channel_to_pool_index() const { return sender_channel_to_pool_index_; }
    const std::vector<uint32_t>& get_sender_channel_to_pool_type() const { return sender_channel_to_pool_type_; }
    const std::vector<size_t>& get_receiver_channel_to_pool_index() const { return receiver_channel_to_pool_index_; }
    const std::vector<uint32_t>& get_receiver_channel_to_pool_type() const { return receiver_channel_to_pool_type_; }

private:
    // Sender channel mappings
    std::vector<size_t> sender_channel_to_pool_index_;
    std::vector<uint32_t> sender_channel_to_pool_type_;  // FabricChannelPoolType as uint32_t

    // Receiver channel mappings
    std::vector<size_t> receiver_channel_to_pool_index_;
    std::vector<uint32_t> receiver_channel_to_pool_type_;  // FabricChannelPoolType as uint32_t
};

}  // namespace tt::tt_fabric
