// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_recipe.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

FabricRouterRecipe::FabricRouterRecipe(
    std::vector<ChannelPoolDefinition> pool_definitions,
    std::vector<size_t> sender_channel_to_pool_index,
    std::vector<size_t> receiver_channel_to_pool_index,
    size_t num_sender_channels,
    size_t num_receiver_channels) :
    pool_definitions_(std::move(pool_definitions)),
    sender_channel_to_pool_index_(std::move(sender_channel_to_pool_index)),
    receiver_channel_to_pool_index_(std::move(receiver_channel_to_pool_index)) {
    TT_FATAL(
        sender_channel_to_pool_index_.size() == num_sender_channels,
        "Sender channel mapping size {} does not match num_sender_channels {}",
        sender_channel_to_pool_index_.size(),
        num_sender_channels);

    TT_FATAL(
        receiver_channel_to_pool_index_.size() == num_receiver_channels,
        "Receiver channel mapping size {} does not match num_receiver_channels {}",
        receiver_channel_to_pool_index_.size(),
        num_receiver_channels);

    // Build reverse mappings to track which channels map to which pools
    for (size_t ch = 0; ch < sender_channel_to_pool_index_.size(); ++ch) {
        size_t pool_idx = sender_channel_to_pool_index_[ch];
        TT_FATAL(pool_idx < pool_definitions_.size(), "Sender channel {} maps to invalid pool index {}", ch, pool_idx);
        pool_definitions_[pool_idx].mapped_sender_channels.push_back(ch);
    }

    for (size_t ch = 0; ch < receiver_channel_to_pool_index_.size(); ++ch) {
        size_t pool_idx = receiver_channel_to_pool_index_[ch];
        TT_FATAL(
            pool_idx < pool_definitions_.size(), "Receiver channel {} maps to invalid pool index {}", ch, pool_idx);
        pool_definitions_[pool_idx].mapped_receiver_channels.push_back(ch);
    }

    validate();
}

FabricRouterRecipe FabricRouterRecipe::create_default_single_static_pool_recipe(
    size_t num_sender_channels, size_t num_receiver_channels) {
    // Create a single static pool
    std::vector<ChannelPoolDefinition> pool_definitions;
    pool_definitions.emplace_back(FabricChannelPoolType::STATIC);

    // Map all channels to pool 0
    std::vector<size_t> sender_channel_to_pool_index(num_sender_channels, 0);
    std::vector<size_t> receiver_channel_to_pool_index(num_receiver_channels, 0);

    return FabricRouterRecipe(
        std::move(pool_definitions),
        std::move(sender_channel_to_pool_index),
        std::move(receiver_channel_to_pool_index),
        num_sender_channels,
        num_receiver_channels);
}

void FabricRouterRecipe::validate() const {
    TT_FATAL(!pool_definitions_.empty(), "Recipe must have at least one pool");
    TT_FATAL(!receiver_channel_to_pool_index_.empty(), "Recipe must have at least one receiver channel mapping");
    // We don't validate sender channels right now because we could be doing a channel mapping for remote
    // channels (over eth) and in those cases we don't necessarily have (or care about) the sender channels.

    // Validate that all pool indices in mappings are valid
    for (size_t ch = 0; ch < sender_channel_to_pool_index_.size(); ++ch) {
        size_t pool_idx = sender_channel_to_pool_index_[ch];
        TT_FATAL(
            pool_idx < pool_definitions_.size(),
            "Sender channel {} maps to out-of-range pool index {} (max {})",
            ch,
            pool_idx,
            pool_definitions_.size() - 1);
    }

    for (size_t ch = 0; ch < receiver_channel_to_pool_index_.size(); ++ch) {
        size_t pool_idx = receiver_channel_to_pool_index_[ch];
        TT_FATAL(
            pool_idx < pool_definitions_.size(),
            "Receiver channel {} maps to out-of-range pool index {} (max {})",
            ch,
            pool_idx,
            pool_definitions_.size() - 1);
    }

    // Note: We don't validate memory capacity here because that requires
    // knowledge of the actual allocator instance and available memory.
    // That validation will be done when allocators are constructed.
}

}  // namespace tt::tt_fabric
