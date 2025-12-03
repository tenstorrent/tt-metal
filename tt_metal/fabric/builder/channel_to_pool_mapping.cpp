// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "channel_to_pool_mapping.hpp"

namespace tt::tt_fabric {

ChannelToPoolMapping::ChannelToPoolMapping(const FabricRouterRecipe& recipe) :
    sender_channel_to_pool_index_(recipe.get_sender_channel_to_pool_index()),
    receiver_channel_to_pool_index_(recipe.get_receiver_channel_to_pool_index()) {
    // Convert sender channel pool types to uint32_t
    sender_channel_to_pool_type_.reserve(sender_channel_to_pool_index_.size());
    for (size_t ch = 0; ch < sender_channel_to_pool_index_.size(); ++ch) {
        FabricChannelPoolType pool_type = recipe.get_sender_channel_pool_type(ch);
        sender_channel_to_pool_type_.push_back(static_cast<uint32_t>(pool_type));
    }

    // Convert receiver channel pool types to uint32_t
    receiver_channel_to_pool_type_.reserve(receiver_channel_to_pool_index_.size());
    for (size_t ch = 0; ch < receiver_channel_to_pool_index_.size(); ++ch) {
        FabricChannelPoolType pool_type = recipe.get_receiver_channel_pool_type(ch);
        receiver_channel_to_pool_type_.push_back(static_cast<uint32_t>(pool_type));
    }
}

void ChannelToPoolMapping::emit_ct_args(std::vector<uint32_t>& ct_args) const {
    // Emit sender channel to pool index mapping (step 5)
    for (size_t pool_idx : sender_channel_to_pool_index_) {
        ct_args.push_back(static_cast<uint32_t>(pool_idx));
    }

    // Emit sender channel to pool type mapping (step 6)
    ct_args.insert(ct_args.end(), sender_channel_to_pool_type_.begin(), sender_channel_to_pool_type_.end());

    // Emit receiver channel to pool index mapping (step 7)
    for (size_t pool_idx : receiver_channel_to_pool_index_) {
        ct_args.push_back(static_cast<uint32_t>(pool_idx));
    }

    // Emit receiver channel to pool type mapping (step 8)
    ct_args.insert(ct_args.end(), receiver_channel_to_pool_type_.begin(), receiver_channel_to_pool_type_.end());
}

}  // namespace tt::tt_fabric
