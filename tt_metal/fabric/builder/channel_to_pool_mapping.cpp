// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "channel_to_pool_mapping.hpp"

namespace tt::tt_fabric {

static bool is_valid_pool_type(FabricChannelPoolType pool_type) {
    return pool_type == FabricChannelPoolType::STATIC || pool_type == FabricChannelPoolType::ELASTIC;
}

ChannelToPoolMapping::ChannelToPoolMapping(const FabricRouterRecipe& recipe) :
    sender_channel_to_pool_index_(recipe.get_sender_channel_to_pool_index()),
    receiver_channel_to_pool_index_(recipe.get_receiver_channel_to_pool_index()) {
    // Convert sender channel pool types to uint32_t
    sender_channel_to_pool_type_.reserve(sender_channel_to_pool_index_.size());
    for (size_t ch = 0; ch < sender_channel_to_pool_index_.size(); ++ch) {
        FabricChannelPoolType pool_type = recipe.get_sender_channel_pool_type(ch);
        TT_FATAL(is_valid_pool_type(pool_type), "Sender channel pool type must be STATIC or ELASTIC");
        sender_channel_to_pool_type_.push_back(pool_type);
    }

    // Convert receiver channel pool types to uint32_t
    receiver_channel_to_pool_type_.reserve(receiver_channel_to_pool_index_.size());
    for (size_t ch = 0; ch < receiver_channel_to_pool_index_.size(); ++ch) {
        FabricChannelPoolType pool_type = recipe.get_receiver_channel_pool_type(ch);
        TT_FATAL(is_valid_pool_type(pool_type), "Receiver channel pool type must be STATIC or ELASTIC");
        receiver_channel_to_pool_type_.push_back(pool_type);
    }
}

void ChannelToPoolMapping::emit_ct_args(std::vector<uint32_t>& ct_args) const {
    TT_FATAL(
        sender_channel_to_pool_index_.size() == sender_channel_to_pool_type_.size(),
        "Sender channel to pool index and type size mismatch");
    for (size_t i = 0; i < sender_channel_to_pool_index_.size(); ++i) {
        // log_info(tt::LogFabric, "Sender channel [{}] -> pool index: {}, type={}", i,
        // sender_channel_to_pool_index_.at(i), sender_channel_to_pool_type_.at(i));
        TT_FATAL(
            is_valid_pool_type(sender_channel_to_pool_type_.at(i)),
            "Sender channel pool type must be STATIC or ELASTIC");
    }
    TT_FATAL(
        receiver_channel_to_pool_index_.size() == receiver_channel_to_pool_type_.size(),
        "Receiver channel to pool index and type size mismatch");
    for (size_t i = 0; i < receiver_channel_to_pool_index_.size(); ++i) {
        // log_info(tt::LogFabric, "Receiver channel [{}] -> pool index: {}, type={}", i,
        // receiver_channel_to_pool_index_.at(i), receiver_channel_to_pool_type_.at(i));
        TT_FATAL(
            is_valid_pool_type(receiver_channel_to_pool_type_.at(i)),
            "Receiver channel pool type must be STATIC or ELASTIC");
    }
    // log_info(tt::LogFabric,
    //     "Emitting ChannelToPoolMapping CT args:"
    //     "\n  Sender channel to pool index: {}"
    //     "\n  Sender channel to pool type: {}"
    //     "\n  Receiver channel to pool index: {}"
    //     "\n  Receiver channel to pool type: {}",
    //     sender_channel_to_pool_index_,
    //     sender_channel_to_pool_type_,
    //     receiver_channel_to_pool_index_,
    //     receiver_channel_to_pool_type_
    // );

    TT_FATAL(
        sender_channel_to_pool_index_.size() == sender_channel_to_pool_type_.size(),
        "Sender channel to pool index and type size mismatch");
    // Emit sender channel to pool index mapping (step 5)
    ct_args.insert(ct_args.end(), sender_channel_to_pool_index_.begin(), sender_channel_to_pool_index_.end());

    // Emit sender channel to pool type mapping (step 6)
    std::transform(
        sender_channel_to_pool_type_.begin(),
        sender_channel_to_pool_type_.end(),
        std::back_inserter(ct_args),
        [](FabricChannelPoolType pool_type) { return static_cast<uint32_t>(pool_type); });

    TT_FATAL(
        receiver_channel_to_pool_index_.size() == receiver_channel_to_pool_type_.size(),
        "Receiver channel to pool index and type size mismatch");
    // Emit receiver channel to pool index mapping (step 7)
    ct_args.insert(ct_args.end(), receiver_channel_to_pool_index_.begin(), receiver_channel_to_pool_index_.end());

    // Emit receiver channel to pool type mapping (step 8)
    std::transform(
        receiver_channel_to_pool_type_.begin(),
        receiver_channel_to_pool_type_.end(),
        std::back_inserter(ct_args),
        [](FabricChannelPoolType pool_type) { return static_cast<uint32_t>(pool_type); });
}

}  // namespace tt::tt_fabric
