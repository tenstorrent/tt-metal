// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_remote_channels_allocator.hpp"

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_router_recipe.hpp"  // for FabricChannelPoolType
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

FabricRemoteChannelsAllocator::FabricRemoteChannelsAllocator(
    const FabricStaticSizedChannelsAllocator& static_allocator)
    : FabricChannelAllocator(
          static_allocator.topology_,
          static_allocator.options_,
          static_allocator.memory_regions_),
      num_used_receiver_channels_(static_allocator.get_num_receiver_channels()) {
    // Extract remote receiver channel information from the static allocator
    // Using friend class access to private members
    for (size_t i = 0; i < builder_config::num_receiver_channels; i++) {
        this->remote_receiver_channels_base_address_[i] = static_allocator.remote_receiver_channels_base_address[i];
        this->remote_receiver_channels_num_buffers_[i] = static_allocator.remote_receiver_channels_num_buffers[i];
    }
}

void FabricRemoteChannelsAllocator::emit_ct_args(
    std::vector<uint32_t>& ct_args,
    size_t num_fwd_paths,
    size_t num_used_sender_channels,
    size_t num_used_receiver_channels) const {
    // This is now called by MultiPoolChannelAllocator, which handles num_pools and pool_type emission.
    // We only emit the pool data itself.

    // Emit pool data in StaticChannelPool format
    // Format: for each receiver channel: (base_address, num_buffers, remote_address, remote_num_buffers)
    // Note: remote_address and remote_num_buffers are unused but required for StaticChannelPool format
    for (size_t i = 0; i < this->num_used_receiver_channels_; ++i) {
        ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_base_address_[i]));
        ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_num_buffers_[i]));
        ct_args.push_back(0);  // remote_address (unused for remote pools)
        ct_args.push_back(0);  // remote_num_buffers (unused for remote pools)
    }
}

size_t FabricRemoteChannelsAllocator::get_remote_receiver_channel_base_address(size_t channel_id) const {
    TT_FATAL(
        channel_id < builder_config::num_receiver_channels,
        "Channel ID {} out of bounds (max {})",
        channel_id,
        builder_config::num_receiver_channels - 1);
    return this->remote_receiver_channels_base_address_[channel_id];
}

size_t FabricRemoteChannelsAllocator::get_remote_receiver_channel_num_buffers(size_t channel_id) const {
    TT_FATAL(
        channel_id < builder_config::num_receiver_channels,
        "Channel ID {} out of bounds (max {})",
        channel_id,
        builder_config::num_receiver_channels - 1);
    return this->remote_receiver_channels_num_buffers_[channel_id];
}

}  // namespace tt::tt_fabric
