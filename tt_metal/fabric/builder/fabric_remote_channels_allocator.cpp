// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_remote_channels_allocator.hpp"

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_router_recipe.hpp"  // for FabricChannelPoolType
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

FabricRemoteChannelsAllocator::FabricRemoteChannelsAllocator(
    const FabricStaticSizedChannelsAllocator& static_allocator) :
    FabricChannelAllocator(static_allocator.topology_, static_allocator.options_, static_allocator.memory_regions_) {
    // Extract remote receiver channel information from the static allocator for all VCs
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        this->num_used_receiver_channels_per_vc_[vc] = static_allocator.num_used_receiver_channels_per_vc[vc];
        for (size_t i = 0; i < static_allocator.num_used_receiver_channels_per_vc[vc]; i++) {
            this->remote_receiver_channels_base_address_[vc][i] =
                static_allocator.remote_receiver_channels_base_address[vc][i];
            this->remote_receiver_channels_num_buffers_[vc][i] =
                static_allocator.remote_receiver_channels_num_buffers[vc][i];
        }
    }
}

void FabricRemoteChannelsAllocator::emit_ct_args(std::vector<uint32_t>& ct_args) const {
    // This is now called by MultiPoolChannelAllocator, which handles num_pools and pool_type emission.
    // We only emit the pool data itself.

    // Emit pool data in StaticChannelPool format for all VCs sequentially (VC0 first, then VC1)
    // Format: for each receiver channel per VC: (base_address, num_buffers, remote_address, remote_num_buffers)
    // Note: remote_address and remote_num_buffers are unused but required for StaticChannelPool ct arg format
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (size_t i = 0; i < this->num_used_receiver_channels_per_vc_[vc]; ++i) {
            ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_base_address_[vc][i]));
            ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_num_buffers_[vc][i]));
            ct_args.push_back(0);  // remote_address (unused for remote pools)
            ct_args.push_back(0);  // remote_num_buffers (unused for remote pools)
        }
    }
}

size_t FabricRemoteChannelsAllocator::get_remote_receiver_channel_base_address(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < builder_config::num_max_receiver_channels,
        "Channel ID {} out of bounds for VC{} (max {})",
        channel_id,
        vc_id,
        builder_config::num_max_receiver_channels - 1);
    TT_FATAL(
        channel_id < this->num_used_receiver_channels_per_vc_[vc_id],
        "Channel ID {} is not used in VC{} (only {} channels used)",
        channel_id,
        vc_id,
        this->num_used_receiver_channels_per_vc_[vc_id]);
    return this->remote_receiver_channels_base_address_[vc_id][channel_id];
}

size_t FabricRemoteChannelsAllocator::get_remote_receiver_channel_num_buffers(size_t vc_id, size_t channel_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(
        channel_id < builder_config::num_max_receiver_channels,
        "Channel ID {} out of bounds for VC{} (max {})",
        channel_id,
        vc_id,
        builder_config::num_max_receiver_channels - 1);
    TT_FATAL(
        channel_id < this->num_used_receiver_channels_per_vc_[vc_id],
        "Channel ID {} is not used in VC{} (only {} channels used)",
        channel_id,
        vc_id,
        this->num_used_receiver_channels_per_vc_[vc_id]);
    return this->remote_receiver_channels_num_buffers_[vc_id][channel_id];
}

}  // namespace tt::tt_fabric
