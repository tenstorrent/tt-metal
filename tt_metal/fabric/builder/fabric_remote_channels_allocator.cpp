// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_remote_channels_allocator.hpp"

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
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
    // Emit per-entry data in ChannelBufferEntry format for all VCs sequentially (VC0 first, then VC1)
    // Format: for each receiver channel per VC: (base_address, num_buffers, remote_address, remote_num_buffers)
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        if (this->num_used_receiver_channels_per_vc_[vc] > 0) {
            ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_base_address_[vc][0]));
            ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channels_num_buffers_[vc][0]));
            ct_args.push_back(0);  // remote_address (unused for remote pools)
            ct_args.push_back(0);  // remote_num_buffers (unused for remote pools)
        } else {
            ct_args.push_back(0);  // base_address (inactive VC)
            ct_args.push_back(0);  // num_buffers (inactive VC)
            ct_args.push_back(0);  // remote_address (inactive VC)
            ct_args.push_back(0);  // remote_num_buffers (inactive VC)
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

void FabricRemoteChannelsAllocator::emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args, size_t num_used_receiver_channels) const {
    // Tag
    ct_args.push_back(0xabcd1234);

    // num_entries = total receiver channels
    size_t num_entries = get_num_receiver_channels();
    ct_args.push_back(static_cast<uint32_t>(num_entries));

    // Per-entry data (reuse existing emit_ct_args)
    emit_ct_args(ct_args);

    // No sender channels for remote allocator (0 sender entries)
    // Receiver channel-to-entry index (identity mapping)
    for (size_t i = 0; i < num_used_receiver_channels; ++i) {
        ct_args.push_back(static_cast<uint32_t>(i));
    }
}

}  // namespace tt::tt_fabric
