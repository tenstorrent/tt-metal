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
        this->is_receiver_channel_active_per_vc_[vc] = static_allocator.is_receiver_channel_active_per_vc[vc];
        if (static_allocator.is_receiver_channel_active_per_vc[vc]) {
            this->remote_receiver_channel_base_address_[vc] = static_allocator.remote_receiver_channel_base_address[vc];
            this->remote_receiver_channel_num_buffers_[vc] = static_allocator.remote_receiver_channel_num_buffers[vc];
        }
    }
}

void FabricRemoteChannelsAllocator::emit_ct_args(std::vector<uint32_t>& ct_args) const {
    // Emit per-entry data in ChannelBufferEntry format for all VCs sequentially (VC0 first, then VC1)
    // Format: for the active receiver channel per VC: (base_address, num_buffers, remote_address, remote_num_buffers)
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        if (this->is_receiver_channel_active_per_vc_[vc]) {
            ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channel_base_address_[vc]));
            ct_args.push_back(static_cast<uint32_t>(this->remote_receiver_channel_num_buffers_[vc]));
            ct_args.push_back(0);  // remote_address (unused for remote pools)
            ct_args.push_back(0);  // remote_num_buffers (unused for remote pools)
        }
    }
}

size_t FabricRemoteChannelsAllocator::get_remote_receiver_channel_base_address(size_t vc_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(this->is_receiver_channel_active_per_vc_[vc_id], "VC{} has no active receiver channel", vc_id);
    return this->remote_receiver_channel_base_address_[vc_id];
}

size_t FabricRemoteChannelsAllocator::get_remote_receiver_channel_num_buffers(size_t vc_id) const {
    TT_FATAL(
        vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
    TT_FATAL(this->is_receiver_channel_active_per_vc_[vc_id], "VC{} has no active receiver channel", vc_id);
    return this->remote_receiver_channel_num_buffers_[vc_id];
}

void FabricRemoteChannelsAllocator::emit_channel_allocations_ct_args(
    std::vector<uint32_t>& ct_args,
    const std::array<bool, builder_config::MAX_NUM_VCS>& is_receiver_channel_active_per_vc) const {
    // Tag
    ct_args.push_back(0xabcd1234);

    // num_entries = total receiver channels
    size_t num_entries = get_num_receiver_channels();
    ct_args.push_back(static_cast<uint32_t>(num_entries));

    // Per-entry data (reuse existing emit_ct_args)
    emit_ct_args(ct_args);

    // No sender channels for remote allocator (0 sender entries)
    // Receiver channel-to-entry index: emit sequential indices only for active VCs
    size_t entry_index = 0;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        if (is_receiver_channel_active_per_vc[vc]) {
            ct_args.push_back(static_cast<uint32_t>(entry_index));
            ++entry_index;
        }
    }
}

}  // namespace tt::tt_fabric
