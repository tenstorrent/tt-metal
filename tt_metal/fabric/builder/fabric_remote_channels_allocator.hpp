// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_channel_allocator.hpp"
#include "fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace tt::tt_fabric {

/**
 * Remote channels allocator for tracking channel information about the remote ethernet core.
 * 
 * This allocator extracts remote receiver channel information from a FabricStaticSizedChannelsAllocator
 * and manages it separately. It emits compile-time arguments in a similar structure to the static
 * allocator but only for remote receiver channels.
 * 
 * The purpose is to provide a separate pool of channel information specifically for initializing
 * remote_receiver_channels in the fabric router, ensuring the remote channel info is correctly
 * sourced from the remote core's channel configuration.
 */
class FabricRemoteChannelsAllocator : public FabricChannelAllocator {
public:
    /**
     * Construct a remote channels allocator from a static channels allocator.
     * Extracts remote receiver channel base addresses and buffer counts from the static allocator.
     * 
     * @param static_allocator The static allocator containing remote channel information
     */
    explicit FabricRemoteChannelsAllocator(
        const FabricStaticSizedChannelsAllocator& static_allocator);

    /**
     * Emit compile-time arguments for remote receiver channels.
     * 
     * Format (for each remote receiver channel):
     *   - base_address (uint32_t)
     *   - num_buffers (uint32_t)
     * 
     * @param ct_args Vector to append compile-time arguments to
     * @param num_fwd_paths Number of forwarding paths (unused, for interface compatibility)
     * @param num_used_sender_channels Number of sender channels (unused, for interface compatibility)
     * @param num_used_receiver_channels Number of receiver channels to emit
     */
    void emit_ct_args(
        std::vector<uint32_t>& ct_args,
        size_t num_fwd_paths,
        size_t num_used_sender_channels,
        size_t num_used_receiver_channels) const override;

    /**
     * Get the base address for a specific remote receiver channel.
     * @param channel_id Channel ID
     * @return Base address
     */
    size_t get_remote_receiver_channel_base_address(size_t channel_id) const;

    /**
     * Get the number of buffers for a specific remote receiver channel.
     * @param channel_id Channel ID
     * @return Number of buffers
     */
    size_t get_remote_receiver_channel_num_buffers(size_t channel_id) const;

    /**
     * Get the number of used receiver channels.
     * @return Number of used receiver channels
     */
    size_t get_num_receiver_channels() const { return num_used_receiver_channels_; }

private:
    std::array<size_t, builder_config::num_receiver_channels> remote_receiver_channels_base_address_ = {};
    std::array<size_t, builder_config::num_receiver_channels> remote_receiver_channels_num_buffers_ = {};
    size_t num_used_receiver_channels_ = 0;
};

}  // namespace tt::tt_fabric

