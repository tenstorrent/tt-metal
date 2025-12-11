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
 * Supports multiple Virtual Channels (VCs) where each VC has its own set of remote receiver channels.
 *
 * The purpose is to provide a separate pool of channel information specifically for initializing
 * remote_receiver_channels in the fabric router, ensuring the remote channel info is correctly
 * sourced from the remote core's channel configuration.
 */
class FabricRemoteChannelsAllocator : public FabricChannelAllocator {
public:
    /**
     * Construct a remote channels allocator from a static channels allocator.
     * Extracts remote receiver channel base addresses and buffer counts from the static allocator
     * for all VCs.
     *
     * @param static_allocator The static allocator containing remote channel information
     */
    explicit FabricRemoteChannelsAllocator(
        const FabricStaticSizedChannelsAllocator& static_allocator);

    /**
     * Emit compile-time arguments for remote receiver channels.
     * Emits data for all VCs sequentially (VC0 channels first, then VC1 channels).
     *
     * Format (for each remote receiver channel per VC):
     *   - base_address (uint32_t)
     *   - num_buffers (uint32_t)
     *
     * @param ct_args Vector to append compile-time arguments to
     */
    void emit_ct_args(std::vector<uint32_t>& ct_args) const override;

    /**
     * Get the base address for a specific remote receiver channel in a specific VC.
     * @param vc_id Virtual Channel ID (0 for VC0, 1 for VC1)
     * @param channel_id Channel ID within the VC
     * @return Base address
     */
    size_t get_remote_receiver_channel_base_address(size_t vc_id, size_t channel_id) const;

    /**
     * Get the number of buffers for a specific remote receiver channel in a specific VC.
     * @param vc_id Virtual Channel ID (0 for VC0, 1 for VC1)
     * @param channel_id Channel ID within the VC
     * @return Number of buffers
     */
    size_t get_remote_receiver_channel_num_buffers(size_t vc_id, size_t channel_id) const;

    /**
     * Legacy getter for VC0 only (for backward compatibility).
     * @param channel_id Channel ID
     * @return Base address
     */
    size_t get_remote_receiver_channel_base_address(size_t channel_id) const {
        return get_remote_receiver_channel_base_address(0, channel_id);
    }

    /**
     * Legacy getter for VC0 only (for backward compatibility).
     * @param channel_id Channel ID
     * @return Number of buffers
     */
    size_t get_remote_receiver_channel_num_buffers(size_t channel_id) const {
        return get_remote_receiver_channel_num_buffers(0, channel_id);
    }

    /**
     * Get the number of used receiver channels for a specific VC.
     * @param vc_id Virtual Channel ID (0 for VC0, 1 for VC1)
     * @return Number of used receiver channels
     */
    size_t get_num_receiver_channels(size_t vc_id) const {
        TT_FATAL(
            vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
        return num_used_receiver_channels_per_vc_[vc_id];
    }

    /**
     * Legacy getter: Get total number of used receiver channels across all VCs.
     * @return Total number of used receiver channels
     */
    size_t get_num_receiver_channels() const {
        return num_used_receiver_channels_per_vc_[0] + num_used_receiver_channels_per_vc_[1];
    }

    /**
     * Override virtual print method from base class
     */
    void print(std::ostream& os) const override;

private:
    // Per-VC remote receiver channel data (VC × channel)
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        remote_receiver_channels_base_address_ = {};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        remote_receiver_channels_num_buffers_ = {};
    std::array<size_t, builder_config::MAX_NUM_VCS> num_used_receiver_channels_per_vc_ = {0, 0};
};

inline void FabricRemoteChannelsAllocator::print(std::ostream& os) const {
    os << "FabricRemoteChannelsAllocator {\n";
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        os << "  VC" << vc << " num_used_receiver_channels: " << num_used_receiver_channels_per_vc_[vc] << "\n";
        if (num_used_receiver_channels_per_vc_[vc] > 0) {
            os << "  VC" << vc << " Remote Receiver Channels:\n";
            for (size_t i = 0; i < num_used_receiver_channels_per_vc_[vc]; ++i) {
                os << "    VC" << vc << " Channel " << i << ":\n";
                os << "      base_address: 0x" << std::hex << remote_receiver_channels_base_address_[vc][i] << std::dec
                   << "\n";
                os << "      num_buffers: " << remote_receiver_channels_num_buffers_[vc][i] << "\n";
            }
        }
    }
    os << "}";
}

}  // namespace tt::tt_fabric
