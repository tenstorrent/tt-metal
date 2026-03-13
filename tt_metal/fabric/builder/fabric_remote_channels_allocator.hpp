// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_channel_allocator.hpp"
#include "fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"

#include <array>
#include <cstddef>
#include <sstream>
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
     * Emit the complete ChannelAllocations CT arg block for remote receiver channels.
     * Format: [tag] [num_entries] [per-entry data...] [receiver_to_entry_idx...]
     */
    void emit_channel_allocations_ct_args(std::vector<uint32_t>& ct_args, size_t num_used_receiver_channels) const;

    /**
     * Get the base address for the remote receiver channel in a specific VC.
     * Each VC has at most one remote receiver channel.
     * @param vc_id Virtual Channel ID (0 for VC0, 1 for VC1)
     * @return Base address
     */
    size_t get_remote_receiver_channel_base_address(size_t vc_id) const;

    /**
     * Get the number of buffers for the remote receiver channel in a specific VC.
     * Each VC has at most one remote receiver channel.
     * @param vc_id Virtual Channel ID (0 for VC0, 1 for VC1)
     * @return Number of buffers
     */
    size_t get_remote_receiver_channel_num_buffers(size_t vc_id) const;

    /**
     * Get whether a specific VC has an active receiver channel.
     * @param vc_id Virtual Channel ID (0 for VC0, 1 for VC1)
     * @return true if the VC has an active receiver channel
     */
    bool is_receiver_channel_active(size_t vc_id) const {
        TT_FATAL(
            vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
        return is_receiver_channel_active_per_vc_[vc_id];
    }

    /**
     * Get total number of active receiver channels across all VCs.
     * @return Total number of active receiver channels (0, 1, or 2)
     */
    size_t get_num_receiver_channels() const {
        return static_cast<size_t>(is_receiver_channel_active_per_vc_[0]) +
               static_cast<size_t>(is_receiver_channel_active_per_vc_[1]);
    }

    /**
     * Override virtual print method from base class
     */
    void print(std::ostream& os) const override;

private:
    // Per-VC remote receiver channel data (one channel per VC)
    std::array<size_t, builder_config::MAX_NUM_VCS> remote_receiver_channel_base_address_ = {};
    std::array<size_t, builder_config::MAX_NUM_VCS> remote_receiver_channel_num_buffers_ = {};
    std::array<bool, builder_config::MAX_NUM_VCS> is_receiver_channel_active_per_vc_ = {false, false};
};

inline void FabricRemoteChannelsAllocator::print(std::ostream& os) const {
    os << "FabricRemoteChannelsAllocator {\n";
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        os << "  VC" << vc << " is_receiver_channel_active: " << is_receiver_channel_active_per_vc_[vc] << "\n";
        if (is_receiver_channel_active_per_vc_[vc]) {
            os << "  VC" << vc << " Remote Receiver Channel:\n";
            os << "      base_address: 0x" << std::hex << remote_receiver_channel_base_address_[vc] << std::dec << "\n";
            os << "      num_buffers: " << remote_receiver_channel_num_buffers_[vc] << "\n";
        }
    }
    os << "}";
}

}  // namespace tt::tt_fabric

// fmt formatter specialization for FabricRemoteChannelsAllocator
template <>
struct fmt::formatter<tt::tt_fabric::FabricRemoteChannelsAllocator> : fmt::formatter<std::string> {
    auto format(const tt::tt_fabric::FabricRemoteChannelsAllocator& allocator, fmt::format_context& ctx) const {
        std::ostringstream stream;
        stream << allocator;
        return formatter<std::string>::format(stream.str(), ctx);
    }
};
