// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_channel_allocator.hpp"

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/mesh_channel_spec.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"

#include <vector>
#include <ostream>

namespace tt::tt_fabric {

/**
 * Static-sized channels allocator implementation.
 * The `FabricStaticSizedChannelsAllocator` allocates memory for statically sized sender(outbound)
 * and receiver (inbound) fabric router channels. The entire set of channels do not need to be
 * contiguous in memory with each other. However, each individual channel must be contiguous in
 * memory.
 *
 * Each channel is a sequence of 1 or more buffer slots (i.e. packet slots)
 *
 * Supports multiple Virtual Channels (VCs) where each VC has its own set of sender/receiver channels.
 */
class FabricStaticSizedChannelsAllocator : public FabricChannelAllocator {
public:
    FabricStaticSizedChannelsAllocator(
        Topology topology,
        const MeshChannelSpec& spec,
        const FabricEriscDatamoverOptions& options,
        size_t channel_buffer_size_bytes,
        size_t available_channel_buffering_space,
        const std::vector<MemoryRegion>& memory_regions);

    void emit_ct_args(std::vector<uint32_t>& ct_args) const override;

    /**
     * Get the number of slots for a specific sender channel in a VC.
     * @param vc_id Virtual Channel ID (0 or 1)
     * @param channel_id Channel ID within the VC
     * @return Number of slots
     */
    size_t get_sender_channel_number_of_slots(size_t vc_id, size_t channel_id) const;

    /**
     * Get the base address for a specific sender channel in a VC.
     * @param vc_id Virtual Channel ID (0 or 1)
     * @param channel_id Channel ID within the VC
     * @return Base address
     */
    size_t get_sender_channel_base_address(size_t vc_id, size_t channel_id) const;

    /**
     * Get the number of slots for a specific receiver channel in a VC.
     * @param vc_id Virtual Channel ID (0 or 1)
     * @param channel_id Channel ID within the VC
     * @return Number of slots
     */
    size_t get_receiver_channel_number_of_slots(size_t vc_id, size_t channel_id) const;

    /**
     * Get the base address for a specific receiver channel in a VC.
     * @param vc_id Virtual Channel ID (0 or 1)
     * @param channel_id Channel ID within the VC
     * @return Base address
     */
    size_t get_receiver_channel_base_address(size_t vc_id, size_t channel_id) const;

    size_t get_num_sender_channels(size_t vc_id) const {
        TT_FATAL(
            vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
        return num_used_sender_channels_per_vc[vc_id];
    }
    size_t get_num_receiver_channels(size_t vc_id) const {
        TT_FATAL(
            vc_id < builder_config::MAX_NUM_VCS, "VC ID {} out of bounds (max {})", vc_id, builder_config::MAX_NUM_VCS);
        return num_used_receiver_channels_per_vc[vc_id];
    }

    /**
     * Get the channel spec for this allocator
     */
    const MeshChannelSpec& get_spec() const { return spec_; }

    // Legacy getters (assume VC0 for backward compatibility)
    size_t get_sender_channel_number_of_slots(size_t channel_id) const {
        return get_sender_channel_number_of_slots(0, channel_id);
    }
    size_t get_sender_channel_base_address(size_t channel_id) const {
        return get_sender_channel_base_address(0, channel_id);
    }
    size_t get_receiver_channel_number_of_slots(size_t channel_id) const {
        return get_receiver_channel_number_of_slots(0, channel_id);
    }
    size_t get_receiver_channel_base_address(size_t channel_id) const {
        return get_receiver_channel_base_address(0, channel_id);
    }
    // For total counts: return sum across all VCs
    size_t get_num_sender_channels() const {
        return num_used_sender_channels_per_vc[0] + num_used_sender_channels_per_vc[1];
    }
    size_t get_num_receiver_channels() const {
        return num_used_receiver_channels_per_vc[0] + num_used_receiver_channels_per_vc[1];
    }

    // Override virtual print method from base class
    void print(std::ostream& os) const override;

private:
    friend class FabricRemoteChannelsAllocator;
    /*
     * Helper function that decides the number of buffer slots for each channel per VC.
     */
    void configure_buffer_slots_helper(
        Topology topology,
        const tt::tt_fabric::FabricEriscDatamoverOptions& options,
        std::array<
            std::array<size_t, tt::tt_fabric::builder_config::num_max_sender_channels>,
            builder_config::MAX_NUM_VCS>& num_sender_buffer_slots_per_vc,
        std::array<
            std::array<size_t, tt::tt_fabric::builder_config::num_max_sender_channels>,
            builder_config::MAX_NUM_VCS>& num_remote_sender_buffer_slots_per_vc,
        std::array<
            std::array<size_t, tt::tt_fabric::builder_config::num_max_receiver_channels>,
            builder_config::MAX_NUM_VCS>& num_receiver_buffer_slots_per_vc,
        std::array<
            std::array<size_t, tt::tt_fabric::builder_config::num_max_receiver_channels>,
            builder_config::MAX_NUM_VCS>& num_remote_receiver_buffer_slots_per_vc);

    // Configuration parameters
    MeshChannelSpec spec_;
    std::array<size_t, builder_config::MAX_NUM_VCS> num_used_sender_channels_per_vc = {0, 0};
    std::array<size_t, builder_config::MAX_NUM_VCS> num_used_receiver_channels_per_vc = {0, 0};
    size_t channel_buffer_size_bytes = 0;
    size_t available_channel_buffering_space = 0;
    size_t max_l1_loading_size = 0;
    size_t buffer_region_start = 0;

    // Tensix configuration channel counts
    static constexpr size_t num_sender_channels_with_tensix_config =
        builder_config::num_sender_channels_with_tensix_config;

    // Channel size and buffer information per VC (VC × channel)
    std::array<std::array<std::size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        sender_channels_size_bytes = {};
    std::array<std::array<std::size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        receiver_channels_size_bytes = {};
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        sender_channels_num_buffers = {};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        receiver_channels_num_buffers = {};

    // Remote channels sizes, used to calculate the remote buffer addresses.
    std::array<std::array<std::size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        remote_sender_channels_size_bytes = {};
    std::array<std::array<std::size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        remote_receiver_channels_size_bytes = {};
    // Remote recv channels number of buffers, use by the local sender channel to check free slots.
    std::array<std::array<std::size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        remote_sender_channels_num_buffers = {};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        remote_receiver_channels_num_buffers = {};

    // Base addresses per VC
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        sender_channels_base_address = {};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        receiver_channels_base_address = {};
    // the base addr per remote channel, used by local channels.
    std::array<std::array<size_t, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS>
        remote_sender_channels_base_address = {};
    std::array<std::array<size_t, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS>
        remote_receiver_channels_base_address = {};
};

// Implementation of virtual print method
inline void FabricStaticSizedChannelsAllocator::print(std::ostream& os) const {
    os << "FabricStaticSizedChannelsAllocator {\n";

    // Configuration parameters
    os << "  Configuration:\n";
    os << "    num_used_sender_channels_per_vc: [" << num_used_sender_channels_per_vc[0] << ", "
       << num_used_sender_channels_per_vc[1] << "]\n";
    os << "    num_used_receiver_channels_per_vc: [" << num_used_receiver_channels_per_vc[0] << ", "
       << num_used_receiver_channels_per_vc[1] << "]\n";
    os << "    channel_buffer_size_bytes: " << channel_buffer_size_bytes << " B\n";
    os << "    available_channel_buffering_space: " << available_channel_buffering_space << " B\n";
    os << "    buffer_region_start: 0x" << std::hex << buffer_region_start << std::dec << "\n";
    os << "    max_l1_loading_size: 0x" << std::hex << max_l1_loading_size << std::dec << "\n";

    // Per-VC channel information
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        os << "  VC" << vc << ":\n";

        // Sender channels for this VC
        if (num_used_sender_channels_per_vc[vc] > 0) {
            os << "    Sender Channels:\n";
            for (size_t i = 0; i < num_used_sender_channels_per_vc[vc]; ++i) {
                os << "      Channel " << i << ":\n";
                os << "        base_address: 0x" << std::hex << sender_channels_base_address[vc][i] << std::dec << "\n";
                os << "        num_buffers: " << sender_channels_num_buffers[vc][i] << "\n";
                os << "        size_bytes: " << sender_channels_size_bytes[vc][i] << " B\n";
                if (remote_sender_channels_num_buffers[vc][i] > 0) {
                    os << "        remote_base_address: 0x" << std::hex << remote_sender_channels_base_address[vc][i]
                       << std::dec << "\n";
                    os << "        remote_num_buffers: " << remote_sender_channels_num_buffers[vc][i] << "\n";
                    os << "        remote_size_bytes: " << remote_sender_channels_size_bytes[vc][i] << " B\n";
                }
            }
        }

        // Receiver channels for this VC
        if (num_used_receiver_channels_per_vc[vc] > 0) {
            os << "    Receiver Channels:\n";
            for (size_t i = 0; i < num_used_receiver_channels_per_vc[vc]; ++i) {
                os << "      Channel " << i << ":\n";
                os << "        base_address: 0x" << std::hex << receiver_channels_base_address[vc][i] << std::dec << "\n";
                os << "        num_buffers: " << receiver_channels_num_buffers[vc][i] << "\n";
                os << "        size_bytes: " << receiver_channels_size_bytes[vc][i] << " B\n";
                if (remote_receiver_channels_num_buffers[vc][i] > 0) {
                    os << "        remote_base_address: 0x" << std::hex << remote_receiver_channels_base_address[vc][i]
                       << std::dec << "\n";
                    os << "        remote_num_buffers: " << remote_receiver_channels_num_buffers[vc][i] << "\n";
                    os << "        remote_size_bytes: " << remote_receiver_channels_size_bytes[vc][i] << " B\n";
                }
            }
        }
    }

    os << "}";
}

}  // namespace tt::tt_fabric
