// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_channel_allocator.hpp"

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp"
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"

#include <vector>

namespace tt::tt_fabric {

/**
 * Static-sized channels allocator implementation.
 * The `FabricStaticSizedChannelsAllocator` allocates memory for statically sized sender(outbound)
 * and receiver (inbound) fabric router channels. The entire set of channels do not need to be
 * contiguous in memory with each other. However, each individual channel must be contiguous in
 * memory.
 *
 * Each channel is a sequence of 1 or more buffer slots (i.e. packet slots)
 */
class FabricStaticSizedChannelsAllocator : public FabricChannelAllocator {
public:
    FabricStaticSizedChannelsAllocator(
        tt::tt_fabric::Topology topology,
        const FabricEriscDatamoverOptions& options,
        size_t num_used_sender_channels,
        size_t num_used_receiver_channels,
        size_t channel_buffer_size_bytes,
        size_t available_channel_buffering_space,
        const std::vector<MemoryRegion>& memory_regions);

    void emit_ct_args(std::vector<uint32_t>& ct_args, size_t num_fwd_paths, size_t num_used_sender_channels, size_t num_used_receiver_channels) const override;

    /**
     * Get the number of slots for a specific sender channel.
     * @param channel_id Channel ID
     * @return Number of slots
     */
    size_t get_sender_channel_number_of_slots(size_t channel_id) const;

    /**
     * Get the base address for a specific sender channel.
     * @param channel_id Channel ID
     * @return Base address
     */
    size_t get_sender_channel_base_address(size_t channel_id) const;

    /**
     * Get the number of slots for a specific receiver channel.
     * @param channel_id Channel ID
     * @return Number of slots
     */
    size_t get_receiver_channel_number_of_slots(size_t channel_id) const;

    /**
     * Get the base address for a specific receiver channel.
     * @param channel_id Channel ID
     * @return Base address
     */
    size_t get_receiver_channel_base_address(size_t channel_id) const;

    size_t get_num_sender_channels() const { return num_used_sender_channels; }
    size_t get_num_receiver_channels() const { return num_used_receiver_channels; }

private:
    friend class FabricRemoteChannelsAllocator;
    /*
     * Helper function that decides the number of buffer slots for each channel.
    */
    void configure_buffer_slots_helper(
        tt::tt_fabric::Topology topology,
        const tt::tt_fabric::FabricEriscDatamoverOptions& options,
        std::array<size_t, tt::tt_fabric::builder_config::num_sender_channels>& num_sender_buffer_slots,
        std::array<size_t, tt::tt_fabric::builder_config::num_sender_channels>& num_remote_sender_buffer_slots,
        std::array<size_t, tt::tt_fabric::builder_config::num_receiver_channels>& num_receiver_buffer_slots,
        std::array<size_t, tt::tt_fabric::builder_config::num_receiver_channels>& num_remote_receiver_buffer_slots,
        tt::tt_fabric::eth_chan_directions direction);

    // Configuration parameters
    size_t num_used_sender_channels = 0;
    size_t num_used_receiver_channels = 0;
    size_t channel_buffer_size_bytes = 0;
    size_t available_channel_buffering_space = 0;
    size_t max_l1_loading_size = 0;
    size_t buffer_region_start = 0;

    // Tensix configuration channel counts
    static constexpr size_t num_sender_channels_with_tensix_config =
        builder_config::num_sender_channels_with_tensix_config;
    static constexpr size_t num_sender_channels_with_tensix_config_deadlock_avoidance =
        builder_config::num_sender_channels_with_tensix_config_deadlock_avoidance;

    // Dateline channel skip indices - from FabricEriscDatamoverConfig
    static constexpr size_t dateline_sender_channel_skip_idx = 2;
    static constexpr size_t dateline_sender_channel_skip_idx_2d = 4;
    static constexpr size_t dateline_receiver_channel_skip_idx = 0;
    static constexpr size_t dateline_upstream_sender_channel_skip_idx = 1;
    static constexpr size_t dateline_upstream_receiver_channel_skip_idx = 1;
    static constexpr size_t dateline_upstream_adjcent_sender_channel_skip_idx = 2;

    // Channel size and buffer information
    std::array<std::size_t, builder_config::num_sender_channels> sender_channels_size_bytes = {};
    std::array<std::size_t, builder_config::num_receiver_channels> receiver_channels_size_bytes = {};
    std::array<size_t, builder_config::num_sender_channels> sender_channels_num_buffers = {};
    std::array<size_t, builder_config::num_receiver_channels> receiver_channels_num_buffers = {};

    // Remote channels sizes, used to calculate the remote buffer addresses.
    std::array<std::size_t, builder_config::num_sender_channels> remote_sender_channels_size_bytes = {};
    std::array<std::size_t, builder_config::num_receiver_channels> remote_receiver_channels_size_bytes = {};
    // Remote recv channels number of buffers, use by the local sender channel to check free slots.
    std::array<std::size_t, builder_config::num_sender_channels> remote_sender_channels_num_buffers = {};
    std::array<size_t, builder_config::num_receiver_channels> remote_receiver_channels_num_buffers = {};
    // Downstream sender channels number of buffers, used by the local receiver channel to check free slots.

    std::array<size_t, builder_config::num_sender_channels> sender_channels_base_address = {};
    std::array<size_t, builder_config::num_receiver_channels> receiver_channels_base_address = {};
    // the base addr per remote channel, used by local channels.
    std::array<size_t, builder_config::num_sender_channels> remote_sender_channels_base_address = {};
    std::array<size_t, builder_config::num_receiver_channels> remote_receiver_channels_base_address = {};
};

}  // namespace tt::tt_fabric
