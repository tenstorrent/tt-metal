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

class FabricStaticSizedChannelsAllocator : public FabricChannelAllocator {
    FabricStaticSizedChannelsAllocator(
        tt::tt_fabric::Topology topology,
        const FabricEriscDatamoverOptions& options,
        const std::vector<MemoryRegion>& memory_regions);

    void emit_ct_args(std::vector<uint32_t>& ct_args) const override;

    void configure_buffer_slots_helper(
        tt::tt_fabric::Topology topology,
        const tt::tt_fabric::FabricEriscDatamoverOptions& options,
        std::array<size_t, tt::tt_fabric::builder_config::num_sender_channels>& num_sender_buffer_slots,
        std::array<size_t, tt::tt_fabric::builder_config::num_sender_channels>& num_remote_sender_buffer_slots,
        std::array<size_t, tt::tt_fabric::builder_config::num_receiver_channels>& num_receiver_buffer_slots,
        std::array<size_t, tt::tt_fabric::builder_config::num_receiver_channels>& num_remote_receiver_buffer_slots,
        tt::tt_fabric::eth_chan_directions direction);

    std::vector<std::size_t> sender_channels_size_bytes = {};
    std::vector<std::size_t> receiver_channels_size_bytes = {};
    std::vector<std::size_t> sender_channels_num_buffers = {};
    std::vector<std::size_t> receiver_channels_num_buffers = {};

    // Remote channels sizes, used to calculate the remote buffer addresses.
    std::vector<std::size_t> remote_sender_channels_size_bytes = {};
    std::vector<std::size_t> remote_receiver_channels_size_bytes = {};
    // Remote recv channels number of buffers, use by the local sender channel to check free slots.
    std::vector<std::size_t> remote_sender_channels_num_buffers = {};
    std::vector<std::size_t> remote_receiver_channels_num_buffers = {};
    // Downstream sender channels number of buffers, used by the local receiver channel to check free slots.
    std::vector<std::size_t> downstream_sender_channels_num_buffers = {};

    std::vector<std::size_t> sender_channels_base_address = {};
    std::vector<std::size_t> receiver_channels_base_address = {};
    // the base addr per remote channel, used by local channels.
    std::vector<std::size_t> remote_sender_channels_base_address = {};
    std::vector<std::size_t> remote_receiver_channels_base_address = {};
};

}  // namespace tt::tt_fabric
