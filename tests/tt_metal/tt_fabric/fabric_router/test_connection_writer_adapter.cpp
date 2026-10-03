// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"

namespace tt::tt_fabric {
namespace {

SenderWorkerAdapterSpec spec_with_base(size_t buffer_base_address) {
    return SenderWorkerAdapterSpec{
        .edm_buffer_base_addr = buffer_base_address,
        .num_buffers_per_channel = 4,
        .edm_connection_handshake_addr = buffer_base_address + 1,
        .edm_worker_location_info_addr = buffer_base_address + 2,
        .buffer_index_semaphore_id = buffer_base_address + 3};
}

// The adapter takes an allocator but does not read it.
FabricStaticSizedChannelsAllocator make_allocator(
    Topology topology,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& sender_channels,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& receiver_channels) {
    constexpr size_t channel_buffer_size = 14432;
    constexpr size_t available_space = 360800;
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};
    return FabricStaticSizedChannelsAllocator(
        topology,
        FabricEriscDatamoverOptions{},
        sender_channels,
        receiver_channels,
        channel_buffer_size,
        available_space,
        memory_regions);
}

TEST(ConnectionWriterAdapterTest, Mesh2DListsEachConnectionAndTheChannelItsSlotFeeds) {
    auto allocator = make_allocator(Topology::Mesh, {4, 3, 0}, {1, 1, 0});
    StaticSizedChannelConnectionWriterAdapter adapter(allocator, Topology::Mesh, eth_chan_directions::EAST);

    // Added out of slot order. An EAST router's slots are WEST 0, NORTH 1, SOUTH 2, Z 3.
    adapter.add_downstream_connection(spec_with_base(3000), 0, 3, eth_chan_directions::SOUTH, {3, 13}, true);
    adapter.add_downstream_connection(spec_with_base(1000), 0, 1, eth_chan_directions::WEST, {1, 11}, true);
    adapter.add_downstream_connection(spec_with_base(2000), 0, 2, eth_chan_directions::NORTH, {2, 12}, true);
    adapter.add_downstream_connection(spec_with_base(4000), 1, 7, eth_chan_directions::Z, {4, 14}, true);

    // We expect 3 connections for VC0 (SOUTH, WEST, NORTH) (Z not counted because its intermesh so conneted to VC1)
    const auto& vc0 = adapter.get_downstream_connections(0);
    ASSERT_EQ(vc0.size(), 3u);
    EXPECT_EQ(vc0[0].first, eth_chan_directions::SOUTH);
    EXPECT_EQ(vc0[0].second.x, 3u);
    EXPECT_EQ(vc0[0].second.y, 13u);
    EXPECT_EQ(vc0[1].first, eth_chan_directions::WEST);
    EXPECT_EQ(vc0[2].first, eth_chan_directions::NORTH);

    // We expect 1 connection for VC1 (Z intermesh)
    const auto& vc1 = adapter.get_downstream_connections(1);
    ASSERT_EQ(vc1.size(), 1u);
    EXPECT_EQ(vc1[0].first, eth_chan_directions::Z);
    EXPECT_EQ(vc1[0].second.x, 4u);
    EXPECT_EQ(vc1[0].second.y, 14u);

    // Each connection's slot, from the helper the packing uses, is a bit of the packed mask, and together they
    // are the mask.
    for (uint32_t vc = 0; vc <= 1; ++vc) {
        uint32_t mask = 0;
        for (const auto& [direction, noc_xy] : adapter.get_downstream_connections(vc)) {
            mask |= 1u << get_receiver_channel_compact_index(eth_chan_directions::EAST, direction);
        }
        EXPECT_EQ(mask, adapter.get_downstream_edm_mask_for_vc(vc));
    }

    // The channel each slot feeds, unpacked, and empty for an unconnected slot.
    // Params are (vc_idx, slot) for get_downstream_sender_channel_id().
    EXPECT_EQ(adapter.get_downstream_sender_channel_id(0, 0), 1u);
    EXPECT_EQ(adapter.get_downstream_sender_channel_id(0, 1), 2u);
    EXPECT_EQ(adapter.get_downstream_sender_channel_id(0, 2), 3u);
    EXPECT_FALSE(adapter.get_downstream_sender_channel_id(0, 3).has_value());
    EXPECT_EQ(adapter.get_downstream_sender_channel_id(1, 3), 7u);
    EXPECT_FALSE(adapter.get_downstream_sender_channel_id(1, 0).has_value());
}

TEST(ConnectionWriterAdapterTest, OneDimensionalListsItsSingleConnection) {
    auto allocator = make_allocator(Topology::Ring, {2, 0, 0}, {1, 0, 0});
    StaticSizedChannelConnectionWriterAdapter adapter(allocator, Topology::Ring, eth_chan_directions::EAST);

    adapter.add_downstream_connection(spec_with_base(5000), 0, 1, eth_chan_directions::WEST, {5, 6}, false);

    // We expect 1 connection for VC0 (WEST)
    const auto& vc0 = adapter.get_downstream_connections(0);
    ASSERT_EQ(vc0.size(), 1u);
    EXPECT_EQ(vc0[0].first, eth_chan_directions::WEST);
    EXPECT_EQ(vc0[0].second.x, 5u);
    EXPECT_EQ(vc0[0].second.y, 6u);
    EXPECT_EQ(adapter.get_downstream_edm_mask_for_vc(0), 1u);
    EXPECT_FALSE(adapter.get_downstream_sender_channel_id(0, 0).has_value());
}

}  // namespace
}  // namespace tt::tt_fabric
