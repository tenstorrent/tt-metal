// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"

namespace tt::tt_fabric {
namespace {

TEST(FabricStaticSizedChannelsAllocatorTest, MeshAssignsStrandedSlotsToLocalWorkerInjection) {
    constexpr size_t channel_buffer_size = 14432;
    constexpr size_t available_space = 365248;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {4, 3, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 1, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};
    const FabricEriscDatamoverOptions options{};

    for (const auto topology : {Topology::Mesh, Topology::Torus}) {
        const FabricStaticSizedChannelsAllocator allocator(
            topology,
            options,
            sender_channels,
            receiver_channels,
            channel_buffer_size,
            available_space,
            memory_regions);

        // The uniform table consumes 20 of the 25 available packet slots. The
        // remaining five deepen only the live local-worker injection channel.
        EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 0), 7);
        for (size_t channel = 1; channel < sender_channels[0]; ++channel) {
            EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, channel), 2);
        }
        for (size_t channel = 0; channel < sender_channels[1]; ++channel) {
            EXPECT_EQ(allocator.get_sender_channel_number_of_slots(1, channel), 2);
        }
        EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(0, 0), 4);
        EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(1, 0), 2);

        size_t allocated_slots = 0;
        for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            for (size_t channel = 0; channel < sender_channels[vc]; ++channel) {
                allocated_slots += allocator.get_sender_channel_number_of_slots(vc, channel);
            }
            for (size_t channel = 0; channel < receiver_channels[vc]; ++channel) {
                allocated_slots += allocator.get_receiver_channel_number_of_slots(vc, channel);
            }
        }
        EXPECT_EQ(allocated_slots, available_space / channel_buffer_size);
    }
}

TEST(FabricStaticSizedChannelsAllocatorTest, RingKeepsUniformChannelDepth) {
    constexpr size_t channel_buffer_size = 14384;
    constexpr size_t available_space = 366656;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {2, 0, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 0, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};

    const FabricStaticSizedChannelsAllocator allocator(
        Topology::Ring,
        FabricEriscDatamoverOptions{},
        sender_channels,
        receiver_channels,
        channel_buffer_size,
        available_space,
        memory_regions);

    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 0), 8);
    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 1), 8);
    EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(0, 0), 8);
}

}  // namespace
}  // namespace tt::tt_fabric
