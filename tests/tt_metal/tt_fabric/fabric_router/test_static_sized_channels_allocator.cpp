// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace {

// Use the arch of the silicon under test. TODO: query the fixture's MetalEnv once tests hold one.
FabricEriscDatamoverOptions silicon_options() {
    return FabricEriscDatamoverOptions{.arch = tt::tt_metal::MetalContext::instance().hal().get_arch()};
}

TEST(FabricStaticSizedChannelsAllocatorTest, MeshAssignsStrandedSlotsToLocalWorkerInjection) {
    constexpr size_t channel_buffer_size = 14432;
    constexpr size_t available_space = 360800;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {4, 3, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 1, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};

    for (const auto topology : {Topology::Mesh, Topology::Torus}) {
        const FabricStaticSizedChannelsAllocator allocator(
            topology,
            silicon_options(),
            sender_channels,
            receiver_channels,
            channel_buffer_size,
            available_space,
            memory_regions);

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

TEST(FabricStaticSizedChannelsAllocatorTest, MeshCapsLocalWorkerInjectionDepth) {
    // Worker injection depth stays capped no matter how much buffering space is available.
    constexpr size_t channel_buffer_size = 14432;
    constexpr size_t available_space = channel_buffer_size * 10000;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {4, 3, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 1, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};

    const FabricStaticSizedChannelsAllocator allocator(
        Topology::Torus,
        silicon_options(),
        sender_channels,
        receiver_channels,
        channel_buffer_size,
        available_space,
        memory_regions);

    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 0), MAX_CHANNEL_BUFFER_SLOTS);
}

TEST(FabricStaticSizedChannelsAllocatorTest, RingKeepsUniformChannelDepth) {
    constexpr size_t channel_buffer_size = 14384;
    constexpr size_t available_space = 366656;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {2, 0, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 0, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};

    const FabricStaticSizedChannelsAllocator allocator(
        Topology::Ring,
        silicon_options(),
        sender_channels,
        receiver_channels,
        channel_buffer_size,
        available_space,
        memory_regions);

    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 0), 8);
    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 1), 8);
    EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(0, 0), 8);
}

TEST(FabricStaticSizedChannelsAllocatorTest, ChannelBuffersEndAfterTheLastReceiver) {
    constexpr size_t region_start = 0x10000;

    // The allocator lays out every sender, VC by VC, then every receiver, so the last buffer is the last receiver.
    // It sizes channels from a hardcodedtable of options, deepest first, taking the first whose total fits the space.

    // Mesh: 4 VC0 senders (worker + 3 forwarding), 3 VC1 senders (all forwarding), one receiver per VC. The space holds
    // 360800 / 14432 = 25 slots. The VC0+VC1 mesh options, as (VC0 sender, VC0 receiver, VC1 sender, VC1 receiver), using
    // the depth options from the allocator:
    //   (4, 8, 2, 4): 4*4 + 8 + 3*2 + 4 = 34 slots, too many
    //   (4, 8, 2, 2): 4*4 + 8 + 3*2 + 2 = 32 slots, too many
    //   (2, 4, 2, 2): 4*2 + 4 + 3*2 + 2 = 20 slots, fits
    // A mesh then gives the 5 spare slots to the worker channel (2 + 5 = 7), so all 25 are used: the buffers end
    // at the end of the space, after VC1's receiver.
    {
        constexpr size_t channel_buffer_size = 14432;
        constexpr size_t available_space = 360800;
        const FabricStaticSizedChannelsAllocator allocator(
            Topology::Mesh,
            FabricEriscDatamoverOptions{},
            {4, 3, 0},
            {1, 1, 0},
            channel_buffer_size,
            available_space,
            {{region_start, available_space}});

        EXPECT_EQ(
            allocator.get_channel_buffers_end_address(),
            allocator.get_receiver_channel_base_address(1, 0) +
                allocator.get_receiver_channel_number_of_slots(1, 0) * channel_buffer_size);
        EXPECT_EQ(allocator.get_channel_buffers_end_address(), region_start + available_space);
    }

    // Ring: VC0 only, 2 senders (worker + the one neighbour it forwards for) and 1 receiver. The space holds
    // 366656 / 14384 = 25 slots, rounded down. The ring options, as (sender, receiver), after Blackhole's deeper
    // (32, 32) and (16, 32), which are also too many, using the depth options from the allocator:
    //   (16, 16): 2*16 + 16 = 48 slots, too many
    //   (8, 16):  2*8 + 16 = 32 slots, too many
    //   (8, 8):   2*8 + 8 = 24 slots, fits
    // Only a mesh or torus hands out spare slots, so the buffers end 24 slots in, short of the space.
    {
        constexpr size_t channel_buffer_size = 14384;
        constexpr size_t available_space = 366656;
        const FabricStaticSizedChannelsAllocator allocator(
            Topology::Ring,
            FabricEriscDatamoverOptions{},
            {2, 0, 0},
            {1, 0, 0},
            channel_buffer_size,
            available_space,
            {{region_start, available_space}});

        EXPECT_EQ(
            allocator.get_channel_buffers_end_address(),
            allocator.get_receiver_channel_base_address(0, 0) +
                allocator.get_receiver_channel_number_of_slots(0, 0) * channel_buffer_size);
        EXPECT_EQ(allocator.get_channel_buffers_end_address(), region_start + (2 * 8 + 8) * channel_buffer_size);
    }
}

}  // namespace
}  // namespace tt::tt_fabric
