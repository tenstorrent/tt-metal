// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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

// The slot counts the allocator gave, summed over every channel.
size_t total_slots(
    const FabricStaticSizedChannelsAllocator& allocator,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& sender_channels,
    const std::array<size_t, builder_config::MAX_NUM_VCS>& receiver_channels) {
    size_t slots = 0;
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (size_t channel = 0; channel < sender_channels[vc]; ++channel) {
            slots += allocator.get_sender_channel_number_of_slots(vc, channel);
        }
        for (size_t channel = 0; channel < receiver_channels[vc]; ++channel) {
            slots += allocator.get_receiver_channel_number_of_slots(vc, channel);
        }
    }
    return slots;
}

// Both arches use every whole slot (Blackhole's staged allocator; Wormhole's per-VC table with the spare slots spread
// over VC0's sender channels), give a VC's forwarding channels one count (an upstream router keeps one downstream
// buffer count per VC) and give receivers a power of 2.
TEST(FabricStaticSizedChannelsAllocatorTest, MeshUsesEverySlot) {
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

        // VC0's channel 0 is its worker; the rest of VC0's senders and all of VC1's forward.
        for (size_t channel = 2; channel < sender_channels[0]; ++channel) {
            EXPECT_EQ(
                allocator.get_sender_channel_number_of_slots(0, channel),
                allocator.get_sender_channel_number_of_slots(0, 1));
        }
        for (size_t channel = 1; channel < sender_channels[1]; ++channel) {
            EXPECT_EQ(
                allocator.get_sender_channel_number_of_slots(1, channel),
                allocator.get_sender_channel_number_of_slots(1, 0));
        }
        for (size_t vc = 0; vc < 2; ++vc) {
            const size_t receiver_slots = allocator.get_receiver_channel_number_of_slots(vc, 0);
            EXPECT_TRUE(receiver_slots != 0 && (receiver_slots & (receiver_slots - 1)) == 0) << receiver_slots;
        }
        EXPECT_EQ(total_slots(allocator, sender_channels, receiver_channels), available_space / channel_buffer_size);
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

TEST(FabricStaticSizedChannelsAllocatorTest, RingUsesEverySlot) {
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

    // 25 slots: the receiver 8, the worker and the forwarding channel 8 and 9 between them, whichever gets the odd one.
    EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(0, 0), 8);
    const size_t worker = allocator.get_sender_channel_number_of_slots(0, 0);
    const size_t forwarding = allocator.get_sender_channel_number_of_slots(0, 1);
    EXPECT_EQ(worker + forwarding, 17);
    EXPECT_LE(std::max(worker, forwarding) - std::min(worker, forwarding), 1u);
    EXPECT_EQ(total_slots(allocator, sender_channels, receiver_channels), available_space / channel_buffer_size);
}

}  // namespace
}  // namespace tt::tt_fabric
