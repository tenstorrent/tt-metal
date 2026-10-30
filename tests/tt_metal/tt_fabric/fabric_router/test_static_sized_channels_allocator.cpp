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

// Liveness: dead sender slots keep their position but share one parking slot, receivers sit at
// the pool base, and rows are fitted against the live counts. Uses the VC0+VC1 table, which is
// identical on both architectures.
TEST(FabricStaticSizedChannelsAllocatorTest, DeadSenderSlotsSharePublicParkingAndFreeTheirBuffering) {
    constexpr size_t slot = 14432;
    constexpr size_t available_space = slot * 25;
    // Family width 5 VC0 / 4 VC1: an N router on a Y-express chip has worker, from-S and from-Z
    // live on VC0 (slots 0, 3, 4) and from-S, from-Z on VC1 (slots 2, 3).
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {5, 4, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 1, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};
    RouterChannelLiveness liveness;
    liveness.sender_live_mask = {0b11001, 0b1100, 0};
    liveness.link_live_sender_counts = {3, 2, 0};

    const FabricStaticSizedChannelsAllocator allocator(
        Topology::Mesh,
        silicon_options(),
        sender_channels,
        receiver_channels,
        slot,
        available_space,
        memory_regions,
        liveness);

    // Row fit against 3 / 2 live: (4,8,2,4)=28 and (4,8,2,2)=26 overflow 25 slots, (2,4,2,2)=16 fits.
    EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(0, 0), 4);
    EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(1, 0), 2);
    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 3), 2);
    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 4), 2);
    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(1, 2), 2);
    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(1, 3), 2);
    // 16 row slots + 1 parking slot leaves 8 spare, all on the worker channel.
    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 0), 2 + 8);

    // Receivers first, then the parking slot, then the live senders in VC/slot order.
    EXPECT_EQ(allocator.get_receiver_channel_base_address(0, 0), 0u);
    EXPECT_EQ(allocator.get_receiver_channel_base_address(1, 0), 4 * slot);
    const size_t parking = 6 * slot;
    for (const auto [vc, channel] : {std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 0}, std::pair{1, 1}}) {
        EXPECT_FALSE(allocator.is_sender_channel_live(vc, channel));
        EXPECT_EQ(allocator.get_sender_channel_number_of_slots(vc, channel), 1);
        EXPECT_EQ(allocator.get_sender_channel_base_address(vc, channel), parking);
    }
    EXPECT_EQ(allocator.get_sender_channel_base_address(0, 0), 7 * slot);
    EXPECT_EQ(allocator.get_sender_channel_base_address(0, 3), 17 * slot);
    EXPECT_EQ(allocator.get_sender_channel_base_address(0, 4), 19 * slot);
    EXPECT_EQ(allocator.get_sender_channel_base_address(1, 2), 21 * slot);
    EXPECT_EQ(allocator.get_sender_channel_base_address(1, 3), 23 * slot);
}

TEST(FabricStaticSizedChannelsAllocatorTest, BoundaryWithoutPassThroughFreesVC1AndTheFitReservesParking) {
    constexpr size_t slot = 14432;
    constexpr size_t available_space = slot * 32;
    // A Z-facing intermesh boundary router without VC1 pass-through: every VC0 slot live, every
    // VC1 sender slot dead, VC1 receiver still allocated. Its Z peer is the same router.
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {5, 4, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 1, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};
    RouterChannelLiveness liveness;
    liveness.sender_live_mask = {0b11111, 0, 0};
    liveness.link_live_sender_counts = {5, 0, 0};

    const FabricStaticSizedChannelsAllocator allocator(
        Topology::Mesh,
        silicon_options(),
        sender_channels,
        receiver_channels,
        slot,
        available_space,
        memory_regions,
        liveness);

    // Against 5 / 0 live, row (4,8,2,4) is exactly 32 slots: it would fill the pool and leave no
    // room for the parking slot, so the fit holds one back and lands on (4,8,2,2) = 30.
    EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(0, 0), 8);
    EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(1, 0), 2);
    for (size_t channel = 1; channel < 5; ++channel) {
        EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, channel), 4);
    }
    // 30 row slots + 1 parking slot leaves 1 spare for the worker.
    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 0), 4 + 1);
    const size_t parking = 10 * slot;
    for (size_t channel = 0; channel < 4; ++channel) {
        EXPECT_FALSE(allocator.is_sender_channel_live(1, channel));
        EXPECT_EQ(allocator.get_sender_channel_number_of_slots(1, channel), 1);
        EXPECT_EQ(allocator.get_sender_channel_base_address(1, channel), parking);
    }
    EXPECT_EQ(allocator.get_sender_channel_base_address(0, 0), 11 * slot);
}

TEST(FabricStaticSizedChannelsAllocatorTest, ReceiverDepthFollowsTheLinkLiveCountsNotTheLocalOnes) {
    constexpr size_t slot = 14432;
    constexpr size_t available_space = slot * 30;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {5, 4, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 1, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};

    // Same local liveness; the peer is a full-width router in one case and a narrow one in the other.
    RouterChannelLiveness narrow_link;
    narrow_link.sender_live_mask = {0b11001, 0b1100, 0};
    narrow_link.link_live_sender_counts = {3, 2, 0};
    RouterChannelLiveness wide_link = narrow_link;
    wide_link.link_live_sender_counts = {5, 4, 0};

    const FabricStaticSizedChannelsAllocator narrow(
        Topology::Mesh,
        silicon_options(),
        sender_channels,
        receiver_channels,
        slot,
        available_space,
        memory_regions,
        narrow_link);
    const FabricStaticSizedChannelsAllocator wide(
        Topology::Mesh,
        silicon_options(),
        sender_channels,
        receiver_channels,
        slot,
        available_space,
        memory_regions,
        wide_link);

    // 3 / 2 live fits (4,8,2,2)=26 of 30; 5 / 4 live only fits (2,4,2,2)=20.
    EXPECT_EQ(narrow.get_receiver_channel_number_of_slots(0, 0), 8);
    EXPECT_EQ(wide.get_receiver_channel_number_of_slots(0, 0), 4);
    // Whatever the row, the receiver region starts at the pool base on both.
    EXPECT_EQ(narrow.get_receiver_channel_base_address(0, 0), 0u);
    EXPECT_EQ(wide.get_receiver_channel_base_address(0, 0), 0u);
}

TEST(FabricStaticSizedChannelsAllocatorTest, AllLiveLayoutPlacesReceiversAtThePoolBase) {
    constexpr size_t slot = 14432;
    constexpr size_t available_space = slot * 25;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {4, 3, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 1, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};

    const FabricStaticSizedChannelsAllocator allocator(
        Topology::Mesh, silicon_options(), sender_channels, receiver_channels, slot, available_space, memory_regions);

    EXPECT_EQ(allocator.get_receiver_channel_base_address(0, 0), 0u);
    EXPECT_EQ(
        allocator.get_receiver_channel_base_address(1, 0), allocator.get_receiver_channel_number_of_slots(0, 0) * slot);
    for (size_t vc = 0; vc < 2; ++vc) {
        for (size_t channel = 0; channel < sender_channels[vc]; ++channel) {
            EXPECT_TRUE(allocator.is_sender_channel_live(vc, channel));
            EXPECT_GE(allocator.get_sender_channel_base_address(vc, channel), 6 * slot);
        }
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

}  // namespace
}  // namespace tt::tt_fabric
