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


// A 1D fabric's line VCs (TT_METAL_FABRIC_LINE_VCS): one row for every line VC, an equal share of the 82 slots at
// 4,448 B each: the receiver the largest power of 2 within half of it, forwarding half of the rest, the worker the
// remainder.
TEST(FabricStaticSizedChannelsAllocatorTest, LineVcsOn1DShareTheSlots) {
    constexpr size_t channel_buffer_size = 4448;
    constexpr size_t available_space = 82 * channel_buffer_size;
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};
    struct Row {
        size_t line_vcs, worker, forwarding, receiver;
    };
    for (const auto topology : {Topology::Linear, Topology::Ring}) {
        for (const Row row : {Row{2, 13, 12, 16}, Row{3, 10, 9, 8}}) {
            const size_t vc2 = row.line_vcs == 3 ? 1 : 0;
            const std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {2, 2, 2 * vc2};
            const std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 1, vc2};
            const FabricStaticSizedChannelsAllocator allocator(
                topology,
                silicon_options(),
                sender_channels,
                receiver_channels,
                channel_buffer_size,
                available_space,
                memory_regions);
            for (size_t vc = 0; vc < row.line_vcs; ++vc) {
                EXPECT_EQ(allocator.get_sender_channel_number_of_slots(vc, 0), row.worker) << "VC" << vc;
                EXPECT_EQ(allocator.get_sender_channel_number_of_slots(vc, 1), row.forwarding) << "VC" << vc;
                EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(vc, 0), row.receiver) << "VC" << vc;
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_fabric
