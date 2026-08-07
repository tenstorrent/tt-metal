// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"

namespace tt::tt_fabric {
namespace {

TEST(FabricStaticSizedChannelsAllocatorTest, MeshAssignsStrandedSlotsToLocalWorkerInjection) {
    constexpr size_t channel_buffer_size = 14432;
    constexpr size_t available_space = 360800;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {4, 3, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 1, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};

    for (const auto topology : {Topology::Mesh, Topology::Torus}) {
        const FabricStaticSizedChannelsAllocator allocator(
            topology,
            FabricEriscDatamoverOptions{},
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

// The worker-connected sender channel (VC0 channel 0) may be any depth, but it must fit the 8-bit
// slot-index field of the producer-position handoff word.
//
// History: this channel used to be implicitly restricted to power-of-two depths. The reconnecting
// producer recovered its slot cursor as `persisted_counter % num_buffers`, and the persisted counter
// is a free-running uint32, so the recovery was only correct when num_buffers divided 2^32. Donating
// stranded slots pushed a Blackhole Galaxy config to depth 18, `2^32 % 18 == 4`, and the first wrap
// desynchronised the producer from the router by 4 slots -- silently, with credits still balanced.
//
// That restriction is gone: connection_handoff (edm_fabric_worker_adapters.hpp) now carries the slot
// index explicitly in bits [31:24] rather than re-deriving it, so any depth is safe. What remains is
// the field width -- the index must round-trip through 8 bits, and `num_buffers_per_channel` is a
// uint8 in the connection table besides.
void ExpectWorkerInjectionDepthFitsHandoff(Topology topology, size_t channel_buffer_size, size_t slot_capacity) {
    constexpr size_t kMaxRepresentableDepth = 255;  // connection_handoff slot-index field is 8 bits

    const size_t available_space = channel_buffer_size * slot_capacity;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {4, 0, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 0, 0};
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};

    const FabricStaticSizedChannelsAllocator allocator(
        topology,
        FabricEriscDatamoverOptions{},
        sender_channels,
        receiver_channels,
        channel_buffer_size,
        available_space,
        memory_regions);

    const size_t worker_channel_depth = allocator.get_sender_channel_number_of_slots(0, 0);
    EXPECT_GT(worker_channel_depth, 0u);
    EXPECT_LE(worker_channel_depth, kMaxRepresentableDepth)
        << "Worker-connected sender channel depth " << worker_channel_depth << " (topology "
        << static_cast<int>(topology) << ", slot capacity " << slot_capacity
        << ") does not fit the 8-bit slot-index field of the connection handoff word.";
}

TEST(FabricStaticSizedChannelsAllocatorTest, WorkerInjectionDepthFitsHandoffIndexField) {
    // channel_buffer_size is arbitrary; slot_capacity is what selects the table row and the donation.
    // Capacities 24..38 land on the {4 sender, 8 receiver} row on both Wormhole and Blackhole, so the
    // sweep is architecture independent: base depth 4, donation 0..14, resulting depth 4..18.
    constexpr size_t channel_buffer_size = 4096;
    for (const auto topology : {Topology::Mesh, Topology::Torus}) {
        for (size_t slot_capacity = 24; slot_capacity <= 38; ++slot_capacity) {
            ExpectWorkerInjectionDepthFitsHandoff(topology, channel_buffer_size, slot_capacity);
        }
    }
}

TEST(FabricStaticSizedChannelsAllocatorTest, WorkerInjectionDonationReachesTheInjectionChannel) {
    // Guard that the donation is actually applied to the worker-connected channel and to it alone.
    constexpr size_t channel_buffer_size = 4096;
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels = {4, 0, 0};
    constexpr std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels = {1, 0, 0};

    // Base row is {4, 8} => 4*4 + 8 = 24 slots. A capacity of 28 leaves 4 stranded slots, which takes
    // the injection channel from 4 to 8.
    constexpr size_t slot_capacity = 28;
    constexpr size_t available_space = channel_buffer_size * slot_capacity;
    const std::vector<MemoryRegion> memory_regions = {{0, available_space}};

    const FabricStaticSizedChannelsAllocator allocator(
        Topology::Mesh,
        FabricEriscDatamoverOptions{},
        sender_channels,
        receiver_channels,
        channel_buffer_size,
        available_space,
        memory_regions);

    EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, 0), 8);
    for (size_t channel = 1; channel < sender_channels[0]; ++channel) {
        EXPECT_EQ(allocator.get_sender_channel_number_of_slots(0, channel), 4);
    }
    EXPECT_EQ(allocator.get_receiver_channel_number_of_slots(0, 0), 8);
}

}  // namespace
}  // namespace tt::tt_fabric
