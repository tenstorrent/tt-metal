// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/router_channel_layouts.hpp"
#include "tt_metal/fabric/fabric_init.hpp"
#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_fabric {
namespace {

using Map = RouterChannelLayouts::Map;
using Record = RouterChannelLayouts::Record;

RouterChannelLayout layout_with(
    std::vector<RouterChannelLayout::Channel> vc0_receivers, std::vector<RouterChannelLayout::Channel> vc1_receivers) {
    RouterChannelLayout layout;
    layout.senders[0] = {{0x1000, 8}};
    layout.receivers[0] = std::move(vc0_receivers);
    layout.receivers[1] = std::move(vc1_receivers);
    return layout;
}

// What the all_gather would hand back: each rank's records padded to the largest count.
std::vector<Record> gather(const std::vector<std::vector<Record>>& per_rank, std::vector<uint32_t>& num_records) {
    size_t stride = 0;
    for (const auto& records : per_rank) {
        num_records.push_back(records.size());
        stride = std::max(stride, records.size());
    }
    std::vector<Record> gathered;
    for (auto records : per_rank) {
        records.resize(stride);
        gathered.insert(gathered.end(), records.begin(), records.end());
    }
    return gathered;
}

TEST(RouterChannelLayoutsTest, PackMergeRoundTrip) {
    const FabricNodeId a(MeshId{0}, 0), b(MeshId{0}, 1), c(MeshId{1}, 3);

    // this is rank 1 of three
    Map rank0 = {{{b, 5}, layout_with({{0x2400, 4}}, {})}};
    Map rank1 = {{{a, 4}, layout_with({{0x2000, 4}}, {{0x3000, 2}})}};
    Map rank2 = {
        {{c, 6}, layout_with({{0x2800, 2}, {0x2c00, 2}}, {{0x3400, 1}})},
        {{c, 7}, layout_with({}, {})},
        {{c, 8}, layout_with({{0x2000, 8}}, {})},
    };

    std::vector<uint32_t> num_records;
    const auto gathered = gather(
        {RouterChannelLayouts::pack(rank0), RouterChannelLayouts::pack(rank1), RouterChannelLayouts::pack(rank2)},
        num_records);
    Map merged = rank1;
    RouterChannelLayouts::merge(merged, gathered, num_records, 1);

    // padding records don't show up
    ASSERT_EQ(merged.size(), 5u);

    // local record kept, senders and all
    const auto& own = merged.at({a, 4});
    ASSERT_EQ(own.senders[0].size(), 1u);
    ASSERT_EQ(own.receivers[1].size(), 1u);
    EXPECT_EQ(own.receivers[1][0].base_address, 0x3000u);

    // remote records carry only receivers
    const auto& remote = merged.at({c, 6});
    EXPECT_TRUE(remote.senders[0].empty());
    ASSERT_EQ(remote.receivers[0].size(), 2u);
    EXPECT_EQ(remote.receivers[0][1].base_address, 0x2c00u);
    EXPECT_EQ(remote.receivers[0][1].num_slots, 2u);
    ASSERT_EQ(remote.receivers[1].size(), 1u);
    EXPECT_EQ(remote.receivers[1][0].base_address, 0x3400u);
    EXPECT_EQ(remote.receivers[1][0].num_slots, 1u);

    const auto& padded_peer = merged.at({b, 5});
    ASSERT_EQ(padded_peer.receivers[0].size(), 1u);
    EXPECT_EQ(padded_peer.receivers[0][0].base_address, 0x2400u);
    EXPECT_TRUE(padded_peer.receivers[1].empty());
    EXPECT_TRUE(merged.at({c, 7}).receivers[0].empty());
}

TEST(RouterChannelLayoutsTest, MergeOfNoRecords) {
    Map local;
    RouterChannelLayouts::merge(local, {}, {0, 0}, 0);
    EXPECT_TRUE(local.empty());
}

TEST(RouterChannelLayoutsTest, MergeRejectsARouterBuiltTwice) {
    const FabricNodeId a(MeshId{0}, 0), b(MeshId{0}, 1);
    const Map rank0 = {{{a, 4}, layout_with({{0x2000, 4}}, {})}};
    const Map rank1 = {{{b, 4}, layout_with({{0x2000, 4}}, {})}};
    const Map rank2 = {{{a, 4}, layout_with({{0x2000, 4}}, {})}};
    std::vector<uint32_t> num_records;
    const auto gathered = gather(
        {RouterChannelLayouts::pack(rank0), RouterChannelLayouts::pack(rank1), RouterChannelLayouts::pack(rank2)},
        num_records);

    // against a local key
    Map local = rank0;
    EXPECT_ANY_THROW(RouterChannelLayouts::merge(local, gathered, num_records, 0));
    // between two other ranks
    local = rank1;
    EXPECT_ANY_THROW(RouterChannelLayouts::merge(local, gathered, num_records, 1));
}

TEST(RouterChannelLayoutsTest, PackRejectsTooManyReceivers) {
    std::vector<RouterChannelLayout::Channel> receivers(builder_config::num_max_receiver_channels + 1, {0x2000, 1});
    const Map layouts = {{{FabricNodeId(MeshId{0}, 0), 0}, layout_with(receivers, {})}};
    EXPECT_ANY_THROW(RouterChannelLayouts::pack(layouts));
}

using Header = RouterChannelLayouts::LaunchHeader;

TEST(FabricLaunchGateTest, AllOk) {
    EXPECT_NO_THROW(check_fabric_launch_headers({{5, 0, 3, 2}, {7, 0, 3, 2}}));
    // nobody has routers: fine, the exchange just skips the records
    EXPECT_NO_THROW(check_fabric_launch_headers({{0, 0, 3, 2}, {0, 0, 3, 2}}));
}

TEST(FabricLaunchGateTest, RankFailed) { EXPECT_ANY_THROW(check_fabric_launch_headers({{5, 0, 3, 2}, {0, 1, 3, 2}})); }

// meshes may run different fabric configs (the control plane checks them)
TEST(FabricLaunchGateTest, ConfigsMayDiffer) {
    EXPECT_NO_THROW(check_fabric_launch_headers({{5, 0, 3, 2}, {5, 0, 4, 2}}));
}

TEST(FabricLaunchGateTest, GenerationMismatch) {
    EXPECT_ANY_THROW(check_fabric_launch_headers({{5, 0, 3, 2}, {5, 0, 3, 1}}));
}

// each fabric counts its own launches
TEST(FabricLaunchGateTest, GenerationPerFabric) {
    const auto& world = *tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    std::vector<uint32_t> seen;
    const auto record = [&](const std::vector<Header>& headers) { seen.push_back(headers.at(0).launch_generation); };
    RouterChannelLayouts a, b;
    a.begin_launch();
    a.exchange(world, {}, record);
    a.begin_launch();
    a.exchange(world, {}, record);
    b.begin_launch();
    b.exchange(world, {}, record);
    EXPECT_EQ(seen, (std::vector<uint32_t>{1, 2, 1}));
}

}  // namespace
}  // namespace tt::tt_fabric
