// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/router_channel_layouts.hpp"

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

    Map rank0 = {{{a, 4}, layout_with({{0x2000, 4}}, {{0x3000, 2}})}};
    Map rank1 = {
        {{b, 5}, layout_with({{0x2400, 4}}, {})},
        {{c, 6}, layout_with({{0x2800, 2}, {0x2c00, 2}}, {{0x3400, 1}})},
        // rank 0's own router, as rank 1 would see it; rank 0's copy has to win
        {{a, 4}, layout_with({{0xdead, 1}}, {})},
    };

    std::vector<uint32_t> num_records;
    const auto gathered = gather({RouterChannelLayouts::pack(rank0), RouterChannelLayouts::pack(rank1)}, num_records);
    Map merged = rank0;
    RouterChannelLayouts::merge(merged, gathered, num_records);

    ASSERT_EQ(merged.size(), 3u);

    // local record kept, senders and all
    const auto& own = merged.at({a, 4});
    ASSERT_EQ(own.senders[0].size(), 1u);
    ASSERT_EQ(own.receivers[0].size(), 1u);
    EXPECT_EQ(own.receivers[0][0].base_address, 0x2000u);

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
}

TEST(RouterChannelLayoutsTest, PackRejectsTooManyReceivers) {
    std::vector<RouterChannelLayout::Channel> receivers(builder_config::num_max_receiver_channels + 1, {0x2000, 1});
    const Map layouts = {{{FabricNodeId(MeshId{0}, 0), 0}, layout_with(receivers, {})}};
    EXPECT_ANY_THROW(RouterChannelLayouts::pack(layouts));
}

}  // namespace
}  // namespace tt::tt_fabric
