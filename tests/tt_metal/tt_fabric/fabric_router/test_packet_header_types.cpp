// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Validates how fabric picks a run's packet header from the header types it offers (packet_header_types.hpp).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <hostdevcommon/fabric_common.h>

#include "tt_metal/fabric/packet_header_types.hpp"

namespace tt::tt_fabric {
namespace {

constexpr uint32_t HOPS_PER_WORD = RoutingFieldsConstants::LowLatency::BASE_HOPS;

// Every route length a 1D header can carry should get an offered header with enough routing words for it, and the
// fewest such words, since a bigger header leaves less room for payload.
TEST(PacketHeaderTypes, LowLatencyHeaderHasTheFewestWordsThatHoldTheRoute) {
    const uint32_t most_hops = (PKT_HDR_1D_EXTENSION_WORDS.back() + 1) * HOPS_PER_WORD;
    for (uint32_t hops = 1; hops <= most_hops; ++hops) {
        const uint32_t words = compute_1d_pkt_hdr_extension_words(hops);
        EXPECT_NE(std::ranges::find(PKT_HDR_1D_EXTENSION_WORDS, words), PKT_HDR_1D_EXTENSION_WORDS.end())
            << hops << " hops";
        EXPECT_GE((words + 1) * HOPS_PER_WORD, hops) << hops << " hops";
        EXPECT_TRUE(words == 0 || words * HOPS_PER_WORD < hops) << hops << " hops";
    }
}

// Every action map a 2D header can carry should get the smallest offered route buffer that holds it.
TEST(PacketHeaderTypes, HybridMeshHeaderHasTheSmallestRouteBufferThatHoldsTheActionMap) {
    const auto largest = static_cast<uint32_t>(PKT_HDR_2D_ROUTE_BUFFER_SIZES.back());
    for (uint32_t bytes = 0; bytes <= largest; ++bytes) {
        const uint32_t size = compute_2d_pkt_hdr_route_buffer_size(bytes);
        EXPECT_NE(
            std::ranges::find(PKT_HDR_2D_ROUTE_BUFFER_SIZES, static_cast<int>(size)),
            PKT_HDR_2D_ROUTE_BUFFER_SIZES.end())
            << bytes << " bytes";
        EXPECT_GE(size, bytes) << bytes << " bytes";
        for (const int smaller : PKT_HDR_2D_ROUTE_BUFFER_SIZES) {
            if (static_cast<uint32_t>(smaller) < size) {
                EXPECT_LT(static_cast<uint32_t>(smaller), bytes) << bytes << " bytes fit in " << smaller;
            }
        }
    }
}

// Each offered header's config should lead the visitor back to that header, so the size FabricContext computes is the
// size of the header the kernels are built with.
TEST(PacketHeaderTypes, VisitPicksTheHeaderEachConfigNames) {
    std::size_t headers = 0;
    for_each_packet_header_type([&]<typename Header>(std::type_identity<Header>, const PacketHeaderConfig& config) {
        ++headers;
        const bool same_header = visit_packet_header_type(
            config, []<typename Visited>(std::type_identity<Visited>) { return std::is_same_v<Visited, Header>; });
        EXPECT_TRUE(same_header) << (config.is_2d ? config.route_buffer_size : config.extension_words);
        const std::size_t size = visit_packet_header_type(
            config, []<typename Visited>(std::type_identity<Visited>) { return sizeof(Visited); });
        EXPECT_EQ(size, sizeof(Header));
    });
    EXPECT_EQ(headers, PKT_HDR_1D_EXTENSION_WORDS.size() + PKT_HDR_2D_ROUTE_BUFFER_SIZES.size());
}

// A config naming a header fabric doesn't offer, such as the disabled 80 B 2D header, should fail rather than size a
// header the kernels can't be built with.
TEST(PacketHeaderTypes, VisitRejectsAHeaderNotOffered) {
    const auto size_of = []<typename Header>(std::type_identity<Header>) { return sizeof(Header); };
    EXPECT_ANY_THROW(visit_packet_header_type(PacketHeaderConfig{.is_2d = true, .route_buffer_size = 20}, size_of));
    EXPECT_ANY_THROW(visit_packet_header_type(
        PacketHeaderConfig{.is_2d = false, .extension_words = static_cast<uint32_t>(PKT_HDR_1D_EXTENSION_WORDS.size())},
        size_of));
}

}  // namespace
}  // namespace tt::tt_fabric
