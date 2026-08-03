// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/connection_registry.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

using namespace tt::tt_fabric;

/**
 * ConnectionRegistry Tests
 *
 * The registry is a plain container for RouterConnectionRecord: recording, field round-trip,
 * size, clear, and the four query axes (from-source, to-dest, by-source-node, by-dest-node).
 * Records carry no connection type -- a local turn is identified by its source/destination
 * directions and channels, so there is nothing type-shaped to query.
 *
 * What the records MEAN (which turns get wired, on which VC, in which sender slot) is the
 * connection maps' and the establishment pass's business; see
 * test_router_turn_set.cpp and test_connection_establishment.cpp.
 */

class ConnectionRegistryTest : public ::testing::Test {
protected:
    ConnectionRegistry registry_;

    RouterConnectionRecord make_record(
        uint32_t source_chip,
        RoutingDirection source_dir,
        uint32_t source_vc,
        uint32_t dest_chip,
        RoutingDirection dest_dir,
        uint32_t dest_vc,
        uint32_t dest_slot) {
        return RouterConnectionRecord{
            .source_node = FabricNodeId(MeshId{0}, source_chip),
            .source_direction = source_dir,
            .source_eth_chan = 0,
            .source_vc = source_vc,
            .source_receiver_channel = 0,
            .dest_node = FabricNodeId(MeshId{0}, dest_chip),
            .dest_direction = dest_dir,
            .dest_eth_chan = 0,
            .dest_vc = dest_vc,
            .dest_sender_channel = dest_slot,
        };
    }
};

TEST_F(ConnectionRegistryTest, EmptyRegistry_HasZeroSize) {
    EXPECT_EQ(registry_.size(), 0);
    EXPECT_TRUE(registry_.get_all_connections().empty());
}

TEST_F(ConnectionRegistryTest, RecordConnection_RoundTripsAllFields) {
    registry_.record_connection(RouterConnectionRecord{
        .source_node = FabricNodeId(MeshId{0}, 1),
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 3,
        .source_vc = 1,
        .source_receiver_channel = 2,
        .dest_node = FabricNodeId(MeshId{0}, 2),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 4,
        .dest_vc = 1,
        .dest_sender_channel = 5,
    });

    ASSERT_EQ(registry_.size(), 1);
    const auto& c = registry_.get_all_connections().front();
    EXPECT_EQ(c.source_node, FabricNodeId(MeshId{0}, 1));
    EXPECT_EQ(c.source_direction, RoutingDirection::N);
    EXPECT_EQ(c.source_eth_chan, 3);
    EXPECT_EQ(c.source_vc, 1);
    EXPECT_EQ(c.source_receiver_channel, 2);
    EXPECT_EQ(c.dest_node, FabricNodeId(MeshId{0}, 2));
    EXPECT_EQ(c.dest_direction, RoutingDirection::S);
    EXPECT_EQ(c.dest_eth_chan, 4);
    EXPECT_EQ(c.dest_vc, 1);
    EXPECT_EQ(c.dest_sender_channel, 5);
}

TEST_F(ConnectionRegistryTest, RecordMultiple_SizeTracks) {
    for (uint32_t i = 0; i < 3; ++i) {
        registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, i));
    }
    EXPECT_EQ(registry_.size(), 3);
    EXPECT_EQ(registry_.get_all_connections().size(), 3);
}

TEST_F(ConnectionRegistryTest, Clear_RemovesEverything) {
    for (uint32_t i = 0; i < 5; ++i) {
        registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, i));
    }
    ASSERT_EQ(registry_.size(), 5);

    registry_.clear();

    EXPECT_EQ(registry_.size(), 0);
    EXPECT_TRUE(registry_.get_all_connections().empty());
}

TEST_F(ConnectionRegistryTest, FromSource_SingleMatch) {
    registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, 1));
    registry_.record_connection(make_record(0, RoutingDirection::E, 0, 1, RoutingDirection::W, 0, 1));

    const auto matches = registry_.get_connections_from_source(FabricNodeId(MeshId{0}, 0), RoutingDirection::N);
    ASSERT_EQ(matches.size(), 1);
    EXPECT_EQ(matches[0].source_direction, RoutingDirection::N);
    EXPECT_EQ(matches[0].dest_direction, RoutingDirection::S);
}

TEST_F(ConnectionRegistryTest, FromSource_MultipleMatches) {
    for (uint32_t i = 0; i < 3; ++i) {
        registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, i));
    }
    registry_.record_connection(make_record(1, RoutingDirection::N, 0, 0, RoutingDirection::S, 0, 1));

    const auto matches = registry_.get_connections_from_source(FabricNodeId(MeshId{0}, 0), RoutingDirection::N);
    EXPECT_EQ(matches.size(), 3);
    for (const auto& c : matches) {
        EXPECT_EQ(c.source_node, FabricNodeId(MeshId{0}, 0));
        EXPECT_EQ(c.source_direction, RoutingDirection::N);
    }
}

TEST_F(ConnectionRegistryTest, FromSource_NoMatches) {
    registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, 1));

    EXPECT_TRUE(registry_.get_connections_from_source(FabricNodeId(MeshId{0}, 99), RoutingDirection::E).empty());
    EXPECT_TRUE(registry_.get_connections_from_source(FabricNodeId(MeshId{0}, 0), RoutingDirection::W).empty());
}

TEST_F(ConnectionRegistryTest, ToDest_SingleMatch) {
    registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, 1));
    registry_.record_connection(make_record(0, RoutingDirection::N, 0, 2, RoutingDirection::W, 0, 1));

    const auto matches = registry_.get_connections_to_dest(FabricNodeId(MeshId{0}, 1), RoutingDirection::S);
    ASSERT_EQ(matches.size(), 1);
    EXPECT_EQ(matches[0].dest_node, FabricNodeId(MeshId{0}, 1));
    EXPECT_EQ(matches[0].dest_direction, RoutingDirection::S);
}

TEST_F(ConnectionRegistryTest, ToDest_MultipleMatches) {
    // Four producers feeding one router, one per direction -- the full fan-in of a mesh-facing
    // receiver.
    for (auto producer : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        registry_.record_connection(make_record(0, producer, 0, 1, RoutingDirection::Z, 0, 1));
    }

    const auto matches = registry_.get_connections_to_dest(FabricNodeId(MeshId{0}, 1), RoutingDirection::Z);
    EXPECT_EQ(matches.size(), 4);
}

TEST_F(ConnectionRegistryTest, BySourceNode_CoversAllRoutersOnTheNode) {
    // Two routers on node 0 (N-facing and E-facing) plus an unrelated record elsewhere.
    registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, 1));
    registry_.record_connection(make_record(0, RoutingDirection::E, 0, 1, RoutingDirection::W, 0, 1));
    registry_.record_connection(make_record(1, RoutingDirection::S, 0, 0, RoutingDirection::N, 0, 1));

    const auto matches = registry_.get_connections_by_source_node(FabricNodeId(MeshId{0}, 0));
    EXPECT_EQ(matches.size(), 2);
    for (const auto& c : matches) {
        EXPECT_EQ(c.source_node, FabricNodeId(MeshId{0}, 0));
    }
}

TEST_F(ConnectionRegistryTest, ByDestNode_CoversAllRoutersOnTheNode) {
    registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, 1));
    registry_.record_connection(make_record(2, RoutingDirection::E, 1, 1, RoutingDirection::Z, 1, 3));
    registry_.record_connection(make_record(1, RoutingDirection::S, 0, 0, RoutingDirection::N, 0, 1));

    const auto matches = registry_.get_connections_by_dest_node(FabricNodeId(MeshId{0}, 1));
    EXPECT_EQ(matches.size(), 2);
    for (const auto& c : matches) {
        EXPECT_EQ(c.dest_node, FabricNodeId(MeshId{0}, 1));
    }
}
