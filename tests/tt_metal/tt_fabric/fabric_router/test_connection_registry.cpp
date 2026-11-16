// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/connection_registry.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

using namespace tt::tt_fabric;

// ConnectionRegistry container and query coverage.

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

TEST_F(ConnectionRegistryTest, Clear_RemovesEverything) {
    EXPECT_EQ(registry_.size(), 0);
    for (uint32_t i = 0; i < 5; ++i) {
        registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, i));
    }
    ASSERT_EQ(registry_.size(), 5);

    registry_.clear();

    EXPECT_EQ(registry_.size(), 0);
    EXPECT_TRUE(registry_.get_all_connections().empty());
}

TEST_F(ConnectionRegistryTest, FromSource_FiltersByNodeAndDirection) {
    for (uint32_t i = 0; i < 3; ++i) {
        registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, i));
    }
    registry_.record_connection(make_record(1, RoutingDirection::N, 0, 0, RoutingDirection::S, 0, 1));
    registry_.record_connection(make_record(0, RoutingDirection::E, 0, 1, RoutingDirection::W, 0, 1));

    const auto matches = registry_.get_connections_from_source(FabricNodeId(MeshId{0}, 0), RoutingDirection::N);
    EXPECT_EQ(matches.size(), 3);
    for (const auto& c : matches) {
        EXPECT_EQ(c.source_node, FabricNodeId(MeshId{0}, 0));
        EXPECT_EQ(c.source_direction, RoutingDirection::N);
    }
    EXPECT_TRUE(registry_.get_connections_from_source(FabricNodeId(MeshId{0}, 99), RoutingDirection::E).empty());
    EXPECT_TRUE(registry_.get_connections_from_source(FabricNodeId(MeshId{0}, 0), RoutingDirection::W).empty());
}

TEST_F(ConnectionRegistryTest, ToDest_FiltersByNodeAndDirection) {
    for (auto producer : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        registry_.record_connection(make_record(0, producer, 0, 1, RoutingDirection::Z, 0, 1));
    }
    registry_.record_connection(make_record(0, RoutingDirection::N, 0, 2, RoutingDirection::Z, 0, 1));
    registry_.record_connection(make_record(0, RoutingDirection::N, 0, 1, RoutingDirection::S, 0, 1));

    const auto matches = registry_.get_connections_to_dest(FabricNodeId(MeshId{0}, 1), RoutingDirection::Z);
    EXPECT_EQ(matches.size(), 4);
    EXPECT_TRUE(registry_.get_connections_to_dest(FabricNodeId(MeshId{0}, 99), RoutingDirection::Z).empty());
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
