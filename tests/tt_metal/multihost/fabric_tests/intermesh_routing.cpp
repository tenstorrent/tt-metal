// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "multihost_fabric_fixtures.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests::multihost {

// ========= Data-Movement Tests for 2 Host, 1 T3K bringup machine  =========

TEST_F(InterMesh2x4Fabric2DFixture, RandomizedInterMeshUnicast) {
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::RandomizedInterMeshUnicast(this);
    }
}

TEST_F(InterMesh2x4Fabric2DFixture, MultiMeshEastMulticast) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{0}, 2)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 0)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 3)}, {FabricNodeId(MeshId{1}, 1)}};
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % 4], mcast_start_nodes[i % 2], routing_info, mcast_group_node_ids[i % 2]);
    }
}

TEST_F(InterMesh2x4Fabric2DFixture, MultiMeshSouthMulticast) {
    std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 1)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 1)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 2)}, {FabricNodeId(MeshId{1}, 3)}};
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % 2], mcast_start_nodes[i % 2], routing_info, mcast_group_node_ids[i % 2]);
    }
}

TEST_F(InterMesh2x4Fabric2DFixture, MultiMeshNorthMulticast) {
    std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{0}, 3)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 3)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 0)}, {FabricNodeId(MeshId{1}, 1)}};
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % 2], mcast_start_nodes[i % 2], routing_info, mcast_group_node_ids[i % 2]);
    }
}

// ========= Data-Movement Tests for 2 Loudboxes with Intermesh Connections  =========

TEST_F(InterMeshDual2x4Fabric2DFixture, RandomizedInterMeshUnicast) {
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::RandomizedInterMeshUnicast(this);
    }
}

TEST_F(InterMeshDual2x4Fabric2DFixture, MultiMesh_EW_Multicast) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},  McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 2)},
        {FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 3)},
        {FabricNodeId(MeshId{1}, 4), FabricNodeId(MeshId{1}, 6)},
        {FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 7)}};
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % 4], mcast_start_nodes[i % 4], routing_info, mcast_group_node_ids[i % 4]);
    }
}

TEST_F(InterMeshDual2x4Fabric2DFixture, MultiMesh_EW_MultiHopMulticast) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6)};
    std::vector<std::vector<McastRoutingInfo>> routing_info = {
        {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2}, McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}},
        {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1}, McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2}},
        {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2}, McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}},
        {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1}, McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2}},
    };

    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 3)},
        {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 3)},
        {FabricNodeId(MeshId{1}, 4), FabricNodeId(MeshId{1}, 6), FabricNodeId(MeshId{1}, 7)},
        {FabricNodeId(MeshId{1}, 4), FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 7)}};
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % 4], mcast_start_nodes[i % 4], routing_info[i % 4], mcast_group_node_ids[i % 4]);
    }
}

TEST_F(InterMeshDual2x4Fabric2DFixture, MultiMesh_EW_MulticastWithTurns) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6), FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},  McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 4), FabricNodeId(MeshId{1}, 6)},
        {FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 7)},
        {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 2)},
        {FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 3)}};
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % 4], mcast_start_nodes[i % 4], routing_info, mcast_group_node_ids[i % 4]);
    }
}


}  // namespace fabric_router_tests::multihost
}  // namespace tt::tt_fabric
