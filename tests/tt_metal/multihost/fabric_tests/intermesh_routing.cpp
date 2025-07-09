// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "multihost_fabric_fixtures.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric.hpp>

#include <random>
#include <algorithm>

namespace tt::tt_fabric {
namespace fabric_router_tests::multihost {

// ========= Data-Movement Tests for 2 Host, 1 T3K bringup machine  =========

TEST_F(IntermeshDual2x2FabricFixture, RandomizedInterMeshUnicast) {
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::RandomizedInterMeshUnicast(this);
    }
}

TEST_F(IntermeshDual2x2FabricFixture, MultiMeshEastMulticast) {
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

TEST_F(IntermeshDual2x2FabricFixture, MultiMeshSouthMulticast) {
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

TEST_F(IntermeshDual2x2FabricFixture, MultiMeshNorthMulticast) {
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

TEST_F(InterMeshDual2x4FabricFixture, RandomizedInterMeshUnicast) {
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::RandomizedInterMeshUnicast(this);
    }
}

TEST_F(InterMeshDual2x4FabricFixture, MultiMesh_EW_Multicast) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
    std::vector<FabricNodeId> mcast_start_nodes = {
        FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},
        McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}};
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

TEST_F(InterMeshDual2x4FabricFixture, MultiMesh_EW_MultiHopMulticast) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
    std::vector<FabricNodeId> mcast_start_nodes = {
        FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6)};
    std::vector<std::vector<McastRoutingInfo>> routing_info = {
        {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2},
         McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}},
        {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},
         McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2}},
        {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2},
         McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}},
        {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},
         McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2}},
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

TEST_F(InterMeshDual2x4FabricFixture, MultiMesh_EW_MulticastWithTurns) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
    std::vector<FabricNodeId> mcast_start_nodes = {
        FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6), FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},
        McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}};
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

// ========= Data-Movement Tests for NanoExabox Machines  =========

TEST_F(IntermeshNanoExaboxFabricFixture, RandomizedIntermeshUnicastBwd) {
    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    constexpr uint32_t sender_rank = 1;
    constexpr uint32_t num_iterations = 100;

    if (*(distributed_context->rank()) == sender_rank) {
        std::vector<uint32_t> recv_node_ranks = {0, 2, 3, 4};
        log_info(tt::LogTest, "{} rank starting unicast to all receivers", sender_rank);
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (auto recv_rank : recv_node_ranks) {
                multihost_utils::run_unicast_sender_step(this, tt::tt_metal::distributed::multihost::Rank{recv_rank});
            }
        }
        log_info(tt::LogTest, "{} rank completed unicast to all receivers", sender_rank);
    } else {
        log_info(tt::LogTest, "{} rank processing unicasts", *(distributed_context->rank()));
        for (uint32_t i = 0; i < num_iterations; i++) {
            multihost_utils::run_unicast_recv_step(this, tt::tt_metal::distributed::multihost::Rank{sender_rank});
        }
        log_info(tt::LogTest, "{} rank done processing unicasts", *(distributed_context->rank()));
    }
    distributed_context->barrier();
}

TEST_F(IntermeshNanoExaboxFabricFixture, RandomizedIntermeshUnicastFwd) {
    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    constexpr uint32_t recv_rank = 1;
    constexpr uint32_t num_iterations = 100;

    if (*(distributed_context->rank()) == recv_rank) {
        std::vector<uint32_t> sender_node_ranks = {0, 2, 3, 4};
        log_info(tt::LogTest, "{} rank starting processing unicasts from all senders", recv_rank);
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (auto sender_rank : sender_node_ranks) {
                multihost_utils::run_unicast_recv_step(this, tt::tt_metal::distributed::multihost::Rank{sender_rank});
            }
        }
        log_info(tt::LogTest, "{} rank completed processing unicasts from all senders", recv_rank);
    } else {
        log_info(tt::LogTest, "{} rank starting unicast to receiver", *(distributed_context->rank()));
        for (uint32_t i = 0; i < num_iterations; i++) {
            multihost_utils::run_unicast_sender_step(this, tt::tt_metal::distributed::multihost::Rank{recv_rank});
        }
        log_info(tt::LogTest, "{} rank completed unicast to receiver", *(distributed_context->rank()));
    }
    distributed_context->barrier();
}

}  // namespace fabric_router_tests::multihost
}  // namespace tt::tt_fabric
