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

TEST_F(IntermeshSplit2x2FabricFixture, RandomizedInterMeshUnicast) {
    for (uint32_t i = 0; i < 100; i++) {
        multihost_utils::RandomizedInterMeshUnicast(this);
    }
}

TEST_F(IntermeshSplit2x2FabricFixture, MultiMeshEastMulticast_0) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{0}, 2)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 0)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 3)}, {FabricNodeId(MeshId{1}, 1)}};

    const uint32_t num_mcast_reqs = mcast_req_nodes.size();
    const uint32_t num_mcast_groups = mcast_start_nodes.size();

    for (uint32_t i = 0; i < 100; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % num_mcast_reqs], mcast_start_nodes[i % num_mcast_groups], routing_info, mcast_group_node_ids[i % num_mcast_groups]);
    }
}

TEST_F(IntermeshSplit2x2FabricFixture, MultiMeshEastMulticast_1) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 3), FabricNodeId(MeshId{1}, 2)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{0}, 1)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{0}, 2)}, {FabricNodeId(MeshId{0}, 0)}};

    const uint32_t num_mcast_reqs = mcast_req_nodes.size();
    const uint32_t num_mcast_groups = mcast_start_nodes.size();

    for (uint32_t i = 0; i < 100; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % num_mcast_reqs], mcast_start_nodes[i % num_mcast_groups], routing_info, mcast_group_node_ids[i % num_mcast_groups], 1, 0);
    }
}

// ========= Data-Movement Tests for Multi-Process Tests with 4 Ranks/Meshes =========

TEST_F(InterMeshSplit1x2FabricFixture, MultiHopUnicast) {
    // Route traffic between meshes that are not directily adjacent and require an intermediate mesh
    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    constexpr uint32_t num_iterations = 20;
    auto run_send_recv = [&](uint32_t sender_rank, uint32_t recv_rank) {
        for (int i = 0; i < num_iterations; i++) {
            if (*(distributed_context->rank()) == sender_rank) {
                multihost_utils::run_unicast_sender_step(this, tt::tt_metal::distributed::multihost::Rank{recv_rank});
            } else if (*(distributed_context->rank()) == recv_rank) {
                multihost_utils::run_unicast_recv_step(this, tt::tt_metal::distributed::multihost::Rank{sender_rank});
            }
        }
    };
    // Run send/recv on meshes that are diagonally opposite
    run_send_recv(0, 3);
    run_send_recv(1, 2);
    run_send_recv(2, 1);
    run_send_recv(3, 0);
    distributed_context->barrier();
}

// ========= Data-Movement Tests for 2 Loudboxes with Intermesh Connections  =========

TEST_F(InterMeshDual2x4FabricFixture, RandomizedInterMeshUnicast) {
    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::RandomizedInterMeshUnicast(this);
    }
}

TEST_F(InterMeshDual2x4FabricFixture, MultiMeshSouthMulticast_0) {
    std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1}};

    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 5)}, {FabricNodeId(MeshId{1}, 6)}};

    const uint32_t num_mcast_reqs = mcast_req_nodes.size();
    const uint32_t num_mcast_groups = mcast_start_nodes.size();

    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % num_mcast_reqs], mcast_start_nodes[i % num_mcast_groups], routing_info, mcast_group_node_ids[i % num_mcast_groups]);
    }
}

TEST_F(InterMeshDual2x4FabricFixture, MultiMeshNorthMulticast_0) {
    std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1}};

    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 1)}, {FabricNodeId(MeshId{1}, 2)}};

    const uint32_t num_mcast_reqs = mcast_req_nodes.size();
    const uint32_t num_mcast_groups = mcast_start_nodes.size();

    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % num_mcast_reqs], mcast_start_nodes[i % num_mcast_groups], routing_info, mcast_group_node_ids[i % num_mcast_groups]);
    }
}

TEST_F(InterMeshDual2x4FabricFixture, MultiMeshSouthMulticast_1) {
    std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1}};

    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{0}, 5)}, {FabricNodeId(MeshId{0}, 6)}};

    const uint32_t num_mcast_reqs = mcast_req_nodes.size();
    const uint32_t num_mcast_groups = mcast_start_nodes.size();

    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % num_mcast_reqs], mcast_start_nodes[i % num_mcast_groups], routing_info, mcast_group_node_ids[i % num_mcast_groups], 1, 0);
    }
}

TEST_F(InterMeshDual2x4FabricFixture, MultiMeshNorthMulticast_1) {
    std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1}};

    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{0}, 1)}, {FabricNodeId(MeshId{0}, 2)}};

    const uint32_t num_mcast_reqs = mcast_req_nodes.size();
    const uint32_t num_mcast_groups = mcast_start_nodes.size();

    for (uint32_t i = 0; i < 500; i++) {
        multihost_utils::InterMeshLineMcast(
            this, mcast_req_nodes[i % num_mcast_reqs], mcast_start_nodes[i % num_mcast_groups], routing_info, mcast_group_node_ids[i % num_mcast_groups], 1, 0);
    }
}

// ========= Data-Movement Tests for NanoExabox Machines  =========

TEST_F(IntermeshNanoExabox2x4FabricFixture, RandomizedIntermeshUnicastBwd) {
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    constexpr uint32_t num_iterations = 10;
    std::vector<uint32_t> all_ranks = {0, 1, 2, 3, 4, 5, 6, 7};
    for (uint32_t sender_rank : all_ranks) {
        if (*(distributed_context->rank()) == sender_rank) {
            std::vector<uint32_t> recv_node_ranks;
            std::copy_if(
                all_ranks.begin(),
                all_ranks.end(),
                std::back_inserter(recv_node_ranks),
                [&sender_rank](const uint32_t& item) { return item != sender_rank; });

            log_info(tt::LogTest, "{} rank starting unicast to all receivers", sender_rank);
            std::cout << "Num Iterations: " << num_iterations << std::endl;
            for (uint32_t i = 0; i < num_iterations; i++) {
                for (auto recv_rank : recv_node_ranks) {
                    std::cout << "Send from: " << sender_rank << " to " << recv_rank << std::endl;
                    multihost_utils::run_unicast_sender_step(
                        this, tt::tt_metal::distributed::multihost::Rank{recv_rank});
                }
            }
            log_info(tt::LogTest, "{} rank completed unicast to all receivers", sender_rank);
        } else {
            log_info(tt::LogTest, "{} rank processing unicasts", *(distributed_context->rank()));
            std::cout << "Num Iterations: " << num_iterations << std::endl;
            for (uint32_t i = 0; i < num_iterations; i++) {
                std::cout << "Receive into: " << *(distributed_context->rank()) << " from " << sender_rank << std::endl;
                multihost_utils::run_unicast_recv_step(this, tt::tt_metal::distributed::multihost::Rank{sender_rank});
            }
            log_info(tt::LogTest, "{} rank done processing unicasts", *(distributed_context->rank()));
        }
        distributed_context->barrier();
    }
}

TEST_F(IntermeshNanoExabox2x4FabricFixture, RandomizedIntermeshUnicastFwd) {
    return;
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

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
